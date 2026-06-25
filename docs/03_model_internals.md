# 3. Model Internals — a line-by-line walk

This file maps the math from [file 2](02_concepts_and_math.md) onto the actual code in
`SRNN/`. Read it with the source open. Shapes are written as `(B, T, …)` where `B` =
batch of trials, `T` = time steps (100), `K = num_tv = 2`, `H = hidden_shape = 8`,
`D = input_shape = 20`.

---

## 3.1 `inference_network.py` — `RNNInfer` (the encoder for `h`)

This network implements `q(h | y)` — read the data, output the continuous latent.

**`PositionalEncoding`** (lines ~9–31): standard sinusoidal positional encoding. It adds
a fixed pattern of sines and cosines to each time step's vector so the Transformer can
tell time steps apart (self-attention is otherwise order-blind). Pure bookkeeping, no
trainable weights.

**`RNNInfer.__init__`** (lines ~34–66): builds the encoder.
- `in_proj`: `Linear(D, H)` — projects the 20 neurons up/down to the latent width `H = 8`.
- `pos_enc`: the positional encoding above.
- `encoder`: a `nn.TransformerEncoder` of `n_layers = 2` standard Transformer layers
  (`n_heads = 1` attention head, feed-forward width `4·H`, pre-norm, ReLU). **This is
  why "RNNInfer" is a misnomer — it is a Transformer.** The class comment about a
  "bidirectional RNN" is leftover; the Transformer is inherently bidirectional (every
  step sees every other step), which suits *non-causal* inference (using past *and*
  future to estimate the present `h_t`).
- `mean_proj`: a small MLP `H → 64 → H` that produces the posterior mean.

**`RNNInfer.forward`** (lines ~70–91):
```python
x = self.in_proj(y_train)            # (B, T, H)  project observations
x = self.pos_enc(x)                  # add positional info
bi_output = self.encoder(x, ...)     # (B, T, H)  Transformer encoding
mean_out  = self.mean_proj(bi_output)# (B, T, H)  posterior mean of h

covariance_matrix = 1e-4 * I_H       # fixed tiny covariance
std = covariance_matrix.diag()       # (H,)
ep  = torch.randn_like(mean_out)     # ε ~ N(0, I)
sampled_h = mean_out + ep * std      # REPARAMETERIZATION TRICK → a sample of h

infer_dist = MultivariateNormal(mean_out, covariance_matrix)  # used for entropy
return infer_dist, sampled_h, mean_out
```
- `mean_out` is `μ` of `q`. `sampled_h` is one draw of `h` from `q`, written so gradients
  flow through `μ` (file 2, §2.4).
- `infer_dist` is the Gaussian object itself; later the loss calls `.entropy()` on it —
  that's the entropy term of the ELBO.
- The variance is **fixed** at `1e-4` (not learned), so `q` is a sharp blob around
  `mean_out`. In practice `h ≈ mean_out`.

---

## 3.2 `model_srnn.py` — the generative model

### `Emission` (lines ~33–49)
The observation MLP: `Linear(H,32) → ReLU → Linear(32,64) → ReLU → Linear(64,D)`. Maps a
latent vector `h_t` (8-dim) to the mean of the 20 neurons. No output nonlinearity (neural
activity can be any real number).

### `_diag_mvn_log_prob` (lines ~19–29)
A fast hand-written log-density of a Gaussian with covariance `var·I`:
```
log N(x; mean, var·I) = −½ [ D·log(2π) + D·log(var) + ‖x − mean‖² / var ]
```
This is just the multivariate-normal log-density specialized to a diagonal, equal
variance — used everywhere instead of building a `MultivariateNormal` object (much
faster). `‖x − mean‖²` is the squared distance between the value and the mean.

### `Model.__init__` (lines ~54–65)
```python
self.rnns        = ModuleList([ nn.RNN(D, H) for _ in range(K) ])  # one RNN per mode
self.emission    = Emission(D, H)                                  # shared emission
self.transitions = nn.RNNCell(H, K*K)                              # transition network
self.initials    = nn.Parameter(randn(K))                         # initial-state logits
```
- `self.rnns` — the `K` per-mode dynamics networks (file 2, §2.2c/§2.3 Step 1).
- `self.transitions` — an `RNNCell` mapping a latent `h` (size `H`) to `K·K = 4` numbers,
  later reshaped into a `K×K` transition matrix (Step 2). Its own recurrent state is
  fed zeros each call (`h0_transitions`), so it acts as a per-step function of `h_{t-1}`.
- `self.initials` — the trainable logits behind `π = softmax(initials)` (Step 0).
- `neural_private_shape` is stored but **not used** in `forward` (vestigial, see file 1).

### `Model.forward` (lines ~67–127) — the per-step probability factory
Inputs: `x_input` (zeros), `y_train` (observations), `sampled_h` and `neural_final`
(both = the inferred latent `h` from `RNNInfer`), `device`.

It pre-allocates the four log-probability tensors and fills them in a loop over time:

```python
prob_all_s : (B, T, K, K)   # log transition matrices
prob_all_h : (B, T, K)      # log p(h_t | h_{t-1}, z_t = i), per mode i
prob_all_y : (B, T)         # log p(y_t | h_t)
prob_initial : (B, K)       # log π, broadcast over the batch
```

**Initial probabilities** (lines ~74–76):
```python
log_initial  = self.initials - logsumexp(self.initials, 0)   # = log softmax(initials)
prob_initial = log_initial.expand(B, K)
```

**The temporal loop `for j in range(T)`** (lines ~91–116):

- *At `j == 0`* (lines ~93–99): the transition is set to the identity (no switching into
  the first step), and `prob_all_h[:,0,i]` is the log-density of `h_0` under its own
  mean — the same constant for every mode (uninformative at `t = 0`, which is correct;
  there's no prior dynamics to score yet).

- *At `j > 0`* (lines ~101–112):
  ```python
  trans = transitions(sampled_h[:,j-1,:], h0_transitions).reshape(B, K, K)
  prob_all_s[:,j] = trans - logsumexp(trans, axis=1)[:,None,:]   # normalize columns
  # run every mode's RNN one step from h_{j-1}, score how well it predicts h_j:
  for i in range(K):
      x_out, _ = self.rnns[i](x_slice, h_prev)                  # mode i's prediction of h_j
      prob_all_h[:,j,i] = _diag_mvn_log_prob(h_target=h_j, x_out, h_var)
  ```
  So `prob_all_h[:,j,i]` answers "if mode `i` were active, how well would its RNN explain
  the observed jump `h_{j-1} → h_j`?" The transition matrix is recomputed from `h_{j-1}`
  (recurrent switching).

- *Every step* (lines ~115–116): score the emission —
  ```python
  emission_mean   = self.emission(neural_final[:, j:j+1, :])     # MLP(h_j) → 20-dim mean
  prob_all_y[:,j] = _diag_mvn_log_prob(y_train[:,j,:], emission_mean, y_var)
  ```
  i.e. "how well does the latent at time `j` reconstruct the real neurons at time `j`?"

**Then** (lines ~123–125) it hands these four tensors to Baum–Welch:
```python
forward_prob  = baum_welch.dis_forward_pass(...)
backward_prob = baum_welch.dis_backward_pass(...)
gamma1, delta1 = baum_welch.get_gamma(forward_prob, backward_prob, ...)
return prob_initial, prob_all_s, prob_all_h, prob_all_y, gamma1, delta1, forward_prob, backward_prob
```

### `Model.get_posterior_lk` (lines ~129–133)
```python
posterior_lk = forward_prob + backward_prob
posterior_lk = posterior_lk - logsumexp(posterior_lk, axis=2)[:,:,None]   # normalize over modes
```
This is `log P(z_t = i | all data)` — the per-time **state posterior** (`γ` for all `t`).
It is what the supervised cross-entropy compares to the labels, and what `plot.py`
`argmax`es to read off the inferred regime sequence.

---

## 3.3 `baum_welch.py` — exact forward–backward over modes

> Theory recap in [file 2, §2.6](02_concepts_and_math.md#26-baumwelch-forwardbackward).
> Everything is log-space; `logsumexp` is "add probabilities" in log-space.

**`dis_forward_pass`** (lines ~9–22): the forward recursion for `α_t`.
- `t = 0`: `α_0(i) = π_i + log p(h_0|i) + log p(y_0)`, then normalized.
- `t > 0`: `α_t(j) = log Σ_i exp( α_{t-1}(i) + A_t[j,i] ) + log p(h_t|j) + log p(y_t)`,
  normalized each step (the normalization keeps numbers bounded — a standard trick).

**`dis_backward_pass`** (lines ~24–36): the backward recursion for `β_t`.
- `t = T−1`: `β_{T-1}(i) = 0` (log 1 — nothing after the end).
- `t < T−1`: `β_t(i) = log Σ_j exp( β_{t+1}(j) + A_{t+1}[j,i] + log p(h_{t+1}|j) +
  log p(y_{t+1}) )`, normalized.

**`get_gamma`** (lines ~38–54):
- `gamma1` = `α_0 + β_0`, normalized → posterior over the *first* mode, `(B, K)`.
- `delta1` = `α_{t} + A_{t+1} + β_{t+1} + log p(h_{t+1}) + log p(y_{t+1})`, normalized over
  the `K×K` pairs → posterior over *consecutive mode pairs*, `(B, T−1, K, K)`.

These `gamma1`/`delta1` are the responsibilities that weight the loss next.

---

## 3.4 `loss_function.py` — building the objective

**`get_loss`** (lines ~8–23) computes the **expected complete-data log-likelihood**, the
fit term of the ELBO with `z` marginalized via the responsibilities:
```python
t1 = Σ  exp(gamma1) · ( prob_ini + prob_all_h[:,0] + prob_all_y[:,0] )      # the t = 0 piece
t2 = Σ  exp(delta1) · ( prob_all_s[:,1:] + prob_all_h[:,1:] + prob_all_y[:,1:] )  # the t > 0 sum
```
In words: weight each mode (or mode-pair) by *how responsible Baum–Welch says it is*
(`exp(gamma)`, `exp(delta)`), times the *log-probability that mode assigns* to the
initial state / transition / dynamics / emission. Summing these is exactly the standard
HMM "expected complete-data log-likelihood." (The code comment points to "Equation 15 of
our paper.")

**`get_cross_entropy`** (lines ~25–27):
```python
return (pos * pri).sum()       # pos = state log-posterior, pri = one-hot true labels
```
Because `pri` is one-hot, this picks out the model's **log-probability of the true mode**
at each time step and sums them — i.e. `Σ_t log P(z_t = true_label_t | y)`. Maximizing it
forces the inferred regimes toward the supplied labels. **This is the supervised signal.**

---

## 3.5 `train.py` — the training loop and evaluation

### `train_` (lines ~17–115)
1. **One-hot the labels** (`initialization.one_hot`) into `pp`, shape `(trials, T, K)` —
   the supervision target.
2. **DataLoader** over `(X_train, y_train, pp)` with the configured `batch_size`, shuffled.
3. **For each epoch, for each batch:**
   ```python
   infer_dist, inferred_h, mean_out = rnninfer(y_train_batch)      # q(h|y), sample h
   prob_ini, prob_all_s, prob_all_h, prob_all_y, gamma1, delta1, fwp, bwp \
       = model(X_train_batch, y_train_batch, inferred_h, inferred_h, device)
   t1, t2     = loss_function.get_loss(gamma1, delta1, prob_ini, prob_all_s, prob_all_h, prob_all_y)
   posterior  = model.get_posterior_lk(fwp, bwp)
   cross_en   = loss_function.get_cross_entropy(posterior, pp_batch)
   loss_all   = -(t1.mean() + t2.mean() + coef_cross*cross_en + infer_dist.entropy().mean())
   loss_all.backward()
   clip_grad_norm_(...)        # stability
   optimizer.step(); optimizer_rnn.step()    # update BOTH networks
   ```
   This is the full objective from [file 2, §2.5](02_concepts_and_math.md#25-the-training-objective--the-elbo-plus-a-supervised-twist).
4. **Schedulers** step each epoch; progress prints every 100 epochs with an ETA
   (`utils.compute_time`).
5. **Autosave** every epoch to `result/autosave_…fold_0.pt` (crash recovery), storing
   weights, test data, metric arrays, and the test state-posterior at that epoch.

> Note: `inferred_h` is passed **twice** to `model(...)` — once as `sampled_h` (used to
> score dynamics and transitions) and once as `neural_final` (fed to the emission). In
> the larger multi-region model these were two different latents; here they are the same.

### `eval_` (lines ~118–138)
Puts the networks in eval mode, runs `RNNInfer` then `Model` once (no gradients), and
returns:
- `y_pred_test` — sampled reconstruction of the neurons (from the emission distribution),
- `pos_test` — the test **state log-posterior** `log P(z_t | y)`, `(trials, T, K)`,
- plus the latent (returned several times to fill a fixed-length tuple — the extra slots
  are placeholders; only positions 0, 2 are really used downstream).

### `compute_metric` / `compute_error` (lines ~140–149)
Helpers: mean-squared error of the reconstruction, and `1 − balanced_accuracy` of the
inferred vs. true states. (Not all are wired into the saved metrics; the live arrays
`mse_all`, `error_all` stay at their initialized values in this version — the meaningful
saved signal is `loss_train` and `pos_test_all`.)

---

## 3.6 `generative_check.py` — "dreaming" from the model (optional)

`run(...)` does a **purely generative** rollout to inspect what the dynamics learned:
1. Infer `h` from the test data, and take the inferred mode sequence `pos_infer` (argmax
   of the posterior).
2. Starting from `h_0`, step forward using **only** `self.rnns[mode_t]` — i.e. let the
   model's own dynamics generate the latent trajectory, *not* the inference network.
3. Push that generated latent through the emission MLP to get generated neural data.

Comparing this "dreamed" `y` to the real `y` tests whether the learned per-mode RNNs
actually captured the dynamics (a stronger check than reconstruction, which leans on the
inference network). It is a utility — not called by `train.py` or `plot.py` by default.

---

## 3.7 Cross-references to the rest of the docs
- Want the *why* behind any of this? → [02_concepts_and_math.md](02_concepts_and_math.md).
- Want to *run* it? → [04_usage_guide.md](04_usage_guide.md).
- Lost on a file? → [01_project_layout.md](01_project_layout.md).
