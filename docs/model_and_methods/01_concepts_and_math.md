# 2. Concepts and Math (with everything defined)

This file builds the model up from nothing. Every term is defined the first time it
appears. If you already know a term, skim past its definition box.

---

## 2.1 The setup: what problem are we solving?

We observe a multi-channel time series. In this project that is **neural activity**:
20 channels ("neurons"), recorded over 100 time steps, in each of 50 trials.

We believe two things about this data:

1. **There is a hidden continuous state** driving the observations. We never see it
   directly; we only see the 20 noisy neurons. Call this hidden state `h_t` (a vector
   at each time `t`). Think of it as a compressed, denoised summary of "what the brain
   is doing right now."

2. **The system switches between a few discrete regimes.** At each moment the system is
   in one of `K` modes (here `K = 2`). Within a mode, the hidden state evolves according
   to that mode's own rule. Call the active mode at time `t` the discrete state `z_t`.

So at every time step there are **two** hidden variables: a continuous one `h_t` and a
discrete one `z_t`. Models with this structure are called **switching state-space models**.

> **State-space model (SSM):** a model with a hidden ("latent") state that evolves over
> time, plus an observation rule that produces what you actually measure from that
> hidden state. "Latent" just means *not directly observed* — you infer it.

> **Discrete vs. continuous state:** `z_t` is *discrete* — it takes one of finitely many
> values (mode 0 or mode 1). `h_t` is *continuous* — it is a vector of real numbers (here
> 8-dimensional, because `hidden_shape = 8`).

---

## 2.2 Three classical building blocks

The SRNN is a fusion of three classic ideas. Understanding each in isolation makes the
combination obvious.

### (a) The Hidden Markov Model (HMM) — handles the *discrete* switching
An **HMM** is a model where a discrete state `z_t` hops between `K` values over time,
and each state emits an observation. Two ingredients:

- **Transition probabilities** `A[i, j] = P(z_t = j | z_{t-1} = i)` — the chance of
  moving from mode `i` to mode `j`. Collected into a `K × K` **transition matrix**.
- **Emission/observation probabilities** — how likely the data is, given the current mode.

> **"Markov":** the future depends on the present only, not the full past.
> `P(z_t | z_{t-1}, z_{t-2}, …) = P(z_t | z_{t-1})`. This "memoryless" assumption is what
> makes the math tractable.

The standard algorithm for HMMs is **forward–backward** (a.k.a. the E-step of
**Baum–Welch**), which computes, for every time step, the probability of being in each
mode given *all* the data. The SRNN uses exactly this — see [§2.6](#26-baumwelch-forwardbackward).

### (b) The Linear Dynamical System (LDS) / Kalman filter — handles the *continuous* state
An **LDS** says the continuous hidden state evolves *linearly*:
`h_t = A·h_{t-1} + noise`, and the observation is `y_t = C·h_t + noise`, where `A` and
`C` are matrices. This is the model behind the **Kalman filter**. It is great when the
real dynamics are roughly linear, and limited when they are not.

### (c) The Recurrent Neural Network (RNN) — a *nonlinear* dynamics engine
An **RNN** is a neural network for sequences. It keeps a hidden state and updates it at
each step with a learned nonlinear function:
`h_t = f(x_t, h_{t-1})`, where `f` is a small neural network with trainable weights.
"Recurrent" = it feeds its own previous output back in as input. RNNs can represent
*nonlinear* dynamics that an LDS cannot.

> In this project the RNNs are PyTorch's default `nn.RNN` (a "vanilla"/Elman RNN):
> `h_t = tanh(W_x·x_t + W_h·h_{t-1} + b)`. The input `x_t` here is all zeros (the model
> is "input-free"), so effectively `h_t = tanh(W_h·h_{t-1} + b)` — a learned nonlinear
> autonomous flow.

### From SLDS to SRNN
Combine (a) + (b): one HMM picking among several *linear* dynamical systems =
**Switching Linear Dynamical System (SLDS)**. (This is very likely what "SLDS/SLLD"
refers to — it is the classical ancestor, not a file in this repo.)

Now swap the *linear* systems for *RNNs* (nonlinear), and you get this repository's
model: a **Switching Recurrent Neural Network (SRNN)**. One RNN per discrete mode, an
HMM-like process choosing which RNN is active, and a learned emission turning the latent
state into observations.

This particular SRNN also makes the **transitions depend on the continuous state**
(the chance of switching depends on `h_{t-1}`), which makes it a *recurrent* switching
model — the discrete and continuous parts talk to each other.

---

## 2.3 The generative model — the full "story" of how data is born

A **generative model** is a precise recipe for how the data could have been produced,
written as probability distributions. Training = finding the weights that make this
recipe assign high probability to the *real* data. Here is the SRNN's recipe, step by
step, with the code's variable names.

**Symbols:**
- `K = num_tv = 2` discrete states (`num_tv` = "number of time-varying [regimes]").
- `H = hidden_shape = 8` dimensions in the continuous latent `h_t`.
- `D = input_shape = 20` observed neurons in `y_t`.
- `T = 100` time steps, `B` = batch of trials.

**Step 0 — pick the first mode.**
`z_1 ~ Categorical(π)`, where the initial probabilities `π = softmax(initials)` come
from a trainable length-`K` parameter `self.initials`.

> **Categorical distribution:** a die with `K` faces; `π_i` is the probability of face `i`.
> **Softmax:** turns any `K` real numbers into positive numbers that sum to 1 (valid
> probabilities): `softmax(a)_i = exp(a_i) / Σ_j exp(a_j)`. In code this is done in
> log-space as `initials - logsumexp(initials)` (same thing, numerically safer).

**Step 1 — evolve the continuous state with the active mode's RNN.**
Given the previous latent `h_{t-1}` and the active mode `z_t = i`:
`h_t ~ Normal( RNN_i(x_t, h_{t-1}), σ²_h · I )`, with `σ²_h = 1e-4`.
There are `K` separate RNNs (`self.rnns[i]`); the one indexed by the current mode drives
the dynamics.

> **Normal (Gaussian) distribution `N(μ, Σ)`:** the bell curve, generalized to vectors.
> `μ` is the mean (center), `Σ` is the covariance (spread). Here `Σ = σ²·I` means each
> dimension is independent with the same small variance `σ² = 1e-4` — i.e. `h_t` sits
> very tightly around the RNN's output. **`I`** is the identity matrix; **diagonal
> covariance** = no correlations between dimensions, just per-dimension variance.

**Step 2 — decide whether to switch, based on the continuous state.**
The transition probabilities are produced by a tiny network from `h_{t-1}`:
`A_t = normalize( reshape( TransitionNet(h_{t-1}) , K×K ) )`.
So the `K×K` transition matrix is *recomputed at every time step* from the current
latent — this is the "recurrent switching" coupling.

**Step 3 — emit the observation.**
`y_t ~ Normal( Emission(h_t), σ²_y · I )`, with `σ²_y = 1e-4`. `Emission` is a small MLP
(multi-layer perceptron): `8 → 32 → 64 → 20` with ReLU nonlinearities. It maps the
8-dim latent to the 20-dim neural activity.

> **MLP (multi-layer perceptron):** the basic feed-forward neural network — alternating
> linear maps and nonlinearities. **ReLU** ("rectified linear unit") is the nonlinearity
> `ReLU(x) = max(0, x)`; it zeroes out negatives and is the standard cheap nonlinearity.

Putting it together, the joint probability the model defines is:

```
p(y_{1:T}, h_{1:T}, z_{1:T})
   = π(z_1) · p(h_1)                                  ← start
     · Π_t  P(z_t | h_{t-1})   · p(h_t | h_{t-1}, z_t) ← transition + dynamics
     · Π_t  p(y_t | h_t)                               ← emission
```

Training tries to make this big product large for the observed `y`.

---

## 2.4 The inference problem — and why we need an inference network

We can *write down* the story above, but to train it (and to use it) we need the
**posterior**: given the data `y`, what were the hidden states `h` and `z`?

> **Posterior:** the distribution of the unknowns *after* seeing the data,
> `p(h, z | y)`. By Bayes' rule it is proportional to `p(y, h, z)` (the generative
> story), but the normalizing constant is an intractable integral over all possible `h`.

The SRNN splits this into two parts, handling each with the right tool:

- **The discrete states `z`** can be marginalized *exactly* with the forward–backward
  (Baum–Welch) algorithm, because — conditioned on a fixed `h` trajectory — the `z`
  process is just an HMM. ([§2.6](#26-baumwelch-forwardbackward))

- **The continuous states `h`** cannot be done exactly (the RNNs and MLP are nonlinear),
  so we approximate the posterior `p(h | y)` with a neural network. This is called
  **amortized variational inference**.

> **Variational inference (VI):** instead of computing the true posterior `p(h|y)`, pick
> a simpler distribution `q(h)` and tune it to be as close as possible to the true
> posterior. "Variational" = you optimize over a family of distributions.
>
> **Amortized:** rather than re-optimizing `q` from scratch for every data point, train
> *one network* that takes the data and outputs the parameters of `q`. The cost of
> inference is "amortized" across all data — once trained, inference is a single forward
> pass. That network is the **inference network**.

In this repo the inference network is `RNNInfer` (in `inference_network.py`). Despite
the name, **it is a Transformer encoder**, not an RNN. It reads the whole observed
sequence `y_{1:T}` and outputs, for each time step, the mean of `q(h_t | y)`. It samples
`h` from `q` using the **reparameterization trick**.

> **Transformer encoder:** a sequence model built on **self-attention**, where every
> time step can directly look at every other time step (no step-by-step recurrence).
> Strong at capturing long-range structure. **Positional encoding** is added so the model
> knows the order of time steps (attention itself is order-agnostic).
>
> **Reparameterization trick:** to sample `h = μ + σ·ε` with `ε ~ N(0, I)` instead of
> sampling `h ~ N(μ, σ²)` directly. Algebraically identical, but now the randomness `ε`
> is separate from the parameters `μ, σ`, so gradients can flow through `μ` and `σ`
> during backpropagation. This is what lets you train a network that *samples*.

Here `σ²` is fixed tiny (`1e-4`), so `q(h_t|y) = N(mean_out, 1e-4·I)` is a very sharp
Gaussian — the network is nearly outputting a point estimate of `h`, with a whisker of
noise to keep gradients well-defined.

---

## 2.5 The training objective — the ELBO, plus a supervised twist

We train by maximizing a quantity called the **ELBO**.

> **ELBO (Evidence Lower BOund):** variational inference cannot maximize the data
> likelihood `log p(y)` directly (intractable), so it maximizes a *lower bound* on it.
> Pushing the bound up pushes the true likelihood up. The ELBO has the form:
>
> ```
> ELBO = E_q[ log p(y, h, z) ]   −   E_q[ log q(h) ]
>        \_______________________/   \_____________/
>        expected complete-data        entropy of q
>        log-likelihood ("fit")        ("don't collapse")
> ```
>
> - The **first term** rewards `h` samples (from `q`) under which the generative story
>   assigns the data high probability — i.e. the model *fits*.
> - The **second term** is the **entropy** of `q`: it rewards `q` for staying spread out
>   / uncertain, preventing it from collapsing to a single overconfident point.

> **Expectation `E_q[…]`:** the average value of "…" when `h` is drawn from `q`. In code
> it is estimated by actually sampling `h` (the reparameterized `sampled_h`) and
> plugging it in.
>
> **Entropy `H(q)`:** a measure of how spread out a distribution is (high entropy =
> uncertain, low entropy = concentrated). For a Gaussian it has a closed form, which is
> why the code can call `infer_dist.entropy()` directly.

In this model the discrete `z` is **marginalized analytically** inside the first term
using the Baum–Welch posterior (the `gamma`/`delta` responsibilities below), so you
never have to sample `z`. The first term becomes the standard "expected complete-data
log-likelihood" of an HMM, evaluated at the sampled `h`.

**The supervised twist.** On top of the ELBO, this model adds a term that uses the
ground-truth labels:

```
total objective = ELBO  +  coef_cross · ( supervised cross-entropy between
                                          the model's inferred state-posterior
                                          and the true one-hot labels )
```

> **Cross-entropy:** a standard measure of disagreement between a predicted probability
> distribution and a target. Minimizing it pushes the predicted distribution toward the
> target. Here the "prediction" is the model's posterior over `z_t` (which mode it thinks
> it's in) and the "target" is the true label as a one-hot vector. **One-hot** = a vector
> that is 1 at the true class and 0 elsewhere (e.g. mode 1 → `[0, 1]`).

`coef_cross` (`= 0.5` in the config) sets how strongly the labels steer training. At 0
you'd recover a fully unsupervised switching model; larger values force the inferred
regimes to line up with the provided labels. **This supervised term is the defining
feature of "Supervised_SRNN."**

In code (`train.py`) the whole thing is assembled and *negated* (because PyTorch
optimizers *minimize*, and we want to *maximize* the objective):

```python
loss_all = -(t1.mean() + t2.mean() + coef_cross*cross_en + infer_dist.entropy().mean())
#             \_______________t1+t2 = expected complete-data LL____/   \__entropy__/
#                            (z marginalized via Baum–Welch)
```

`t1` is the `t = 0` part of the expected log-likelihood; `t2` is the sum over `t > 0`.
Details in [03_model_internals.md](02_model_internals.md).

---

## 2.6 Baum–Welch / forward–backward

This is the exact bookkeeping over the discrete states. Given (for a fixed `h`
trajectory) the per-step quantities the generative model produced —

- `prob_initial` — log `π`, the initial state log-probabilities, shape `(B, K)`,
- `prob_all_s` — log transition matrices per step, shape `(B, T, K, K)`,
- `prob_all_h` — `log p(h_t | h_{t-1}, z_t = i)` for each mode `i`, shape `(B, T, K)`,
- `prob_all_y` — `log p(y_t | h_t)`, shape `(B, T)` (does not depend on the mode),

— the **forward pass** computes `α_t(i)` = (log) probability of the data up to time `t`
*and* being in mode `i` at `t`; the **backward pass** computes `β_t(i)` = (log)
probability of the *future* data given mode `i` at `t`. Both are recursions: forward
sweeps `t = 1 → T`, backward sweeps `t = T → 1`.

> Everything is done in **log-space** with `logsumexp` (compute `log Σ exp(·)` stably).
> This avoids numerical underflow: probabilities of long sequences are astronomically
> small numbers that would round to 0 in normal arithmetic, but their logarithms are
> well-behaved. `logsumexp` is the log-space version of "add probabilities."

From `α` and `β` we get the two posterior quantities the loss needs:

- **`gamma` (γ):** the posterior probability of each mode at each time,
  `γ_t(i) = P(z_t = i | all data)`. (The code computes `gamma1`, the `t=1` marginal, and
  `delta1` covers the rest.) These are the "responsibilities" — how much each mode is
  responsible for each time step.
- **`delta` (δ, standard symbol ξ):** the posterior probability of *consecutive pairs*,
  `δ_t(i, j) = P(z_{t-1} = i, z_t = j | all data)` — needed to score the transitions.

These responsibilities are exactly the weights in the expected complete-data
log-likelihood (the `t1`, `t2` terms). And the per-time mode posterior is also what gets
compared to the true labels in the supervised cross-entropy, and what `plot.py` turns
into the "inferred states" via `argmax`.

Reference in the code comments: <https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm>.

---

## 2.7 How it all fits together (one training step)

```
        y (observed neural data, one batch of trials)
        │
        ▼
  ┌─────────────────────┐
  │ Inference network    │  RNNInfer  (a Transformer)
  │  q(h | y)            │  → mean_out, and a sample h = mean + ε·σ   (reparam. trick)
  └─────────────────────┘
        │ sampled latent trajectory h
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │ Generative model  (model_srnn.Model)                      │
  │  • per-mode RNNs  → prob_all_h  (how well each mode        │
  │                      explains the step h_{t-1}→h_t)        │
  │  • transition net → prob_all_s  (switch probabilities)     │
  │  • emission MLP   → prob_all_y  (how well h explains y)     │
  │  • initials       → prob_initial                           │
  └─────────────────────────────────────────────────────────┘
        │ per-step log-probs
        ▼
  ┌─────────────────────┐
  │ Baum–Welch           │  forward + backward  →  gamma, delta
  │ (baum_welch.py)      │  and the state-posterior  P(z_t | y)
  └─────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │ Loss (loss_function.py + train.py)            │
  │  −( expected complete-data LL                 │  ← fit
  │     + coef_cross · cross-entropy(posterior,   │  ← SUPERVISION (true labels)
  │                                   true labels)│
  │     + entropy(q) )                            │  ← keep q honest
  └─────────────────────────────────────────────┘
        │
        ▼  backprop → update inference network + generative model
```

The two networks (inference + generative) are trained **jointly**, each with its own
Adam optimizer (`optimizer_rnn` for the inference network, `optimizer` for the
generative model). Gradients are clipped to norm 1.0 for stability, and both have a
learning-rate scheduler that decays the step size over time.

> **Adam:** a popular gradient-descent optimizer that adapts the step size per parameter.
> **Gradient clipping:** cap the size of the update so a single huge gradient can't
> destabilize training. **LR scheduler / StepLR:** multiply the learning rate by a factor
> (`gamma = 0.8`) every `step_size` epochs, so learning slows down as it converges.

Continue to [03_model_internals.md](02_model_internals.md) for the line-by-line code, or
[04_usage_guide.md](03_usage_guide.md) to just run it.
