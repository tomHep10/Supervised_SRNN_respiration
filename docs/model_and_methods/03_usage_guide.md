# 4. Usage Guide — running, training, and adapting the model

This file is the practical "how do I actually use it" companion. It covers the simulated
data first (your stated first goal), then how to load a trained model, then how to point
it at your own data.

All commands assume you are in the repo root: `c:\Users\thoma\Code\ResearchCode\Supervised_SRNN`.

---

## 4.1 One-time setup

```bash
conda env create -f environment.yml   # creates the "SSRNN" environment
conda activate SSRNN
```

This installs Python 3.11 and PyTorch 2.8 built for **CUDA 12.9** (NVIDIA GPU). If you do
not have an NVIDIA GPU, see [§4.6](#46-running-on-cpu-no-gpu).

If the env build fails, the README's advice holds: `pip install <missing-package>` — all
deps are standard scientific-Python packages.

---

## 4.2 Run on the simulated data (train one fold)

```bash
python array_hidden8.py --config config.yaml --fold 0
```

What happens:
- Loads `data/simulation.npy` (50×100×20) and `data/labels.npy` (50×100×1).
- Holds out fold 0 (10 trials) for test, trains on the other 40.
- Trains for `epochs: 2000` (from `config.yaml`).
- Prints the loss every 100 epochs plus an estimated time remaining.
- Autosaves every epoch to `result/autosave_sim_model_hidden8_fold_0.pt`.
- On completion writes the final `result/sim_model_hidden8_fold0.pt`.

> **Expect this to take a while.** 2000 epochs with a per-step Python time-loop over
> `T = 100` is not fast. To do a quick smoke-test first, shrink it:
> ```bash
> python array_hidden8.py --config config.yaml --fold 0 --epochs 50
> ```
> You can also override `--lr`, `--num_tv`, `--hidden_shape` on the command line; they
> take precedence over `config.yaml`.

### Full 5-fold run
Each fold is a separate process. On Windows `cmd`:
```bat
for %i in (0 1 2 3 4) do python array_hidden8.py --config config.yaml --fold %i
```
In **PowerShell** (your default shell) use:
```powershell
foreach ($i in 0..4) { python array_hidden8.py --config config.yaml --fold $i }
```
This runs all five held-out splits sequentially, producing five `…fold0..4.pt` files.

---

## 4.3 Make the figures

```bash
python plot.py
```
By default it loads `result/sim_model_hidden8_fold0.pt`, re-runs the model on that
fold's test trials, and writes:
- **`plot/neural_recon.png`** — true vs. reconstructed neural activity (best test trial).
  Top block = truth, bottom block = model. Similar-looking blocks ⇒ good reconstruction.
- **`plot/states.png`** — true vs. inferred discrete-regime sequence across trials.
  Matching colors top↔bottom ⇒ the supervised switching worked.

To plot a different fold, edit `result_path` near the top of `plot.py`.

---

## 4.4 Loading a trained model yourself (for analysis / reuse)

The checkpoints are plain dictionaries. The pattern (mirrors `plot.py`) is:

```python
import torch
from SRNN import model_srnn, inference_network, train

ckpt = torch.load('result/sim_model_hidden8_fold0.pt', weights_only=False, map_location='cpu')

# rebuild the two networks with the SAME sizes they were trained with
input_shape          = ckpt['X_test'].shape[2]      # = 20
num_tv               = ckpt['num_tv']               # = 2
hidden_shape         = ckpt['hidden_shape']         # = 8
neural_private_shape = ckpt['neural_private_shape']

model    = model_srnn.Model(input_shape, num_tv, hidden_shape, neural_private_shape).to('cpu')
rnninfer = inference_network.RNNInfer(input_shape, hidden_shape).to('cpu')
model.load_state_dict(ckpt['model_state_dict'])
rnninfer.load_state_dict(ckpt['rnninfer_state_dict'])

# run inference on test data
import torch
y_test = torch.tensor(ckpt['y_test'], dtype=torch.float32)
X_test = torch.tensor(ckpt['X_test'], dtype=torch.float32)
y_pred, _, pos_test, *_ = train.eval_(model, rnninfer, X_test, y_test, 'cpu')

import numpy as np
inferred_states = np.exp(pos_test).argmax(-1)   # (trials, time): the mode the model infers
reconstruction  = y_pred                        # (trials, time, 20): predicted neural data
```

**Useful keys inside a checkpoint:**

| Key | Meaning |
|-----|---------|
| `model_state_dict`, `rnninfer_state_dict` | the trained weights |
| `y_test`, `X_test`, `label_test` | the held-out data and true labels |
| `num_tv`, `hidden_shape`, `neural_private_shape`, `coef_cross`, `lr` | the settings used |
| `loss_train` | training loss per epoch (learning curve) |
| `pos_test_all` | the test state-posterior at **every** epoch `(epochs, trials, T, K)` |

> `weights_only=False` is required because these checkpoints store NumPy arrays, not just
> tensors. Only load checkpoints you trust (this flag allows arbitrary unpickling).

### Inspecting the learned dynamics ("dreaming")
To see what the per-mode RNNs generate on their own (not relying on the inference net):
```python
from SRNN import generative_check
y_dreamed, h_dreamed = generative_check.run(model, rnninfer, X_test, y_test, pos_test, 'cpu')
```
See [03_model_internals.md §3.6](02_model_internals.md#36-generative_checkpy--dreaming-from-the-model-optional).

---

## 4.5 Using your own data

The model is data-agnostic — it just needs the two arrays in the right shape.

1. **Format your data** as:
   - observations: `float` array of shape **(trials, time, features)** → save as
     `simulation.npy` (or any path).
   - labels: `int` array of shape **(trials, time, 1)** with values `0 … K−1` (the true
     regime at each time step) → save as `labels.npy`.
   ```python
   import numpy as np
   np.save('data/my_obs.npy',    my_obs)     # (N, T, D)
   np.save('data/my_labels.npy', my_labels)  # (N, T, 1), ints in {0..K-1}
   ```
2. **Point the config at them** and set `num_tv` to your number of regimes `K`:
   ```yaml
   experiment:
     data_path:  "./data/my_obs.npy"
     data_label: "./data/my_labels.npy"
   model:
     num_tv: 3            # however many discrete regimes your labels use
     hidden_shape: 8      # size of the continuous latent — tune this
   ```
   Everything downstream (`input_shape`, the 5-fold split, the networks) adapts to the
   array shapes automatically. `input_shape` is read from your feature count; you do not
   set it by hand.
3. **Train** exactly as in §4.2.

> If your trials have **different lengths**, you would need to pad/mask them — the current
> code assumes a fixed `T` across trials (it allocates dense `(B, T, …)` tensors). For
> equal-length trials there's nothing to do.

> **Don't have labels?** This repo is the *supervised* variant and expects them. With
> `coef_cross = 0` the supervised term vanishes and it becomes unsupervised in spirit,
> but `train.py` still builds the one-hot target from whatever is in `labels.npy`, so
> you'd need to supply *some* placeholder label array of the right shape. For a truly
> label-free workflow you'd use an unsupervised SRNN/SLDS instead.

---

## 4.6 Running on CPU (no GPU)

`config.yaml` sets `system.device: "cuda"`. On a machine without an NVIDIA GPU:
- `array_hidden8.py` already falls back to CPU at runtime (it re-derives `device` from
  `torch.cuda.is_available()` on line ~120), so training will run on CPU — just slowly.
- `plot.py` runs on CPU regardless.
- You can also set `device: "cpu"` in the config to be explicit.
- For CPU you'd typically `pip install torch` from the regular PyPI index rather than the
  CUDA wheel in `environment.yml`.

---

## 4.7 Key knobs and what they do (quick reference)

| Setting (`config.yaml`) | Effect |
|---|---|
| `model.num_tv` | Number of discrete regimes `K`. Must match the label range. |
| `model.hidden_shape` | Size of the continuous latent `h`. Bigger = more expressive, slower, more overfit-prone. The "8" in `array_hidden8.py`. |
| `train.epochs` | Training iterations over the data. 2000 is the default; lower for quick tests. |
| `train.lr` | Adam learning rate (`0.001`). |
| `train.coef_cross` | **Strength of supervision** (`0.5`). ↑ forces inferred states to match labels; ↓ lets the data speak more. |
| `train.batch_size` | Trials per gradient step (`64`; with only 40 train trials this means one batch). |
| `experiment.seed` | Random seed for reproducibility (`131`). |
| `experiment.fold` | Which CV fold (0–4) is the test set. Overridden by `--fold`. |
| `model.bottleneck_shape`, `model.neural_private_shape` | **Unused** in this simplified model (see [file 1](../01_project_layout.md)). Leftovers; changing them does nothing. |

---

## 4.8 A sensible first session

```bash
conda activate SSRNN

# 1) quick smoke test (a few minutes) to confirm everything runs
python array_hidden8.py --config config.yaml --fold 0 --epochs 50

# 2) look at the autosaved result
python plot.py            # (edit result_path to the autosave_ file if needed)

# 3) once happy, do the real 2000-epoch run (PowerShell)
foreach ($i in 0..4) { python array_hidden8.py --config config.yaml --fold $i }

# 4) final figures
python plot.py
```

From there, open the checkpoints in a notebook (§4.4) to study `pos_test_all` (how the
inferred regimes sharpen over epochs), the reconstruction quality, and — if you want to
probe the learned dynamics — the generative rollout (§3.6).
