# 1. Project Layout

This file is a guided tour of every folder and file in the repository, and why each
one matters to you.

```
Supervised_SRNN/
├── array_hidden8.py        ← MAIN ENTRY POINT: trains the model on the data
├── plot.py                 ← Post-training analysis & figures
├── config.yaml             ← All knobs (hyperparameters, paths) in one place
├── environment.yml         ← Conda environment specification
├── README.md               ← Short install/run instructions (the official one)
│
├── data/                   ← Input data
│   ├── simulation.npy       ← The neural recordings  (50, 100, 20)
│   └── labels.npy           ← Ground-truth regime labels (50, 100, 1)
│
├── SRNN/                   ← THE MODEL: the Python package with all the math
│   ├── model_srnn.py        ← Generative model (emission, per-state RNNs, transitions)
│   ├── inference_network.py ← Inference network (Transformer that reads the data)
│   ├── baum_welch.py        ← Forward–backward algorithm over the discrete states
│   ├── loss_function.py     ← The training objective (ELBO + supervised term)
│   ├── train.py             ← Training loop and evaluation routine
│   ├── initialization.py    ← Small helper: one-hot encoding of labels
│   ├── generative_check.py  ← Optional: roll the model forward to "dream" data
│   └── utils.py             ← Small helper: ETA / time formatting
│
├── result/                 ← Saved trained models (.pt checkpoint files)
│   ├── sim_model_hidden8_fold0.pt
│   └── autosave_sim_model_hidden8_fold_0.pt
│
└── plot/                   ← Output figures from plot.py
    ├── neural_recon.png
    └── states.png
```

---

## Top-level files

### `array_hidden8.py` — the main script you run
This is the file you execute to train the model. The odd name comes from the
`hidden_shape = 8` setting (the size of the continuous latent state — see file 2). It:

1. Parses command-line arguments and loads `config.yaml`.
2. Loads `data/simulation.npy` and `data/labels.npy`.
3. Splits the 50 trials into train/test using **5-fold cross-validation** (see below).
4. Builds the model (`model_srnn.Model`) and the inference network
   (`inference_network.RNNInfer`).
5. Sets up the optimizers and learning-rate schedulers.
6. Calls `train.train_(...)` to actually train.
7. Saves everything (weights + data + metrics) into a `.pt` file in `result/`.

> **"5-fold cross-validation"** means: split the 50 trials into 5 equal groups
> ("folds") of 10. Train on 4 groups (40 trials), test on the held-out group (10
> trials). Do this 5 times so every trial gets used as test data once. The `--fold N`
> argument (N = 0..4) selects *which* group is held out for testing this run.

### `plot.py` — analysis after training
Loads a saved `.pt` checkpoint, re-runs the model on the test trials, and produces two
figures into `plot/`:
- **`neural_recon.png`** — the model's reconstruction of the neural data vs. the truth.
- **`states.png`** — the model's *inferred* discrete regime sequence vs. the *true* one.

### `config.yaml` — your control panel
Every tunable setting lives here so you do not have to edit code. Grouped into:
- `experiment`: random seed, which fold, data paths, where to save.
- `system`: `cuda` (GPU) vs `cpu`, and float precision.
- `model`: model sizes — `num_tv` (number of discrete states), `hidden_shape`
  (continuous latent size), plus `bottleneck_shape` and `neural_private_shape`.
- `train`: `epochs`, learning rate `lr`, the supervised weight `coef_cross`, and `batch_size`.

> **Heads-up (matters for understanding):** `bottleneck_shape` and
> `neural_private_shape` are **read but not actually used** by the simplified model in
> this repo. They are leftovers from a larger multi-region version of the model (the
> commented-out `sharedecoder`, "private" vs "shared" latents in `array_hidden8.py` are
> the other traces of it). You can ignore them; changing them changes nothing here.
> They are documented so you are not confused when you go looking for where they're used.

### `environment.yml` — the software environment
A Conda spec that creates an environment named **`SSRNN`** with Python 3.11, PyTorch
2.8 (CUDA 12.9 build), NumPy, scikit-learn, matplotlib, and PyYAML.

### `README.md`
The original, terse install-and-run instructions. This `docs/` folder is the expanded
version of it.

---

## `data/` — the inputs

Both files are NumPy `.npy` arrays (NumPy's binary save format).

### `simulation.npy` — the observations, shape `(50, 100, 20)`
The convention throughout the code is **(trials, time, features)**:
- **50 trials** — independent repetitions of the experiment (think: 50 recording runs).
- **100 time points** — the length of each trial.
- **20 features** — the 20 simulated "neurons" (channels) recorded at each time point.

In the code this array is called `y_c` and later `y_train` / `y_test`. This is the
data the model tries to explain and reconstruct.

### `labels.npy` — the ground-truth regimes, shape `(50, 100, 1)`
For every trial and every time point, an integer saying which discrete regime the
system was *truly* in. Here the values are `{0, 1}` — i.e. **2 states**, matching
`num_tv: 2` in the config. The trailing `1` is just a singleton dimension.

These labels are what make the training **supervised**: most switching models have to
*guess* the regimes; this one is shown the answers and rewarded for matching them.

---

## `SRNN/` — the model package

This is the heart of the project. Each file is dissected in
[03_model_internals.md](03_model_internals.md); here is the one-line role of each:

| File | Role |
|------|------|
| `model_srnn.py` | The **generative model**: the per-state RNNs, the transition network, the emission MLP, and the loop that produces all the per-timestep probabilities. |
| `inference_network.py` | The **inference network** `RNNInfer` — a Transformer encoder that reads `y` and outputs an estimate of the continuous hidden state `h`. (Note: named "RNN" but it is actually a Transformer.) |
| `baum_welch.py` | The **forward–backward / Baum–Welch** algorithm — exact probability bookkeeping over the discrete states. |
| `loss_function.py` | Builds the **training objective** from the Baum–Welch outputs plus the supervised cross-entropy. |
| `train.py` | The **training loop** (`train_`) and the **evaluation** routine (`eval_`). |
| `initialization.py` | `one_hot(...)` — converts integer labels into one-hot vectors for the cross-entropy term. |
| `generative_check.py` | Optional sanity check: runs the learned dynamics *forward* to generate ("dream") new data — a way to inspect what the model has learned. |
| `utils.py` | `compute_time(...)` — prints an estimated time-remaining during training. |

> **`__pycache__/`** is just Python's compiled-bytecode cache. Ignore it; it is
> regenerated automatically.

---

## `result/` — trained model checkpoints

PyTorch `.pt` files (saved with `torch.save`). Each is a dictionary holding the trained
weights **and** the data/metrics needed to reproduce the analysis. Two kinds appear:

- **`sim_model_hidden8_fold0.pt`** — the *final* checkpoint written at the end of a run
  (by `array_hidden8.py`). This is the one `plot.py` loads by default.
- **`autosave_sim_model_hidden8_fold_0.pt`** — an *every-epoch* autosave written by the
  training loop (`train.py`), so a crashed run is recoverable. Same contents, saved
  continuously.

The filename encodes the settings: `sim` (the `save_name`), `hidden8` (`hidden_shape=8`),
`fold0` (the cross-validation fold).

What's inside a checkpoint (keys you can load): `model_state_dict`,
`rnninfer_state_dict` (the two networks' weights), the train/test data arrays,
`label_test`, training-curve arrays (`loss_train`, `mse_all*`, `error_all*`), and
`pos_test_all` (the inferred-state posterior at every epoch). See file 4 for how to read these.

---

## `plot/` — output figures

Where `plot.py` writes its PNGs.
- **`neural_recon.png`** — top half = true neural activity for the best-reconstructed
  test trial; bottom half = the model's reconstruction. They should look alike if
  training worked.
- **`states.png`** — top = true regime sequence per trial; bottom = the regime sequence
  the model inferred. Agreement here is the main "did the supervised switching work?" check.
