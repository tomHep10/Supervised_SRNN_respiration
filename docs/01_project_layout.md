# 1. Project Layout

A guided tour of every folder and file in the repository and why each one matters. The
repo has **three layers**, and keeping them straight makes everything else clear:

1. **`SRNN/`** — the model itself (the math). Imported everywhere, **never modified**.
2. **The original simulation demo** (`array_hidden8.py`, `plot.py`, `config.yaml`,
   `data/`) — the canned example the model shipped with; useful as a reference, not part of
   the respiration science.
3. **`respiration/`** — the actual research pipeline: turn raw respiration recordings into
   windows, train the SRNN, and test it for social valence. **This is where the science
   lives.**

Plus **`hipergator/`** (SLURM job scripts to run all of the above on the cluster) and
**`docs/`** (this documentation).

```
Supervised_SRNN_respiration/
│
├── docs/                         ← THIS documentation (start at 00_quickstart.md)
│   ├── 00_quickstart.md           ← env, node, how work gets run (read first)
│   ├── 01_project_layout.md       ← you are here
│   ├── 02_concepts_and_math.md    ← the model's ideas + background math
│   ├── 03_model_internals.md      ← line-by-line walk through SRNN/
│   ├── 04_usage_guide.md          ← install / run / read outputs (sim demo)
│   ├── 05_hipergator_guide.md     ← SLURM concepts + the original sim job scripts
│   ├── 06_project_context.md      ← research goal, dataset, findings & decisions
│   └── 07_experiment_runbook.md   ← step-by-step to reproduce the current experiment
│
├── SRNN/                         ← THE MODEL (imported, never edited) — see file 03
│   ├── model_srnn.py              ← generative model (per-state RNNs, transitions, emission)
│   ├── inference_network.py       ← inference net RNNInfer (a Transformer that reads y → h)
│   ├── baum_welch.py              ← forward–backward over the discrete states
│   ├── loss_function.py           ← training objective (ELBO + supervised cross-entropy)
│   ├── train.py                   ← training loop train_() and evaluation eval_()
│   ├── initialization.py          ← one_hot() label helper
│   ├── generative_check.py        ← optional: roll dynamics forward to "dream" data
│   └── utils.py                   ← compute_time() ETA helper
│
├── respiration/                  ← THE RESEARCH PIPELINE — see respiration/README.md
│   ├── config_respiration.yaml        ← params + LOCAL paths (smoke tests)
│   ├── config_respiration_hpg.yaml    ← params + HiPerGator paths (the real run)
│   ├── prepare_respiration.py         ← raw .h5 + BORIS .csv → windowed arrays
│   ├── train_respiration.py           ← train the SRNN (one CV fold per call)
│   ├── plot_respiration.py            ← per-fold figures (recon, states, latent PCA)
│   ├── analyze_valence.py             ← THE CLASSIFIER/ANALYSIS: valence tests across folds
│   ├── data_prepared/                 ← output of prepare_ (observations/labels/meta)
│   ├── result/                        ← trained checkpoints resp_srnn_<split>_h8_fold*.pt
│   └── plot/                          ← analysis figures (PNGs)
│
├── hipergator/                   ← SLURM JOB SCRIPTS (see "Job scripts" below, in order)
│   ├── run_on_hpg.sh                  ← one-shot helper: build env + prepare + submit
│   ├── respiration_job_loso.slurm     ← TRAIN, leave-one-subject-out  (GPU array 0-7)
│   ├── analyze_job.slurm              ← ANALYZE the subject split (CPU)
│   ├── classifier_results.slurm       ← ANALYZE the subject split (CPU, one job)
│   ├── ssrnn_job.slurm                ← (legacy) train the sim demo on GPU
│   └── ssrnn_job_cpu.slurm            ← (legacy) train the sim demo on CPU
│
├── logs/                         ← SLURM logs, <jobname>_<jobid>.log
│
│   ── original simulation demo (reference, not the respiration science) ──
├── array_hidden8.py              ← sim demo entry point (trains on data/simulation.npy)
├── plot.py                       ← sim demo figures
├── config.yaml                   ← sim demo config
├── data/                         ← sim demo inputs (simulation.npy, labels.npy)
├── result/                       ← sim demo checkpoints
├── plot/                         ← sim demo figures (neural_recon.png, states.png)
├── claude_runs/                  ← archived smoke-test run of the sim demo
│
├── environment.yml               ← conda spec → the SSRNN env
├── README.md                     ← terse top-level readme
└── .gitignore
```

---

## `SRNN/` — the model package (do not modify)

The heart of the project; each file is dissected in
[03_model_internals.md](03_model_internals.md). One-line roles:

| File | Role |
|------|------|
| `model_srnn.py` | **Generative model:** per-state RNNs, transition network, emission MLP, the per-timestep probability loop. |
| `inference_network.py` | **Inference network** `RNNInfer` — a Transformer encoder reading `y` → estimate of continuous hidden state `h`. (Named "RNN" but it's a Transformer.) |
| `baum_welch.py` | **Forward–backward / Baum–Welch** — exact probability bookkeeping over discrete states. |
| `loss_function.py` | Builds the **objective** from Baum–Welch outputs + supervised cross-entropy. |
| `train.py` | **Training loop** `train_()` and **evaluation** `eval_()`. |
| `initialization.py` | `one_hot()` — integer labels → one-hot for the cross-entropy term. |
| `generative_check.py` | Optional sanity check: run dynamics forward to generate data. |
| `utils.py` | `compute_time()` — time-remaining estimate during training. |

> **Why "supervised":** unlike usual switching models it can be shown ground-truth regime
> labels and pushed to match them (weight `coef_cross`). With `coef_cross = 0` it becomes
> effectively unsupervised — the mode the respiration experiment uses (see file 06).

---

## `respiration/` — the research pipeline

This is the code you actually run for the science. Order of use matches the pipeline:

| File | Role | When you run it |
|------|------|-----------------|
| `config_respiration_hpg.yaml` | All params + **HiPerGator** data paths, the recording list (15), valence map, windowing, model sizes, training knobs. | Edit to change the experiment; passed as `--config` to every script. |
| `config_respiration.yaml` | Same but **local** paths, for small smoke tests off-cluster. | Local debugging only. |
| `prepare_respiration.py` | Clean + window raw `.h5` + BORIS `.csv` → `data_prepared/`. | **Once**, before training (Step 1 of the runbook). |
| `train_respiration.py` | Train the SRNN for **one** CV fold. Args: `--fold`, `--split {subject,window}`, plus optional `--epochs/--num_tv/--hidden_shape/--coef_cross`. | Called by the training SLURM array, once per fold. |
| `plot_respiration.py` | Per-fold figures (resp reconstruction, inferred states, latent PCA). Args `--fold`, `--split`. | Right after each training fold (the SLURM scripts call it automatically). |
| `analyze_valence.py` | **The classifier / valence analysis.** Pools the held-out folds and runs: (A) recording-level breathing-rate ROC-AUC, (B) pooled latent PCA, (C) rate-controlled LOSO decode, (D) permutation test, (E) LOSO LDA projection. Args `--split subject`, `--pca_fold`, `--n_perm`. | After all folds train (Step 3); via `analyze_job.slurm` / `classifier_results.slurm`. |

**Sub-folders:**
- `data_prepared/` — output of `prepare_respiration.py`: `observations.npy`
  `(147,1500,1)`, `labels.npy`, `meta.npz` (recording/subject/valence per window),
  `label_map.json`.
- `result/` — trained checkpoints, named `resp_srnn_<split>_h8_fold<k>.pt` (e.g.
  `resp_srnn_subject_h8_fold3.pt`), plus per-fold
  `progress_<split>_fold<k>.csv` training curves.
- `plot/` — analysis figures: `permutation_test_subject.png`,
  `lda_projection_subject.png`, `pooled_latent_pca_by_valence.png`, and the
  per-fold `resp_recon.png` / `states.png` / `latent_pca.png` (these overwrite across
  folds — they show the last fold to run).

See [respiration/README.md](../respiration/README.md) for the data design and scientific
caveats, and [07_experiment_runbook.md](07_experiment_runbook.md) to run it end to end.

---

## `hipergator/` — the SLURM job scripts (in order of use)

All cluster runs go through these. They are grouped by the pipeline stage. Cluster
coordinates baked into the respiration scripts: account/qos `npadillacoreano`, GPU
partition `hpg-b200`, CPU partition `hpg-default` (see [00_quickstart.md](00_quickstart.md)).

| # | Script | Stage | GPU? | What it does / when to use |
|---|--------|-------|------|----------------------------|
| 0 | `run_on_hpg.sh` | setup | — | **Convenience one-shot** (not a SLURM file — `bash` it in your own SSH session). Builds the `SSRNN` env if missing, runs `prepare_respiration.py`, auto-detects your account, and submits the subject-out training array. Use it for a clean first run; afterwards prefer the individual steps. |
| 1 | `respiration_job_loso.slurm` | **train** | ✅ `hpg-b200` | **Primary training.** SLURM array `0-7` — one model per fold, leave-one-**subject**-out (holds out *all* of one animal's recordings), `coef_cross=0` (discovery). Runs `train_respiration.py` then `plot_respiration.py` per fold. → `resp_srnn_subject_h8_fold{0..7}.pt`. The clean test of cross-individual generalization (see file 06, finding #4). Submit after `data_prepared/` exists. |
| 2 | `analyze_job.slurm` | **analyze** | ❌ `hpg-default` | Runs `analyze_valence.py` for the **subject** split (the only split). Inference-only, minutes. Produces the valence numbers + figures. |
| 3 | `classifier_results.slurm` | **analyze** | ❌ `hpg-default` | **All-in-one results job.** Runs `analyze_valence.py` for the subject split in a single submission. The easiest way to refresh every result/figure at once. |
| — | `ssrnn_job.slurm` | legacy | ✅ | Trains the **original simulation demo** (`array_hidden8.py`), 5-fold array. Template/reference; has `GROUP/QOS/GPU_TYPE` placeholders, not the respiration run. |
| — | `ssrnn_job_cpu.slurm` | legacy | ❌ | Same demo, CPU-only. Reference. |

Typical flow: **1 → 3** (or **2** for just the analyze step). The runbook
([07_experiment_runbook.md](07_experiment_runbook.md)) walks each one.

---

## The original simulation demo (reference only)

These are the files the SRNN repo shipped with — a canned 2-state, 20-"neuron" simulation.
They're **not** the respiration science, but they're the cleanest example of the model and
are documented fully in [04_usage_guide.md](04_usage_guide.md).

| File / folder | Role |
|---|---|
| `array_hidden8.py` | Entry point: trains the model on `data/simulation.npy` with 5-fold CV (`--fold 0..4`). Name comes from `hidden_shape=8`. |
| `plot.py` | Loads a checkpoint, writes `plot/neural_recon.png` (reconstruction vs. truth) and `plot/states.png` (inferred vs. true regimes). |
| `config.yaml` | The demo's knobs (seed, fold, paths, model sizes, training). |
| `data/simulation.npy` | Observations `(50, 100, 20)` = (trials, time, neurons). |
| `data/labels.npy` | Ground-truth regimes `(50, 100, 1)`, values `{0,1}`. |
| `result/` | Demo checkpoints (`sim_model_hidden8_fold0.pt`, autosave). |
| `plot/` | Demo figures. |
| `claude_runs/` | An archived CPU smoke-test of the demo (its own config/plot/result/log) — proof the pipeline runs end-to-end. |

> **Note on `bottleneck_shape` / `neural_private_shape`:** present in the configs but
> **read and not used** by this simplified model — leftovers from a larger multi-region
> version. Changing them changes nothing. Documented so they don't confuse you.

---

## Top-level support files

- **`environment.yml`** — conda spec that creates the **`SSRNN`** env (Python 3.11,
  PyTorch 2.8 / CUDA 12.9, numpy, scikit-learn, matplotlib, pyyaml). Already built at
  `/blue/npadillacoreano/t.heeps/.conda/envs/SSRNN` — see [00_quickstart.md](00_quickstart.md).
- **`README.md`** — terse top-level pointer; this `docs/` folder is the expanded version.
- **`logs/`** — every SLURM job writes `<jobname>_<jobid>.log` here.
- **`__pycache__/`** — Python bytecode cache; ignore, auto-regenerated.
