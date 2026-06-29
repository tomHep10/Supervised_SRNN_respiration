# 7. Experiment Runbook — the 15-recording valence experiment, step by step

> A **procedural, copy-paste** walkthrough of the current experiment so **you can run the
> whole thing yourself.** Every step says *what it does*, *which node it runs on*, the
> *exact command*, and *what you should see*. For the concepts behind it read
> [06_project_context.md](01_project_context.md); for the environment basics read
> [00_quickstart.md](../00_quickstart.md).

**The experiment in one line:** take 15 respiration recordings (7 positive RI1 / 8
negative RI2), cut each into 30 s windows, train the SRNN in *discovery* mode
(`coef_cross = 0`) with cross-validation, then test whether the learned latent dynamics
separate positive vs. negative social valence.

**Pipeline shape:**
```
prepare  ──►  train (CV folds, GPU array)  ──►  analyze / classify (CPU)  ──►  read results
(once)        respiration_job_loso.slurm       classifier_results.slurm        logs/ + plot/
```

All commands assume you start from the repo root:
```bash
cd /home/t.heeps/blue_npadillacoreano/npadillacoreano/share/respiration-project/Supervised_SRNN_respiration
```

---

## Step 0 — One-time setup (skip if already done)

The env and prepared data already exist, so **you can normally skip this whole step.** It's
here for a fresh machine / fresh agent.

**On the login node** (env build needs internet; compute nodes may not have it):
```bash
module load conda
conda env create -f environment.yml      # builds the SSRNN env (only if it's missing)
conda activate SSRNN
```
Check it's there instead of rebuilding:
```bash
ls /blue/npadillacoreano/t.heeps/.conda/envs/SSRNN/bin/python && echo "env OK"
```

---

## Step 1 — Prepare the windows  (once, light, login node OK)

**What it does:** reads the raw `.h5` respiration + BORIS `.csv`, cleans (downsample→50 Hz,
bandpass 0.1–20 Hz, z-score), tiles each recording into the **top-10 sniffing-richest
non-overlapping 30 s windows**, and writes the arrays to
[`../respiration/data_prepared/`](../../respiration/data_prepared/).

This is fast and CPU-only, so the direct-Python path on the login node is fine:
```bash
/blue/npadillacoreano/t.heeps/.conda/envs/SSRNN/bin/python \
  respiration/valence/prepare_respiration.py --config respiration/valence/config_respiration_hpg.yaml
```

**You should see** `respiration/data_prepared/` containing `observations.npy`,
`labels.npy`, `meta.npz`, `label_map.json`. Verify the shape is the expected 147 windows:
```bash
/blue/npadillacoreano/t.heeps/.conda/envs/SSRNN/bin/python -c \
"import numpy as np; print('observations', np.load('respiration/data_prepared/observations.npy').shape)"
# -> observations (147, 1500, 1)
```

> **Already prepared.** The current `data_prepared/` is the 147-window set; only re-run
> this if you change the recording list, windowing, or preprocessing in the config.

---

## Step 2 — Train the model  (heavy → SLURM GPU array)

Training is **leave-one-SUBJECT-out** only. Leave-one-recording-out was dropped because
every animal appears in both a positive and a negative recording, so holding out one
recording still lets the model see the held-out animal's breathing — the real test of
whether valence generalizes **across individuals** holds out *all* of a subject's
recordings (see [06_project_context.md](01_project_context.md), finding #4).

**What it does:** trains 8 models; fold *k* holds out subject *k* (all of that animal's
recordings) and trains on the rest. One GPU per fold, submitted as a SLURM **array**.
```bash
sbatch hipergator/respiration_job_loso.slurm   # --array=0-7, partition hpg-b200
```
**Writes:** `respiration/result/resp_srnn_subject_h8_fold{0..7}.pt` (+ per-fold
`progress_subject_fold*.csv`).

### Monitor the run
```bash
squeue -u t.heeps                          # PD = pending, R = running; empty = all done
tail -f logs/resp_loso_fold0_*.log         # watch one subject fold (Ctrl-C to stop watching)
```
Each fold trains 2000 epochs (~hours). The array runs folds in parallel as GPUs free up.
Wait until `squeue` shows no `resp-srnn*` jobs before analyzing.

> **Already trained.** All 8 subject folds are present in
> `respiration/result/`. Only re-run Step 2 if you re-prepared the data or changed the
> model/training config.

---

## Step 3 — Run the classifier / analysis  (CPU → SLURM, minutes)

**What it does:** loads the trained checkpoints, runs inference (no training), and produces
the valence results: breathing-rate ROC-AUC, leakage-free decoding numbers, permutation
test, LDA projection, and latent PCA. **One job runs the subject split:**
```bash
sbatch hipergator/classifier_results.slurm
```
**Watch / read it:**
```bash
squeue -j <JOBID>                                  # wait for it to finish (~4 min)
cat logs/classifier_<JOBID>.log                    # all the numbers print here
```

**Figures land in** [`../respiration/plot/`](../../respiration/plot/):
`permutation_test_subject.png`, `lda_projection_subject.png`,
`pooled_latent_pca_by_valence.png`.

### Variants (if you want just the analyze step)
```bash
# the subject split's full analysis:
sbatch hipergator/analyze_job.slurm            # subject split

# quick peek without SLURM (small, CPU, login node tolerable):
/blue/npadillacoreano/t.heeps/.conda/envs/SSRNN/bin/python \
  respiration/valence/analyze_valence.py --config respiration/valence/config_respiration_hpg.yaml --split subject
```

---

## Step 4 — Read the results

In the log from Step 3, the sections that matter:

| Section | Question it answers | What "good" looks like |
|---|---|---|
| **(A) breathing rate** | Does rate alone separate valence? | positive ≈ 7.8 Hz vs negative ≈ 6.3 Hz, ROC-AUC 1.0 — *real but a confound* |
| **(C) signal beyond rate** | Is there valence info after removing rate? | `latent, rate regressed out` LOSO value stays well above ~0.5 |
| **(D) permutation test** | Is it statistically real (not luck)? | observed beats the shuffled null, small `p` (e.g. 0.001) |
| **(E) LDA projection** | Visualize the separating axis | per-recording means split by valence |

**Read** the `analyze_valence.py` single-model latent numbers — that is the trustworthy
result. (An older `collect_folds.py` cross-fold decoder pooled latents from
separately-trained folds whose coordinate systems aren't aligned, so its numbers were not
trustworthy; that decoder has since been removed from the pipeline.)

---

## Cheat sheet — the whole experiment from scratch

```bash
cd /home/t.heeps/blue_npadillacoreano/npadillacoreano/share/respiration-project/Supervised_SRNN_respiration
PY=/blue/npadillacoreano/t.heeps/.conda/envs/SSRNN/bin/python

# 1. prepare (once)
$PY respiration/valence/prepare_respiration.py --config respiration/valence/config_respiration_hpg.yaml

# 2. train (GPU array) — wait for it to finish in squeue
sbatch hipergator/respiration_job_loso.slurm   # leave-one-subject-out  (8)

# 3. classify / analyze (CPU) — read the log when done
sbatch hipergator/classifier_results.slurm
```

> Reminder: only **prepare** is safe to run directly on the login node. **Training** and
> the **full analysis** go through `sbatch`. See [00_quickstart.md](../00_quickstart.md).
