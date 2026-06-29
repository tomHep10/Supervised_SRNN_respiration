# respiration/ — SRNN on respiration for social-valence discovery (created by Claude)

Pipeline: raw `.h5` respiration + BORIS `.csv` → **per-recording, non-overlapping 30 s
windows from sniffing-rich periods** → train the SRNN → test whether the learned
**latent dynamics and switch statistics differ between positive (RI1) and negative (RI2)
recordings**. `SRNN/` is imported, never modified.

## Files
| File | Role |
|---|---|
| `config_respiration.yaml` | params + **local** data paths (for smoke tests) |
| `config_respiration_hpg.yaml` | same, with **HiPerGator** data paths |
| `prepare_respiration.py` | clean + window the recordings → `data_prepared/` |
| `train_respiration.py` | train the SRNN (lean loop, subject-level split) |
| `plot_respiration.py` | per-fold: resp reconstruction, states, latent PCA by valence |
| `analyze_valence.py` | **leakage-free** cross-fold valence test (latent + switch stats) |

## Data design (matches the agreed spec)
- 4 recordings = 4 independent sequences. **RI1 = positive, RI2 = negative.**
- Respiration cleaned with your pipeline: lowpass(N=4, cut=fs/2) → downsample(→50 Hz) →
  bandpass(0.1–20 Hz). (`nk.signal_filter` butterworth == scipy `sosfiltfilt`.)
- Each recording tiled into **non-overlapping contiguous 30 s windows (T=1500)**; the
  **10 richest in sniffing** kept. No overlap, no stitching, no event-centering.
- Result: **40 windows** `(40, 1500, 1)`, 20 positive / 20 negative.
- Behavior states = 3 sniff types only (body=1, anogenital=2, facial=3; 0=nonsocial).
  Posturing/chasing/fighting are disregarded → map to 0.

## Run order
```bash
# local smoke test (CPU, sleap-new env):
conda run -n sleap-new python respiration/prepare_respiration.py --config respiration/config_respiration.yaml
conda run -n sleap-new python respiration/train_respiration.py  --config respiration/config_respiration.yaml --fold 0 --split subject --epochs 50
conda run -n sleap-new python respiration/plot_respiration.py   --config respiration/config_respiration.yaml --fold 0 --split subject

# HiPerGator (GPU): see ../hipergator/respiration_job_loso.slurm — runs folds 0-7, then:
python respiration/analyze_valence.py --config respiration/config_respiration_hpg.yaml --split subject
```

## ⚠️ Key scientific caveats
1. **Supervision confound (decide `coef_cross`).** RI2 (negative) windows have ~0%
   sniffing, so the sniff-state labels essentially *only* fire for RI1. With
   `coef_cross > 0` the model is pushed to put sniff-states in RI1 and nonsocial in RI2 —
   which means any valence separation could be **driven by the labels, not discovered
   from respiration.** For a genuine *discovery* claim, run `coef_cross: 0`
   (unsupervised switches) — `--coef_cross 0`. Running both (0 and 0.5) and comparing is
   ideal. **Recommended primary run: `coef_cross = 0`.**
2. **Pilot power.** Valence is 2 vs 2 recordings. `analyze_valence.py --split subject` is
   leakage-free, but n=4 → treat any separation as suggestive, not significant.
3. **Behavior–physiology sync** (video ~715 s vs physiology ~621 s): assumes t=0 start;
   affects only the sniff labels, not the respiration signal or per-recording valence.
   With `coef_cross=0` it's irrelevant.

## Key knobs (config)
| Param | Meaning |
|---|---|
| `windowing.window_sec` / `windows_per_recording` | 30 s windows; how many per recording (10) |
| `preprocess.target_fs`, bandpass params | your cleaning pipeline (50 Hz, 0.1–20 Hz) |
| `model.num_tv`, `hidden_shape` | # discrete states (4), latent dim (8) |
| `train.coef_cross` | **0 = discovery (recommended), 0.5 = behaviorally supervised** |
| `train.split_mode` | `subject` (leave-one-subject-out, no leakage) |

## What `analyze_valence.py --split subject` reports
For the pooled held-out windows: switch-rate and state-occupancy **by valence**, plus
leakage-free leave-one-subject-out decoding of valence from (a) the latent `h`,
(b) switch statistics, (c) both.

<!-- ======================================================================= -->
<!-- NEW ADDITION (everything above this line is the original doc)            -->
<!-- Last updated 2026-06-26 — describes the experiment currently being run   -->
<!-- ======================================================================= -->

## Current experiment (running now — updated 2026-06-26)

> Supersedes the "Data design" section above, which describes the original 4-recording
> pilot. The experiment now running uses the **full 15-recording set in discovery mode**.

**What changed from the original pilot**

| | Original pilot (above) | **Current run** |
|---|---|---|
| Recordings | 4 (2 positive / 2 negative) | **15 — 7 positive (RI1) + 8 negative (RI2)** |
| Windows | 40 `(40,1500,1)` | **147 `(147,1500,1)`** (top-10 sniffing-rich 30 s windows/recording) |
| `train.coef_cross` | 0.5 (behaviorally supervised) | **0.0 — DISCOVERY** (switches learned from respiration, not sniff labels) |
| Cross-validation | array 0–3 | **array 0–7** (leave-one-subject-out across all 8 subjects) |

**Why discovery mode.** RI2 (negative) windows have ~0% sniffing, so with `coef_cross > 0`
the sniff-state labels essentially only fire for RI1 and any valence separation could be
label-driven. `coef_cross = 0` makes the switches **discovered from respiration** (see caveat
#1 above). The behaviorally-supervised variant (`coef_cross = 0.5`) can be run separately for
comparison.

**Recordings (15), valence from the `RIx` prefix** (`RI1 = positive = 1`, `RI2 = negative = 0`):
```
positive (RI1): RI1_s1_1  RI1_s1_2  RI1_s2_3  RI1_s2_4  RI1_s3_6  RI1_s4_7  RI1_s4_8
negative (RI2): RI2_s1_1  RI2_s1_2  RI2_s2_3  RI2_s2_4  RI2_s3_5  RI2_s3_6  RI2_s4_7  RI2_s4_8
```

**Fixed settings (unchanged):** `model.num_tv = 4`, `hidden_shape = 8`, `bottleneck_shape = 16`,
`neural_private_shape = 8`; preprocess → 50 Hz, bandpass 0.1–20 Hz, z-scored, subject-only;
`train.epochs = 2000`, `lr = 0.001`, `batch_size = 256` (all train windows in one batch),
`split_mode = subject`, `seed = 131`.

**How it's launched** — `hipergator/respiration_job_loso.slurm` as a SLURM array
(`--array=0-7`, partition `hpg-b200`, 1 GPU/fold), each task running:
```bash
python respiration/train_respiration.py --config respiration/config_respiration_hpg.yaml --fold $SLURM_ARRAY_TASK_ID --split subject
python respiration/plot_respiration.py  --config respiration/config_respiration_hpg.yaml --fold $SLURM_ARRAY_TASK_ID --split subject
```
Writes checkpoints `respiration/result/resp_srnn_subject_h8_fold{0..7}.pt`. After all 8
folds finish, pool them for the leakage-free valence number:
```bash
python respiration/analyze_valence.py --config respiration/config_respiration_hpg.yaml --split subject
```

> **Figures note:** per-fold PNGs in `respiration/plot/` (`resp_recon.png`, `states.png`,
> `latent_pca.png`) share filenames across folds, so they show only the **last fold to run**.
> Checkpoints for all folds are intact, so `analyze_valence.py` still uses everything. Add a
> `_fold{k}` suffix to the figure filenames to keep all 8.
