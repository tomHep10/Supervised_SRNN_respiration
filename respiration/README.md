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
| `train_respiration.py` | train the SRNN (lean loop, recording-level split) |
| `plot_respiration.py` | per-fold: resp reconstruction, states, latent PCA by valence |
| `collect_folds.py` | **leakage-free** cross-fold valence test (latent + switch stats) |

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
conda run -n sleap-new python respiration/train_respiration.py  --config respiration/config_respiration.yaml --fold 0 --split recording --epochs 50
conda run -n sleap-new python respiration/plot_respiration.py   --config respiration/config_respiration.yaml --fold 0 --split recording

# HiPerGator (GPU): see ../hipergator/respiration_job.slurm — runs folds 0-3, then:
python respiration/collect_folds.py --config respiration/config_respiration_hpg.yaml
```

## ⚠️ Key scientific caveats
1. **Supervision confound (decide `coef_cross`).** RI2 (negative) windows have ~0%
   sniffing, so the sniff-state labels essentially *only* fire for RI1. With
   `coef_cross > 0` the model is pushed to put sniff-states in RI1 and nonsocial in RI2 —
   which means any valence separation could be **driven by the labels, not discovered
   from respiration.** For a genuine *discovery* claim, run `coef_cross: 0`
   (unsupervised switches) — `--coef_cross 0`. Running both (0 and 0.5) and comparing is
   ideal. **Recommended primary run: `coef_cross = 0`.**
2. **Pilot power.** Valence is 2 vs 2 recordings. `collect_folds.py` is leakage-free, but
   n=4 → treat any separation as suggestive, not significant.
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
| `train.split_mode` | `recording` (leave-one-recording-out, no leakage) |

## What `collect_folds.py` reports
For the pooled held-out windows: switch-rate and state-occupancy **by valence**, plus
leakage-free leave-one-recording-out decoding of valence from (a) the latent `h`,
(b) switch statistics, (c) both.
