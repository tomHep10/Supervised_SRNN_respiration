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

<!-- ======================================================================= -->
<!-- NEW ADDITION (everything above this line is the original doc)            -->
<!-- Added 2026-06-26 — first full 4-fold HiPerGator run                      -->
<!-- ======================================================================= -->

## Results — first full 4-fold HiPerGator run (run completed 2026-06-25)

**Run:** `coef_cross = 0.5` (behaviorally supervised), `hidden_shape = 8`, `num_tv = 4`,
`split_mode = recording`, 2000 epochs/fold. Checkpoints
`respiration/result/resp_srnn_recording_h8_fold{0..3}.pt`.

### Training — all 4 folds finished cleanly
All four checkpoints written; every log ends in `done. checkpoint -> …`, no tracebacks.

| Fold | Held-out valence | Final test MSE |
|---|---|---|
| 0 | positive (RI1) | 0.0054 |
| 1 | positive (RI1) | 0.0034 |
| 2 | negative (RI2) | 0.0200 |
| 3 | negative (RI2) | 0.0137 |

(The per-fold logs print `valence decoding: skipped` — expected: under leave-one-recording-out
each fold's test set is a *single* valence, so valence is only decodable by pooling across
folds in `collect_folds.py`.)

### Pooled valence test (`collect_folds.py`)
```
pooled windows=40  pos=20 neg=20
  positive(RI1): switch-rate=0.295  state-occupancy=[0.322, 0.0, 0.0, 0.678]
  negative(RI2): switch-rate=0.243  state-occupancy=[0.3, 0.0, 0.0, 0.7]

LEAKAGE-FREE valence decoding (leave-one-recording-out on the decoder):
  [latent h     ] LORO balanced-acc=0.000 acc=0.000
  [switch stats ] LORO balanced-acc=0.000 acc=0.000
  [latent+switch] LORO balanced-acc=0.000 acc=0.000
(chance=0.5; n=4 recordings -> treat as a pilot signal, not significance)
```
Per-fold mean switch-rates: fold0=0.309, fold1=0.280 (positive) vs fold2=0.252,
fold3=0.234 (negative).

### Reading
- **Descriptive signal in the expected direction.** Switch-rate is higher for positive
  (0.295) than negative (0.243), and this **rank-orders perfectly across all 4 recordings**
  (both positives above both negatives, no overlap). Signal is in switching *dynamics*, not
  state occupancy (occupancy ≈ identical; only states 0 and 3 are ever used → effectively
  bistable).
- **The 0.000 LORO accuracy is an n=4 artifact, not evidence against the effect.** With 2
  recordings/class, leave-one-recording-out can't calibrate a decision boundary from the one
  remaining same-class example, so it flips systematically (a clean 0.000 rather than ~0.5
  noise). This metric is untrustworthy in either direction at this n.
- **Caveat carried forward:** this run used `coef_cross = 0.5`, so the switch separation
  could be label-driven (see caveat #1 above). The discovery-mode run (`coef_cross = 0`) is
  still the recommended primary.

### Suggested next steps
1. More recordings — the only real fix for the LORO metric (even n=6–8 lets the decoder calibrate).
2. Interim defensible statistic: window-level decode (`split_mode='window'`) or a permutation
   test on per-recording switch-rates.
3. Re-run in discovery mode (`coef_cross = 0`) and compare to this supervised run.

> **Figures note:** per-fold PNGs in `respiration/plot/` (`resp_recon.png`, `states.png`,
> `latent_pca.png`) currently show only **fold 3** — every fold overwrites the same filenames.
> Checkpoints for all folds are intact, so `collect_folds.py` still uses everything. Add a
> `_fold{k}` suffix to the figure filenames to save all four.
