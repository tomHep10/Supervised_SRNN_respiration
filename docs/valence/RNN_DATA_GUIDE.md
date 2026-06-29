# Valence Classification — Data Guide for an SRNN Preprocessing Agent

> **Audience:** an agent (or engineer) who must turn this respiration dataset into
> sequence input for the **Supervised SRNN** that predicts a recording's social
> **Valence** (positive / negative) from breathing during sniffing interactions.
>
> **Sibling doc:** [`../rank/RNN_DATA_GUIDE.md`](../rank/RNN_DATA_GUIDE.md) is the same guide
> for the **rank** experiment (Dominant/Subordinate, 400 Hz, leave-one-*cage*-out, includes a
> BLA cohort). This valence guide downsamples to **50 Hz**, targets **valence**, splits
> **leave-one-subject-out**, and **ignores BLA entirely**. Don't cross the wires.
>
> **Unlike the rank guide, the valence preprocessing is already written.** The whole
> pipeline below is implemented in [`../../respiration/valence/prepare_respiration.py`](../../respiration/valence/prepare_respiration.py),
> driven by [`../../respiration/valence/config_respiration_hpg.yaml`](../../respiration/valence/config_respiration_hpg.yaml).
> Read this to understand what that script does (and how to adapt it), not to rebuild it from
> scratch. Procedural runbook: [02_experiment_runbook.md](02_experiment_runbook.md).

---

## 0. The one thing that will trip you up

A bout-level feature matrix (one row per bout of BreathMetrics summary stats) is for
**classical ML, not an SRNN.** A sequence model needs the **raw waveform sliced into windows**
— here the 50 Hz respiration trace cut into fixed **30 s windows** (`T = 1500` samples) shaped
`(T, 1)`. `prepare_respiration.py` already emits exactly this:
`observations.npy` of shape `(n_windows, 1500, 1)` plus per-sample behavior `labels.npy`.

So if you are adapting the pipeline: keep the recording catalog, the valence map, the BORIS
loader, and the signal loader; the windowing **keeps the time series** (it does not summarize
it).

---

## 1. How to enumerate the usable recordings

Do **not** glob the H5 directories blindly. The recordings (and their H5/BORIS pairing) are
declared explicitly in the config so the set is reproducible:

```yaml
# config_respiration_hpg.yaml
recordings:                 # 15 total: 7 positive (RI1) + 8 negative (RI2)
  - "RI1_s1_1"  ...  - "RI2_s4_8"
paths:
  h5_dir:  ...   # raw respiration .h5 (key "resp", 20 kHz)
  csv_dir: ...   # matching BORIS .csv exports
```

`prepare_respiration.py` matches each `RIx_sN_M` stem to its `.h5` and `.csv` by that
3-token prefix (`find_one`). Trial keys are `RI1_s1_1`-style: the **`RIx` prefix is the
valence**, the **`sN_M` token is the subject** (see §3, §4).

---

## 2. Which recordings / windows are GOOD to use (the filters that already run)

Reproduce these, or the SRNN trains on garbage / mislabeled data. All are applied inside
`prepare_respiration.py`.

**A. Behavior labels = the 3 sniff types only.** Everything else maps to `0` (nonsocial).
Posturing / chasing / fighting are disregarded.
```yaml
behavior_states: {"body sniffing": 1, "anogenital sniffing": 2, "facial sniffing": 3}
```

**B. Dead-signal removal (BEFORE windowing).** Flat / low-amplitude "dead" respiration is
detected by local amplitude (rolling std over `dead_win_sec = 1 s`) and flagged where it falls
below `dead_rel_thresh = 0.25` × the **p75** of local std (a robust "typical active amplitude"
reference; the median would be dragged down by dead-heavy recordings). Runs shorter than
`dead_min_sec = 2 s` are spared (don't nick brief between-breath dips). Then:
- dead samples have their **behavior labels zeroed** (a BORIS sniff bout sitting on dead
  signal no longer counts as sniffing), and
- any window more than `max_window_dead_frac = 0.20` dead is **dropped from the candidate pool**
  before window selection.

These thresholds are **data-tuned, not guessed** — see
[01_project_context.md](01_project_context.md) and the `dead_signal_mask` docstring. Bimodal
auto-thresholding (Otsu/GMM) **fails** here (the dead tail is unimodal, ~1–5%); don't "make it
dynamic."

**C. Window selection.** Each recording is tiled into **non-overlapping contiguous 30 s
windows**; the **top `windows_per_recording = 10`** by sniffing fraction are kept (a short
recording yields fewer). No overlap, no stitching, no event-centering.

> Current set: **147 windows** `(147, 1500, 1)`, 70 positive / 77 negative. (Count is stable
> across the dead-signal change because dead windows are *replaced* by cleaner ones, not
> subtracted — see the runbook.)

---

## 3. Valence labels (the target)

Valence comes straight from the `RIx` prefix — **no hand-keyed map of subjects**:
```yaml
valence_map:
  RI1: 1   # positive
  RI2: 0   # negative
```
So `RI1_s2_3 → 1`, `RI2_s2_3 → 0`. The **same animal** (`s2_3`) contributes both a positive
and a negative recording — which is exactly why the split must hold out *subjects*, not
recordings (§4).

> **Rank is not the target here**, but rank is still a real per-animal attribute of these mice
> (the cage's lower global index is Subordinate, the higher is Dominant — see the rank guide).
> Carry it as metadata if useful; never train on it in this experiment.

---

## 4. Subject / caging structure (critical for the train/test split)

Animals interact **only with cagemates**. A subject id is `sCAGE_ANIMAL` — the **first index
is the cage**, the trailing number is the global individual. **You must split by subject (we
use leave-one-SUBJECT-out) — never randomly by window** — or the model leaks an individual's
breathing signature between train and test.

### The 8 subjects (4 cages × 2 mice)
| Cage | Subjects (valences present) |
|------|-----------------------------|
| 1    | s1_1 (RI1+RI2), s1_2 (RI1+RI2) |
| 2    | s2_3 (RI1+RI2), s2_4 (RI1+RI2) |
| 3    | s3_5 (RI2 only), s3_6 (RI1+RI2) |
| 4    | s4_7 (RI1+RI2), s4_8 (RI1+RI2) |

- Every subject except `s3_5` appears in **both** valences, so leave-one-subject-out tests
  whether valence generalizes **across individuals** rather than memorizing one animal's
  breathing. **8 subjects → 8 folds.**
- **Why subject and not recording:** leave-one-*recording*-out leaks — a held-out recording's
  subject is still in training. That split was **removed** from this repo; subject is the only
  CV. (The rank experiment goes one coarser — leave-one-*cage*-out.)
- Subject = the **full `sN_M` token**. Grouping by the leading `sN` alone (cage) would be the
  rank-style split; for valence we hold out the individual.

The split is implemented in `train_respiration.py::split_indices(mode="subject")`; the SRNN
array is [`../../hipergator/respiration_job_loso.slurm`](../../hipergator/respiration_job_loso.slurm)
(`--array=0-7`, `--split subject`).

---

## 5. How a recording becomes SRNN input

```python
# inside prepare_respiration.py, per recording:
# 1) load the raw 20 kHz respiration trace (H5 key "resp")
raw, fs, trial = load_resp(h5_path, "resp")

# 2) clean -> 50 Hz: Butterworth low-pass (order 4) -> downsample to target_fs=50
#    -> band-pass 0.1-20 Hz (order 2).  DON'T double-filter.
resp = preprocess_resp_to_target_rate(raw, fs, 50, lowpass_order=4,
                                       bandpass_low=0.1, bandpass_high=20, bandpass_order=2)

# 3) z-score over the WHOLE recording (dead stretches barely move mean/std)
resp = (resp - resp.mean()) / (resp.std() + 1e-8)

# 4) rasterize BORIS bouts to per-sample behavior labels, zero dead samples,
#    then tile into non-overlapping 30 s windows (T=1500) and keep the top-10 sniffing-rich
```

Outputs to `data_prepared/`: `observations.npy (n,1500,1)`, `labels.npy (n,1500,1)`,
`meta.npz` (`valence`, `recording_id`, `recording_names`, `window_starts`,
`window_dead_frac`), `label_map.json`.

### Signal facts you need
- Final rate is **50 Hz** (re-derived from the raw 20 kHz `resp` key). `T = 30 s × 50 Hz = 1500`.
- Preprocessing is **already applied** in step 2 — don't band-pass again downstream.
- z-scoring uses the **whole recording**, intentionally (dead stretches are low-amplitude and
  barely shift mean/std). Change only if asked.
- Sanity-check `nan/inf/min/max/std` per recording before training, as in any resp pipeline.

---

## 6. The BORIS gotcha (will silently mislabel actors)

In raw BORIS CSVs the column named **`Subject`** is the **actor** (`subject` /
`social_agent`), **not** the mouse id. The loader keeps only the focal subject's bouts when
`subject_only: true` (the config default). If you read the CSVs yourself, replicate that or you
will assign breaths to the wrong animal. Remember the **respiration H5 is the focal subject's
signal**, so a `social_agent`-initiated bout is still the focal subject's breathing during that
interaction.

---

## 7. Recommended end-to-end recipe

1. Confirm the 15 recordings + paths in `config_respiration_hpg.yaml` (§1).
2. Run prep (CPU, fast): `sbatch hipergator/prepare_respiration.slurm` — applies the 3-sniff
   label map, dead-signal removal, and top-10 window selection (§2), writes `data_prepared/`.
3. **Read the printout**: per-recording `dropped(dead)` / `dead % of rec`; confirm the window
   count and shape `(n, 1500, 1)`.
4. Train leave-one-**subject**-out: `sbatch hipergator/respiration_job_loso.slurm`
   (`--array=0-7`). → `respiration/result/resp_srnn_subject_h8_fold{0..7}.pt`.
5. Analyze (CPU): `sbatch hipergator/classifier_results.slurm` → leakage-free LOSO valence
   decoding + permutation test + figures.
6. Never split windows randomly; never train on rank; never feed in BLA recordings here.
