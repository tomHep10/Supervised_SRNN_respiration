# Rank Classification — Data Guide for an RNN Preprocessing Agent

> **Scope note (this repo's docs split):** this `rank/` guide describes the **rank**
> experiment (Dominant/Subordinate, 400 Hz, leave-one-*cage*-out, BLA cohort included). It is
> kept as a cross-experiment reference; the helper files it cites
> (`resp_helper_functions.py`, `PREPROCESSING_NOTES.md`, the `-rank.ipynb`) live in the
> **rank project**, not in this Supervised_SRNN repo, so the links below may not resolve here.
> For the *valence* experiment this repo actually runs, see the sibling
> [`../valence/RNN_DATA_GUIDE.md`](../valence/RNN_DATA_GUIDE.md) (50 Hz, valence target,
> leave-one-*subject*-out, no BLA).
>
> **Audience:** an agent (or engineer) who must turn this respiration dataset into
> sequence input for an RNN that predicts a mouse's social **Rank** (Dominant /
> Subordinate) from breathing during sniffing interactions.
>
> **Read this first, then read** [PREPROCESSING_NOTES.md](./PREPROCESSING_NOTES.md) for the
> human narrative. Code you will reuse lives in
> [resp_helper_functions.py](../../resp_helper_functions.py); the worked example is
> [feature_extraction-interactions-rank.ipynb](../feature_extraction-interactions-rank.ipynb).

---

## 0. The one thing that will trip you up

The existing pipeline outputs a **bout-level feature matrix** (one row = one sniffing bout,
~27 BreathMetrics summary stats). **That is for classical ML, not an RNN.** For a sequence
model you almost certainly want the **raw 400 Hz respiration waveform sliced per bout** (or a
per-breath sequence). So:

- **Reuse** the catalog builder, the rank map, the BORIS loader, and the signal loader.
- **Replace** `extract_bout_features` (the BreathMetrics summarization step) with your own
  window-slicing that keeps the **time series**.

Two viable RNN input representations:
1. **Raw waveform windows** — slice the 400 Hz signal to each bout window → `(T, 1)` per bout.
2. **Per-breath feature sequences** — use BreathMetrics breath landmarks to emit one vector
   per breath → `(n_breaths, n_features)` per bout.

---

## 1. How to enumerate the usable recordings

Do **not** glob the H5 directories blindly. Use the catalog builder — it pairs each
respiration H5 with the correct BORIS export and handles the naming chaos:

```python
from resp_helper_functions import (
    build_interaction_boris_catalog,
    load_rank_resp_signal_interactions,
    load_clean_boris,
    build_upstream_exclusion_set,
    filter_interaction_paths,
    normalize_subject_id, subject_id_from_trial_key, cohort_from_trial_key,
)

resp_paths, boris_paths, bla_rank_map, bla_sa_cm_pairs = build_interaction_boris_catalog(
    aim1_h5_dir, aim1_boris_dir, bla_cm_h5_dir, bla_cm_boris_dir,
    aim1_skip_boris_stems={"CM_s3_6_sub3_5_20250623_174348"},   # no BORIS
    aim1_exclude_boris_names={"CM_s4_8_sub4_7_20250623.csv"},    # bad duplicate
)
# resp_paths / boris_paths: trial_key -> absolute path
```

Trial keys look like `AIM1cm_1_1_1_2_20250623_111352` or
`BLAcm_1_1_1_3_20260107_163611_VT_SA`.

---

## 2. Which recordings are GOOD to use (apply these filters)

Reproduce the same exclusions the feature pipeline uses, or your RNN will train on garbage /
mislabeled data.

**A. Drop subjects/trials before extraction (cohort-aware):**
```python
behavior_df = ...  # concat of load_clean_boris(path, subject_only=False) over all trials
exclude_by_cohort, low_sniff_trials, report = build_upstream_exclusion_set(
    behavior_df, imbalance_ratio=0.4, drop_aim1_subjects=None, min_sniffs_per_trial=10,
)
resp_paths, boris_paths, dropped = filter_interaction_paths(
    resp_paths, boris_paths,
    exclude_subjects_by_cohort=exclude_by_cohort, exclude_trials=low_sniff_trials,
)
```

**B. Dedupe BLA double-scored sessions:** keep the **`VT_SA`** (subject) export per H5; drop
the matching `VT_CM` (social-agent) export. `bla_sa_cm_pairs` lists the affected sessions.

**C. Bout-level filters (when you slice windows):**
- Keep only `facial sniffing`, `body sniffing`, `anogenital sniffing`.
- Drop `Duration < 0.5 s`.
- Drop the ~400 s anogenital bout in `AIM1cm_2_3_2_4_20250623_151153`.
- Require `>= 2` breaths in the window if you need breath structure.

**D. Hard drops:**
- `Trial == "RI1_3_5"`.
- Any row with `Rank` missing (covers BLA `i`-role recordings — they have **no rank**).
- **Aim1 `1_1` and `1_2`** (fighting sessions) — excluded in the balanced variant.
- `min_windows < 10` per subject in the balanced variant.

> If you want the conservative, paper-ready set, mirror `master_df_balanced`. If you want
> maximum data and will handle imbalance in the loss, start from the unbalanced `master_df`
> filters but **still** drop missing-rank rows and the explicit one-offs above.

---

## 3. Rank labels (the target)

### Aim1 — hard-coded map (use verbatim)
```python
aim1_rank_map = {
    "1_1": "Subordinate", "1_2": "Dominant",
    "2_3": "Subordinate", "2_4": "Dominant",
    "3_5": "Subordinate", "3_6": "Dominant",
    "4_7": "Subordinate", "4_8": "Dominant",
}
```

### BLA — derived from the H5 filename role letter, returned as `bla_rank_map`
`d → Dominant`, `s → Subordinate`, `i → no label (drop the recording)`.

### ⚠️ Do NOT naively merge the two maps
They share keys (`1_1`, `1_2`, `2_3`) that refer to **different mice**. The map is keyed by
subject id only, so a blind merge **overwrites Aim1 ranks with BLA ranks**. Safe approach for
the RNN: build a **`(cohort, subject_id)` → rank** lookup instead:

```python
def rank_for(cohort, subj):
    if cohort == "Aim1":
        return aim1_rank_map.get(subj)
    if cohort == "BLA":
        return bla_rank_map.get(subj)
    return None
```
Derive `cohort` and `subj` from the trial key via `cohort_from_trial_key()` and
`subject_id_from_trial_key()`.

---

## 4. Caging structure (critical for train/test splits)

Animals interact **only with cagemates**. The **first index** of a subject id is the cage.
**You must split by cage (or at least by subject) — never randomly by bout** — or the model
leaks identity/cage between train and test.

### Aim1 — 4 cages × 2 mice (global animal numbering 1–8)
| Cage | Mice (rank)                 |
|------|-----------------------------|
| 1    | 1_1 (Sub), 1_2 (Dom) — *excluded, fighting* |
| 2    | 2_3 (Sub), 2_4 (Dom)        |
| 3    | 3_5 (Sub), 3_6 (Dom — no BORIS) |
| 4    | 4_7 (Sub), 4_8 (Dom)        |

### BLA — up to 8 cages × ~3 mice (per-cage numbering `X_1, X_2, X_3`)
| Cage | Mice observed          |
|------|------------------------|
| 1    | 1_1, 1_2, 1_3          |
| 2    | 2_1, 2_2, 2_3          |
| 3    | 3_1, 3_2, 3_3          |
| 4    | 4_1, 4_2, 4_3          |
| 5    | 5_1, 5_2, 5_3          |
| 6    | 6_2, 6_3               |
| 7    | 7_1, 7_2               |
| 8    | 8_1, 8_2               |

- BLA cagemates interact **round-robin** (1_1↔1_2, 1_1↔1_3, 1_2↔1_3, …).
- A recording's `subject` and `social_agent` are **always same-cage**. No cross-cage data.
- **Recommended CV:** `GroupKFold` with `group = (cohort, cage)`, falling back to
  `group = (cohort, subject)` if you need more folds. The notebook imports `GroupKFold` for
  exactly this reason.

> Numbering differs by cohort (Aim1 global pairs vs BLA per-cage triads). Always carry
> `cohort` alongside the id so `1_1` is unambiguous.

---

## 5. How to parse a recording into RNN input

```python
# 1) Load the cleaned 400 Hz respiration trace (raw-preferred pipeline)
signal, time, fs, meta = load_rank_resp_signal_interactions(
    h5_path, target_rate=400, prefer_raw=True, verbose=False
)   # signal: 1-D float @ 400 Hz; time: seconds; fs == 400.0

# 2) Load the matching behavior bouts (keep both actors; or subject_only=True)
bouts = load_clean_boris(boris_path, subject_only=False)
#   columns: Behavior, Initiator ('subject'/'social_agent'), Start, Stop, Duration

# 3) Slice the signal per bout window -> a sequence per bout
WIN = 2.0  # seconds, onset-anchored (matches the feature pipeline)
for _, b in bouts.iterrows():
    t0 = b["Start"]
    t1 = b["Start"] + WIN              # onset anchor; or use b["Stop"] for full bout
    m  = (time >= t0) & (time < t1)
    seq = signal[m]                    # shape (T,), T ~= WIN * fs = 800 samples
    # -> stack into (T, 1), pad/truncate to fixed T, z-score per sequence/subject
```

Metadata to attach to every sequence sample:
`Trial, Subject, social_agent, Cohort, Rank, Behavior, Initiator, Start, Stop, Duration,
h5_path, boris_path`.

### Signal facts you need
- Final rate is **400 Hz** (re-derived from raw 20 kHz). Stored `resp_clean` is 100 Hz —
  the loader bypasses it unless raw is missing.
- Preprocessing already applied: Butterworth low-pass → downsample → NeuroKit band-pass
  0.1–20 Hz. **Don't double-filter.**
- Expect occasional NaN/Inf guards; the notebook prints signal `nan/inf/min/max/std`
  sanity checks per session — replicate those before training.
- `time[0]` is not always 0; always mask by `time`, don't assume sample index 0 = t=0.

---

## 6. The BORIS gotcha (will silently mislabel actors)

In raw BORIS CSVs the column named **`Subject`** is the **actor** (`subject` /
`social_agent`), **not** the mouse id. `load_clean_boris` renames it to **`Initiator`**. The
mouse id (`Subject`) is added later from the trial key. If you read the CSVs yourself,
replicate this rename or you will assign breaths to the wrong animal.

- `subject_only=True` keeps only bouts the focal subject initiated.
- `subject_only=False` keeps both — useful if you also want the cagemate's breathing, but
  remember the **respiration H5 is the focal subject's signal**; a `social_agent`-initiated
  bout is still the focal subject's breathing during that interaction.

---

## 7. Recommended end-to-end recipe for the RNN agent

1. Build catalog (§1) with the two Aim1 skip/exclude sets.
2. Build `behavior_df` and apply `build_upstream_exclusion_set` + `filter_interaction_paths` (§2A).
3. Dedupe BLA `VT_SA`/`VT_CM` → keep `VT_SA` (§2B).
4. Build the `(cohort, subject) → rank` lookup (§3); drop trials with no rank.
5. For each surviving trial: load 400 Hz signal (§5), load bouts, apply bout filters (§2C),
   slice onset-anchored windows, z-score, pad/truncate to fixed length.
6. Drop the explicit one-offs: `RI1_3_5`, the 400 s anogenital bout, Aim1 `1_1`/`1_2` if
   matching the balanced set.
7. Split with `GroupKFold` on `(cohort, cage)` (§4). Never split bouts randomly.
8. Keep a manifest (parquet/csv) of every emitted sequence with its metadata for traceability.


