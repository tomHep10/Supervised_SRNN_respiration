# Handoff — dead-signal removal added; retrain on HiPerGator

**Context for the next agent.** The previous session added **dead-signal removal** to the
respiration preprocessing. Code + configs are changed but **nothing has been re-prepared or
retrained yet**. Your job: get the changes onto HiPerGator, regenerate the data, retrain,
and re-run the analysis. The `docs/07_experiment_runbook.md` "already prepared / already
trained" notes are now **STALE** — ignore them; data and models must be rebuilt.

---

## What changed this session (already in the working tree)

1. **`respiration/prepare_respiration.py`** — dead-signal filtering applied **before** windowing:
   - New `dead_signal_mask()` — flags "dead" respiration (flat / low-amplitude noisy
     stretches) by local amplitude = rolling std over a 1 s window. A sample is dead if its
     local std falls below `dead_rel_thresh` × the **p75** of local std (a robust "typical
     active amplitude" reference). New `_remove_short_runs()` drops dead runs shorter than
     `dead_min_sec` so brief between-breath dips survive.
   - In `main()`: dead samples have their **behavior labels zeroed** (`labels[dead] = 0`), so
     a BORIS sniff bout sitting on dead signal no longer counts as sniffing; and any window
     more than `max_window_dead_frac` dead is **dropped from the candidate pool** before the
     top-N-by-sniffing selection. `meta.npz` gains `window_dead_frac`; the per-recording
     printout now shows `dropped(dead)` and `dead % of rec`.

2. **`respiration/config_respiration.yaml`** and **`respiration/config_respiration_hpg.yaml`**
   — new `preprocess` keys (identical in both):
   `remove_dead_signal: true`, `dead_win_sec: 1.0`, `dead_ref_pct: 75.0`,
   `dead_rel_thresh: 0.25`, `dead_abs_thresh: 0.0`, `dead_min_sec: 2.0`,
   `max_window_dead_frac: 0.20`.

3. **(unrelated, earlier in session)** `analyze_valence.py`: `loro_decode` was renamed
   `logo_decode` (no external references — self-contained).

### These thresholds are data-tuned, not guessed
The local-amplitude distribution was measured on all 15 raw recordings:
- **Bimodal auto-thresholding (Otsu/GMM) FAILS here** — the distribution is unimodal (dead
  is only a ~1–5% low tail), so Otsu splits the *active* signal and flags ~50–78%. Do not
  "make it dynamic" with a bimodal method.
- The chosen p75-relative cut at 0.25 was validated per-recording: it catches the real dead
  chunks — **RI2_s2_4 (27.6% dead, longest run 58 s)**, RI2_s1_2 (11.6%, 46 s),
  RI2_s2_3 (8.4%, 49 s), RI1_s2_3 (4.6%, 28 s) — leaves clean recordings at 0–3%, and
  correctly does **not** flag **RI2_s3_6** (low amplitude but alive — this is why the
  reference is p75, not the median, which RI2_s2_4's dead chunk would drag down).

---

## What to do on HiPerGator (in order)

Repo on HPG:
`/home/t.heeps/blue_npadillacoreano/npadillacoreano/share/respiration-project/Supervised_SRNN_respiration`
Env python: `PY=/blue/npadillacoreano/t.heeps/.conda/envs/SSRNN/bin/python`

0. **Sync the code** so HPG has the new `prepare_respiration.py` + both configs
   (git pull on HPG after the user pushes, or rsync). Confirm `dead_signal_mask` is present:
   `grep -n dead_signal_mask respiration/prepare_respiration.py`.

1. **Re-prepare the windows** (login node OK, CPU, fast):
   ```bash
   $PY respiration/prepare_respiration.py --config respiration/config_respiration_hpg.yaml
   ```
   - **Read the printout.** Expect `RI2_s2_4` to drop most of its windows; total window
     count will be **lower than the old 147**. That is expected and correct.
   - Sanity-check the shape changed:
     `$PY -c "import numpy as np; print(np.load('respiration/data_prepared/observations.npy').shape)"`

2. **Retrain BOTH arrays** (the old checkpoints were trained on dead-contaminated windows —
   overwrite them). Wait for `squeue -u t.heeps` to clear between submit and analysis.
   ```bash
   sbatch hipergator/respiration_job.slurm        # leave-one-recording-out (15 folds)
   sbatch hipergator/respiration_job_loso.slurm   # leave-one-subject-out  (8 folds)
   ```

3. **Re-run the analysis** (CPU):
   ```bash
   sbatch hipergator/classifier_results.slurm
   cat logs/classifier_<JOBID>.log
   ```

4. **Check the results** against the pre-cleanup baseline (rate-removed latent ≈ 0.73,
   permutation p ≈ 0.001; see `docs/06_project_context.md`). Two things to confirm:
   - **`respiration/plot/resp_recon.png` no longer reconstructs flat dead signal** — this
     was the original motivation for the change.
   - The permutation test / decoding numbers in
     `respiration/plot/permutation_test_{recording,subject}.png` are still meaningful (and
     ideally cleaner now that dead windows aren't diluting training).

Full procedural detail (commands, what each step writes, monitoring) is in
[docs/07_experiment_runbook.md](docs/07_experiment_runbook.md).

---

## Tuning knobs (if the cuts look wrong in the printout)
- Too much flagged dead → lower `dead_rel_thresh` (e.g. 0.20) and re-prepare.
- Dead signal still leaking through → raise `dead_rel_thresh` (e.g. 0.30) or set a small
  `dead_abs_thresh` (z-scored units).
- Want only fully-clean windows → set `max_window_dead_frac: 0.0`.
All live in the `preprocess` block of `config_respiration_hpg.yaml`.

## Notes
- Diagnostic scripts used to tune the thresholds live in the previous session's scratchpad
  (not in the repo); the logic is preserved in `dead_signal_mask`'s docstring.
- z-scoring still uses the whole recording (dead stretches barely affect mean/std since
  they're low-amplitude); left as-is intentionally. Change only if asked.
