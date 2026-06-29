# Presentation Handoff — Respiration SRNN: Social Valence & Rank

> **For the agent building the presentation.** This document is a self-contained brief plus an
> index of every artifact (figures, logs, consolidated reports) you need. All paths are
> **relative to the repository root** (`.../Supervised_SRNN_respiration/`). Read the
> consolidated per-run reports first (they hold every number); pull figures by the paths listed
> here. Numbers in this brief are the headline results as of the latest runs — if a report .txt
> disagrees, trust the report (it is regenerated each run).

---

## 1. The question

Can a mouse's **social state** be read out from its **respiration (breathing) signal alone**,
using an unsupervised-dynamics model of the breathing waveform? Two separate experiments:

- **Valence** — does breathing during a resident–intruder interaction distinguish a
  **positive (RI1)** from a **negative (RI2)** social encounter?
- **Rank** — does breathing during a cagemate interaction distinguish a **Dominant** from a
  **Subordinate** animal?

The model is a **Supervised Switching Recurrent Neural Network (SRNN)**: it learns a small set
of discrete latent "states" with their own nonlinear dynamics from the breathing waveform, plus
a latent trajectory. We then ask whether that latent separates the social classes — in a
**leakage-free** way.

## 2. What was done (methods — the story)

1. **Raw data** — 20 kHz respiration `.h5` + BORIS behavior `.csv` (sniffing bouts) per recording.
2. **Preprocessing** (identical across experiments): low-pass → **downsample to 50 Hz** →
   band-pass 0.1–20 Hz → z-score.
3. **Dead-signal removal** — flat/low-amplitude "dead" stretches of the breathing trace are
   detected (rolling-amplitude vs a robust p75 reference) and excluded *before* windowing, so
   the model never trains on or reconstructs dead signal. Thresholds were data-tuned (a bimodal
   auto-threshold fails here; the dead tail is a unimodal ~1–5%).
4. **Windowing** — each recording is cut into non-overlapping **30 s windows (T = 1500)**; the
   **top-10 sniffing-richest** windows per recording are kept.
5. **Model** — per-state RNN dynamics + a transformer inference network; trained in **discovery
   mode** (`coef_cross = 0`, i.e. the discrete states are learned from the breathing itself, not
   from the sniff labels), 2000 epochs/fold.
6. **Leakage-free cross-validation** — this is the crux:
   - **Valence → leave-one-SUBJECT-out (LOSO).** Every animal appears in both a positive and a
     negative recording, so holding out a whole subject prevents the model from memorizing an
     individual's breathing. (Leave-one-*recording*-out was explicitly removed — it leaks.)
   - **Rank → leave-one-CAGE-out (LOCO).** Animals interact only with cagemates, so holding out
     a whole cage prevents cagemate-identity leakage.
7. **Leakage-aware analysis** (per experiment): (A) recording-level breathing-rate → class
   ROC-AUC; (B) pooled latent PCA; (C) **decode of class from the latent with breathing-rate
   regressed out** (the key "signal beyond rate" test); (D) **permutation test** (labels shuffled
   at the recording level); (E) held-out **LDA projection**; rank also has (F) a permutation test
   on the per-recording LDA separation.

## 3. Headline results

### Valence — a real signal beyond breathing rate ✅
- **Data:** 147 windows, 8 subjects (LOSO, 8 folds).
- **Breathing rate alone separates valence perfectly — but it's a confound:** ROC-AUC = **1.00**
  (positives breathe faster, ~7.9 vs ~6.5 Hz).
- **Signal beyond rate (the real claim):** with breathing rate regressed out of the latent,
  leave-one-subject-out decode = **0.756 balanced-acc**, **permutation p = 0.001**. Full latent =
  0.762, p = 0.001. → valence information in the breathing dynamics **generalizes across
  individuals**, not explained by rate alone.
- **Reconstruction MSE** (held-out): 0.0025–0.021 across folds (model fits the breathing well).

### Rank (base cohort) — no significant signal (and underpowered) ⚠️
- **Data:** 70 windows, 7 recordings, 4 cages (LOCO, 4 folds). One cage is single-rank in this
  cohort (the Dominant `s3_6` was dropped for having no scored behavior).
- **Breathing rate carries no rank signal:** ROC-AUC = **0.333** (≈ chance/anti-correlated).
- **Latent decode (LOCO):** rate-only 0.604, full latent 0.446, rate-removed 0.550.
- **Permutation tests:** decode (rate-removed) **p = 0.18**; per-recording LDA separation
  observed AUC 0.667 but **p = 0.40**. → the apparent LDA separation is **not statistically
  real**; with only 4 cages the test is very underpowered.

### Rank (incl_s3_6 cohort) — IN PROGRESS ⏳
- Re-run with the Dominant `s3_6` added back so **every cage has both ranks** (80 windows,
  balanced 40 Dom / 40 Sub, 4 cages). Training array was submitted; results will land in
  `reports/rank_incl_s3_6/report_latest.txt` and `respiration/plot_rank_incl_s3_6/`. **Check
  those for the final numbers before presenting the rank conclusion.**

**Caveat to state plainly in the talk:** the rank experiment is **small-n** (4 cages); a null
result is "not detected here / underpowered," not "proven absent." Valence is also a pilot
(n = 8 subjects) but clears a permutation test at p = 0.001.

## 4. Figure index (relative paths)

### Valence — `respiration/plot/`
| File | What it shows | Use for |
|---|---|---|
| `respiration/plot/resp_recon.png` | True vs SRNN-reconstructed breathing for held-out windows | "model fits breathing" / methods |
| `respiration/plot/states.png` | Inferred discrete states over time (the two = inhale/exhale phases) | methods, what the model learns |
| `respiration/plot/pooled_latent_pca_by_valence.png` | All windows' latent (one model), colored by valence | descriptive geometry |
| `respiration/plot/lda_projection_subject.png` | **Leakage-free** LOSO LDA separation by valence (per-window + per-recording) | the key valence result figure |
| `respiration/plot/permutation_test_subject.png` | Permutation null vs observed decode (p = 0.001) | significance of the valence result |

> Ignore `respiration/plot/permutation_test_recording.png` and `lda_projection_recording.png` —
> they are **stale** leftovers from the removed leave-one-recording-out split. Use the
> `*_subject` versions.

### Rank base — `respiration/plot_rank/`
| File | What it shows | Use for |
|---|---|---|
| `respiration/plot_rank/pooled_latent_pca_by_rank.png` | Latent colored by rank (Dom/Sub) | descriptive |
| `respiration/plot_rank/lda_projection_cage.png` | Held-out LOCO LDA separation by rank | the rank result figure (note overlap) |
| `respiration/plot_rank/permutation_test_cage.png` | Permutation null vs observed decode (p = 0.18) | "not significant" |
| `respiration/plot_rank/lda_permutation_cage.png` | Permutation test on the per-recording LDA AUC (obs 0.667, p = 0.40) | confirms the LDA separation is chance |
| `respiration/plot_rank/resp_recon.png`, `states.png` | reconstruction / states (rank data) | methods, if needed |

### Rank incl_s3_6 — `respiration/plot_rank_incl_s3_6/` (populates when training finishes)
Same figure names as rank base, in that folder.

## 5. Where the numbers live (consolidated reports + raw logs)

**Read these first — each is the complete text of one run** (config, data, per-fold
reconstruction MSE, full decode/permutation output, artifact paths):
- Valence: `reports/valence/report_latest.txt`
- Rank base: `reports/rank_base/report_latest.txt`
- Rank incl_s3_6: `reports/rank_incl_s3_6/report_latest.txt` *(after training finishes)*

Raw analysis logs (same numbers, unconsolidated), latest per experiment:
- Valence: `logs/classifier_*.log` (most recent)
- Rank: `logs/rank_classifier_*.log` (most recent; the one with section F is the current base)

Per-fold training curves (reconstruction MSE per epoch): `respiration/result*/progress_*_fold*.csv`.

## 6. Suggested slide outline

1. **Question & motivation** — read social state from breathing alone.
2. **Methods** — pipeline diagram (50 Hz → dead-signal removal → 30 s windows → SRNN);
   discovery mode; **why LOSO / LOCO** (the leakage argument is a highlight).
3. **Valence result** — rate confound (AUC 1.0) → *but* signal survives rate-removal &
   permutation (0.756, p = 0.001). Figures: `lda_projection_subject.png`,
   `permutation_test_subject.png`.
4. **Rank result** — no rate signal; latent doesn't separate rank in a leakage-free test
   (p = 0.18 / 0.40); honest "underpowered, 4 cages." Figures: `lda_projection_cage.png`,
   `lda_permutation_cage.png`. Update with the incl_s3_6 cohort if it changes the picture.
5. **Takeaways & caveats** — valence is a promising pilot; rank is inconclusive at this n;
   next step is more cages / rate-matched design.

## 7. Project provenance (for deeper detail)
- Full project + methods docs: `docs/` (start at `docs/README.md`; valence specifics in
  `docs/valence/`, rank in `docs/rank/`, model/methods in `docs/model_and_methods/`).
- Code: shared pipeline `respiration/resp_pipeline.py`, `train_respiration.py`,
  `plot_respiration.py`; valence in `respiration/valence/`, rank in `respiration/rank/`.
- Report generator: `respiration/make_report.py` (produces the `reports/<cohort>/report.txt`).
