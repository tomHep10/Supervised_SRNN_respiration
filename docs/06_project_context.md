# 6. Project Context

> A standing record of *why* this repo is being used and the open questions driving it.
> This is research context (not derivable from the code). The model in this repo is the
> **supervised SRNN**, not SSLD — see [05 model comparison facts in README](README.md#a-note-on-srnn-vs-slld)
> and the SRNN/SSLD analysis from the papers.

## Goal

Investigate whether **latent dynamical models** can uncover an underlying
**social state / valence** (positive vs. negative social interaction) from
**respiration recordings**.

**Hypothesis:** latent dynamical structure may separate positive and negative valence
*even if it is not obvious from the raw respiration signal*.

---

## Dataset

Currently available:

- **4 recordings** total
  - 2 positive-valence recordings
  - 2 negative-valence recordings
- Each recording ≈ **10 minutes**
- Respiration sampled at **50 Hz**
- ≈ **30,000 timepoints per recording** (~120,000 total)
- Synchronized **SLEAP pose estimation** also available.

**Behavior:** primarily different sniffing behaviors (facial sniff, anogenital sniff,
etc.) plus nonsocial behavior. Typical sniff bouts are ≈ **1 second** (not several seconds).

---

## Original SSLD idea

Initially considered **SSLD (Switching Shared Latent Dynamics)** with two views:

- **View 1:** respiration
- **View 2:** SLEAP pose / kinematics

Potential switching states:
- facial sniff
- anogenital sniff
- body sniff
- nonsocial
- (possibly other behavioral states)

The scientific goal is **not** behavioral classification. The goal is to determine
whether the learned latent representations encode **social valence** — e.g. via
PCA/UMAP of the shared latent, or decoding valence from the latent.

---

## What I learned about SSLD

- There is **one continuous shared latent trajectory** `s_t`.
- The user chooses its dimensionality `D_s`.
- There are **K discrete switching states** `z_t`.
- `z_t` is **not** a latent vector. It is a **discrete latent variable** whose value
  selects one of K different RNN dynamics models.
- Every dynamics model operates on the **same shared latent space**.
- SSLD additionally has a **private neural latent** and a **private behavioral latent**.
- The switching state is inferred by an inference network.
- The model learns switching states because they improve reconstruction and latent
  dynamics; SSLD additionally nudges them toward known behavioral states via the
  supervised `L_switch` loss.

---

## Current concern

SSLD was designed for continuous trajectories with natural behavioral transitions.
My recordings are long, so I am concerned about:

- computational feasibility
- whether to use the full recordings
- whether to splice recordings
- whether event-centered windows make sense

Current leaning: **avoid artificial stitching** — it introduces fake behavioral transitions.

---

## Options under consideration

**Option 1 — Original SRNN (single modality) on continuous respiration.**
- preserves natural dynamics
- no second modality required
- no artificial stitching
- possibly allows much longer continuous recordings

**Option 2 — SSLD with respiration + SLEAP on continuous 20–60 s chunks** that naturally
contain multiple behaviors.

---

## Open questions (to answer from the SRNN paper / experiments)

1. Is SRNN computationally feasible on recordings of this size?
2. Is it intended to train on full continuous recordings?
3. How does it handle long sequences?
4. Would SRNN be a better fit than SSLD for my data?
5. Can SRNN still reveal latent structure related to positive vs. negative social valence?
6. If so, what latent analyses (PCA, UMAP, decoding, transition matrices, etc.) would be
   most informative?

---

## Notes relevant to these questions (from reading this codebase)

These are starting observations, **not** final answers — flagged for the analysis to come:

- **Sequence length is a real constraint here.** The generative model loops over time
  steps in Python ([model_srnn.py:91](../SRNN/model_srnn.py#L91)) and the Baum–Welch
  passes are sequential recursions over `T`. The shipped sim uses `T = 100`. At
  `T ≈ 30,000` per recording, a single forward pass would be ~300× longer per step and
  memory for the `(B, T, K, K)` tensors grows linearly in `T` — so **full 10-min
  recordings are likely infeasible without windowing** (relevant to Q1–Q3).
- This supports considering **event-centered or fixed-length windows** rather than full
  recordings — but note your concern about fake transitions applies to *stitching*, not
  to *cutting* a long recording into contiguous windows (cutting preserves real dynamics
  within each window).
- The repo is the **supervised** SRNN: it already supports a behavioral-label prior via
  `coef_cross` ([train.py:57](../SRNN/train.py#L57)). With `coef_cross = 0` it becomes
  effectively unsupervised — useful if you want the switches to be *discovered* rather
  than tied to sniff labels (relevant to Q5).
- Latent analyses (Q6) map onto saved outputs: the continuous latent `h` (inference
  network output) → PCA/UMAP/decoding; the state posterior `pos_test` → inferred
  switch sequence and transition matrices.

<!-- ======================================================================= -->
<!-- NEW ADDITION (everything above this line is the original context doc)    -->
<!-- Last updated 2026-06-26 — describes the experiment currently being run   -->
<!-- ======================================================================= -->

---

## Current experiment (running now — updated 2026-06-26)

*Not part of the original context above. The **Dataset** section above describes the
original 4-recording pilot; the experiment now running uses the full set below.*

**Dataset has grown to 15 recordings** (the "4 recordings" in the Dataset section is
superseded):
- **7 positive** (RI1): `RI1_s1_1, s1_2, s2_3, s2_4, s3_6, s4_7, s4_8`
- **8 negative** (RI2): `RI2_s1_1, s1_2, s2_3, s2_4, s3_5, s3_6, s4_7, s4_8`
- Each recording tiled into the **top-10 sniffing-richest non-overlapping 30 s windows**
  (T=1500) → **147 windows total** `(147, 1500, 1)`. Valence read from the `RIx` prefix
  (`RI1 = 1`, `RI2 = 0`).

**Model = supervised SRNN run in DISCOVERY mode (`coef_cross = 0`).** This is the key
configuration choice for Q5: RI2 (negative) windows have ~0% sniffing, so a behaviorally
supervised run (`coef_cross = 0.5`) would make any valence separation **label-driven**.
With `coef_cross = 0` the switching states are **discovered from respiration itself**, so a
valence signal would be a genuine discovery. (The supervised variant can be run separately
for comparison.)

**Cross-validation:** leave-one-recording-out across all 15 recordings — SLURM array
`0–14`, one GPU per fold (`hipergator/respiration_job.slurm`, partition `hpg-b200`).

**Fixed settings:** `num_tv = 4`, `hidden_shape = 8`, `bottleneck_shape = 16`,
`neural_private_shape = 8`; 50 Hz, bandpass 0.1–20 Hz, z-scored, subject-only;
`epochs = 2000`, `lr = 0.001`, `batch_size = 256`, `seed = 131`.

**Which open questions this run targets:** Q1–Q3 (feasibility via windowing — full 10-min
recordings are not used), Q5 (does latent / switching structure separate valence, now at
n=15 instead of n=4), and Q6 (analyses on the saved `h` and `pos_test`: switch statistics,
state-occupancy, PCA/UMAP, leakage-free LORO valence decoding via `collect_folds.py`).

*Operational details (exact recording list, launch commands, output paths) live in
[respiration/README.md](../respiration/README.md#current-experiment-running-now--updated-2026-06-26).*

---

## Findings & design decisions (run completed — 2026-06-26)

All 15 LORO folds trained (2000 epochs; held-out reconstruction MSE 0.0014–0.019 — the
30 s windowing is computationally fine, answering Q1–Q3). Analysis via
`respiration/analyze_valence.py` (CPU, inference-only; launch with
`hipergator/analyze_job.slurm`).

**1. Discovery mode collapses to 2 of 4 states = breathing phase, not behavior.**
With `coef_cross = 0`, only 2 of `num_tv = 4` states are ever used; they are the
inhale/exhale phases of the respiration cycle. This is expected, not a bug: an
unsupervised *reconstruction* objective spends its state budget where signal variance is
largest, and for a quasi-periodic ~8 Hz signal that is the oscillation phase. Social /
valence structure is a low-variance modulation the objective has no incentive to give a
state to. **Consequence:** the discrete states are a dynamics mechanism (they let the
model use different dynamics for inhale vs exhale), not interpretable "social states."

**2. The one clean valence signal so far is breathing rate — a confound, not a discovery.**
Because states = breath phase, switch rate is a proxy for breathing frequency
(`breathing_Hz = switch_rate · fs/2`). Recording-level (leakage-free, each recording
scored by the fold that held it out): **positive (RI1) ≈ 7.83 Hz vs negative (RI2)
≈ 6.28 Hz**, perfectly separable (min_pos − max_neg = +0.56 Hz, ROC-AUC = 1.000, n = 15).
Real, but it is essentially *"positive animals breathe faster."* The descriptive 8-D
latent decode (single fold-0 model, LORO-by-recording) was balanced-acc ≈ 0.71 — and the
latent is itself rate-dominated.

**3. The real test = is there valence signal BEYOND rate? YES — significant across subjects.**
`analyze_valence.py` parts (C)–(D), on the recording-trained fold-0 model (decoder + permutation
leakage-free; the SRNN feature extractor still saw the subjects — LOSO retrain pending for the
fully-clean confirmation):

| feature | LORO (recording) | **LOSO (subject, n=8)** | permutation p (LOSO) |
|---|---|---|---|
| rate only (switch rate) | 0.679 | 0.699 | — |
| full latent h | 0.714 | **0.770** | **0.001** (null 0.39 ± 0.11) |
| latent, rate regressed out | 0.716 | **0.730** | **0.001** (null 0.40 ± 0.10) |

The LOSO decode did **not** collapse (so it is not individual-respiration leakage) and is
**significant under a structure-respecting null** (labels shuffled at the recording level;
observed beat all 1000 shuffles, ~3 SD out). Critically the **rate-removed** latent is still
0.730 / p=0.001 → a valence signal that **generalizes across held-out animals AND survives
breathing-rate removal**. NB the permutation null centers at ~0.40, not 0.5 — comparing to
chance=0.5 would have mis-stated the baseline. Effect-size estimate (0.77) still has wide CI at
n=8; the permutation p is the trustworthy claim. Plots: `permutation_test_recording.png`,
`lda_projection_recording.png`.

**4. SUBJECT LEAKAGE — LORO is not enough; use LOSO.** Subject = the **full `sX_Y` token**
(trailing number = global individual id; leading `sX` = session/pair). The prepared 15-recording
set has **8 subjects**: s1_1, s1_2, s2_3, s2_4, s3_6, s4_7, s4_8 span both valences; s3_5 is
negative-only. (The h5 folder on disk also has a 9th subject, **s5_13**, in both valences — it
was *not* in the 15-recording config; add it to `recordings:` + re-prepare to use n=9 subjects.)
*Caveat: an earlier version of the split grouped by the leading `sX` (4 session groups) — fixed
to the full token (8 subjects).* So leave-one-*recording*-out keeps the held-out recording's
subject in training at **both** the SRNN-training level (the 15 folds are leave-one-recording-out)
**and** the decoder-CV level — and respiration has strong individual signatures. The valid test
of whether valence generalizes *across individuals* is **leave-one-subject-out (LOSO, 8 folds)**:
- Decoder LOSO (regroup CV by subject, no retrain) is reported as the second column in
  `analyze_valence.py` part (C) — run `sbatch hipergator/analyze_job.slurm` to refresh it.
- Fully leakage-free LOSO (SRNN also never sees the held-out subject) needs retraining:
  `sbatch hipergator/respiration_job_loso.slurm` (`--split subject`, array 0–7,
  → `resp_srnn_subject_h8_fold{0..7}.pt`), then `sbatch hipergator/analyze_job.slurm subject`.
- Caveat: LOSO has only n=8 subject groups → coarse CV, pilot-level. The cleanest leakage-free
  *scalar* is breathing rate under LOSO models (part A); the latent decode stays descriptive
  (pooling latents across the 4 separately-trained LOSO models reintroduces cross-model
  non-alignment, so it is not both leakage-free *and* comparable).

**Design decisions for the NEXT experiment**
- **Rate-matched recordings** (planned) are the correct design: matching breathing rate
  across valence strips the confound and forces the question to be about non-rate
  structure. The current AUC=1.0 will (and should) disappear there.
- **Supervised mode (`coef_cross > 0`) does not help the valence goal.** Valence is one
  label per *recording*, not a per-timepoint state, so there is nothing to supervise the
  switches with; supervising with sniff behaviors is label-driven (RI2 ≈ 0% sniffing). Use
  it only to align states to behaviors for interpretability within RI1.
- **If using respiration features, feed feature *time-series* (instantaneous rate,
  amplitude envelope, irregularity), never static per-window scalars** — static features
  discard the dynamics that justify an RNN over the existing logreg/SVM. The SRNN earns its
  place only if valence lives in the *temporal/non-linear dynamics*, not in summary stats.
- **Most promising route to genuine social states: SSLD with SLEAP pose as a second view.**
  A pose view anchors the shared latent to behavior (pose has no trivial 8 Hz oscillation),
  breaking the "phase dominates reconstruction" trap that keeps the single-view SRNN
  biological.

**New analysis artifacts:** `respiration/analyze_valence.py` (recording-level breathing-rate
test + pooled single-model latent PCA + rate-controlled decode, with **LORO and LOSO** decoder
groupings and a `--split recording|subject` flag), `hipergator/analyze_job.slurm` (CPU; takes
an optional `recording|subject` arg), `hipergator/respiration_job_loso.slurm` (leave-one-
subject-out training array), a `subject` split mode in `train_respiration.py`, and
`respiration/plot/pooled_latent_pca_by_valence.png`. Part (D) adds a **permutation test**
(`--n_perm`, default 1000): null = valence labels shuffled at the recording level,
recomputing the LOSO-by-subject decode → p-value + `respiration/plot/permutation_test_{split}.png`.
Part (E) adds a **leave-one-subject-out LDA projection** (the supervised separating axis PCA
can't show; projection is held-out so not circular) → `respiration/plot/lda_projection_{split}.png`.
The old `collect_folds.py` cross-fold latent decode is **not trustworthy** (it pools latents
from separately-trained folds whose coordinate systems are not aligned — non-identifiability;
use the single-model latent in `analyze_valence.py` instead).
