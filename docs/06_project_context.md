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
