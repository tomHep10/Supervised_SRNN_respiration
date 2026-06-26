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
<!-- Added 2026-06-26 — results from the first full 4-fold HiPerGator run     -->
<!-- ======================================================================= -->

---

## Results so far (first full 4-fold HiPerGator run, completed 2026-06-25)

*These are findings from the first complete run, not in the original context above. Run
config: supervised SRNN, `coef_cross = 0.5`, `hidden_shape = 8`, `num_tv = 4`,
leave-one-recording-out (`split_mode = recording`), 2000 epochs/fold.*

**Feasibility (Q1–Q3):** confirmed feasible with windowing. All 4 folds trained to 2000
epochs and converged to low reconstruction error (test MSE 0.003–0.020) on the
**30 s / T=1500 contiguous windows** — full 10-min recordings were *not* used, consistent
with the sequence-length constraint noted above.

**Valence signal (Q5):** a **suggestive descriptive signal**, not yet a certified decode.
- Respiratory-state **switch-rate** is higher for positive than negative valence
  (pooled 0.295 vs 0.243) and **rank-orders perfectly across all 4 recordings**
  (positives 0.309, 0.280 > negatives 0.252, 0.234; no overlap).
- The signal is in switching **dynamics**, not **state occupancy** (occupancy ≈ identical
  across valence; only 2 of the 4 discrete states are ever used → effectively bistable).
- **Leakage-free LORO decoding returned balanced-acc = 0.000** for latent `h`, switch
  stats, and both combined. This is an **n=4 artifact** (2 recordings/class can't calibrate
  a held-out boundary, so it flips systematically) — *not* evidence against the effect, and
  not trustworthy in either direction at this sample size.

**Most informative analyses (Q6), observed:** switch-rate / switch statistics were the
discriminating feature; state-occupancy and the latent `h` (via LORO) were not, at n=4.

**Caveat (ties to README caveat #1):** this run used `coef_cross = 0.5`, so the switch
separation could be **label-driven** rather than discovered. The discovery-mode run
(`coef_cross = 0`) remains the recommended primary and is still **to do**.

**Next steps:** (1) more recordings — the only real fix for LORO power; (2) interim
defensible statistic via window-level decode or a permutation test on per-recording
switch-rates; (3) re-run in discovery mode (`coef_cross = 0`) and compare.

*Full numeric output and the operational write-up live in
[respiration/README.md](../respiration/README.md#results--first-full-4-fold-hipergator-run-run-completed-2026-06-25).*
