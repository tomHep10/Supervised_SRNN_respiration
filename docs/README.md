# Documentation — Supervised SRNN (respiration)

This repo runs respiration-based **Supervised Switching Recurrent Neural Network
(Supervised_SRNN)** classification for two distinct experiments. The docs are organized so
the **transferable** model-and-methods material is shared, while each experiment's
specifics stay in its own folder and the two never get crossed.

| Experiment | Target | Sample rate | Leakage-safe CV | BLA cohort |
|------------|--------|-------------|-----------------|------------|
| **Valence** (this repo's live experiment) | positive (RI1) / negative (RI2) | **50 Hz** | leave-one-**subject**-out | not used |
| **Rank** (cross-experiment reference) | Dominant / Subordinate | 400 Hz | leave-one-**cage**-out | included |

**The one rule when working across both:** valence and rank share animals, recordings, and
most preprocessing **insights** (the BORIS actor-column gotcha, the "don't double-filter",
the per-subject leakage warning) — but they differ in **target, sample rate, and split
granularity** (subject vs. cage). When adapting code or notes from one to the other, change
those three things and leave the shared insights intact.

---

## Navigation

### Start here
- [`00_quickstart.md`](00_quickstart.md) — **Read first.** Environment, which node, how work
  gets run (login vs. SLURM), current status, and a one-paragraph hand-off for a new agent.
  Experiment-agnostic onboarding.
- [`01_project_layout.md`](01_project_layout.md) — Every folder and file across the repo, what
  it is and why it exists — including all SLURM scripts, in order of use.

### Understand the model & methods (transferable)
- [`model_and_methods/01_concepts_and_math.md`](model_and_methods/01_concepts_and_math.md) —
  The big-picture idea and all background terms (state-space models, HMMs, ELBO, variational
  inference) defined slowly.
- [`model_and_methods/02_model_internals.md`](model_and_methods/02_model_internals.md) — A
  line-by-line walk through the actual code: the generative model, the inference network,
  Baum–Welch, and the loss.
- [`model_and_methods/03_usage_guide.md`](model_and_methods/03_usage_guide.md) — How to
  install, run on the simulated data, train, plot, and adapt it to your own data.
- [`model_and_methods/04_hipergator_guide.md`](model_and_methods/04_hipergator_guide.md) —
  Running the model as a batch job on UF HiPerGator (SLURM); ready-to-edit job scripts in
  [`../hipergator/`](../hipergator/).

### Valence experiment
- [`valence/01_project_context.md`](valence/01_project_context.md) — The research goal (social
  valence from respiration), dataset, SRNN-vs-SSLD decision, findings, and open questions.
- [`valence/02_experiment_runbook.md`](valence/02_experiment_runbook.md) — Procedural,
  copy-paste walkthrough of the current experiment: prepare → train → classify → read results.
- [`valence/RNN_DATA_GUIDE.md`](valence/RNN_DATA_GUIDE.md) — **Data → SRNN-input guide.** How
  the raw `.h5` + BORIS data becomes 50 Hz, 30 s windows: recording catalog, valence labels,
  dead-signal removal, the subject (LOSO) split, and the BORIS actor-column gotcha.

### Rank experiment
- [`rank/RNN_DATA_GUIDE.md`](rank/RNN_DATA_GUIDE.md) — Raw data → RNN input for the rank
  experiment (400 Hz, Dominant/Subordinate, leave-one-cage-out). Kept for reference; some
  files it cites live in the separate rank project, not here.

**Your respiration pipeline** lives in [`../respiration/`](../respiration/) — scripts that
turn your raw `.h5` + BORIS `.csv` files into windows, train the SRNN, and analyze the
latent for social valence. See [respiration/README.md](../respiration/README.md).

---

## TL;DR of what this model is

You have multi-channel time-series data (here: 20 simulated "neurons" recorded over
100 time steps, across 50 trials). You believe the system switches between a small
number of **discrete regimes** (here: 2 states, labeled 0/1), and within each regime
the data follows its own **continuous nonlinear dynamics**.

The SRNN learns:

- a separate small **RNN** (recurrent neural network) for each discrete regime, which
  describes how the hidden state evolves *while in that regime*,
- a **transition model** that decides when to switch regimes,
- an **emission model** (a small MLP) that turns the hidden state into the observed
  neural activity,
- an **inference network** (a Transformer) that reads the data and estimates the
  hidden state.

It is **"supervised"** because, unlike the usual unsupervised switching models, it is
*also* given the ground-truth regime labels during training and is pushed to match
them (a cross-entropy term). This is the key twist of this repository.

## A note on "SRNN vs. SLLD"

- **SRNN is here** — it is the entire `SRNN/` package.
- **There is no SLLD model in this repository** (I searched the whole repo and your
  other projects). The closest standard term is **SLDS = Switching Linear Dynamical
  System**, the classical statistical model that the SRNN generalizes by replacing its
  *linear* dynamics with *recurrent neural networks*. See
  [model_and_methods/01_concepts_and_math.md](model_and_methods/01_concepts_and_math.md#from-slds-to-srnn)
  for that lineage. If you have an "SLLD" model from a different source, it is not in this
  codebase.
