# Supervised SRNN — Documentation

This folder is a from-scratch explanation of the **Supervised Switching Recurrent
Neural Network (Supervised_SRNN)** project. It is written to do two things at once:

1. **Help you *use* the code** — run the simulation, train the model, read the outputs.
2. **Help you *understand* the model** — every mechanism, every term, every line of
   math is defined and unpacked, assuming only that you know basic linear algebra,
   probability, and that a neural network is a function with trainable weights.

Read the files in order if you want the full story, or jump to what you need.

| File | What it covers |
|------|----------------|
| [00_quickstart.md](00_quickstart.md) | **Read first.** Environment, which node, how work gets run (login vs. SLURM), current status, and a one-paragraph hand-off for a new agent. |
| [01_project_layout.md](01_project_layout.md) | Every folder and file, what it is and why it exists — including all SLURM scripts, in order of use. |
| [02_concepts_and_math.md](02_concepts_and_math.md) | The big-picture idea and all the background terms (state-space models, HMMs, ELBO, variational inference) defined slowly. |
| [03_model_internals.md](03_model_internals.md) | A line-by-line walk through the actual code: the generative model, the inference network, Baum–Welch, and the loss. |
| [04_usage_guide.md](04_usage_guide.md) | How to install, run on the simulated data, train, plot, and adapt it to your own data. |
| [05_hipergator_guide.md](05_hipergator_guide.md) | Running the model as a batch job on UF HiPerGator (SLURM); ready-to-edit job scripts in [`../hipergator/`](../hipergator/). |
| [06_project_context.md](06_project_context.md) | The research goal (social valence from respiration), dataset, SRNN-vs-SSLD decision, findings, and open questions. |
| [07_experiment_runbook.md](07_experiment_runbook.md) | Procedural, copy-paste walkthrough of the current 15-recording experiment: prepare → train → classify → read results. |

**Your respiration pipeline** lives in [`../respiration/`](../respiration/) — scripts that
turn your raw `.h5` + BORIS `.csv` files into windows, train the SRNN, and analyze the
latent for social valence. See [respiration/README.md](../respiration/README.md).

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
  [02_concepts_and_math.md](02_concepts_and_math.md#from-slds-to-srnn) for that lineage.
  If you have an "SLLD" model from a different source, it is not in this codebase.
