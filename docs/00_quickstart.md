# 0. Quickstart — environment, node, and how work actually gets run

> **Read this first.** It exists so a *new agent* (Claude or otherwise) and *you* are on
> the same page about **where things run, how to activate the environment, and which
> commands belong on the login node vs. a compute node.** Once an agent has read this it
> can pick up the project without re-discovering any of it.
>
> If you are a human: the box **["What to tell a new agent"](#what-to-tell-a-new-agent)**
> is the one-paragraph hand-off you can paste.

---

## The 30-second mental model

- This is **UF HiPerGator** (a SLURM cluster). You log into a **login node**; real
  compute runs on **compute nodes** via the **SLURM** scheduler (`sbatch`).
- The project conda environment is **`SSRNN`** and already exists at:
  ```
  /blue/npadillacoreano/t.heeps/.conda/envs/SSRNN
  ```
  (Python 3.11, PyTorch 2.8 + CUDA 12.9, scikit-learn, numpy, matplotlib, pyyaml.)
- The repo lives at:
  ```
  /home/t.heeps/blue_npadillacoreano/npadillacoreano/share/respiration-project/Supervised_SRNN_respiration
  ```
  (this is a symlinked view of `/blue/npadillacoreano/share/...` — `/blue` is the fast
  work storage; never put heavy data on `/home`).

---

## The one gotcha that trips up every new agent

An agent's shell opens on the **login node (`login10.ufhpc`)** in **non-interactive**
mode. In that shell **`module` and `conda` are NOT on the PATH** — so
`conda activate SSRNN` fails with `conda: command not found`. That is *expected*, not a
broken setup. There are two correct ways to work, depending on the task:

### A) Quick, light, read-only checks → call the env's Python by full path
For listing results, reading a checkpoint, a tiny sanity script — anything that runs in
seconds and won't load the login node — skip activation entirely and use the absolute
interpreter path:

```bash
/blue/npadillacoreano/t.heeps/.conda/envs/SSRNN/bin/python -c "import torch; print(torch.__version__)"
```

This always works because it doesn't need `conda activate`.

### B) Any real compute (training, full analysis, anything > a minute) → submit a SLURM job
**Do not run training or the full analysis on the login node.** Write/submit a batch
script. *Inside* a SLURM script the module system IS available, so the normal incantation
works there:

```bash
module load conda
conda activate SSRNN
```

Submit with `sbatch`, then poll `squeue`. Ready-made scripts are in
[`../hipergator/`](../hipergator/) (see [01_project_layout.md](01_project_layout.md) for
what each one does and [07_experiment_runbook.md](07_experiment_runbook.md) for the
step-by-step).

> Rule of thumb: **if it would take more than ~30 s or use real CPU/GPU, it goes through
> `sbatch`, not the login shell.**

---

## Cluster coordinates (filled in — don't re-discover these)

| Setting | Value | Note |
|---|---|---|
| Login node (agent shell) | `login10.ufhpc` | non-interactive; no `module`/`conda` on PATH |
| Conda env | `SSRNN` | `/blue/npadillacoreano/t.heeps/.conda/envs/SSRNN` |
| Env Python (direct) | `/blue/npadillacoreano/t.heeps/.conda/envs/SSRNN/bin/python` | for quick checks |
| SLURM account | `npadillacoreano` | `--account=npadillacoreano` |
| SLURM QOS | `npadillacoreano` (or `npadillacoreano-b` burst) | `--qos=...` |
| GPU partition | `hpg-b200` | `--partition=hpg-b200 --gres=gpu:1` (training) |
| CPU partition | `hpg-default` | analysis / inference (no GPU needed) |
| Email for job mail | `t.heeps@ufl.edu` | |

Verify any of these any time with:
```bash
sacctmgr show assoc where user=t.heeps format=account,qos%50
sinfo -o "%P %a" | head        # available partitions
```

---

## Day-to-day SLURM commands

| Action | Command |
|---|---|
| Submit a job | `sbatch hipergator/<script>.slurm` |
| List my jobs | `squeue -u t.heeps` |
| Status of one job | `squeue -j JOBID -o "%.10i %.12P %.18j %.8T %.10M %R"` |
| Watch a log live | `tail -f logs/<name>_*.log` |
| Cancel | `scancel JOBID` |
| Efficiency of a finished job | `seff JOBID` |

Logs land in [`../logs/`](../logs/) named `<jobname>_<jobid>.log` (array jobs add the fold
index). An agent waiting on a job should poll `squeue` in a loop rather than block.

---

## Where things are

- **Pipeline code & configs:** [`../respiration/`](../respiration/) — see its
  [README.md](../respiration/README.md).
- **The model package (don't modify):** [`../SRNN/`](../SRNN/).
- **Job scripts:** [`../hipergator/`](../hipergator/).
- **Trained checkpoints:** [`../respiration/result/`](../respiration/result/)
  (`resp_srnn_<split>_h8_fold*.pt`).
- **Figures:** [`../respiration/plot/`](../respiration/plot/).
- **Full file-by-file tour:** [01_project_layout.md](01_project_layout.md).
- **Why this project exists + latest findings:** [06_project_context.md](06_project_context.md).
- **Step-by-step to reproduce the current experiment:** [07_experiment_runbook.md](07_experiment_runbook.md).

---

## Current status (kept current — last updated 2026-06-26)

- **Data:** 15 recordings (7 positive RI1 / 8 negative RI2), 8 subjects, windowed to
  **147 × 30 s windows** in [`../respiration/data_prepared/`](../respiration/data_prepared/).
- **Trained:** all 8 leave-one-**subject**-out folds
  (`resp_srnn_subject_h8_fold{0..7}.pt`).
- **Latest result:** breathing rate alone separates valence (ROC-AUC 1.0, but a confound);
  a valence signal **beyond** rate survives leave-one-subject-out and a permutation test
  (rate-removed latent ≈ 0.73, p ≈ 0.001). Details in
  [06_project_context.md](06_project_context.md#findings--design-decisions-run-completed--2026-06-26).
- **Re-run the classifier any time:** `sbatch hipergator/classifier_results.slurm`.

---

## What to tell a new agent

> Paste this to get an agent productive immediately:
>
> *"We're on UF HiPerGator (SLURM). Read `docs/00_quickstart.md` first. The conda env is
> `SSRNN` — your login shell can't `conda activate` it, so use
> `/blue/npadillacoreano/t.heeps/.conda/envs/SSRNN/bin/python` for quick checks and submit
> a SLURM job (`sbatch hipergator/...`) for anything heavy. SLURM account/qos are
> `npadillacoreano`, GPU partition `hpg-b200`, CPU partition `hpg-default`. The repo is at
> `.../Supervised_SRNN_respiration`. Don't run training or full analysis on the login
> node."*
