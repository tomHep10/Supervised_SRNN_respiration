# 5. Running on UF HiPerGator (batch jobs)

This guide takes the code you developed locally and runs it on **UF HiPerGator** as a
**batch job** so it runs on the cluster's resources until finished — you can log off and
it keeps going.

It is written to be **copy-paste friendly for both a human and a Claude agent**:
- Fill in the four blanks in the table below **once**.
- Every command block is self-contained; replace the `ALL_CAPS` placeholders.
- Ready-to-edit job scripts live in [`../hipergator/`](../../hipergator/) — a GPU version
  and a CPU version.

---

## What "batch job" means (the concept)

You do **not** run heavy work on the login node. Instead you write a small text file (a
**job script**) that declares *what resources you want* and *what command to run*, then
hand it to the **SLURM** scheduler with `sbatch`. SLURM queues it, runs it on a compute
node when resources are free, writes all output to a log file, and keeps running after
you disconnect. That is exactly the "send it and let it run until done" workflow.

---

## Fill these in once

| Placeholder | What it is | How to find it |
|---|---|---|
| `GROUP` | your HiPerGator account/group | run the `sacctmgr` command in Step 4 |
| `QOS` | your quality-of-service | usually same as `GROUP`; `GROUP-b` = "burst" |
| `EMAIL` | where job emails go | your `@ufl.edu` address (e.g. `t.heeps@ufl.edu`) |
| `GPU_TYPE` | GPU model + partition | `a100` is the common default; confirm in Step 4 |

Your login is `t.heeps@hpg.rc.ufl.edu`; paths below assume username `t.heeps` — change if needed.

---

## Step 1 — Copy the code to HiPerGator

Put it on **`/blue`** (fast work storage), **not** `/home` (40 GB quota, slow). From a
local terminal (Git Bash on Windows):

```bash
scp -r "/c/Users/thoma/Code/ResearchCode/Supervised_SRNN" \
    t.heeps@hpg.rc.ufl.edu:/blue/GROUP/t.heeps/
```

For repeated re-syncs while you iterate, prefer `rsync`:

```bash
rsync -avz --exclude '.git' --exclude '__pycache__' \
    "/c/Users/thoma/Code/ResearchCode/Supervised_SRNN/" \
    t.heeps@hpg.rc.ufl.edu:/blue/GROUP/t.heeps/Supervised_SRNN/
```

(If the repo is on GitHub you can instead `git clone` / `git pull` on HiPerGator.)

---

## Step 2 — Build the conda env once (on the login node)

The login node has internet; compute nodes may not — so create the environment here,
once, before submitting any job.

```bash
ssh t.heeps@hpg.rc.ufl.edu
module load conda
cd /blue/GROUP/t.heeps/Supervised_SRNN
conda env create -f environment.yml          # creates the "SSRNN" env (one time)
```

> If `conda env create` is slow or the CUDA build is troublesome, you can instead make a
> minimal CPU env — see [Step 6](#step-6--cpu-only-alternative-simplest).

---

## Step 3 — The job script

A ready-to-edit script is at [`../hipergator/ssrnn_job.slurm`](../../hipergator/ssrnn_job.slurm).
It uses a **job array** (`--array=0-4`) so all five cross-validation folds run as five
independent jobs from a single `sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=ssrnn
#SBATCH --output=logs/ssrnn_fold%a_%j.log    # %a = array/fold index, %j = job id
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=EMAIL
#SBATCH --account=GROUP
#SBATCH --qos=QOS
#SBATCH --partition=gpu
#SBATCH --gres=gpu:GPU_TYPE:1                 # one GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=08:00:00
#SBATCH --array=0-4                           # runs folds 0,1,2,3,4

module load conda
conda activate SSRNN
cd /blue/GROUP/t.heeps/Supervised_SRNN
mkdir -p logs

python array_hidden8.py --config config.yaml --fold $SLURM_ARRAY_TASK_ID
```

To run a **single fold** instead of all five, delete the `--array` line and replace
`$SLURM_ARRAY_TASK_ID` with a number (e.g. `--fold 0`).

---

## Step 4 — Discover your GROUP / QOS / GPU

On HiPerGator:

```bash
# your account + qos options:
sacctmgr show assoc where user=t.heeps format=account,user,qos%40
```

- The **account** column → `GROUP`.
- The **qos** column → `QOS` (a `...-b` entry means a burst QOS is available).
- For GPU types/partitions available to your group, check the HiPerGator GPU docs or:
  ```bash
  module spider cuda          # and see the RC GPU docs for current partition names
  ```
  `--partition=gpu --gres=gpu:a100:1` is the long-standing default; newer nodes
  (e.g. L4, B200) use different partition names, so confirm before relying on a type.

---

## Step 5 — Submit, monitor, retrieve

```bash
cd /blue/GROUP/t.heeps/Supervised_SRNN
sbatch hipergator/ssrnn_job.slurm        # queue the job(s)

squeue -u t.heeps                        # see queued/running jobs
tail -f logs/ssrnn_fold0_*.log           # watch one fold live (Ctrl-C to stop watching)
scancel JOBID                            # cancel a job (JOBID from squeue)
```

Results are written to `result/`. Pull them back to your laptop and plot locally:

```bash
# run locally:
scp -r t.heeps@hpg.rc.ufl.edu:/blue/GROUP/t.heeps/Supervised_SRNN/result ./
python plot.py
```

---

## Step 6 — CPU-only alternative (simplest)

This model is small and bottlenecked by a per-timestep Python loop, so a **CPU job is
totally viable** and skips the GPU queue entirely. The script
[`../hipergator/ssrnn_job_cpu.slurm`](../../hipergator/ssrnn_job_cpu.slurm) is the same as
above with the GPU lines removed:

```bash
#!/bin/bash
#SBATCH --job-name=ssrnn-cpu
#SBATCH --output=logs/ssrnn_fold%a_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=EMAIL
#SBATCH --account=GROUP
#SBATCH --qos=QOS
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=12:00:00
#SBATCH --array=0-4

module load conda
conda activate SSRNN
cd /blue/GROUP/t.heeps/Supervised_SRNN
mkdir -p logs

python array_hidden8.py --config config.yaml --fold $SLURM_ARRAY_TASK_ID
```

The code auto-detects no GPU and runs on CPU with no edits
([array_hidden8.py:120](../../array_hidden8.py#L120)).

---

## Quick reference

| Action | Command |
|---|---|
| Submit | `sbatch hipergator/ssrnn_job.slurm` |
| List my jobs | `squeue -u t.heeps` |
| Watch a log | `tail -f logs/ssrnn_fold0_*.log` |
| Cancel | `scancel JOBID` |
| My limits | `sacctmgr show assoc where user=t.heeps format=account,qos%40` |
| Detailed usage of a finished job | `seff JOBID` |

Sources: UFIT-RC Sample Slurm Scripts (`docs.rc.ufl.edu/scheduler/sample_job_scripts/`),
UFIT-RC GPU Access (`docs.rc.ufl.edu/scheduler/gpu_access/`).
