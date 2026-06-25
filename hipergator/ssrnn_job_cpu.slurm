#!/bin/bash
# =============================================================================
# HiPerGator CPU-only batch job for Supervised_SRNN  (created by Claude)
# Same as ssrnn_job.slurm but no GPU — this model is small and bottlenecked by
# a per-timestep Python loop, so CPU is viable and skips the GPU queue.
# The code auto-detects no GPU and runs on CPU with no edits.
#
# BEFORE SUBMITTING, replace: GROUP, QOS, EMAIL, and the path in `cd`.
#   sacctmgr show assoc where user=$USER format=account,user,qos%40
#
# Submit:   sbatch hipergator/ssrnn_job_cpu.slurm
# =============================================================================
#SBATCH --job-name=ssrnn-cpu
#SBATCH --output=logs/ssrnn_fold%a_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=EMAIL
#SBATCH --account=GROUP
#SBATCH --qos=QOS
#SBATCH --partition=hpg-default                 # CPU partition
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
