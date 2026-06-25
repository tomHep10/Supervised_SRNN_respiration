#!/bin/bash
# =============================================================================
# HiPerGator GPU batch job for Supervised_SRNN  (created by Claude)
# Runs all 5 cross-validation folds as a job array (one job each).
#
# BEFORE SUBMITTING, replace the placeholders: GROUP, QOS, EMAIL, GPU_TYPE,
# and the path in `cd`. Find GROUP/QOS with:
#   sacctmgr show assoc where user=$USER format=account,user,qos%40
#
# Submit:   sbatch hipergator/ssrnn_job.slurm
# Monitor:  squeue -u $USER   ;   tail -f logs/ssrnn_fold0_*.log
# =============================================================================
#SBATCH --job-name=ssrnn
#SBATCH --output=logs/ssrnn_fold%a_%j.log     # %a = fold index, %j = job id
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=EMAIL                      # e.g. t.heeps@ufl.edu
#SBATCH --account=GROUP                        # your group (sacctmgr)
#SBATCH --qos=QOS                              # usually GROUP, or GROUP-b for burst
#SBATCH --partition=gpu                        # GPU partition (confirm for your group)
#SBATCH --gres=gpu:GPU_TYPE:1                  # one GPU, e.g. gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=08:00:00
#SBATCH --array=0-4                            # folds 0,1,2,3,4 (delete for a single fold)

module load conda
conda activate SSRNN

cd /blue/GROUP/t.heeps/Supervised_SRNN         # <-- path to the repo on /blue
mkdir -p logs

python array_hidden8.py --config config.yaml --fold $SLURM_ARRAY_TASK_ID
