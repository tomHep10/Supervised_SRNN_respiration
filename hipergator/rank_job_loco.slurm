#!/bin/bash
# =============================================================================
# HiPerGator GPU job: LEAVE-ONE-CAGE-OUT cagemate RANK SRNN  (created by Claude)
#
# Why LOCO: animals interact ONLY with cagemates, so a held-out recording whose cagemate
# is still in the training set leaks identity/cage. Holding out a WHOLE cage (both its mice)
# tests whether the rank signal generalizes ACROSS cages. Cage = first index of the subject
# token (sCAGE_ANIMAL). The cagemate set has 4 cages -> array 0-3. (Bump --array if you add
# cages; check the prepare_rank.py printout for the cage count.)
#
# Same model/training as the valence SRNN, just --split cage and the rank config. Writes
# checkpoints respiration/result_rank/resp_srnn_cage_h8_fold{0..3}.pt and per-fold figures.
#
# PREREQ: build the windows first ->  sbatch hipergator/prepare_rank.slurm
# Submit:                              sbatch hipergator/rank_job_loco.slurm
# =============================================================================
#SBATCH --job-name=rank-srnn-loco
#SBATCH --output=logs/rank_loco_fold%a_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.heeps@ufl.edu
#SBATCH --account=npadillacoreano
#SBATCH --qos=npadillacoreano
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=12:00:00          # ~4-5h/fold
#SBATCH --array=0-3              # 4 cages, leave-one-cage-out (one fold per cage)

module load conda
conda activate SSRNN

cd /home/t.heeps/blue_npadillacoreano/npadillacoreano/share/respiration-project/Supervised_SRNN_respiration
mkdir -p logs

CFG=${1:-respiration/rank/config_rank_hpg.yaml}   # pass a cohort config as arg 1 to run a different cohort

# data_prepared_rank/ must already exist (build it first: sbatch hipergator/prepare_rank.slurm).
python respiration/train_respiration.py --config $CFG --fold $SLURM_ARRAY_TASK_ID --split cage
python respiration/plot_respiration.py  --config $CFG --fold $SLURM_ARRAY_TASK_ID --split cage
