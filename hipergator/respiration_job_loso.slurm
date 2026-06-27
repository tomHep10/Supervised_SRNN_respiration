#!/bin/bash
# =============================================================================
# HiPerGator GPU job: LEAVE-ONE-SUBJECT-OUT respiration SRNN  (created by Claude)
#
# Why: every subject (s1..s4) appears in BOTH RI1 (positive) and RI2 (negative),
# so leave-one-RECORDING-out (respiration_job.slurm) leaks individual respiration
# signatures -- the held-out recording's subject is still in the training set.
# Holding out ALL of a subject's recordings tests whether valence generalizes
# ACROSS individuals (the real claim). Subject = full sX_Y token (trailing number =
# individual). The current prepared set has 8 subjects -> array 0-7. (If you re-prepare
# with more recordings, e.g. add s5_13, bump --array to match the subject count.)
#
# Writes checkpoints respiration/result/resp_srnn_subject_h8_fold{0..3}.pt
# After all 4 folds finish, the leakage-free (across-subject) analysis:
#   sbatch hipergator/analyze_job.slurm subject
#   # or: python respiration/analyze_valence.py --config <hpg cfg> --split subject
# =============================================================================
#SBATCH --job-name=resp-srnn-loso
#SBATCH --output=logs/resp_loso_fold%a_%j.log
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
#SBATCH --time=12:00:00          # ~4-5h/fold; train set is larger here (3 subjects) but batch_size covers all
#SBATCH --array=0-7              # 8 subjects, leave-one-subject-out (one fold per individual)

module load conda
conda activate SSRNN

cd /home/t.heeps/blue_npadillacoreano/npadillacoreano/share/respiration-project/Supervised_SRNN_respiration
mkdir -p logs

CFG=respiration/config_respiration_hpg.yaml

# data_prepared/ must already exist (built once on the login node, see respiration_job.slurm header).
python respiration/train_respiration.py --config $CFG --fold $SLURM_ARRAY_TASK_ID --split subject
python respiration/plot_respiration.py  --config $CFG --fold $SLURM_ARRAY_TASK_ID --split subject
