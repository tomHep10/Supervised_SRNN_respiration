#!/bin/bash
# =============================================================================
# HiPerGator CPU job for the leakage-aware valence analysis  (created by Claude)
# Inference only (no training, no GPU). Produces:
#   (A) recording-level breathing-rate / switch-rate test (leakage-free)  -> stdout/log
#   (B) pooled latent PCA colored by valence                              -> respiration/plot/
#
# Submit from the repo root (subject = leave-one-subject-out is the only split):
#   sbatch hipergator/analyze_job.slurm            # leakage-free across individuals
#                                                  #   (run the LOSO training first)
# Watch it:
#   squeue -u $USER ;  tail -f logs/analyze_*.log
# =============================================================================
#SBATCH --job-name=resp-analyze
#SBATCH --output=logs/analyze_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.heeps@ufl.edu
#SBATCH --account=npadillacoreano
#SBATCH --qos=npadillacoreano
#SBATCH --partition=hpg-default     # CPU partition (no GPU needed). Change if your CPU partition differs.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:30:00             # inference over 147 windows is minutes; generous buffer

module load conda
conda activate SSRNN

cd /home/t.heeps/blue_npadillacoreano/npadillacoreano/share/respiration-project/Supervised_SRNN_respiration
mkdir -p logs

CFG=respiration/valence/config_respiration_hpg.yaml
SPLIT=${1:-subject}            # only 'subject' (leave-one-subject-out) is supported now
python respiration/valence/analyze_valence.py --config $CFG --split $SPLIT --pca_fold 0
