#!/bin/bash
# =============================================================================
# HiPerGator CPU job: run the leakage-aware valence classifier on the trained
# SRNN folds and produce all result figures + the leakage-free decoding numbers.
# Inference only (no training, no GPU). Runs BOTH cross-validation splits that
# have trained checkpoints:
#   - recording  (leave-one-recording-out, folds 0-14)
#   - subject    (leave-one-subject-out,   folds 0-7)
# Outputs:
#   (A) recording-level breathing-rate / switch-rate ROC-AUC      -> log
#   (B) pooled latent PCA colored by valence                      -> respiration/plot/
#   (C) leakage-free LORO valence decoding (latent / switch / both)-> log
#   (D) permutation test + LDA projection                         -> respiration/plot/
#
# Submit from the repo root:
#   sbatch hipergator/classifier_results.slurm
# Watch it:
#   squeue -u $USER ;  tail -f logs/classifier_*.log
# =============================================================================
#SBATCH --job-name=resp-classifier
#SBATCH --output=logs/classifier_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.heeps@ufl.edu
#SBATCH --account=npadillacoreano
#SBATCH --qos=npadillacoreano
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:30:00

module load conda
conda activate SSRNN

cd /home/t.heeps/blue_npadillacoreano/npadillacoreano/share/respiration-project/Supervised_SRNN_respiration
mkdir -p logs

CFG=respiration/config_respiration_hpg.yaml

echo "############################################################"
echo "### collect_folds.py  (leakage-free LORO valence decode) ###"
echo "############################################################"
python respiration/collect_folds.py --config $CFG

for SPLIT in recording subject; do
  echo ""
  echo "############################################################"
  echo "### analyze_valence.py  --split $SPLIT"
  echo "############################################################"
  python respiration/analyze_valence.py --config $CFG --split $SPLIT --pca_fold 0
done

echo ""
echo "DONE. Figures in respiration/plot/ ; numbers above in this log."
