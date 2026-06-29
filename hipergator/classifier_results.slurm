#!/bin/bash
# =============================================================================
# HiPerGator CPU job: run the leakage-aware valence classifier on the trained
# SRNN folds and produce all result figures + the leakage-free decoding numbers.
# Inference only (no training, no GPU).
#
# LOSO-ONLY (2026-06-28): leave-one-recording-out was dropped because it leaks —
# the same subject appears in multiple recordings, so a held-out recording still
# shares a subject with the training set. We now run only the subject split
# (leave-one-subject-out, folds 0-7). collect_folds.py is omitted: it is a pure
# leave-one-recording-out decoder and is superseded by analyze_valence's
# leave-one-subject-out (logo) decoding.
# Outputs:
#   (A) recording-level breathing-rate / switch-rate ROC-AUC      -> log
#   (B) pooled latent PCA colored by valence                      -> respiration/plot/
#   (C) leakage-free LOSO valence decoding (latent / switch / both)-> log
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

CFG=respiration/valence/config_respiration_hpg.yaml

echo "############################################################"
echo "### analyze_valence.py  --split subject  (leave-one-subject-out)"
echo "############################################################"
python respiration/valence/analyze_valence.py --config $CFG --split subject --pca_fold 0

# consolidate config + data + per-fold MSE + this analysis into one report .txt
python respiration/make_report.py --config $CFG --analysis-log logs/classifier_${SLURM_JOB_ID}.log

echo ""
echo "DONE. Figures in respiration/plot/ ; full report in reports/valence/report_latest.txt"
