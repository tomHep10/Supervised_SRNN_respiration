#!/bin/bash
# =============================================================================
# HiPerGator CPU job: leakage-aware RANK analysis on the trained leave-one-cage-out
# SRNN folds. Inference only (no training, no GPU). Produces:
#   (A) recording-level breathing-rate -> rank ROC-AUC                 -> log
#   (B) pooled latent PCA colored by rank                             -> plot_rank/
#   (C) leave-one-cage-out rank decoding (rate / latent / both)       -> log
#   (D) permutation test (LOCO-by-cage)                               -> plot_rank/
#   (E) LDA projection (leave-one-cage-out)                           -> plot_rank/
#
# PER-TIMESTEP (2026-06-30): (B)-(F) now use EVERY timestep's latent (~1500/window) instead
# of the per-window mean, so the permutation tests (D, F) are ~1500x more rows. They are now
# fanned across cores (joblib honors --cpus-per-task=16 below), so walltime is 2 h. To go
# faster still: append --latent_stride N and/or --n_perm 200, or raise --cpus-per-task.
#
# PREREQ: rank LOCO training finished (respiration/result_rank/resp_srnn_cage_h8_fold*.pt).
# Submit from the repo root:
#   sbatch hipergator/analyze_rank.slurm
# =============================================================================
#SBATCH --job-name=rank-classifier
#SBATCH --output=logs/rank_classifier_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.heeps@ufl.edu
#SBATCH --account=npadillacoreano
#SBATCH --qos=npadillacoreano
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24gb
#SBATCH --time=02:00:00

module load conda
conda activate SSRNN

cd /home/t.heeps/blue_npadillacoreano/npadillacoreano/share/respiration-project/Supervised_SRNN_respiration
mkdir -p logs

# convenience: always-current pointer to THIS run's log -> tail -f logs/rank_classifier_latest.log
ln -sf rank_classifier_${SLURM_JOB_ID}.log logs/rank_classifier_latest.log

CFG=${1:-respiration/rank/config_rank_hpg.yaml}   # pass a cohort config as arg 1 to run a different cohort

echo "############################################################"
echo "### analyze_rank.py  --split cage  (leave-one-cage-out)"
echo "############################################################"
python respiration/rank/analyze_rank.py --config $CFG --split cage --pca_fold 0

# consolidate config + data + per-fold MSE + this analysis into one report .txt
python respiration/make_report.py --config $CFG --analysis-log logs/rank_classifier_${SLURM_JOB_ID}.log

echo ""
echo "DONE. Figures in respiration/plot_rank/ ; full report in reports/<cohort>/report_latest.txt"
