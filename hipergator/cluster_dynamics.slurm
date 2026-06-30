#!/bin/bash
# =============================================================================
# HiPerGator CPU job: WITHIN-WINDOW latent micro-state clustering (descriptive).
# Runs respiration/cluster_dynamics.py on the trained fold-0 model: pools EVERY
# timestep's latent (~1500/window), K-Means clusters them into micro-states, and
# reports occupancy / dwell time / transitions / micro-state-vs-SRNN-state, plus
# figures. Inference only (no training, no GPU).
#
# Usage (run from the repo root):
#   sbatch hipergator/cluster_dynamics.slurm              # valence (default), auto-pick k
#   sbatch hipergator/cluster_dynamics.slurm rank         # rank experiment, auto-pick k
#   sbatch hipergator/cluster_dynamics.slurm valence "" 4 # valence, fixed k=4
#   sbatch hipergator/cluster_dynamics.slurm rank respiration/rank/config_rank_incl_s3_6.yaml 0
#     args: $1 = target {valence|rank}, $2 = config (""=default for target), $3 = k (0=auto)
#
# PREREQ: the matching folds are trained (valence: result/resp_srnn_subject_h8_fold0.pt;
#         rank: result_rank/resp_srnn_cage_h8_fold0.pt).
# Watch it:  squeue -u $USER ;  tail -f logs/cluster_*.log
# Outputs (target = valence|rank):
#   cluster_k_selection_<target>.png   (only when k=0: silhouette + inertia vs k)
#   cluster_pca_<target>.png           (latent PCA colored by micro-state)
#   cluster_summary_<target>.png       (centroids | occupancy-by-label | transitions)
#   cluster_trajectories_<target>.png  (example within-window state trajectories)
#   ...written to respiration/plot/ (valence) or respiration/plot_rank/ (rank).
# =============================================================================
#SBATCH --job-name=resp-cluster
#SBATCH --output=logs/cluster_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.heeps@ufl.edu
#SBATCH --account=npadillacoreano
#SBATCH --qos=npadillacoreano
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=01:00:00

module load conda
conda activate SSRNN

cd /home/t.heeps/blue_npadillacoreano/npadillacoreano/share/respiration-project/Supervised_SRNN_respiration
mkdir -p logs

# convenience: always-current pointer to THIS run's log -> tail -f logs/cluster_latest.log
ln -sf cluster_${SLURM_JOB_ID}.log logs/cluster_latest.log

TARGET=${1:-valence}
K=${3:-0}                                       # 0 = auto-pick k by silhouette
if [ "$TARGET" = "rank" ]; then
    CFG=${2:-respiration/rank/config_rank_hpg.yaml}
    SPLIT=cage
else
    CFG=${2:-respiration/valence/config_respiration_hpg.yaml}
    SPLIT=subject
fi
# allow an empty $2 placeholder ("") to mean "use the default config for this target"
[ -z "$CFG" ] && { [ "$TARGET" = "rank" ] && CFG=respiration/rank/config_rank_hpg.yaml || CFG=respiration/valence/config_respiration_hpg.yaml; }

echo "############################################################"
echo "### cluster_dynamics.py  target=$TARGET  split=$SPLIT  k=$K"
echo "### config=$CFG"
echo "############################################################"
python respiration/cluster_dynamics.py --config "$CFG" --target "$TARGET" \
       --split "$SPLIT" --pca_fold 0 --k "$K"

echo ""
PLOTDIR=$([ "$TARGET" = "rank" ] && echo respiration/plot_rank || echo respiration/plot)
echo "DONE. Cluster figures (cluster_*_${TARGET}.png) in ${PLOTDIR}/"
