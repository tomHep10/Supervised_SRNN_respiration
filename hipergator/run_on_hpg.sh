#!/bin/bash
# run_on_hpg.sh  (created by Claude)
# Run this ON HiPerGator, inside your own SSH session (you type your password + DUO;
# Claude never sees them). It prepares the windows and submits the 4-fold GPU array.
#
#   bash hipergator/run_on_hpg.sh
set -euo pipefail

REPO="/home/t.heeps/blue_npadillacoreano/npadillacoreano/share/respiration-project/Supervised_SRNN_respiration"
CFG="respiration/config_respiration_hpg.yaml"

cd "$REPO"
module load conda
eval "$(conda shell.bash hook)"

# build the SSRNN env once if it's not there yet
if ! conda env list | grep -qw SSRNN; then
  echo ">> building SSRNN env (one-time, a few minutes)..."
  conda env create -f environment.yml
fi
conda activate SSRNN

echo ">> preparing windows (4 chosen recordings -> 40x 30s windows)..."
python respiration/prepare_respiration.py --config "$CFG"

echo ""
echo ">> your slurm associations (account / qos):"
sacctmgr show assoc where user="$USER" format=account,qos%40 || true
ACCT="$(sacctmgr -nP show assoc where user="$USER" format=account | sort -u | head -1 || true)"
echo ">> auto-detected account: ${ACCT:-<none-found>}"

if [ -z "${ACCT:-}" ]; then
  echo "!! Could not auto-detect your account. Edit hipergator/respiration_job.slurm"
  echo "   (--account / --qos) using the table above, then run: sbatch hipergator/respiration_job.slurm"
  exit 1
fi

mkdir -p logs
echo ">> submitting 4-fold leave-one-recording-out GPU array (coef_cross=0, discovery)..."
sbatch --account="$ACCT" --qos="$ACCT" hipergator/respiration_job.slurm

echo ""
echo "=========================================================================="
echo " submitted. monitor:   squeue -u $USER"
echo " watch a fold:         tail -f logs/resp_fold0_*.log"
echo " AFTER all 4 finish:   conda activate SSRNN && \\"
echo "                       python respiration/collect_folds.py --config $CFG"
echo "=========================================================================="
echo " NOTE: if sbatch was rejected on --qos, rerun with the right qos from the"
echo "       table above, e.g.:  sbatch --account=$ACCT --qos=${ACCT}-b hipergator/respiration_job.slurm"
echo " NOTE: if prepare failed to find h5 files, fix paths.h5_dir in $CFG (the"
echo "       Resp_h5 folder that actually holds RI2_s2_3 & RI2_s3_6 on HPG)."
