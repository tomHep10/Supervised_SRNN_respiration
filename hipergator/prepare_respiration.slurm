#!/bin/bash
# =============================================================================
# HiPerGator CPU job: (re)prepare the respiration windows with dead-signal
# removal applied BEFORE windowing. Writes respiration/data_prepared/
# (observations.npy, labels.npy, meta.npz, label_map.json).
#
# CPU only, fast (minutes). Submit from the repo root:
#   sbatch hipergator/prepare_respiration.slurm
# Then READ the log — expect total windows < old 147, RI2_s2_4 dropping most of
# its windows, and the per-recording "dropped(dead)" / "dead % of rec" columns:
#   cat logs/prepare_*.log
# =============================================================================
#SBATCH --job-name=resp-prepare
#SBATCH --output=logs/prepare_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.heeps@ufl.edu
#SBATCH --account=npadillacoreano
#SBATCH --qos=npadillacoreano
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:20:00

module load conda
conda activate SSRNN

cd /home/t.heeps/blue_npadillacoreano/npadillacoreano/share/respiration-project/Supervised_SRNN_respiration
mkdir -p logs

CFG=respiration/config_respiration_hpg.yaml

python respiration/prepare_respiration.py --config $CFG

echo ""
echo "=== observations shape ==="
python -c "import numpy as np; print(np.load('respiration/data_prepared/observations.npy').shape)"
echo "DONE. Inspect window counts above before submitting training."
