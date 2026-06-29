#!/bin/bash
# =============================================================================
# HiPerGator CPU job: (re)prepare the CAGEMATE/RANK windows. Same signal pipeline as
# the valence prep (50 Hz, dead-signal removal, 30 s top-N-sniffing windows) but the
# target is RANK and the data is the cagemate baseline recordings. Writes
# respiration/data_prepared_rank/ (observations.npy, labels.npy, meta.npz, label_map.json).
#
# CPU only, fast (minutes). Submit from the repo root:
#   sbatch hipergator/prepare_rank.slurm
# Then READ the log — confirm each cagemate recording, its subject/cage/rank, the
# dead %, and the Dom/Sub window counts. The second s3_5 recording (no csv) is dropped
# by default (discovery.skip_no_csv).
#   cat logs/prepare_rank_*.log
# =============================================================================
#SBATCH --job-name=rank-prepare
#SBATCH --output=logs/prepare_rank_%j.log
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

CFG=${1:-respiration/rank/config_rank_hpg.yaml}   # pass a cohort config as arg 1 to run a different cohort

python respiration/rank/prepare_rank.py --config $CFG

echo ""
echo "=== observations shape ==="
python -c "import yaml,os,numpy as np; p=yaml.safe_load(open('$CFG'))['paths']['out_dir']; print(np.load(os.path.join(p,'observations.npy')).shape)"
echo "DONE. Inspect window counts / rank balance above before submitting LOCO training."
