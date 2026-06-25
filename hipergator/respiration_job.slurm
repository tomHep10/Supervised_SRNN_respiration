#!/bin/bash
# =============================================================================
# HiPerGator GPU job for the respiration SRNN  (created by Claude)
# Trains 4 leave-one-recording-out folds (array 0-3) on the GPU.
#
# ONE-TIME, on the LOGIN node, before submitting:
#   module load conda
#   cd /home/t.heeps/blue_npadillacoreano/npadillacoreano/share/respiration-project/Supervised_SRNN_respiration
#   conda env create -f environment.yml          # builds SSRNN (once)
#   conda activate SSRNN
#   python respiration/prepare_respiration.py --config respiration/config_respiration_hpg.yaml
#
# Then fill in GROUP/QOS/GPU_TYPE below (verify: sacctmgr show assoc where user=$USER
# format=account,user,qos%40) and submit:
#   sbatch hipergator/respiration_job.slurm
# After all 4 folds finish, pool them for the leakage-free valence number:
#   python respiration/collect_folds.py --config respiration/config_respiration_hpg.yaml
# =============================================================================
#SBATCH --job-name=resp-srnn
#SBATCH --output=logs/resp_fold%a_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.heeps@ufl.edu
#SBATCH --account=npadillacoreano          # verify with sacctmgr (your blue dir => npadillacoreano)
#SBATCH --qos=npadillacoreano              # or npadillacoreano-b for burst
#SBATCH --partition=hpg-b200                 # HiPerGator B200 GPU partition (gpu/hpg-ai are retired)
#SBATCH --gres=gpu:1                         # 1 GPU of whatever type the partition has
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --time=08:00:00
#SBATCH --array=0-3                         # 4 recordings, leave-one-out

module load conda
conda activate SSRNN

cd /home/t.heeps/blue_npadillacoreano/npadillacoreano/share/respiration-project/Supervised_SRNN_respiration
mkdir -p logs

CFG=respiration/config_respiration_hpg.yaml

# NOTE: run prepare_respiration.py ONCE on the login node first (see header) so
# respiration/data_prepared/ exists before this array starts.
python respiration/train_respiration.py --config $CFG --fold $SLURM_ARRAY_TASK_ID --split recording
python respiration/plot_respiration.py  --config $CFG --fold $SLURM_ARRAY_TASK_ID --split recording
