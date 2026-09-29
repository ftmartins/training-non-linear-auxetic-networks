#!/bin/bash
#SBATCH -t 5-00:00:00
#SBATCH --qos=low
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --begin=now
#SBATCH --job-name=exp21_allo
#SBATCH --output=/home1/felipetm/auxetic_networks/ensemble_training/Logs/%x_%A_%a.out
#SBATCH --error=/home1/felipetm/auxetic_networks/ensemble_training/Logs/%x_%A_%a.err

# 2026-09-21 explorations, allosteric z-sweep. Output -> /data2/shared/felipetm/explore_sep21/allo_z/
#   sbatch --array=0-249%80 submit_explore_sep21_allo.sh
# Resubmit with EXTRA="--training-steps 12000" to extend (resumes). LR 10 default (as production).

cd /home1/felipetm/auxetic_networks/ensemble_training/training/runners
eval "$(conda shell.bash hook)"
conda activate auxetic_nets
echo "case=${SLURM_ARRAY_TASK_ID} extra=${EXTRA}"
python -u explore_sep21_allo.py --case-index ${SLURM_ARRAY_TASK_ID} ${EXTRA}
EXIT_CODE=$?
echo "Finished exit ${EXIT_CODE}"
exit ${EXIT_CODE}
