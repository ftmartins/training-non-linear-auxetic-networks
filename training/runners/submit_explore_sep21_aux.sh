#!/bin/bash
#SBATCH -t 5-00:00:00
#SBATCH --qos=low
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --begin=now
#SBATCH --job-name=exp21_aux
#SBATCH --output=/home1/felipetm/auxetic_networks/ensemble_training/Logs/%x_%A_%a.out
#SBATCH --error=/home1/felipetm/auxetic_networks/ensemble_training/Logs/%x_%A_%a.err

# 2026-09-21 explorations, auxetic side. Output -> /data2/shared/felipetm/explore_sep21/<FAMILY>/
# (separate from the production ensemble). Resubmitting the same array resumes from checkpoints.
#   sbatch --array=0-249%60 --export=ALL,FAMILY=aux_z submit_explore_sep21_aux.sh
#   sbatch --array=0-14     --export=ALL,FAMILY=aux_stretch submit_explore_sep21_aux.sh
# Optional: EXTRA="--steps 10000" or EXTRA="--overwrite --learning-rate 1e-3".

cd /home1/felipetm/auxetic_networks/ensemble_training/training/runners
eval "$(conda shell.bash hook)"
conda activate auxetic_nets
echo "family=${FAMILY} case=${SLURM_ARRAY_TASK_ID} extra=${EXTRA}"
python -u explore_sep21_aux.py --family ${FAMILY} --case-index ${SLURM_ARRAY_TASK_ID} ${EXTRA}
EXIT_CODE=$?
echo "Finished exit ${EXIT_CODE}"
exit ${EXIT_CODE}
