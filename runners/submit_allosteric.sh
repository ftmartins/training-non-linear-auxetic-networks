#!/bin/bash
#SBATCH -t 5-00:00:00
#SBATCH --qos=low
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4gb
#SBATCH --array=0-99%20
#SBATCH --job-name=allosteric
#SBATCH --output=/home1/felipetm/auxetic_networks/ensemble_training/Logs/allosteric_%A_%a.out
#SBATCH --error=/home1/felipetm/auxetic_networks/ensemble_training/Logs/allosteric_%A_%a.err

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node:          $(hostname)"
echo "Start time:    $(date)"
echo "=========================================="

cd $SLURM_SUBMIT_DIR

eval "$(conda shell.bash hook)"
conda activate auxetic_nets

echo "Python: $(which python)"
echo "Conda env: ${CONDA_DEFAULT_ENV}"

NETWORK_ID=${SLURM_ARRAY_TASK_ID}
BASE_SEED=314159265
SEED=$((BASE_SEED + NETWORK_ID))

echo ""
echo "Network ID: ${NETWORK_ID}"
echo "Seed:       ${SEED}"
echo ""

python allosteric_trainer.py \
    --network-id     ${NETWORK_ID} \
    --seed           ${SEED} \
    --training-steps 10000 \
    --output-dir     /data2/shared/felipetm/allosteric_nets

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Finished with exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "=========================================="

exit ${EXIT_CODE}
