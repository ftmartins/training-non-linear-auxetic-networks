#!/bin/bash
#SBATCH -t 5-00:00:00
#SBATCH --qos=low
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4gb
#SBATCH --array=0-124%20
#SBATCH --job-name=allosteric
#SBATCH --output=/home1/felipetm/auxetic_networks/ensemble_training/Logs/allosteric_%A_%a.out
#SBATCH --error=/home1/felipetm/auxetic_networks/ensemble_training/Logs/allosteric_%A_%a.err

# ============================================================================
# SLURM array for allosteric ensemble: 5 geometries × 5 tasks × 5 realizations
# = 125 jobs total (array indices 0-124).
#
# Index encoding:
#   geometry_id    = SLURM_ARRAY_TASK_ID / 25        (0-4)
#   task_id        = (SLURM_ARRAY_TASK_ID / 5) % 5  (0-4)
#   realization_id = SLURM_ARRAY_TASK_ID % 5         (0-4)
#
# Output: <OUTPUT_DIR>/geometry_<g>/task_<t>/realization_<r>/
# ============================================================================

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

GEOMETRY_ID=$((SLURM_ARRAY_TASK_ID / 25))
TASK_ID=$(((SLURM_ARRAY_TASK_ID / 5) % 5))
REALIZATION_ID=$((SLURM_ARRAY_TASK_ID % 5))

echo ""
echo "Geometry ID:    ${GEOMETRY_ID}"
echo "Task ID:        ${TASK_ID}"
echo "Realization ID: ${REALIZATION_ID}"
echo ""

python allosteric_trainer.py \
    --geometry-id    ${GEOMETRY_ID} \
    --task-id        ${TASK_ID} \
    --realization-id ${REALIZATION_ID} \
    --training-steps 100000 \
    --output-dir     /data2/shared/felipetm/allosteric_nets

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Finished with exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "=========================================="

exit ${EXIT_CODE}
