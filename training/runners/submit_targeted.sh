#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH --qos=low
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20gb
#SBATCH --array=0-5%50
#SBATCH --begin=now
#SBATCH --job-name=targeted_auxetic
#SBATCH --output=/home1/felipetm/auxetic_networks/ensemble_training/Logs/targeted_%A_%a.out
#SBATCH --error=/home1/felipetm/auxetic_networks/ensemble_training/Logs/targeted_%A_%a.err

# ============================================================================
# SLURM Job Array Script for Targeted Training of Auxetic Networks
# ============================================================================
#
# Total jobs: 300 (30 tasks × 10 realizations)
#
# Array indices 0-299 map to:
#   task_id        = SLURM_ARRAY_TASK_ID / 10
#   realization    = SLURM_ARRAY_TASK_ID % 10
#
# ============================================================================

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

cd $SLURM_SUBMIT_DIR

eval "$(conda shell.bash hook)"
conda activate auxetic_nets

echo "Python: $(which python)"
echo "Conda env: ${CONDA_DEFAULT_ENV}"

TASK_ID=$((SLURM_ARRAY_TASK_ID / 1))
REALIZATION=$((SLURM_ARRAY_TASK_ID % 1))

echo ""
echo "Running targeted training:"
echo "  Task ID:     ${TASK_ID}"
echo "  Realization: ${REALIZATION}"
echo ""

python targeted_ensemble_runner.py \
    --mode single \
    --task ${TASK_ID} \
    --realization ${REALIZATION} \
    --verbose \
    --gradient-method jax

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed with exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "=========================================="

exit ${EXIT_CODE}
