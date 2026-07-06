#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH --qos=low
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20gb
#SBATCH --array=0-299%50
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

TASK_ID=$((SLURM_ARRAY_TASK_ID / 10))
REALIZATION=$((SLURM_ARRAY_TASK_ID % 10))

# Network generation method: 'jammed' (default) or 'lattice'.
# Override at submission time, e.g.: sbatch --export=NETWORK_TYPE=lattice submit_targeted.sh
NETWORK_TYPE="${NETWORK_TYPE:-jammed}"

echo ""
echo "Running targeted training:"
echo "  Task ID:     ${TASK_ID}"
echo "  Realization: ${REALIZATION}"
echo "  Network type: ${NETWORK_TYPE}"
echo ""

python targeted_ensemble_runner.py \
    --mode single \
    --task ${TASK_ID} \
    --realization ${REALIZATION} \
    --network-type ${NETWORK_TYPE} \
    --verbose

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "Running post-training timestep-sweep analysis..."
    python post_training_sweep.py --task-type targeted --task ${TASK_ID} --realization ${REALIZATION}
    python verify_and_plot_loss.py --task-type targeted --task ${TASK_ID} --realization ${REALIZATION}
fi

echo ""
echo "=========================================="
echo "Job completed with exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "=========================================="

exit ${EXIT_CODE}
