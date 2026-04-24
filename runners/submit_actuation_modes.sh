#!/bin/bash
# ============================================================================
# SLURM Job Array Script for Actuation Mode Analysis
# ============================================================================
#
# Each job runs compute_actuation_modes.py for one task seed, covering all
# qualifying realizations (filtered by relative min-loss threshold).
#
# Array index maps directly to task seed:
#   task_seed = SLURM_ARRAY_TASK_ID
#
# Submit for tasks 10-20:
#   sbatch --array=10-20 submit_actuation_modes.sh
#
# Submit a subset:
#   sbatch --array=11,12,13,14,17 submit_actuation_modes.sh
#
# Output per task:
#   <OUTPUT_DIR>/task_<NN>/all_results.pkl    – actuation + Hessian mode data
#   <OUTPUT_DIR>/task_<NN>/all_mode_data.pkl  – overlap, correlation, feature vectors
#
# ============================================================================
#SBATCH -t 3-00:00:00
#SBATCH --qos=low
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --job-name=actuation_modes
#SBATCH --output=/home1/felipetm/auxetic_networks/ensemble_training/Logs/actuation_modes_%A_%a.out
#SBATCH --error=/home1/felipetm/auxetic_networks/ensemble_training/Logs/actuation_modes_%A_%a.err

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node:          $(hostname)"
echo "Start time:    $(date)"
echo "=========================================="

cd $SLURM_SUBMIT_DIR

eval "$(conda shell.bash hook)"
conda activate auxetic_nets

echo "Python: $(which python)"
echo "Conda env: ${CONDA_DEFAULT_ENV}"

TASK_SEED=${SLURM_ARRAY_TASK_ID}

echo ""
echo "Task seed: ${TASK_SEED}"
echo ""

DATA_DIR=/data2/shared/felipetm/auxetic_networks/ensemble_training_new/results_new/
OUTPUT_DIR=/data2/shared/felipetm/auxetic_networks/ensemble_training_new/results_new/figure_data/

python compute_actuation_modes.py \
    --task           ${TASK_SEED} \
    --data-dir       ${DATA_DIR} \
    --output-dir     ${OUTPUT_DIR} \
    --loss-threshold 0.0001 \
    --traj-subsample 4 \
    --n-modes        25 \
    --n-corr-modes   10

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Exit code: ${EXIT_CODE}"
echo "End time:  $(date)"
echo "=========================================="

exit ${EXIT_CODE}
