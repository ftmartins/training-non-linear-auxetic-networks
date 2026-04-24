#!/bin/bash
# ============================================================================
# SLURM Job Array Script for Cost Hessian Eigenvector Computation
# ============================================================================
#
# Each job computes the Lanczos top-k eigenvalues/vectors of the cost Hessian
# for one (task, realization) pair and saves to:
#   data/results/task_XX/realization_XX/cost_hessian_eigs.npz
#
# Array indices 0-199 map to:
#   task_seed        = SLURM_ARRAY_TASK_ID / 20
#   realization_seed = SLURM_ARRAY_TASK_ID % 20
#
# Submit all:
#   sbatch --array=0-199%20 submit_cost_hessian.sh
#
# Submit specific pairs (e.g. tasks 0-4):
#   sbatch --array=0-99%20 submit_cost_hessian.sh
#
# ============================================================================
#SBATCH -t 2-00:00:00
#SBATCH --qos=low
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8gb
#SBATCH --job-name=cost_hessian
#SBATCH --output=/home1/felipetm/auxetic_networks/ensemble_training/Logs/cost_hessian_%A_%a.out
#SBATCH --error=/home1/felipetm/auxetic_networks/ensemble_training/Logs/cost_hessian_%A_%a.err

echo "=========================================="
echo "Job ID:           ${SLURM_JOB_ID}"
echo "Array Task ID:    ${SLURM_ARRAY_TASK_ID}"
echo "Node:             $(hostname)"
echo "Start time:       $(date)"
echo "=========================================="

cd $SLURM_SUBMIT_DIR

eval "$(conda shell.bash hook)"
conda activate auxetic_nets

echo "Python: $(which python)"
echo "Conda env: ${CONDA_DEFAULT_ENV}"

# Map array index → (task, realization)
TASK_SEED=$((SLURM_ARRAY_TASK_ID / 20))
REALIZATION_SEED=$((SLURM_ARRAY_TASK_ID % 20))

echo ""
echo "Task seed:        ${TASK_SEED}"
echo "Realization seed: ${REALIZATION_SEED}"
echo ""

python ../analysis/compute_cost_hessian.py \
    --task         ${TASK_SEED} \
    --realization  ${REALIZATION_SEED} \
    --data_dir     /data2/shared/felipetm/auxetic_networks/ensemble_training_new/results_new/ \
    --n_jobs       4 \
    --k_eigs       20 \
    --epsilon      1e-5 \
    --hvp_epsilon  1e-4

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Exit code: ${EXIT_CODE}"
echo "End time:  $(date)"
echo "=========================================="

exit ${EXIT_CODE}
