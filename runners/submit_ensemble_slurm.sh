#!/bin/bash
#SBATCH -t 5-00:00:00
#SBATCH --qos=low
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5gb
#SBATCH --array=0-200%20
#SBATCH --begin=now
#SBATCH --job-name=auxetic_ensemble
#SBATCH --output=/home1/felipetm/auxetic_networks/ensemble_training/Logs/auxetic_training_%A_%a.out
#SBATCH --error=/home1/felipetm/auxetic_networks/ensemble_training/Logs/auxetic_training_%A_%a.err

# ============================================================================
# SLURM Job Array Script for Ensemble Training of Auxetic Networks
# ============================================================================
#
# This script runs ensemble training with checkpointing support.
# Total jobs: 500 (10 tasks × 50 realizations)
#
# Array indices 0-499 map to:
#   task_seed = SLURM_ARRAY_TASK_ID / 50
#   realization_seed = SLURM_ARRAY_TASK_ID % 50
#
# Features:
#   - Automatic checkpoint/resume on failure
#   - Skips already completed jobs
#   - 8 CPUs for parallel gradient computation
#   - 5GB memory per job
#   - Max 20 jobs running simultaneously (%20)
#
# ============================================================================

# Print job info
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# Navigate to submission directory (runners/)
cd $SLURM_SUBMIT_DIR

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate auxetic_nets

# Verify environment
echo "Python: $(which python)"
echo "Conda env: ${CONDA_DEFAULT_ENV}"

# Calculate task and realization seeds from array index
TASK_SEED=$((SLURM_ARRAY_TASK_ID / 20))
REALIZATION_SEED=$((SLURM_ARRAY_TASK_ID % 20))

echo ""
echo "Running training:"
echo "  Task seed: ${TASK_SEED}"
echo "  Realization seed: ${REALIZATION_SEED}"
echo ""

# Run training with checkpoint support
python ensemble_runner.py \
    --mode single \
    --task ${TASK_SEED} \
    --realization ${REALIZATION_SEED} \
    --verbose

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed with exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "=========================================="

exit ${EXIT_CODE}
