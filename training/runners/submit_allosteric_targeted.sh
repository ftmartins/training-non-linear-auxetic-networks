#!/bin/bash
#SBATCH -t 5-00:00:00
#SBATCH --qos=low
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20gb
#SBATCH --begin=now
#SBATCH --array=0-24%25
#SBATCH --job-name=allosteric_targeted
#SBATCH --output=/home1/felipetm/auxetic_networks/ensemble_training/Logs/allosteric_targeted_%A_%a.out
#SBATCH --error=/home1/felipetm/auxetic_networks/ensemble_training/Logs/allosteric_targeted_%A_%a.err

# ============================================================================
# SLURM array for the TARGETED allosteric ensemble: 5 tasks × 5 realizations
# = 25 jobs total (array indices 0-24), all sharing one fixed geometry
# (--targeted-ensemble makes allosteric_trainer.py ignore --geometry-id and
# use TARGETED_ENSEMBLE's fixed shared geometry instead).
#
# Index encoding:
#   task_id        = SLURM_ARRAY_TASK_ID / 5   (0-4)
#   realization_id = SLURM_ARRAY_TASK_ID % 5   (0-4)
#
# Output: <OUTPUT_DIR>/geometry_targeted/task_<t>/realization_<r>/
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

TASK_ID=$((SLURM_ARRAY_TASK_ID / 5))
REALIZATION_ID=$((SLURM_ARRAY_TASK_ID % 5))

echo ""
echo "Task ID:        ${TASK_ID}"
echo "Realization ID: ${REALIZATION_ID}"
echo ""

python allosteric_trainer.py \
    --task-id        ${TASK_ID} \
    --realization-id ${REALIZATION_ID} \
    --training-steps 5000 \
    --output-dir     /data2/shared/felipetm/allosteric_nets_aug \
    --targeted-ensemble

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "Running post-training timestep-sweep analysis..."
    python post_training_sweep.py --task-type allosteric \
        --task ${TASK_ID} --realization ${REALIZATION_ID} --targeted-ensemble

    python verify_and_plot_loss.py --task-type allosteric \
        --task ${TASK_ID} --realization ${REALIZATION_ID} --targeted-ensemble
fi

echo ""
echo "=========================================="
echo "Finished with exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "=========================================="

exit ${EXIT_CODE}
