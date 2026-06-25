#!/bin/bash
# ============================================================================
# SLURM Job Array Script — Figure Cache Generation
# ============================================================================
#
# Generates all .figure_notebook_cache/ entries needed by AllFigures.ipynb.
# Run each MODE separately.  All modes are idempotent (skip existing cache).
#
# ── Modes and array index mappings ──────────────────────────────────────────
#
#  global_nt   (one job per (task, real) pair for data/results/)
#    Array 0-599:  task = ID / 20,  real = ID % 20   (skips nonexistent)
#    Submit:  sbatch --export=CACHE_MODE=global_nt --array=0-599%40 submit_figure_cache.sh
#
#  local_nt    (one job per (geom, task, real) triplet for allosteric_nets/)
#    Array 0-124:  geom = ID / 25,  task = (ID % 25) / 5,  real = ID % 5
#    Submit:  sbatch --export=CACHE_MODE=local_nt --array=0-124%40 submit_figure_cache.sh
#
#  global_rep  (single job for the global representative network)
#    Submit:  sbatch --export=CACHE_MODE=global_rep submit_figure_cache.sh
#
#  local_rep   (single job for the local representative network)
#    Submit:  sbatch --export=CACHE_MODE=local_rep submit_figure_cache.sh
#
#  modesens_global  (one job per (task, real) pair — same mapping as global_nt)
#    Submit:  sbatch --export=CACHE_MODE=modesens_global --array=0-599%40 submit_figure_cache.sh
#
#  modesens_local   (one job per (geom, task, real) — same mapping as local_nt)
#    Submit:  sbatch --export=CACHE_MODE=modesens_local --array=0-124%40 submit_figure_cache.sh
#
# ── Recommended submission order ─────────────────────────────────────────────
#
#   1. Representatives (fast, single jobs — good for a quick sanity check):
#        sbatch --export=CACHE_MODE=global_rep submit_figure_cache.sh
#        sbatch --export=CACHE_MODE=local_rep  submit_figure_cache.sh
#
#   2. Non-targeted ensembles (bulk — most of the runtime):
#        sbatch --export=CACHE_MODE=global_nt --array=0-599%40 submit_figure_cache.sh
#        sbatch --export=CACHE_MODE=local_nt  --array=0-124%40 submit_figure_cache.sh
#
#   3. Mode sensitivity (requires shorter trajectories — faster than step 2):
#        sbatch --export=CACHE_MODE=modesens_global --array=0-599%40 submit_figure_cache.sh
#        sbatch --export=CACHE_MODE=modesens_local  --array=0-124%40 submit_figure_cache.sh
#
# ============================================================================
#SBATCH -t 0-01:00:00
#SBATCH --qos=liu
#SBATCH --partition=liu_compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4gb
#SBATCH --job-name=fig_cache
#SBATCH --output=/home1/felipetm/auxetic_networks/ensemble_training/Logs/fig_cache_%A_%a.out
#SBATCH --error=/home1/felipetm/auxetic_networks/ensemble_training/Logs/fig_cache_%A_%a.err

echo "=========================================="
echo "Job ID:           ${SLURM_JOB_ID}"
echo "Array Task ID:    ${SLURM_ARRAY_TASK_ID:-none}"
echo "Node:             $(hostname)"
echo "Start time:       $(date)"
echo "Cache mode:       ${CACHE_MODE}"
echo "=========================================="

cd $SLURM_SUBMIT_DIR

eval "$(conda shell.bash hook)"
conda activate auxetic_nets

echo "Python: $(which python)"
echo "Conda env: ${CONDA_DEFAULT_ENV}"
echo ""

# ── Validate CACHE_MODE ──────────────────────────────────────────────────────
if [[ -z "${CACHE_MODE}" ]]; then
    echo "ERROR: CACHE_MODE is not set. Use --export=CACHE_MODE=<mode> when submitting."
    exit 1
fi

# ── Build argument string based on mode ─────────────────────────────────────
ID=${SLURM_ARRAY_TASK_ID:-0}

case "${CACHE_MODE}" in

  global_nt | modesens_global)
    # task = ID / 20,  real = ID % 20
    TASK_SEED=$((ID / 20))
    REAL_SEED=$((ID % 20))
    echo "Task seed:        ${TASK_SEED}"
    echo "Realization seed: ${REAL_SEED}"
    EXTRA_ARGS="--task ${TASK_SEED} --real ${REAL_SEED}"
    ;;

  local_nt | modesens_local)
    # geom = ID / 25,  task = (ID % 25) / 5,  real = ID % 5
    GEOM_SEED=$((ID / 25))
    TASK_SEED=$(((ID % 25) / 5))
    REAL_SEED=$((ID % 5))
    echo "Geometry seed:    ${GEOM_SEED}"
    echo "Task seed:        ${TASK_SEED}"
    echo "Realization seed: ${REAL_SEED}"
    EXTRA_ARGS="--geom ${GEOM_SEED} --task ${TASK_SEED} --real ${REAL_SEED}"
    ;;

  global_rep)
    echo "Global representative (task 13, real 0)"
    EXTRA_ARGS=""
    ;;

  local_rep)
    echo "Local representative (task 0, real 0)"
    EXTRA_ARGS=""
    ;;

  *)
    echo "ERROR: Unknown CACHE_MODE '${CACHE_MODE}'"
    echo "Valid modes: global_nt, local_nt, global_rep, local_rep, modesens_global, modesens_local"
    exit 1
    ;;
esac

echo ""

python ../analysis/compute_figure_cache.py \
    --mode ${CACHE_MODE} \
    ${EXTRA_ARGS}

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Exit code: ${EXIT_CODE}"
echo "End time:  $(date)"
echo "=========================================="

exit ${EXIT_CODE}
