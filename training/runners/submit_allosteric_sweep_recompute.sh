#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH --qos=low
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --begin=now
#SBATCH --array=0-599%60
#SBATCH --job-name=allo_swp_bestk
#SBATCH --output=/home1/felipetm/auxetic_networks/ensemble_training/Logs/allo_swp_bestk_%A_%a.out
#SBATCH --error=/home1/felipetm/auxetic_networks/ensemble_training/Logs/allo_swp_bestk_%A_%a.err

# ============================================================================
# Sweep-ONLY recompute for the allosteric _aug ensemble (no retraining).
# Regenerates timestep_sweep.npz with cost_hessian_after_* evaluated at
# best_stiffnesses.npy (the trainer's per-step argmin-loss K*) instead of the
# coarse checkpoint-grid argmin -- see analysis/notebooks/figures/best_stiffnesses_audit.md
# and analysis/timestep_sweep.py::sweep_allosteric (best_stiffnesses kwarg).
#
# 600 jobs = 6 geometries x 5 tasks x 20 realizations.
#   gi   = SLURM_ARRAY_TASK_ID / 100     (0..4 -> geometry_<gi>; 5 -> geometry_targeted)
#   task = (SLURM_ARRAY_TASK_ID / 20) % 5
#   real =  SLURM_ARRAY_TASK_ID % 20
#
# Cost Hessian backend: jax_fire (autodiff). The FD-LAMMPS "match" path never
# finishes within wall limits for these networks (base gradient alone is ~3-6 h;
# 0/143 allosteric_sweep_all logs ever completed a single eigsh) -- see
# memory project_costhessian_solver_match. --k-eigs 4, --n-hessian-traj-steps 6
# match the last successful sweep-only run (job 1039442, submit_allosteric_sweep_jax.sh).
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
echo "Python: $(which python)   Conda env: ${CONDA_DEFAULT_ENV}"

GI=$((SLURM_ARRAY_TASK_ID / 100))
TASK_ID=$(((SLURM_ARRAY_TASK_ID / 20) % 5))
REALIZATION_ID=$((SLURM_ARRAY_TASK_ID % 20))

OUTPUT_DIR=/data2/shared/felipetm/allosteric_nets_aug

if [ "${GI}" -eq 5 ]; then
    GEOM_DIR=geometry_targeted
    GEOM_ARGS="--targeted-ensemble"
else
    GEOM_DIR=geometry_${GI}
    GEOM_ARGS="--geometry ${GI}"
fi

REAL_DIR="${OUTPUT_DIR}/${GEOM_DIR}/task_${TASK_ID}/realization_${REALIZATION_ID}"
echo "Target: ${REAL_DIR}"

if [ ! -f "${REAL_DIR}/stiffnesses_traj.npy" ] || [ ! -f "${REAL_DIR}/mse1.npy" ]; then
    echo "SKIP: no trained realization at ${REAL_DIR} (missing stiffnesses_traj.npy / mse1.npy)"
    exit 0
fi

python post_training_sweep.py --task-type allosteric \
    --task ${TASK_ID} --realization ${REALIZATION_ID} ${GEOM_ARGS} \
    --output-dir ${OUTPUT_DIR} \
    --cost-hessian-solver jax_fire \
    --k-eigs 4 \
    --n-hessian-traj-steps 6

EXIT_CODE=$?
echo "=========================================="
echo "post_training_sweep exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "=========================================="
exit ${EXIT_CODE}
