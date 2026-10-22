#!/bin/bash
#SBATCH -t 5-00:00:00
#SBATCH --qos=low
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --begin=now
#SBATCH --job-name=netA_supp
#SBATCH --output=/home1/felipetm/auxetic_networks/ensemble_training/Logs/%x_%A_%a.out
#SBATCH --error=/home1/felipetm/auxetic_networks/ensemble_training/Logs/%x_%A_%a.err

# Supplementary auxetic targeted runs on the stored Network A
# (docs/adr/0001, 0002). One array index per line of LOOKUP (0-based):
#     TASK REAL SEED [LR] [STEPS]
# SEED must equal the seed table entry for (TASK, REAL); the job refuses to
# train otherwise. MODE selects how the runner is invoked:
#     fresh     - first attempt, config defaults (LR 3e-3, 5000 steps)
#     extend    - resume with more steps (STEPS required)
#     overwrite - wipe and restart at LR [and STEPS]
# Submit, e.g.:
#     sbatch --array=0-105 --export=ALL,MODE=fresh,LOOKUP=netA_supplement_bc.txt submit_netA_supplement.sh
#     sbatch --array=0-58 --export=ALL,MODE=fresh,LOOKUP=netA_supplement_new.txt submit_netA_supplement.sh

MODE=${MODE:-fresh}
LOOKUP=${LOOKUP:-netA_supplement_bc.txt}

cd /home1/felipetm/auxetic_networks/ensemble_training/training/runners
eval "$(conda shell.bash hook)"
conda activate auxetic_nets

read TASK REAL SEED LR STEPS < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${LOOKUP}")
echo "Node: $(hostname)  Start: $(date)"
echo "MODE=${MODE} TASK=${TASK} REAL=${REAL} SEED=${SEED} LR=${LR:-default} STEPS=${STEPS:-default}"

TABLE_SEED=$(python -c "
import sys; sys.path.insert(0, '../..')
from training.src.good_realizations import get_realization_seed
print(get_realization_seed('auxetic_targeted', ${TASK}, ${REAL}))")
if [ "${TABLE_SEED}" != "${SEED}" ]; then
    echo "ERROR: seed table gives ${TABLE_SEED} for task ${TASK} realization ${REAL}, lookup says ${SEED}"
    exit 2
fi

EXTRA=()
case ${MODE} in
    fresh) ;;
    extend) EXTRA+=(--steps ${STEPS}) ;;
    overwrite)
        EXTRA+=(--overwrite --learning-rate ${LR})
        [ -n "${STEPS}" ] && EXTRA+=(--steps ${STEPS}) ;;
    *) echo "ERROR: unknown MODE=${MODE}"; exit 2 ;;
esac

python -u targeted_ensemble_runner.py --mode single --task ${TASK} --realization ${REAL} \
    --gradient-method jax --network-type jammed --verbose \
    --network-file ../../base/networks/auxetic_network_A.npz "${EXTRA[@]}"
EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    python -u post_training_sweep.py --task-type targeted --task ${TASK} --realization ${REAL} \
        --network-type jammed --gradient-method jax --k-eigs 4 --n-hessian-traj-steps 20
    python -u verify_and_plot_loss.py --task-type targeted --task ${TASK} --realization ${REAL} \
        --network-type jammed --gradient-method jax
fi

echo "Exit: ${EXIT_CODE}  End: $(date)"
exit ${EXIT_CODE}
