#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH --qos=low
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --begin=now
#SBATCH --job-name=netA_pilot
#SBATCH --output=/home1/felipetm/auxetic_networks/ensemble_training/Logs/%x_%j.out
#SBATCH --error=/home1/felipetm/auxetic_networks/ensemble_training/Logs/%x_%j.err

# Pilot for docs/adr/0002: retrain an existing Network A run on the stored
# network into a scratch tree and require a bit-identical trajectory.
# Default: targeted task 0 realization 7 (seed 1000013, 14 steps, one
# invocation on 4276f68). Override: sbatch --export=ALL,TASK=..,REAL=.. ...

TASK=${TASK:-0}
REAL=${REAL:-7}
REF_ROOT=/data2/shared/felipetm/auxetic_networks/targeted_results_sqr_aug
PILOT_ROOT=/data2/shared/felipetm/auxetic_networks/netA_pilot

echo "Node: $(hostname)  Start: $(date)"
cd /home1/felipetm/auxetic_networks/ensemble_training/training/runners
eval "$(conda shell.bash hook)"
conda activate auxetic_nets

rm -rf "${PILOT_ROOT}/jax/task_$(printf %02d ${TASK})/realization_$(printf %02d ${REAL})"

python -u targeted_ensemble_runner.py --mode single --task ${TASK} --realization ${REAL} \
    --gradient-method jax --network-type jammed --verbose \
    --network-file ../../base/networks/auxetic_network_A.npz \
    --results-dir ${PILOT_ROOT}
EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    python -u check_netA_pilot.py --task ${TASK} --realization ${REAL} \
        --ref-root ${REF_ROOT}/jax --pilot-root ${PILOT_ROOT}/jax
    EXIT_CODE=$?
fi

echo "Exit: ${EXIT_CODE}  End: $(date)"
exit ${EXIT_CODE}
