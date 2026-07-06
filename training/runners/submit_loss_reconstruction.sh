#!/bin/bash
#SBATCH -t 02:00:00
#SBATCH --qos=low
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20gb
#SBATCH --begin=now
#SBATCH --job-name=loss_recon_verify
#SBATCH --output=/home1/felipetm/auxetic_networks/ensemble_training/Logs/loss_recon_%j.out
#SBATCH --error=/home1/felipetm/auxetic_networks/ensemble_training/Logs/loss_recon_%j.err

# ============================================================================
# Runs loss_reconstruction_verification.py on the cluster (--mode train, by
# default; see usage in the script for the cluster -> download -> local
# --mode verify workflow).
# ============================================================================

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

cd $SLURM_SUBMIT_DIR

eval "$(conda shell.bash hook)"
conda activate auxetic_nets

echo "Python: $(which python)"
echo "Conda env: ${CONDA_DEFAULT_ENV}"

RESULTS_DIR="${RESULTS_DIR:-$SLURM_SUBMIT_DIR/loss_reconstruction_results}"

# Network generation method: 'jammed' (default) or 'lattice'.
# Override at submission time, e.g.: sbatch --export=NETWORK_TYPE=lattice submit_loss_reconstruction.sh
NETWORK_TYPE="${NETWORK_TYPE:-jammed}"

echo ""
echo "Results dir: ${RESULTS_DIR}"
echo "Network type: ${NETWORK_TYPE}"
echo ""

python loss_reconstruction_verification.py \
    --mode all \
    --results-dir "${RESULTS_DIR}" \
    --network-type "${NETWORK_TYPE}"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed with exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "=========================================="

exit ${EXIT_CODE}
