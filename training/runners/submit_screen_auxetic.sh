#!/bin/bash
#SBATCH -t 0-01:00:00
#SBATCH --qos=low
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --begin=now
#SBATCH --job-name=screen_auxetic
#SBATCH --output=/home1/felipetm/auxetic_networks/ensemble_training/Logs/screen_auxetic_%A_%a.out
#SBATCH --error=/home1/felipetm/auxetic_networks/ensemble_training/Logs/screen_auxetic_%A_%a.err

# Realization screening for auxetic training — see docs/realization_screening.md.
# Submit with explicit --array (POOL_SIZE * n_tasks - 1) and --export=KIND=targeted|general,
# e.g.:
#   sbatch --array=0-74%50   --export=KIND=targeted,RESULTS_DIR=<dir> submit_screen_auxetic.sh
#   sbatch --array=0-449%50  --export=KIND=general,RESULTS_DIR=<dir>  submit_screen_auxetic.sh
#
# 8GB is generous headroom post-lax.scan (base/simulate.py's
# compute_quasistatic_trajectory_auxetic_jax): the same compile that used to
# need 25+GB now peaks under 1GB — see docs/jax_solver_speedup.md history /
# git log for the lax.scan change.

cd $SLURM_SUBMIT_DIR
eval "$(conda shell.bash hook)"
conda activate auxetic_nets
echo "Python: $(which python)"
echo "KIND=${KIND} RESULTS_DIR=${RESULTS_DIR} ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

python screen_realizations_auxetic.py --kind "${KIND}" --results-dir "${RESULTS_DIR}"
