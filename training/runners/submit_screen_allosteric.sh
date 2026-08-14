#!/bin/bash
#SBATCH -t 0-02:30:00
#SBATCH --qos=low
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20gb
#SBATCH --begin=now
#SBATCH --job-name=screen_allosteric
#SBATCH --output=/home1/felipetm/auxetic_networks/ensemble_training/Logs/screen_allosteric_%A_%a.out
#SBATCH --error=/home1/felipetm/auxetic_networks/ensemble_training/Logs/screen_allosteric_%A_%a.err

# Realization screening for allosteric training — see docs/realization_screening.md.
# Submit with explicit --array (POOL_SIZE * n_grid_cells - 1) and --export=KIND=targeted|general,
# e.g.:
#   sbatch --array=0-74%50  --export=KIND=targeted,RESULTS_DIR=<dir> submit_screen_allosteric.sh
#   sbatch --array=0-374%50 --export=KIND=general,RESULTS_DIR=<dir>  submit_screen_allosteric.sh

cd $SLURM_SUBMIT_DIR
eval "$(conda shell.bash hook)"
conda activate auxetic_nets
echo "Python: $(which python)"
echo "KIND=${KIND} RESULTS_DIR=${RESULTS_DIR} ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

python screen_realizations_allosteric.py --kind "${KIND}" --results-dir "${RESULTS_DIR}"
