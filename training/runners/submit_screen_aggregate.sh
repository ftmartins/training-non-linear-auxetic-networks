#!/bin/bash
#SBATCH -t 0-00:15:00
#SBATCH --qos=low
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2gb
#SBATCH --begin=now
#SBATCH --job-name=screen_aggregate
#SBATCH --output=/home1/felipetm/auxetic_networks/ensemble_training/Logs/screen_aggregate_%A_%a.out
#SBATCH --error=/home1/felipetm/auxetic_networks/ensemble_training/Logs/screen_aggregate_%A_%a.err

# Aggregates one kind's screening trial results into the good_realizations
# lookup table. Submit with --dependency=afterok:<screening_array_job_id> and
# --export=PHYSICS=auxetic|allosteric,KIND=targeted|general,RESULTS_DIR=<dir>.

cd $SLURM_SUBMIT_DIR
eval "$(conda shell.bash hook)"
conda activate auxetic_nets
echo "Python: $(which python)"
echo "PHYSICS=${PHYSICS} KIND=${KIND} RESULTS_DIR=${RESULTS_DIR}"

python screen_aggregate.py --physics "${PHYSICS}" --kind "${KIND}" --results-dir "${RESULTS_DIR}"
