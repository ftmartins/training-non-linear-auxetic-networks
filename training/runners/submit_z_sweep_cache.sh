#!/bin/bash
#SBATCH -t 06:00:00
#SBATCH --qos=low
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --begin=now
#SBATCH --job-name=z_sweep_cache
#SBATCH --output=/home1/felipetm/auxetic_networks/ensemble_training/Logs/%x_%A_%a.out
#SBATCH --error=/home1/felipetm/auxetic_networks/ensemble_training/Logs/%x_%A_%a.err

# explorations_sep21.ipynb section 2 (z-sweep susceptibilities/cost-Hessian), computed on the
# cluster instead of in the notebook. Writes z_<family>_<case_id>.pkl into
# /data2/shared/felipetm/explore_sep21/z_sweep_cache/ -- rsync that dir into the local
# figure_data/explore_sep21/_cache/ and the notebook picks the results up with no recompute
# (it already skips any case_id whose pkl exists).
#   sbatch --array=0-$((N-1))%80 submit_z_sweep_cache.sh   # N = line count of z_sweep_lookup.txt

cd /home1/felipetm/auxetic_networks/ensemble_training/training/runners
eval "$(conda shell.bash hook)"
conda activate auxetic_nets
echo "case ${SLURM_ARRAY_TASK_ID}"
python -u compute_z_sweep_cache.py --case-index ${SLURM_ARRAY_TASK_ID} --lookup-file z_sweep_lookup.txt
EXIT_CODE=$?
echo "Finished arr ${SLURM_ARRAY_TASK_ID} exit ${EXIT_CODE}"
exit ${EXIT_CODE}
