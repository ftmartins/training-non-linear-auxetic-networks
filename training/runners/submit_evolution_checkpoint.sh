#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --qos=low
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --begin=now
#SBATCH --job-name=evol_ckpt
#SBATCH --output=/home1/felipetm/auxetic_networks/ensemble_training/Logs/%x_%A_%a.out
#SBATCH --error=/home1/felipetm/auxetic_networks/ensemble_training/Logs/%x_%A_%a.err

# explorations_sep21.ipynb section 5b (mode-overlap evolution through training), computed
# per-checkpoint on the cluster instead of serially in the notebook (this cell repeatedly
# overran nbconvert's timeout). One task per checkpoint; FAMILY=allo|aux, N_CKPT=8 (notebook
# default). After every checkpoint for a family completes, run:
#   python compute_evolution_checkpoint.py --assemble --family FAMILY --n-ckpt N_CKPT
# which writes the notebook's own cache pkl directly -- rsync it back and the notebook
# reads it with no recompute.
#   sbatch --export=ALL,FAMILY=allo,N_CKPT=8 --array=0-7 submit_evolution_checkpoint.sh
#   sbatch --export=ALL,FAMILY=aux,N_CKPT=8  --array=0-$((N-1)) submit_evolution_checkpoint.sh
#     (N = `python compute_evolution_checkpoint.py --count --family aux --n-ckpt 8`)

cd /home1/felipetm/auxetic_networks/ensemble_training/training/runners
eval "$(conda shell.bash hook)"
conda activate auxetic_nets
echo "family=${FAMILY} n_ckpt=${N_CKPT} ckpt_idx=${SLURM_ARRAY_TASK_ID}"
python -u compute_evolution_checkpoint.py --family ${FAMILY} --n-ckpt ${N_CKPT} --ckpt-idx ${SLURM_ARRAY_TASK_ID}
EXIT_CODE=$?
echo "Finished arr ${SLURM_ARRAY_TASK_ID} exit ${EXIT_CODE}"
exit ${EXIT_CODE}
