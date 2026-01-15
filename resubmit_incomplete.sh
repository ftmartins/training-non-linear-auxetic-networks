#!/bin/bash
# Script to identify and resubmit incomplete jobs

echo "=========================================="
echo "Checking ensemble training status..."
echo "=========================================="

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate auxetic_nets

# Get list of incomplete jobs
python -c "
from checkpoint_manager import get_incomplete_jobs, get_complete_jobs
from config import N_TASKS, N_REALIZATIONS

complete = get_complete_jobs()
incomplete = get_incomplete_jobs()
total = N_TASKS * N_REALIZATIONS

print(f'')
print(f'Total jobs: {total}')
print(f'Complete: {len(complete)} ({100*len(complete)/total:.1f}%)')
print(f'Incomplete: {len(incomplete)} ({100*len(incomplete)/total:.1f}%)')
print(f'')

if len(incomplete) > 0:
    print('Incomplete job array indices:')
    array_indices = [task * 50 + real for task, real in incomplete]
    # Print in batches of 50 for readability
    for i in range(0, len(array_indices), 50):
        batch = array_indices[i:i+50]
        print(','.join(map(str, batch)))

    # Write to file for easy resubmission
    with open('incomplete_jobs.txt', 'w') as f:
        f.write(','.join(map(str, array_indices)))
    print(f'')
    print('Incomplete job indices saved to: incomplete_jobs.txt')
    print(f'')
    print('To resubmit incomplete jobs, use:')
    print('  sbatch --array=$(cat incomplete_jobs.txt) submit_ensemble_slurm.sh')
else:
    print('All jobs complete!')
"

echo "=========================================="
