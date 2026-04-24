from checkpoint_manager import get_incomplete_jobs, get_complete_jobs
from config import N_TASKS, N_REALIZATIONS

data_dir = '/data2/shared/felipetm/auxetic_networks/ensemble_training_new/results_new/'

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
    array_indices = [task * N_REALIZATIONS + real for task, real in incomplete]
#    # Print in batches of 50 for readability
#    for i in range(0, len(array_indices), N_REALIZATIONS):
#        batch = array_indices[i:i+N_REALIZATIONS]
#        print(','.join(map(str, batch)))
    incomplete_per_task = {}
    for (task, real) in incomplete:
        if task in incomplete_per_task.keys():
            incomplete_per_task[task].append(real)
        else:
            incomplete_per_task[task] = [real]
    for task in incomplete_per_task.keys():
        print(f'Task {task}')
        print(incomplete_per_task[task])

    # Write to file for easy resubmission
    with open('incomplete_jobs.txt', 'w') as f:
        f.write(','.join(map(str, array_indices)))
    print(f'')
    print('Incomplete job indices saved to: incomplete_jobs.txt')
    print(f'')
    print('To resubmit incomplete jobs, use:')
    print('  sbatch --array=$(cat incomplete_jobs.txt) submit_ensemble_slurm.sh')
