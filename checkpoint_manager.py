"""
Checkpoint management and result I/O for ensemble training.

This module handles saving and loading training results, as well as
tracking which jobs are complete for resume capability.

Completion criteria (either one satisfies):
1. Standard completion: training_complete.txt exists
2. Loss-based completion: loss reduced by ≥1000× (3 orders of magnitude)
   and training_complete_small_loss.txt created
"""

import os
import pickle
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from config import RESULTS_DIR, N_TASKS, N_REALIZATIONS


def get_training_result_path(task_seed, realization_seed, results_dir=None):
    """
    Get the directory path for a specific training result.

    Args:
        task_seed: Task index
        realization_seed: Realization index
        results_dir: Results directory (default: from config)

    Returns:
        result_path: Path object for this training result
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    
    task_dir = Path(results_dir) / f"task_{task_seed:02d}"
    return task_dir / f"realization_{realization_seed:02d}"


def mark_training_complete_small_loss(task_seed, realization_seed, reduction_ratio=None, results_dir=None):
    """
    Mark a training job as complete based on loss reduction criterion.

    Creates training_complete_small_loss.txt marker file indicating that
    the job achieved ≥1000× loss reduction (3 orders of magnitude).

    Args:
        task_seed: Task index
        realization_seed: Realization index
        reduction_ratio: The actual loss reduction ratio achieved (optional)
        results_dir: Results directory (default: from config)
    """
    result_path = get_training_result_path(task_seed, realization_seed, results_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    completion_marker = result_path / "training_complete_small_loss.txt"

    with open(completion_marker, 'w') as f:
        f.write(f"Completed (loss reduction criterion) at {datetime.now().isoformat()}\n")
        if reduction_ratio is not None:
            f.write(f"Loss reduction ratio: {reduction_ratio:.2e} (initial/min)\n")
            f.write(f"Orders of magnitude: {np.log10(reduction_ratio):.2f}\n")


def check_loss_reduction_criterion(task_seed, realization_seed, results_dir=None):
    """
    Check if training achieved 3+ orders of magnitude loss reduction.

    If the minimum loss is at least 1000× smaller than the initial loss,
    create an alternative completion marker and return True.

    Args:
        task_seed: Task index
        realization_seed: Realization index
        results_dir: Results directory (default: from config)

    Returns:
        criterion_met: Boolean indicating if loss reduction criterion is satisfied
    """
    result_path = get_training_result_path(task_seed, realization_seed, results_dir)
    loss_file = result_path / "loss_trajectory.npy"

    # Check if loss file exists
    if not loss_file.exists():
        return False

    try:
        # Load loss trajectory
        loss = np.load(loss_file)

        # Check if we have at least 2 data points
        if len(loss) < 2:
            return False

        initial_loss = loss[0]
        min_loss = np.min(loss)

        # Avoid division by zero or negative values
        if min_loss <= 0 or initial_loss <= 0:
            return False

        # Calculate reduction ratio
        reduction_ratio = initial_loss / min_loss

        # Check if 3+ orders of magnitude (1000×)
        if reduction_ratio >= 10_000_000.0:
            # Create alternative completion marker
            mark_training_complete_small_loss(
                task_seed, realization_seed,
                reduction_ratio=reduction_ratio,
                results_dir=results_dir
            )
            return True

        return False

    except Exception as e:
        # Handle corrupted files or other errors gracefully
        print(f"Warning: Could not check loss criterion for task {task_seed}, "
              f"realization {realization_seed}: {e}")
        return False


def has_nan_in_results(task_seed, realization_seed, results_dir=None):
    """
    Check if previously saved training results contain NaN values.

    Loads ``stiffness_trajectory.npy`` and ``loss_trajectory.npy`` from the
    result directory and returns True if either file contains any NaN.
    Returns False when the files do not exist (no results saved yet).

    Args:
        task_seed: Task index
        realization_seed: Realization index
        results_dir: Results directory (default: from config)

    Returns:
        has_nan: True if NaN detected in saved stiffnesses or losses.
    """
    result_path = get_training_result_path(task_seed, realization_seed, results_dir)

    stiffness_file = result_path / "stiffness_trajectory.npy"
    loss_file = result_path / "loss_trajectory.npy"

    try:
        if stiffness_file.exists():
            stiffnesses = np.load(stiffness_file)
            if np.isnan(stiffnesses).any():
                return True
        if loss_file.exists():
            losses = np.load(loss_file)
            if np.isnan(losses).any():
                return True
    except Exception as e:
        print(f"Warning: Could not check NaN for task {task_seed}, "
              f"realization {realization_seed}: {e}")

    return False


def get_last_good_step(task_seed, realization_seed, results_dir=None):
    """
    Return the index of the last training step that contains no NaN in either
    stiffnesses or loss.

    Loads ``stiffness_trajectory.npy`` (n_steps, E) and
    ``loss_trajectory.npy`` (n_steps,) and finds the last row in which
    neither array has a NaN.

    Args:
        task_seed: Task index
        realization_seed: Realization index
        results_dir: Results directory (default: from config)

    Returns:
        last_good: Index of last clean step, or -1 if no clean step exists
                   or the trajectory files are missing.
    """
    result_path = get_training_result_path(task_seed, realization_seed, results_dir)

    stiffness_file = result_path / "stiffness_trajectory.npy"
    loss_file = result_path / "loss_trajectory.npy"

    if not stiffness_file.exists() or not loss_file.exists():
        return -1

    try:
        stiffnesses = np.load(stiffness_file)  # (n_steps, E)
        losses = np.load(loss_file)             # (n_steps,)

        n_steps = min(len(stiffnesses), len(losses))
        if n_steps == 0:
            return -1

        stiffnesses = stiffnesses[:n_steps]
        losses = losses[:n_steps]

        # Good steps: no NaN in stiffnesses row and no NaN in loss
        stiff_ok = ~np.isnan(stiffnesses).any(axis=1)  # (n_steps,)
        loss_ok = ~np.isnan(losses)                     # (n_steps,)
        good_mask = stiff_ok & loss_ok

        good_indices = np.where(good_mask)[0]
        if len(good_indices) == 0:
            return -1
        return int(good_indices[-1])

    except Exception as e:
        print(f"Warning: Could not find last good step for task {task_seed}, "
              f"realization {realization_seed}: {e}")
        return -1


def is_training_complete(task_seed, realization_seed, results_dir=None):
    """
    Check if a training job is marked as complete.

    Completion criteria (either one satisfies):
    1. Standard completion: training_complete.txt exists
    2. Loss-based completion: loss reduced by ≥1000× (3 orders of magnitude)
       and training_complete_small_loss.txt created

    Args:
        task_seed: Task index
        realization_seed: Realization index
        results_dir: Results directory (default: from config)

    Returns:
        is_complete: Boolean indicating if job is complete by either criterion
    """
    result_path = get_training_result_path(task_seed, realization_seed, results_dir)

    # Fast path 1: Check standard completion marker
    #if (result_path / "training_complete.txt").exists():
    #    return True

    # Fast path 2: Check alternative completion marker
    if (result_path / "training_complete_small_loss.txt").exists():
        return True

    # Slow path: Evaluate loss reduction criterion
    return check_loss_reduction_criterion(task_seed, realization_seed, results_dir)


def mark_training_complete(task_seed, realization_seed, results_dir=None):
    """
    Mark a training job as complete by creating a marker file.

    Args:
        task_seed: Task index
        realization_seed: Realization index
        results_dir: Results directory (default: from config)
    """
    result_path = get_training_result_path(task_seed, realization_seed, results_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    completion_marker = result_path / "training_complete.txt"
    with open(completion_marker, 'w') as f:
        f.write(f"Completed at {datetime.now().isoformat()}\n")


def save_training_results(task_seed, realization_seed, history, network, task_config, results_dir=None):
    """
    Save all training results to organized directory structure.

    Directory structure:
        results/
            task_00/
                realization_00/
                    history.pkl           # Full training trajectory
                    loss_trajectory.npy   # Loss values (n_steps,)
                    stiffness_trajectory.npy  # Stiffnesses (n_steps, n_edges)
                    final_network.pkl     # Final network state
                    task_config.json      # Task configuration
                    training_complete.txt # Standard completion marker
                    training_complete_small_loss.txt  # Alternative marker (loss-based)

    The history.pkl contains the complete training trajectory:
        - 'stiffnesses': Array of stiffness values at each step (n_steps, n_edges)
        - 'loss': Array of loss values at each step (n_steps,)
        - 'positions': List of position arrays at each step (n_steps, n_nodes, 2)
        - 'freetraj': (optional) Free trajectory data

    Separate numpy files are also saved for quick access:
        - loss_trajectory.npy: Loss values at each step
        - stiffness_trajectory.npy: Stiffness values at each step

    Completion markers (either indicates job is complete):
        - training_complete.txt: Created when full training completes normally
        - training_complete_small_loss.txt: Created when loss reduces by ≥1000×

    Args:
        task_seed: Task index
        realization_seed: Realization index
        history: Training history dictionary with full trajectory
        network: Final network object (ElasticNetwork)
        task_config: Task configuration dictionary
        results_dir: Results directory (default: from config)
    """
    result_path = get_training_result_path(task_seed, realization_seed, results_dir)
    result_path.mkdir(parents=True, exist_ok=True)

    # Convert lists to numpy arrays for more efficient storage
    stiffness_array = np.array(history.get('stiffnesses', []))  # (n_steps, n_edges)
    loss_array = np.array(history.get('loss', []))               # (n_steps,)

    # Save history (includes full training trajectory)
    history_to_save = {
        'stiffnesses': stiffness_array,
        'loss': loss_array,
        'positions': history.get('positions', []),  # List of arrays
        'freetraj': history.get('freetraj', [])     # Optional
    }

    with open(result_path / "history.pkl", "wb") as f:
        pickle.dump(history_to_save, f)

    # Save loss and stiffness trajectories as separate numpy files
    np.save(result_path / "loss_trajectory.npy", loss_array)
    np.save(result_path / "stiffness_trajectory.npy", stiffness_array)

    # Save final network state
    network_dict = {
        'positions': network.positions,
        'edges': network.edges,
        'stiffnesses': network.stiffnesses,
        'rest_lengths': network.rest_lengths
    }
    with open(result_path / "final_network.pkl", "wb") as f:
        pickle.dump(network_dict, f)

    # Save task configuration
    with open(result_path / "task_config.json", "w") as f:
        json.dump(task_config, f, indent=2)

    # Mark complete
    mark_training_complete(task_seed, realization_seed, results_dir)


def save_checkpoint(task_seed, realization_seed, history, network, task_config,
                   current_step, results_dir=None):
    """
    Save a training checkpoint (intermediate state).

    Args:
        task_seed: Task index
        realization_seed: Realization index
        history: Training history dictionary (up to current step)
        network: Current network object (ElasticNetwork)
        task_config: Task configuration dictionary
        current_step: Current training step number
        results_dir: Results directory (default: from config)
    """
    result_path = get_training_result_path(task_seed, realization_seed, results_dir)
    result_path.mkdir(parents=True, exist_ok=True)

    checkpoint_data = {
        'task_seed': task_seed,
        'realization_seed': realization_seed,
        'current_step': current_step,
        'history': history,
        'network': {
            'positions': network.positions,
            'edges': network.edges,
            'stiffnesses': network.stiffnesses,
            'rest_lengths': network.rest_lengths
        },
        'task_config': task_config
    }

    # Save checkpoint
    checkpoint_file = result_path / "checkpoint.pkl"
    with open(checkpoint_file, "wb") as f:
        pickle.dump(checkpoint_data, f)

    # Also save task config separately for easy access
    with open(result_path / "task_config.json", "w") as f:
        json.dump(task_config, f, indent=2)


def load_checkpoint(task_seed, realization_seed, results_dir=None):
    """
    Load a training checkpoint if it exists.

    Args:
        task_seed: Task index
        realization_seed: Realization index
        results_dir: Results directory (default: from config)

    Returns:
        checkpoint_data: Dictionary with checkpoint data, or None if no checkpoint exists
    """
    result_path = get_training_result_path(task_seed, realization_seed, results_dir)
    checkpoint_file = result_path / "checkpoint.pkl"
 
    print(checkpoint_file, checkpoint_file.exists())
 
    if not checkpoint_file.exists():
        return None

    try:
        with open(checkpoint_file, "rb") as f:
            checkpoint_data = pickle.load(f)
        return checkpoint_data
    except Exception as e:
        print(f"Warning: Failed to load checkpoint for task {task_seed}, realization {realization_seed}: {e}")
        return None


def has_checkpoint(task_seed, realization_seed, results_dir=None):
    """
    Check if a checkpoint exists for this training job.

    Args:
        task_seed: Task index
        realization_seed: Realization index
        results_dir: Results directory (default: from config)

    Returns:
        has_ckpt: Boolean indicating if checkpoint exists
    """
    result_path = get_training_result_path(task_seed, realization_seed, results_dir)
    checkpoint_file = result_path / "checkpoint.pkl"
    return checkpoint_file.exists()


def remove_checkpoint(task_seed, realization_seed, results_dir=None):
    """
    Remove checkpoint file after successful completion.

    Args:
        task_seed: Task index
        realization_seed: Realization index
        results_dir: Results directory (default: from config)
    """
    result_path = get_training_result_path(task_seed, realization_seed, results_dir)
    checkpoint_file = result_path / "checkpoint.pkl"

    if checkpoint_file.exists():
        checkpoint_file.unlink()


def get_incomplete_jobs(n_tasks=None, n_realizations=None, results_dir=None):
    """
    Get list of incomplete jobs (not marked as complete).

    Args:
        n_tasks: Number of tasks (default: from config)
        n_realizations: Number of realizations per task (default: from config)
        results_dir: Results directory (default: from config)

    Returns:
        incomplete_jobs: List of (task_seed, realization_seed) tuples
    """
    if n_tasks is None:
        n_tasks = N_TASKS
    if n_realizations is None:
        n_realizations = N_REALIZATIONS

    incomplete = []
    for task_seed in range(n_tasks):
        for realization_seed in range(n_realizations):
            if not is_training_complete(task_seed, realization_seed, results_dir):
                incomplete.append((task_seed, realization_seed))

    return incomplete


def get_complete_jobs(n_tasks=None, n_realizations=None, results_dir=None):
    """
    Get list of complete jobs.

    Args:
        n_tasks: Number of tasks (default: from config)
        n_realizations: Number of realizations per task (default: from config)
        results_dir: Results directory (default: from config)

    Returns:
        complete_jobs: List of (task_seed, realization_seed) tuples
    """
    if n_tasks is None:
        n_tasks = N_TASKS
    if n_realizations is None:
        n_realizations = N_REALIZATIONS

    complete = []
    for task_seed in range(n_tasks):
        for realization_seed in range(n_realizations):
            if is_training_complete(task_seed, realization_seed, results_dir):
                complete.append((task_seed, realization_seed))

    return complete


def print_progress_summary(results_dir=None):
    """
    Print a summary of ensemble training progress.

    Args:
        results_dir: Results directory (default: from config)
    """
    total_jobs = N_TASKS * N_REALIZATIONS
    complete_jobs = get_complete_jobs(results_dir=results_dir)
    incomplete_jobs = get_incomplete_jobs(results_dir=results_dir)

    print(f"\n{'='*80}")
    print(f"ENSEMBLE TRAINING PROGRESS")
    print(f"{'='*80}")
    print(f"Total jobs: {total_jobs}")
    print(f"Complete: {len(complete_jobs)} ({100*len(complete_jobs)/total_jobs:.1f}%)")
    print(f"Incomplete: {len(incomplete_jobs)} ({100*len(incomplete_jobs)/total_jobs:.1f}%)")
    print(f"{'='*80}\n")

    # Progress by task
    print("Progress by task:")
    for task_seed in range(N_TASKS):
        task_complete = sum(1 for t, r in complete_jobs if t == task_seed)
        print(f"  Task {task_seed:02d}: {task_complete:2d}/{N_REALIZATIONS} complete")


if __name__ == '__main__':
    # Test checkpoint manager
    print("Testing checkpoint manager...")

    # Create a test results directory
    test_dir = Path("/tmp/test_ensemble_results")
    test_dir.mkdir(exist_ok=True)

    # Test 1: Save and load results
    print("\n1. Testing save/load functionality:")

    # Create dummy data
    test_task = 0
    test_real = 0
    test_history = {
        'loss': [1.0, 0.5, 0.25],
        'stiffnesses': np.random.rand(3, 100)
    }
    test_network_dict = {
        'positions': np.random.rand(50, 2),
        'edges': np.array([[0, 1], [1, 2]]),
        'stiffnesses': np.random.rand(2),
        'rest_lengths': np.random.rand(2)
    }

    # Mock network object
    class MockNetwork:
        def __init__(self, data):
            self.positions = data['positions']
            self.edges = data['edges']
            self.stiffnesses = data['stiffnesses']
            self.rest_lengths = data['rest_lengths']

    test_network = MockNetwork(test_network_dict)
    test_config = {'task_seed': 0, 'packing_seed': 0}

    # Save results
    save_training_results(test_task, test_real, test_history, test_network, test_config, results_dir=test_dir)
    print(f"   Saved results to {get_training_result_path(test_task, test_real, test_dir)}")

    # Check if marked complete
    is_complete = is_training_complete(test_task, test_real, results_dir=test_dir)
    print(f"   Marked as complete: {is_complete}")

    # Test 2: Get incomplete jobs
    print("\n2. Testing incomplete jobs detection:")
    # Mark a few jobs as complete
    for i in range(3):
        mark_training_complete(0, i, results_dir=test_dir)
        mark_training_complete(1, i, results_dir=test_dir)

    incomplete = get_incomplete_jobs(n_tasks=3, n_realizations=5, results_dir=test_dir)
    complete = get_complete_jobs(n_tasks=3, n_realizations=5, results_dir=test_dir)
    print(f"   Total jobs: 15 (3 tasks × 5 realizations)")
    print(f"   Complete: {len(complete)}")
    print(f"   Incomplete: {len(incomplete)}")

    # Test 3: Progress summary
    print("\n3. Testing progress summary:")
    # Clean up first
    import shutil
    shutil.rmtree(test_dir)

    print("\nCheckpoint manager tests complete!")
