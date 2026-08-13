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
from base.config import RESULTS_DIR, N_TASKS, N_REALIZATIONS, NETWORK_TYPE
from training.src.run_provenance import (
    save_run_provenance,
    save_training_meta,
    save_code_snapshot,
    has_critical_mismatch,
)


def get_training_result_path(task_seed, realization_seed, results_dir=None):
    """
    Get the directory path for a specific training result.

    Directory depth/layout is the same regardless of network_type — see
    `_nt_filename` for how network_type is instead encoded into individual
    filenames within this directory.

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


def results_dir_for_gradient_method(base_dir, gradient_method):
    """
    Append a `gradient_method` subdirectory to `base_dir`, idempotently.

    Results are partitioned by gradient_method (newton/newton_fd/fire/parallel/jax
    write to separate subdirectories so they don't clobber each other's checkpoints).
    This is idempotent — if `base_dir`'s last path component already equals
    `gradient_method`, it is returned unchanged instead of appending a second
    time — so callers can safely apply it to a `results_dir` that may already
    be gradient_method-specific (e.g. a user-supplied --results-dir override)
    without ever producing a doubled-up path like `.../newton/newton/...`.
    """
    base_dir = Path(base_dir)
    if base_dir.name == gradient_method:
        return base_dir
    return base_dir / gradient_method


def _nt_filename(filename, network_type):
    """
    Insert `network_type` into a filename unless it is 'jammed' (the
    original/default network type), so that pre-existing 'jammed' files keep
    their original names (backward compatible) while other network types
    (e.g. 'lattice') get distinct filenames within the same directory and
    never overwrite/collide with a 'jammed' run of the same task/realization.

    Example: _nt_filename('history.pkl', 'lattice') -> 'history_lattice.pkl'
    """
    stem, dot, ext = filename.partition('.')
    return f"{stem}_{network_type}{dot}{ext}"


def mark_training_complete_small_loss(task_seed, realization_seed, reduction_ratio=None, results_dir=None,
                                       network_type=NETWORK_TYPE):
    """
    Mark a training job as complete based on loss reduction criterion.

    Creates training_complete_small_loss.txt marker file indicating that
    the job achieved ≥1000× loss reduction (3 orders of magnitude).

    Args:
        task_seed: Task index
        realization_seed: Realization index
        reduction_ratio: The actual loss reduction ratio achieved (optional)
        results_dir: Results directory (default: from config)
        network_type: 'jammed' or 'lattice' (see create_auxetic_network)
    """
    result_path = get_training_result_path(task_seed, realization_seed, results_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    completion_marker = result_path / _nt_filename("training_complete_small_loss.txt", network_type)

    with open(completion_marker, 'w') as f:
        f.write(f"Completed (loss reduction criterion) at {datetime.now().isoformat()}\n")
        if reduction_ratio is not None:
            f.write(f"Loss reduction ratio: {reduction_ratio:.2e} (initial/min)\n")
            f.write(f"Orders of magnitude: {np.log10(reduction_ratio):.2f}\n")


def check_loss_reduction_criterion(task_seed, realization_seed, results_dir=None, network_type=NETWORK_TYPE):
    """
    Check if training achieved 3+ orders of magnitude loss reduction.

    If the minimum loss is at least 1000× smaller than the initial loss,
    create an alternative completion marker and return True.

    Args:
        task_seed: Task index
        realization_seed: Realization index
        results_dir: Results directory (default: from config)
        network_type: 'jammed' or 'lattice' (see create_auxetic_network)

    Returns:
        criterion_met: Boolean indicating if loss reduction criterion is satisfied
    """
    result_path = get_training_result_path(task_seed, realization_seed, results_dir)
    loss_file = result_path / _nt_filename("loss_trajectory.npy", network_type)

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
                results_dir=results_dir,
                network_type=network_type,
            )
            return True

        return False

    except Exception as e:
        # Handle corrupted files or other errors gracefully
        print(f"Warning: Could not check loss criterion for task {task_seed}, "
              f"realization {realization_seed}: {e}")
        return False




def is_training_complete(task_seed, realization_seed, results_dir=None, network_type=NETWORK_TYPE):
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
        network_type: 'jammed' or 'lattice' (see create_auxetic_network)

    Returns:
        is_complete: Boolean indicating if job is complete by either criterion
    """
    result_path = get_training_result_path(task_seed, realization_seed, results_dir)

    # Fast path 1: Check standard completion marker
    #if (result_path / "training_complete.txt").exists():
    #    return True

    # Fast path 2: Check alternative completion marker
    if (result_path / _nt_filename("training_complete_small_loss.txt", network_type)).exists():
        return True

    # Slow path: Evaluate loss reduction criterion
    return check_loss_reduction_criterion(task_seed, realization_seed, results_dir, network_type)


def mark_training_complete(task_seed, realization_seed, results_dir=None, network_type=NETWORK_TYPE):
    """
    Mark a training job as complete by creating a marker file.

    Args:
        task_seed: Task index
        realization_seed: Realization index
        results_dir: Results directory (default: from config)
        network_type: 'jammed' or 'lattice' (see create_auxetic_network)
    """
    result_path = get_training_result_path(task_seed, realization_seed, results_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    completion_marker = result_path / _nt_filename("training_complete.txt", network_type)
    with open(completion_marker, 'w') as f:
        f.write(f"Completed at {datetime.now().isoformat()}\n")


def save_training_results(task_seed, realization_seed, history, network, task_config,
                           boundary_dict=None, results_dir=None, network_type=NETWORK_TYPE):
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
                    boundary_nodes.json   # Boundary node indices used in training
                    training_complete.txt # Standard completion marker
                    training_complete_small_loss.txt  # Alternative marker (loss-based)

    The history.pkl contains the complete training trajectory:
        - 'stiffnesses': Array of stiffness values at each step (n_steps, n_edges)
        - 'loss': Array of loss values at each step (n_steps,)
        - 'positions': List of position arrays at each step (n_steps, n_nodes, 2)
        - 'freetraj': (optional) Free trajectory data
        - 'boundary': (optional) dict with 'top'/'bottom'/'left'/'right' node index lists

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
        boundary_dict: Dict of boundary node indices (keys 'top', 'bottom', 'left',
            'right') used to set up the training task. Saved to history.pkl and to
            a separate boundary_nodes.json file. Optional for backward compatibility.
        results_dir: Results directory (default: from config)
        network_type: 'jammed' or 'lattice' (see create_auxetic_network)
    """
    result_path = get_training_result_path(task_seed, realization_seed, results_dir)
    result_path.mkdir(parents=True, exist_ok=True)

    # Convert lists to numpy arrays for more efficient storage
    stiffness_array = np.array(history.get('stiffnesses', []))  # (n_steps, n_edges)
    loss_array = np.array(history.get('loss', []))               # (n_steps,)

    boundary_to_save = None
    if boundary_dict is not None:
        boundary_to_save = {k: np.asarray(v).tolist() for k, v in boundary_dict.items()}

    # Save history (includes full training trajectory)
    history_to_save = {
        'stiffnesses': stiffness_array,
        'loss': loss_array,
        'positions': history.get('positions', []),  # List of arrays
        'freetraj': history.get('freetraj', []),    # Optional
        'boundary': boundary_to_save,
    }

    with open(result_path / _nt_filename("history.pkl", network_type), "wb") as f:
        pickle.dump(history_to_save, f)

    # Save boundary nodes separately for quick access
    if boundary_to_save is not None:
        with open(result_path / _nt_filename("boundary_nodes.json", network_type), "w") as f:
            json.dump(boundary_to_save, f, indent=2)

    # Save loss and stiffness trajectories as separate numpy files
    np.save(result_path / _nt_filename("loss_trajectory.npy", network_type), loss_array)
    np.save(result_path / _nt_filename("stiffness_trajectory.npy", network_type), stiffness_array)
    np.save(result_path / _nt_filename("edges.npy", network_type), np.array(network.edges))

    # Save final network state
    network_dict = {
        'positions': network.positions,
        'edges': network.edges,
        'stiffnesses': network.stiffnesses,
        'rest_lengths': network.rest_lengths
    }
    with open(result_path / _nt_filename("final_network.pkl", network_type), "wb") as f:
        pickle.dump(network_dict, f)

    # Save task configuration
    with open(result_path / _nt_filename("task_config.json", network_type), "w") as f:
        json.dump(task_config, f, indent=2)

    # Mark complete
    mark_training_complete(task_seed, realization_seed, results_dir, network_type)


def save_checkpoint(task_seed, realization_seed, history, network, task_config,
                   current_step, results_dir=None, network_type=NETWORK_TYPE):
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
        network_type: 'jammed' or 'lattice' (see create_auxetic_network)
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
    checkpoint_file = result_path / _nt_filename("checkpoint.pkl", network_type)
    with open(checkpoint_file, "wb") as f:
        pickle.dump(checkpoint_data, f)

    # Also save task config separately for easy access
    with open(result_path / _nt_filename("task_config.json", network_type), "w") as f:
        json.dump(task_config, f, indent=2)


def load_checkpoint(task_seed, realization_seed, results_dir=None, network_type=NETWORK_TYPE):
    """
    Load a training checkpoint if it exists.

    Args:
        task_seed: Task index
        realization_seed: Realization index
        results_dir: Results directory (default: from config)
        network_type: 'jammed' or 'lattice' (see create_auxetic_network)

    Returns:
        checkpoint_data: Dictionary with checkpoint data, or None if no checkpoint exists
    """
    result_path = get_training_result_path(task_seed, realization_seed, results_dir)
    checkpoint_file = result_path / _nt_filename("checkpoint.pkl", network_type)

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


def has_checkpoint(task_seed, realization_seed, results_dir=None, network_type=NETWORK_TYPE):
    """
    Check if a checkpoint exists for this training job.

    Args:
        task_seed: Task index
        realization_seed: Realization index
        results_dir: Results directory (default: from config)
        network_type: 'jammed' or 'lattice' (see create_auxetic_network)

    Returns:
        has_ckpt: Boolean indicating if checkpoint exists
    """
    result_path = get_training_result_path(task_seed, realization_seed, results_dir)
    checkpoint_file = result_path / _nt_filename("checkpoint.pkl", network_type)
    return checkpoint_file.exists()


def remove_checkpoint(task_seed, realization_seed, results_dir=None, network_type=NETWORK_TYPE):
    """
    Remove checkpoint file after successful completion.

    Args:
        task_seed: Task index
        realization_seed: Realization index
        results_dir: Results directory (default: from config)
        network_type: 'jammed' or 'lattice' (see create_auxetic_network)
    """
    result_path = get_training_result_path(task_seed, realization_seed, results_dir)
    checkpoint_file = result_path / _nt_filename("checkpoint.pkl", network_type)

    if checkpoint_file.exists():
        checkpoint_file.unlink()


def job_has_critical_mismatch(task_seed, realization_seed, hyperparams, results_dir=None,
                               network_type=NETWORK_TYPE):
    """
    Read-only check: do this job's critical hyperparameters (solver/
    gradient_method/learning_rate/k_min/k_max/force_tol/optimizer/
    opt_fire_finc by default) conflict
    with what's already recorded in its training_meta.json?

    Call this BEFORE loading any checkpoint or doing NaN/completion checks —
    if it returns True and the caller wants to proceed anyway (overwrite),
    the job must be wiped with reset_job() first rather than resumed, or the
    saved metadata will describe a trajectory that never actually happened
    under those settings. See training.src.run_provenance.has_critical_mismatch.

    Args:
        task_seed: Task index
        realization_seed: Realization index
        hyperparams: Dict of hyperparameters this run would use.
        results_dir: Results directory (default: from config)
        network_type: 'jammed' or 'lattice' (see create_auxetic_network)
    """
    result_path = get_training_result_path(task_seed, realization_seed, results_dir)
    return has_critical_mismatch(result_path, hyperparams,
                                  filename=_nt_filename("training_meta.json", network_type))


def reset_job(task_seed, realization_seed, results_dir=None, network_type=NETWORK_TYPE):
    """
    Delete this job's saved state for `network_type` (checkpoint, results,
    training_meta, run_provenance — everything whose filename carries the
    `_<network_type>` suffix from _nt_filename) so it restarts completely
    from scratch.

    Use this when overwrite=True and job_has_critical_mismatch() is True:
    resuming a checkpoint trained under the OLD hyperparameters and merely
    relabeling it with the NEW ones in training_meta.json would silently mix
    two different settings into one saved trajectory while the metadata
    claims otherwise — wiping and restarting from scratch is the only way
    overwrite=True stays honest.

    Only deletes `network_type`-suffixed files, not the whole (task,
    realization) directory: 'jammed' and 'lattice' runs for the same task/
    realization coexist side-by-side in one directory (see _nt_filename), so
    a jammed-only mismatch must not destroy an unrelated lattice run living
    in the same folder. Code snapshot .gz files (not network_type-suffixed,
    shared across network_types) are left alone; run_provenance's git-commit
    trail is the authoritative record regardless.

    Args:
        task_seed: Task index
        realization_seed: Realization index
        results_dir: Results directory (default: from config)
        network_type: 'jammed' or 'lattice' (see create_auxetic_network)
    """
    result_path = get_training_result_path(task_seed, realization_seed, results_dir)
    if not result_path.exists():
        return
    suffix = f"_{network_type}"
    for f in result_path.iterdir():
        if f.is_file() and suffix in f.stem:
            f.unlink()


def save_run_metadata(task_seed, realization_seed, hyperparams, results_dir=None, network_type=NETWORK_TYPE,
                       overwrite=False):
    """
    Record run provenance and a quick-check hyperparameter snapshot for a
    training job, alongside its checkpoint/results.

    Writes two files into the job's result directory (see
    training.src.run_provenance for details of each):
      - training_meta[_<network_type>].json: `hyperparams`, written once.
        Critical keys (solver/gradient_method/learning_rate/k_min/k_max/
        force_tol/optimizer/opt_fire_finc) are enforced to match on every subsequent call — a
        mismatch raises HyperparameterMismatchError unless overwrite=True,
        so a job can't silently resume under different physics/optimizer
        settings. n_steps always tracks the current value (training longer
        is fine).
      - run_provenance[_<network_type>].json: one entry appended per call,
        so a job resumed under a different git commit keeps its full
        code-version history.

    Args:
        task_seed: Task index
        realization_seed: Realization index
        hyperparams: Dict of hyperparameters actually used for this run
            (e.g. learning_rate, k_min, k_max, n_steps, force_tol,
            gradient_method)
        results_dir: Results directory (default: from config)
        network_type: 'jammed' or 'lattice' (see create_auxetic_network)
        overwrite: If True, replace conflicting critical hyperparameters
            instead of raising. See training.src.run_provenance.save_training_meta.

    Raises:
        HyperparameterMismatchError: A critical hyperparameter conflicts
            with what's already recorded and overwrite=False.
    """
    result_path = get_training_result_path(task_seed, realization_seed, results_dir)
    save_training_meta(result_path, hyperparams, filename=_nt_filename("training_meta.json", network_type),
                        overwrite=overwrite)
    save_run_provenance(result_path, extra=hyperparams, filename=_nt_filename("run_provenance.json", network_type))


def save_run_code_snapshot(task_seed, realization_seed, files, results_dir=None, network_type=NETWORK_TYPE):
    """
    Gzip-archive `files` (the training script + relevant src/base modules)
    into the job's result directory. See training.src.run_provenance.save_code_snapshot.

    Args:
        task_seed: Task index
        realization_seed: Realization index
        files: Iterable of file paths to archive.
        results_dir: Results directory (default: from config)
        network_type: 'jammed' or 'lattice' (see create_auxetic_network)
    """
    result_path = get_training_result_path(task_seed, realization_seed, results_dir)
    save_code_snapshot(result_path, files)


def get_incomplete_jobs(n_tasks=None, n_realizations=None, results_dir=None, network_type=NETWORK_TYPE):
    """
    Get list of incomplete jobs (not marked as complete).

    Args:
        n_tasks: Number of tasks (default: from config)
        n_realizations: Number of realizations per task (default: from config)
        results_dir: Results directory (default: from config)
        network_type: 'jammed' or 'lattice' (see create_auxetic_network)

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
            if not is_training_complete(task_seed, realization_seed, results_dir, network_type):
                incomplete.append((task_seed, realization_seed))

    return incomplete


def get_complete_jobs(n_tasks=None, n_realizations=None, results_dir=None, network_type=NETWORK_TYPE):
    """
    Get list of complete jobs.

    Args:
        n_tasks: Number of tasks (default: from config)
        n_realizations: Number of realizations per task (default: from config)
        results_dir: Results directory (default: from config)
        network_type: 'jammed' or 'lattice' (see create_auxetic_network)

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
            if is_training_complete(task_seed, realization_seed, results_dir, network_type):
                complete.append((task_seed, realization_seed))

    return complete


def print_progress_summary(results_dir=None, network_type=NETWORK_TYPE):
    """
    Print a summary of ensemble training progress.

    Args:
        results_dir: Results directory (default: from config)
        network_type: 'jammed' or 'lattice' (see create_auxetic_network)
    """
    total_jobs = N_TASKS * N_REALIZATIONS
    complete_jobs = get_complete_jobs(results_dir=results_dir, network_type=network_type)
    incomplete_jobs = get_incomplete_jobs(results_dir=results_dir, network_type=network_type)

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
