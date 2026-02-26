#!/usr/bin/env python
"""
Moduli ensemble training runner for elastic networks.

Runs 18 training tasks across 6 categories of elastic moduli objectives,
each with 3 realizations (54 total jobs). All tasks share the same network
topology (PACKING_SEED=42). Targets are multipliers of the natural moduli
from the uniform-stiffness reference network.

Usage:
    # Run all tasks sequentially
    python moduli_ensemble_runner.py --mode sequential

    # Run specific task (for debugging)
    python moduli_ensemble_runner.py --mode single --task 0 --verbose

    # Resume incomplete tasks
    python moduli_ensemble_runner.py --mode sequential --resume

    # Check progress
    python moduli_ensemble_runner.py --mode status
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path

# Add necessary paths
sys.path.append(str(Path(__file__).parent.parent / 'instruments'))
sys.path.append(str(Path(__file__).parent.parent / 'production'))
sys.path.append(str(Path(__file__).parent.parent.parent / 'cl_mech_repo' / 'physical_learning'))

# Import shared config
from config import (
    N_NODES, FORCE_TYPE, BOUNDARY_MARGIN,
    LEARNING_RATE, FORCE_TOL, VMIN, VMAX,
)

# Import moduli task definitions
from moduli_task_generator import (
    get_moduli_task_config,
    get_all_moduli_task_configs,
    print_moduli_tasks_summary,
    compute_reference_moduli,
    resolve_training_goals,
    N_TASKS, N_REALIZATIONS, N_STEPS, N_STRAIN_STEPS,
    PACKING_SEED, MODULI_RESULTS_DIR, ALL_COMPRESSION_STRAINS,
)

# Import shared utilities
from task_generator import generate_realization_stiffnesses
from network_utils import create_auxetic_network
from checkpoint_manager import (
    is_training_complete,
    save_training_results,
    get_incomplete_jobs,
    get_complete_jobs,
    load_checkpoint,
    remove_checkpoint,
    has_nan_in_results,
    get_last_good_step,
    get_training_result_path,
)
import pickle

# Learning-rate reduction factor applied when NaN is detected in saved results
LR_NAN_REDUCTION = 0.1

# Import moduli training
try:
    from moduli_training import run_moduli_training
    TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import moduli training functions: {e}")
    TRAINING_AVAILABLE = False


# ============================================================================
# Reference moduli (computed once, cached globally)
# ============================================================================

_reference_moduli = None


def get_reference_moduli(verbose=False):
    """
    Compute or return cached reference moduli for the uniform network.

    Creates the reference network (PACKING_SEED=42, uniform k=1),
    compresses to each required strain, and evaluates the elasticity tensor.
    """
    global _reference_moduli
    if _reference_moduli is not None:
        return _reference_moduli

    if verbose:
        print("Computing reference moduli for uniform network...")

    # Create reference network with uniform stiffnesses
    network, boundary_dict = create_auxetic_network(
        n_nodes=N_NODES,
        packing_seed=PACKING_SEED,
        force_type=FORCE_TYPE,
        boundary_margin=BOUNDARY_MARGIN,
    )
    # Ensure uniform stiffnesses
    network.stiffnesses = np.ones(len(network.edges))

    _reference_moduli = compute_reference_moduli(
        network, boundary_dict,
        compression_strains=ALL_COMPRESSION_STRAINS,
        n_strain_steps=N_STRAIN_STEPS,
        force_type=FORCE_TYPE,
    )

    if verbose:
        print("Reference moduli computed:")
        for strain, moduli in _reference_moduli.items():
            print(f"  strain={strain}: B={moduli['B']:.4f}, "
                  f"G={moduli['G']:.4f}, nu={moduli['nu']:.4f}")
        print()

    return _reference_moduli


# ============================================================================
# Single Training Job
# ============================================================================


def run_single_training(task_id, realization_seed=0, verbose=False,
                        use_checkpoint=True):
    """
    Run a single moduli training job.

    Args:
        task_id: Task index (0 to N_TASKS-1)
        realization_seed: Realization index (default: 0)
        verbose: Print detailed progress
        use_checkpoint: Whether to use checkpointing

    Returns:
        success: Boolean indicating success
    """
    print(f"\n{'='*80}")
    print(f"Starting Moduli Task {task_id}, Realization {realization_seed}")
    print(f"{'='*80}")

    # Check for NaN in previously saved results (takes priority over completion check)
    nan_detected = has_nan_in_results(task_id, realization_seed,
                                      results_dir=MODULI_RESULTS_DIR)
    recovery_mode = None   # None | 'from_last_good' | 'from_scratch'
    recovery_lr_scale = 1.0

    if nan_detected:
        last_good = get_last_good_step(task_id, realization_seed,
                                       results_dir=MODULI_RESULTS_DIR)
        if last_good >= 0:
            recovery_mode = 'from_last_good'
        else:
            recovery_mode = 'from_scratch'
        recovery_lr_scale = LR_NAN_REDUCTION
        print(f"NaN detected in saved results — recovery_mode={recovery_mode}, "
              f"LR scale={recovery_lr_scale}")
    elif is_training_complete(task_id, realization_seed, results_dir=MODULI_RESULTS_DIR):
        print(f"Job already completed! Skipping...")
        print(f"{'='*80}\n")
        return True

    start_time = time.time()

    try:
        if not TRAINING_AVAILABLE:
            raise ImportError("Moduli training functions not available. Check imports.")

        # 1. Get task configuration
        if verbose:
            print("Step 1: Loading moduli task configuration...")
        task_config = get_moduli_task_config(task_id)
        print(f"  Category: {task_config['category']} ({task_config['category_name']})")

        # 2. Compute reference moduli and resolve targets
        if verbose:
            print("Step 2: Computing reference moduli...")
        ref_moduli = get_reference_moduli(verbose=verbose)
        training_goals = resolve_training_goals(task_config, ref_moduli)

        print(f"  Resolved training goals:")
        for strain, targets in training_goals.items():
            target_strs = [f"{k}={v:.4f}" for k, v in targets.items()]
            print(f"    strain={strain}: {', '.join(target_strs)}")

        # 3. Create base network structure
        if verbose:
            print("Step 3: Creating network from packing...")
        network, boundary_dict = create_auxetic_network(
            n_nodes=N_NODES,
            packing_seed=task_config['packing_seed'],
            force_type=FORCE_TYPE,
            boundary_margin=BOUNDARY_MARGIN,
        )
        print(f"  Network created: {len(network.positions)} nodes, "
              f"{len(network.edges)} edges")

        # 4. Determine initial state based on recovery mode or checkpoint
        history = {}
        start_step = 0

        if recovery_mode == 'from_last_good':
            last_good = get_last_good_step(task_id, realization_seed,
                                           results_dir=MODULI_RESULTS_DIR)
            result_path = get_training_result_path(task_id, realization_seed,
                                                   results_dir=MODULI_RESULTS_DIR)
            with open(result_path / "history.pkl", 'rb') as f:
                saved_history = pickle.load(f)
            n_trim = last_good + 1
            history = {
                'stiffnesses': list(np.array(saved_history['stiffnesses'])[:n_trim]),
                'loss':        list(np.array(saved_history['loss'])[:n_trim]),
                'positions':   list(saved_history['positions'])[:n_trim],
            }
            network.stiffnesses = np.array(saved_history['stiffnesses'])[last_good]
            network.positions = np.array(saved_history['positions'][last_good])
            start_step = n_trim
            print(f"  Restored from last good step {last_good} — continuing from step {start_step}")

        elif recovery_mode == 'from_scratch':
            n_edges = len(network.edges)
            initial_stiffnesses = generate_realization_stiffnesses(realization_seed, n_edges)
            network.stiffnesses = initial_stiffnesses
            network.save_original_parameters()
            print(f"  Restarting from scratch with reduced LR "
                  f"[{initial_stiffnesses.min():.2e}, {initial_stiffnesses.max():.2e}]")

        else:
            # Normal path: try checkpoint first, else fresh start
            checkpoint = None
            if use_checkpoint:
                checkpoint = load_checkpoint(task_id, realization_seed,
                                             results_dir=MODULI_RESULTS_DIR)
                if checkpoint is not None:
                    print(f"Found checkpoint at step {checkpoint['current_step']}")
                    network.positions = checkpoint['network']['positions']
                    network.stiffnesses = checkpoint['network']['stiffnesses']
                    network.rest_lengths = checkpoint['network']['rest_lengths']
                    network.edges = checkpoint['network']['edges']
                    history = checkpoint['history']
                    start_step = checkpoint['current_step']
                    print(f"  Resuming from step {start_step}/{N_STEPS}")

            if checkpoint is None:
                if verbose:
                    print("Step 4: Initializing random stiffnesses...")
                n_edges = len(network.edges)
                initial_stiffnesses = generate_realization_stiffnesses(realization_seed, n_edges)
                network.stiffnesses = initial_stiffnesses
                network.save_original_parameters()
                print(f"  Stiffnesses initialized: range "
                      f"[{initial_stiffnesses.min():.2e}, {initial_stiffnesses.max():.2e}]")

        effective_lr = LEARNING_RATE * recovery_lr_scale
        remaining_steps = N_STEPS - start_step

        print(f"  Training parameters:")
        print(f"    Learning rate: {effective_lr} (scale={recovery_lr_scale})")
        print(f"    Steps remaining: {remaining_steps:,} / {N_STEPS:,}")
        print(f"    Strain steps: {N_STRAIN_STEPS}")
        print(f"    Force tolerance: {FORCE_TOL}")

        # 5. Run training
        if verbose:
            print("Step 5: Running moduli training...")

        def _run_train(net, hist, lr, n_steps):
            return run_moduli_training(
                network=net,
                training_goals=training_goals,
                learning_rate=lr,
                n_steps=n_steps,
                top_nodes=boundary_dict['top'],
                bottom_nodes=boundary_dict['bottom'],
                left_nodes=boundary_dict['left'],
                right_nodes=boundary_dict['right'],
                force_type=FORCE_TYPE,
                n_strain_steps=N_STRAIN_STEPS,
                history=hist,
                force_tol=FORCE_TOL,
                vmin=VMIN,
                vmax=VMAX,
                task_seed=task_id,
                realization_seed=realization_seed,
                save_interval=5,
                task_config=task_config,
                TARGETED_RESULTS_DIR=MODULI_RESULTS_DIR,
            )

        if remaining_steps > 0:
            history, trained_network = _run_train(network, history, effective_lr, remaining_steps)
        else:
            trained_network = network
            print("  Training already complete!")

        # 5b. Second NaN recovery
        if recovery_mode is not None and has_nan_in_results(
                task_id, realization_seed, results_dir=MODULI_RESULTS_DIR):
            print(f"\n{'!'*60}")
            print(f"NaN persists after {recovery_mode} recovery.")
            print(f"Restarting from scratch with LR scale={LR_NAN_REDUCTION}.")
            print(f"{'!'*60}\n")
            n_edges = len(network.edges)
            initial_stiffnesses = generate_realization_stiffnesses(realization_seed, n_edges)
            trained_network.stiffnesses = initial_stiffnesses
            trained_network.save_original_parameters()
            history = {}
            history, trained_network = _run_train(
                trained_network, history, LEARNING_RATE * LR_NAN_REDUCTION, N_STEPS)

        # 6. Save final results
        if verbose:
            print("Step 6: Saving results...")
        save_training_results(
            task_seed=task_id,
            realization_seed=realization_seed,
            history=history,
            network=trained_network,
            task_config=task_config,
            results_dir=MODULI_RESULTS_DIR,
        )

        # Remove checkpoint after success
        if use_checkpoint:
            remove_checkpoint(task_id, realization_seed, results_dir=MODULI_RESULTS_DIR)

        elapsed = time.time() - start_time
        loss_list = history.get('loss', [])
        final_loss = loss_list[-1] if loss_list else float('nan')

        print(f"\n{'='*80}")
        print(f"SUCCESS: Moduli Task {task_id}, Realization {realization_seed}")
        print(f"Time elapsed: {elapsed/60:.2f} minutes")
        print(f"Final loss: {final_loss:.4e}")
        print(f"Training steps completed: {len(loss_list)}")
        print(f"{'='*80}\n")

        return True

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"ERROR: Moduli Task {task_id}, Realization {realization_seed}")
        print(f"Time elapsed: {elapsed/60:.2f} minutes")
        print(f"Exception: {e}")
        print(f"{'='*80}\n")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Sequential Runner
# ============================================================================


def run_all_moduli(resume=True, verbose=False):
    """
    Run all moduli training jobs sequentially.

    Args:
        resume: Skip already completed jobs
        verbose: Print detailed progress
    """
    print(f"\n{'#'*80}")
    print(f"# MODULI ENSEMBLE TRAINING: SEQUENTIAL MODE")
    print(f"# Total jobs: {N_TASKS} tasks x {N_REALIZATIONS} realizations = "
          f"{N_TASKS * N_REALIZATIONS}")
    print(f"# Resume mode: {resume}")
    print(f"{'#'*80}\n")

    # Compute reference moduli once
    ref_moduli = get_reference_moduli(verbose=True)
    print_moduli_tasks_summary(reference_moduli=ref_moduli)

    if resume:
        jobs = get_incomplete_jobs(
            n_tasks=N_TASKS,
            n_realizations=N_REALIZATIONS,
            results_dir=MODULI_RESULTS_DIR,
        )
        print(f"Found {len(jobs)} incomplete jobs "
              f"(out of {N_TASKS * N_REALIZATIONS} total)")
    else:
        jobs = [
            (task, real)
            for task in range(N_TASKS)
            for real in range(N_REALIZATIONS)
        ]
        print(f"Running all {len(jobs)} jobs from scratch")

    if len(jobs) == 0:
        print("No jobs to run! All training complete.")
        return

    success_count = 0
    failure_count = 0
    start_time_overall = time.time()

    for idx, (task_id, realization_seed) in enumerate(jobs):
        print(f"\n[Job {idx+1}/{len(jobs)}]")
        success = run_single_training(task_id, realization_seed, verbose=verbose)

        if success:
            success_count += 1
        else:
            failure_count += 1

        elapsed = time.time() - start_time_overall
        avg_time = elapsed / (idx + 1)
        remaining = avg_time * (len(jobs) - (idx + 1))

        print(f"\n{'~'*80}")
        print(f"Progress: {idx+1}/{len(jobs)} jobs completed")
        print(f"Success: {success_count}, Failed: {failure_count}")
        print(f"Average time per job: {avg_time/60:.2f} minutes")
        print(f"Estimated time remaining: {remaining/60:.2f} minutes")
        print(f"{'~'*80}\n")

    total_elapsed = time.time() - start_time_overall

    print(f"\n{'#'*80}")
    print(f"# MODULI TRAINING COMPLETE")
    print(f"# Successful: {success_count}/{len(jobs)}")
    print(f"# Failed: {failure_count}/{len(jobs)}")
    print(f"# Total time: {total_elapsed/60:.2f} minutes")
    print(f"{'#'*80}\n")

    print_moduli_progress()


# ============================================================================
# Status
# ============================================================================


def print_moduli_progress():
    """Print progress summary for moduli tasks."""
    complete = get_complete_jobs(
        n_tasks=N_TASKS,
        n_realizations=N_REALIZATIONS,
        results_dir=MODULI_RESULTS_DIR,
    )
    incomplete = get_incomplete_jobs(
        n_tasks=N_TASKS,
        n_realizations=N_REALIZATIONS,
        results_dir=MODULI_RESULTS_DIR,
    )
    total = N_TASKS * N_REALIZATIONS

    print(f"\n{'='*80}")
    print(f"MODULI TRAINING PROGRESS")
    print(f"{'='*80}")
    print(f"Total jobs: {total}")
    print(f"Complete: {len(complete)} ({100*len(complete)/total:.0f}%)")
    print(f"Incomplete: {len(incomplete)} ({100*len(incomplete)/total:.0f}%)")
    print()

    all_configs = get_all_moduli_task_configs()
    current_cat = None
    for task_id in range(N_TASKS):
        config = all_configs[task_id]
        cat = config['category']
        cat_name = config['category_name']

        if cat != current_cat:
            current_cat = cat
            print(f"  --- Category {cat}: {cat_name} ---")

        n_complete = sum(1 for t, r in complete if t == task_id)
        status_str = f"{n_complete}/{N_REALIZATIONS}"
        if n_complete == N_REALIZATIONS:
            status_str += " DONE"

        goals = config['training_goals_multipliers']
        parts = []
        for strain, targets in goals.items():
            target_strs = [f"{k}x{v}" for k, v in targets.items()]
            parts.append(f"strain={strain}: {', '.join(target_strs)}")

        print(f"    Task {task_id:2d}: [{status_str:>10s}]  {' | '.join(parts)}")

    print(f"{'='*80}\n")


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description='Moduli ensemble training for elastic networks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single task (debugging)
  python moduli_ensemble_runner.py --mode single --task 0 --verbose

  # Run all 18 tasks sequentially
  python moduli_ensemble_runner.py --mode sequential

  # Resume incomplete tasks
  python moduli_ensemble_runner.py --mode sequential --resume

  # Check progress
  python moduli_ensemble_runner.py --mode status
        """
    )

    parser.add_argument(
        '--mode',
        choices=['sequential', 'single', 'status'],
        default='single',
        help='Execution mode',
    )
    parser.add_argument(
        '--task',
        type=int,
        help=f'Task ID for single mode (0-{N_TASKS-1})',
    )
    parser.add_argument(
        '--realization',
        type=int,
        default=0,
        help='Realization seed for single mode (default: 0)',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Resume from incomplete jobs (sequential mode)',
    )
    parser.add_argument(
        '--no-resume',
        action='store_false',
        dest='resume',
        help='Start from scratch (sequential mode)',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output',
    )

    args = parser.parse_args()

    if args.mode == 'single':
        if args.task is None:
            parser.error("--task required for single mode")
        if args.task < 0 or args.task >= N_TASKS:
            parser.error(f"--task must be between 0 and {N_TASKS-1}")

        # Compute reference moduli once
        get_reference_moduli(verbose=args.verbose)
        print_moduli_tasks_summary()
        success = run_single_training(args.task, args.realization,
                                      verbose=args.verbose)
        sys.exit(0 if success else 1)

    elif args.mode == 'sequential':
        run_all_moduli(resume=args.resume, verbose=args.verbose)

    elif args.mode == 'status':
        print_moduli_tasks_summary()
        print_moduli_progress()


if __name__ == '__main__':
    main()
