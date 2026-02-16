#!/usr/bin/env python
"""
Targeted ensemble training runner for auxetic networks.

Runs 5 specific training tasks with large compression strains (-0.4, -0.2)
and specific Poisson ratio targets. All tasks share the same network topology.

Usage:
    # Run all 5 tasks sequentially
    python targeted_ensemble_runner.py --mode sequential

    # Run specific task (for debugging)
    python targeted_ensemble_runner.py --mode single --task 0 --verbose

    # Resume incomplete tasks
    python targeted_ensemble_runner.py --mode sequential --resume

    # Check progress
    python targeted_ensemble_runner.py --mode status
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

# Import shared config (not validate_config)
from config import (
    N_NODES, FORCE_TYPE, BOUNDARY_MARGIN,
    LEARNING_RATE, FORCE_TOL, VMIN, VMAX,
)

# Import targeted config and task definitions
from targeted_task_generator import (
    get_targeted_task_config,
    get_all_targeted_task_configs,
    print_targeted_tasks_summary,
    N_TASKS, N_REALIZATIONS, N_STEPS, N_STRAIN_STEPS,
    TARGETED_RESULTS_DIR,
)


# Import shared utilities
from task_generator import generate_realization_stiffnesses, compute_target_extensions
from network_utils import create_auxetic_network
from checkpoint_manager import (
    is_training_complete,
    save_training_results,
    get_incomplete_jobs,
    get_complete_jobs,
    save_checkpoint,
    load_checkpoint,
    has_checkpoint,
    remove_checkpoint,
)

# Import training functions
try:
    from training_functions_with_toggle import (
        finish_training_GD_auxetic_batch,
        finish_training_GD_auxetic_batch_jax,
    )
    TRAINING_FUNCTIONS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import training functions: {e}")
    TRAINING_FUNCTIONS_AVAILABLE = False


def run_single_training(task_id, realization_seed=0, verbose=False, use_checkpoint=True,
                        gradient_method='jax'):
    """
    Run a single targeted training job with checkpoint support.

    Args:
        task_id: Task index (0 to 4)
        realization_seed: Realization index (default: 0)
        verbose: Print detailed progress
        use_checkpoint: Whether to use checkpointing
        gradient_method: 'parallel' (finite-difference) or 'jax' (autodiff)

    Returns:
        success: Boolean indicating success
    """
    print(f"\n{'='*80}")
    print(f"Starting Targeted Task {task_id}, Realization {realization_seed}")
    print(f"{'='*80}")

    # Check if already complete
    if is_training_complete(task_id, realization_seed, results_dir=TARGETED_RESULTS_DIR):
        print(f"Job already completed! Skipping...")
        print(f"{'='*80}\n")
        return True

    start_time = time.time()

    try:
        if not TRAINING_FUNCTIONS_AVAILABLE:
            raise ImportError("Training functions not available. Check imports.")

        # Try to load checkpoint
        checkpoint = None
        if use_checkpoint:
            checkpoint = load_checkpoint(task_id, realization_seed,
                                         results_dir=TARGETED_RESULTS_DIR)
            if checkpoint is not None:
                print(f"Found checkpoint at step {checkpoint['current_step']}")
                print(f"Resuming from checkpoint...")

        # 1. Get task configuration (or load from checkpoint)
        if checkpoint is not None:
            task_config = checkpoint['task_config']
            if verbose:
                print("Step 1: Loaded task configuration from checkpoint...")
        else:
            if verbose:
                print("Step 1: Loading targeted task configuration...")
            task_config = get_targeted_task_config(task_id)

        print(f"  Compression strains: {task_config['compression_strains']}")
        print(f"  Target Poisson ratios: {task_config['target_poisson_ratios']}")

        # 2. Create network or restore from checkpoint
        if checkpoint is not None:
            if verbose:
                print("Step 2: Restoring network from checkpoint...")
            network, boundary_dict = create_auxetic_network(
                n_nodes=N_NODES,
                packing_seed=task_config['packing_seed'],
                force_type=FORCE_TYPE,
                boundary_margin=BOUNDARY_MARGIN
            )
            network.positions = checkpoint['network']['positions']
            network.stiffnesses = checkpoint['network']['stiffnesses']
            network.rest_lengths = checkpoint['network']['rest_lengths']
            network.edges = checkpoint['network']['edges']
            print(f"  Network restored: {len(network.positions)} nodes, {len(network.edges)} edges")
        else:
            if verbose:
                print("Step 2: Creating network from packing...")
            network, boundary_dict = create_auxetic_network(
                n_nodes=N_NODES,
                packing_seed=task_config['packing_seed'],
                force_type=FORCE_TYPE,
                boundary_margin=BOUNDARY_MARGIN
            )
            print(f"  Network created: {len(network.positions)} nodes, {len(network.edges)} edges")

            # 3. Initialize stiffnesses
            if verbose:
                print("Step 3: Initializing random stiffnesses...")
            n_edges = len(network.edges)
            initial_stiffnesses = generate_realization_stiffnesses(
                realization_seed, n_edges
            )
            network.stiffnesses = initial_stiffnesses
            network.save_original_parameters()
            print(f"  Stiffnesses initialized: range [{initial_stiffnesses.min():.2e}, {initial_stiffnesses.max():.2e}]")

        # 4. Prepare training parameters
        compression_strains = task_config['compression_strains']
        target_poisson_ratios = task_config['target_poisson_ratios']
        target_extensions = compute_target_extensions(compression_strains, target_poisson_ratios)

        if verbose:
            print(f"  Target extensions: {target_extensions}")

        # 5. Run training
        if verbose:
            print("Step 4: Running training...")
        print(f"  Training parameters:")
        print(f"    Learning rate: {LEARNING_RATE}")
        print(f"    Steps: {N_STEPS:,}")
        print(f"    Strain steps: {N_STRAIN_STEPS}")
        print(f"    Force tolerance: {FORCE_TOL}")

        # Initialize or restore history
        if checkpoint is not None:
            history = checkpoint['history']
            start_step = checkpoint['current_step']
            print(f"  Resuming from step {start_step}/{N_STEPS}")
        else:
            history = {}
            start_step = 0

        remaining_steps = N_STEPS - start_step

        if remaining_steps > 0:
            train_fn = (finish_training_GD_auxetic_batch_jax
                        if gradient_method == 'jax'
                        else finish_training_GD_auxetic_batch)
            history, trained_network = train_fn(
                network=network,
                history=history,
                learning_rate=LEARNING_RATE,
                n_steps=remaining_steps,
                top_nodes=boundary_dict['top'],
                bottom_nodes=boundary_dict['bottom'],
                left_nodes=boundary_dict['left'],
                right_nodes=boundary_dict['right'],
                force_type=FORCE_TYPE,
                n_strain_steps=N_STRAIN_STEPS,
                source_compression_strain_list=compression_strains,
                desired_target_extension_list=target_extensions,
                force_tol=FORCE_TOL,
                vmin=VMIN,
                vmax=VMAX,
                task_seed=task_id,
                realization_seed=realization_seed,
                save_interval=5,
                task_config=task_config,
                TARGETED_RESULTS_DIR=TARGETED_RESULTS_DIR,
            )
        else:
            trained_network = network
            print("  Training already complete from checkpoint!")

        # 6. Save final results
        if verbose:
            print("Step 5: Saving results...")
        save_training_results(
            task_seed=task_id,
            realization_seed=realization_seed,
            history=history,
            network=trained_network,
            task_config=task_config,
            results_dir=TARGETED_RESULTS_DIR,
        )

        # Remove checkpoint after success
        if use_checkpoint:
            remove_checkpoint(task_id, realization_seed,
                              results_dir=TARGETED_RESULTS_DIR)

        elapsed = time.time() - start_time
        final_loss = history['loss'][-1] if 'loss' in history and history['loss'] else float('nan')

        print(f"\n{'='*80}")
        print(f"SUCCESS: Targeted Task {task_id}, Realization {realization_seed}")
        print(f"Time elapsed: {elapsed/60:.2f} minutes")
        print(f"Final loss: {final_loss:.4e}")
        print(f"Training steps completed: {len(history.get('loss', []))}")
        print(f"{'='*80}\n")

        return True

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"ERROR: Targeted Task {task_id}, Realization {realization_seed}")
        print(f"Time elapsed: {elapsed/60:.2f} minutes")
        print(f"Exception: {e}")
        print(f"{'='*80}\n")
        import traceback
        traceback.print_exc()
        return False


def run_all_targeted(resume=True, verbose=False, gradient_method='jax'):
    """
    Run all 5 targeted training jobs sequentially.

    Args:
        resume: Skip already completed jobs
        verbose: Print detailed progress
        gradient_method: 'parallel' (finite-difference) or 'jax' (autodiff)
    """
    print(f"\n{'#'*80}")
    print(f"# TARGETED ENSEMBLE TRAINING: SEQUENTIAL MODE")
    print(f"# Total jobs: {N_TASKS} tasks x {N_REALIZATIONS} realization = {N_TASKS * N_REALIZATIONS}")
    print(f"# Resume mode: {resume}")
    print(f"{'#'*80}\n")

    print_targeted_tasks_summary()

    if resume:
        jobs = get_incomplete_jobs(
            n_tasks=N_TASKS,
            n_realizations=N_REALIZATIONS,
            results_dir=TARGETED_RESULTS_DIR,
        )
        print(f"Found {len(jobs)} incomplete jobs (out of {N_TASKS * N_REALIZATIONS} total)")
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
        success = run_single_training(task_id, realization_seed, verbose=verbose,
                                      gradient_method=gradient_method)

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
    print(f"# TARGETED TRAINING COMPLETE")
    print(f"# Successful: {success_count}/{len(jobs)}")
    print(f"# Failed: {failure_count}/{len(jobs)}")
    print(f"# Total time: {total_elapsed/60:.2f} minutes")
    print(f"{'#'*80}\n")

    print_targeted_progress()


def print_targeted_progress():
    """Print progress summary for targeted tasks."""
    complete = get_complete_jobs(
        n_tasks=N_TASKS,
        n_realizations=N_REALIZATIONS,
        results_dir=TARGETED_RESULTS_DIR,
    )
    incomplete = get_incomplete_jobs(
        n_tasks=N_TASKS,
        n_realizations=N_REALIZATIONS,
        results_dir=TARGETED_RESULTS_DIR,
    )
    total = N_TASKS * N_REALIZATIONS

    print(f"\n{'='*80}")
    print(f"TARGETED TRAINING PROGRESS")
    print(f"{'='*80}")
    print(f"Total jobs: {total}")
    print(f"Complete: {len(complete)} ({100*len(complete)/total:.0f}%)")
    print(f"Incomplete: {len(incomplete)} ({100*len(incomplete)/total:.0f}%)")
    print()

    all_configs = get_all_targeted_task_configs()
    for task_id in range(N_TASKS):
        config = all_configs[task_id]
        status = "DONE" if (task_id, 0) in complete else "pending"
        pairs = list(zip(config['compression_strains'], config['target_poisson_ratios']))
        pairs_str = ", ".join(f"nu={p} @ comp={c}" for c, p in pairs)
        print(f"  Task {task_id}: [{status:>7s}]  {pairs_str}")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Targeted ensemble training for auxetic networks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single task (debugging)
  python targeted_ensemble_runner.py --mode single --task 0 --verbose

  # Run all 5 tasks sequentially
  python targeted_ensemble_runner.py --mode sequential

  # Resume incomplete tasks
  python targeted_ensemble_runner.py --mode sequential --resume

  # Check progress
  python targeted_ensemble_runner.py --mode status
        """
    )

    parser.add_argument(
        '--mode',
        choices=['sequential', 'single', 'status'],
        default='single',
        help='Execution mode'
    )
    parser.add_argument(
        '--task',
        type=int,
        help='Task ID for single mode (0-4)'
    )
    parser.add_argument(
        '--realization',
        type=int,
        default=0,
        help='Realization seed for single mode (default: 0)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Resume from incomplete jobs (sequential mode)'
    )
    parser.add_argument(
        '--no-resume',
        action='store_false',
        dest='resume',
        help='Start from scratch (sequential mode)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--gradient-method',
        choices=['parallel', 'jax'],
        default='jax',
        help='Gradient computation method: parallel (finite-difference, default) or jax (autodiff)'
    )

    args = parser.parse_args()

    if args.mode == 'single':
        if args.task is None:
            parser.error("--task required for single mode")
        if args.task < 0 or args.task >= N_TASKS:
            parser.error(f"--task must be between 0 and {N_TASKS-1}")

        print_targeted_tasks_summary()
        success = run_single_training(args.task, args.realization, verbose=args.verbose,
                                      gradient_method=args.gradient_method)
        sys.exit(0 if success else 1)

    elif args.mode == 'sequential':
        run_all_targeted(resume=args.resume, verbose=args.verbose,
                         gradient_method=args.gradient_method)

    elif args.mode == 'status':
        print_targeted_tasks_summary()
        print_targeted_progress()


if __name__ == '__main__':
    main()
