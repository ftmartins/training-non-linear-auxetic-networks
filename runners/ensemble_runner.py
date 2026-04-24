#!/usr/bin/env python
"""
Ensemble training runner for auxetic networks.

Usage:
    # Run all jobs sequentially
    python ensemble_runner.py --mode sequential

    # Run specific job (for debugging)
    python ensemble_runner.py --mode single --task 0 --realization 0

    # Resume incomplete jobs
    python ensemble_runner.py --mode sequential --resume

    # Verbose output
    python ensemble_runner.py --mode single --task 0 --realization 0 --verbose
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import config and modules
from config import *
from config import get_n_nodes, get_n_strain_steps
from network_utils import create_auxetic_network
from task_generator import (
    generate_task_config,
    generate_realization_stiffnesses,
    compute_target_extensions
)
from checkpoint_manager import (
    is_training_complete,
    save_training_results,
    get_incomplete_jobs,
    print_progress_summary,
    save_checkpoint,
    load_checkpoint,
    has_checkpoint,
    remove_checkpoint
)

# Import training functions
try:
    from training_functions_with_toggle import (
        finish_training_GD_auxetic_batch,
        finish_training_GD_auxetic_batch_jax,
        compute_quasistatic_trajectory_auxetic,
        poisson_loss_batch_parallel,
        finite_difference_gradient_parallel_batch
    )
    TRAINING_FUNCTIONS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import training functions: {e}")
    print("Will need to extract from notebook if running actual training.")
    TRAINING_FUNCTIONS_AVAILABLE = False


def run_single_training(task_seed, realization_seed, verbose=False, use_checkpoint=True,
                        gradient_method='parallel'):
    """
    Run a single training job with checkpoint support.

    Args:
        task_seed: Task index (0 to N_TASKS-1)
        realization_seed: Realization index (0 to N_REALIZATIONS-1)
        verbose: Print detailed progress
        use_checkpoint: Whether to use checkpointing (default: True)
        gradient_method: 'parallel' (finite-difference) or 'jax' (autodiff)

    Returns:
        success: Boolean indicating success
    """
    print(f"\n{'='*80}")
    print(f"Starting Task {task_seed}, Realization {realization_seed}")
    print(f"{'='*80}")

    # Check if already complete
    if is_training_complete(task_seed, realization_seed):
        print(f"Job already completed! Skipping...")
        print(f"{'='*80}\n")
        return True

    start_time = time.time()

    try:
        # Check if training functions are available
        if not TRAINING_FUNCTIONS_AVAILABLE:
            raise ImportError("Training functions not available. Check imports.")

        # Try to load checkpoint
        checkpoint = None
        if use_checkpoint:
            checkpoint = load_checkpoint(task_seed, realization_seed)
            if checkpoint is not None:
                print(f"Found checkpoint at step {checkpoint['current_step']}")
                print(f"Resuming from checkpoint...")

        # 1. Generate task configuration (or load from checkpoint)
        if checkpoint is not None:
            task_config = checkpoint['task_config']
            if verbose:
                print("Step 1: Loaded task configuration from checkpoint...")
        else:
            if verbose:
                print("Step 1: Generating task configuration...")
            task_config = generate_task_config(task_seed)

        print(f"  Compression strains: {task_config['compression_strains']}")
        print(f"  Target Poisson ratios: {task_config['target_poisson_ratios']}")

        # 2. Create network (unique per task via packing_seed) or restore from checkpoint
        if checkpoint is not None:
            if verbose:
                print("Step 2: Restoring network from checkpoint...")
            # Create base network structure
            network, boundary_dict = create_auxetic_network(
                n_nodes=get_n_nodes(task_seed),
                packing_seed=task_config['packing_seed'],
                force_type=FORCE_TYPE,
                boundary_margin=BOUNDARY_MARGIN
            )
            # Restore network state from checkpoint
            network.positions = checkpoint['network']['positions']
            network.stiffnesses = checkpoint['network']['stiffnesses']
            network.rest_lengths = checkpoint['network']['rest_lengths']
            network.edges = checkpoint['network']['edges']
            print(f"  Network restored: {len(network.positions)} nodes, {len(network.edges)} edges")
        else:
            if verbose:
                print("Step 2: Creating network from packing...")
            network, boundary_dict = create_auxetic_network(
                n_nodes=get_n_nodes(task_seed),
                packing_seed=task_config['packing_seed'],
                force_type=FORCE_TYPE,
                boundary_margin=BOUNDARY_MARGIN
            )
            print(f"  Network created: {len(network.positions)} nodes, {len(network.edges)} edges")

            # 3. Initialize stiffnesses (unique per realization)
            if verbose:
                print("Step 3: Initializing random stiffnesses...")
            n_edges = len(network.edges)
            initial_stiffnesses = generate_realization_stiffnesses(
                realization_seed,
                n_edges
            )
            network.stiffnesses = initial_stiffnesses
            network.save_original_parameters()
            print(f"  Stiffnesses initialized: range [{initial_stiffnesses.min():.2e}, {initial_stiffnesses.max():.2e}]")

        # 4. Prepare training parameters
        compression_strains = task_config['compression_strains']
        target_poisson_ratios = task_config['target_poisson_ratios']

        # Convert Poisson ratios to target extensions
        # For auxetic: ν = -(lateral_strain / vertical_strain)
        # So: lateral_strain = -ν * vertical_strain
        target_extensions = compute_target_extensions(compression_strains, target_poisson_ratios)

        if verbose:
            print(f"  Target extensions: {target_extensions}")

        # 5. Run training (with checkpoint support)
        if verbose:
            print("Step 4: Running training...")
        print(f"  Training parameters:")
        print(f"    Learning rate: {LEARNING_RATE}")
        print(f"    Steps: {N_STEPS:,}")
        print(f"    Strain steps: {get_n_strain_steps(task_seed)}")
        print(f"    Force tolerance: {FORCE_TOL}")

        # Initialize or restore history
        if checkpoint is not None:
            history = checkpoint['history']
            start_step = checkpoint['current_step']
            print(f"  Resuming from step {start_step}/{N_STEPS}")
        else:
            history = {}
            start_step = 0

        # Note: The training function doesn't natively support resuming,
        # so we call it with remaining steps
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
                n_strain_steps=get_n_strain_steps(task_seed),
                source_compression_strain_list=compression_strains,
                desired_target_extension_list=target_extensions,
                force_tol=FORCE_TOL,
                vmin=VMIN,
                vmax=VMAX,
                task_seed=task_seed,
                realization_seed=realization_seed,
                save_interval=500,
            )
        else:
            trained_network = network
            print("  Training already complete from checkpoint!")

        # 6. Save final results
        if verbose:
            print("Step 5: Saving results...")
        save_training_results(
            task_seed=task_seed,
            realization_seed=realization_seed,
            history=history,
            network=trained_network,
            task_config=task_config
        )

        # Remove checkpoint file after successful completion
        if use_checkpoint:
            remove_checkpoint(task_seed, realization_seed)

        elapsed = time.time() - start_time
        final_loss = history['loss'][-1] if 'loss' in history and history['loss'] else float('nan')

        print(f"\n{'='*80}")
        print(f"SUCCESS: Task {task_seed}, Realization {realization_seed}")
        print(f"Time elapsed: {elapsed/60:.2f} minutes")
        print(f"Final loss: {final_loss:.4e}")
        print(f"Training steps completed: {len(history.get('loss', []))}")
        print(f"{'='*80}\n")

        return True

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"ERROR: Task {task_seed}, Realization {realization_seed}")
        print(f"Time elapsed: {elapsed/60:.2f} minutes")
        print(f"Exception: {e}")
        print(f"{'='*80}\n")
        import traceback
        traceback.print_exc()
        return False


def run_ensemble_sequential(resume=True, verbose=False, gradient_method='parallel'):
    """
    Run all ensemble jobs sequentially.

    Args:
        resume: Skip already completed jobs
        verbose: Print detailed progress
        gradient_method: 'parallel' (finite-difference) or 'jax' (autodiff)
    """
    print(f"\n{'#'*80}")
    print(f"# ENSEMBLE TRAINING: SEQUENTIAL MODE")
    print(f"# Total jobs: {N_TASKS} tasks × {N_REALIZATIONS} realizations = {N_TASKS * N_REALIZATIONS}")
    print(f"# Resume mode: {resume}")
    print(f"{'#'*80}\n")

    if resume:
        jobs = get_incomplete_jobs()
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

    for idx, (task_seed, realization_seed) in enumerate(jobs):
        print(f"\n[Job {idx+1}/{len(jobs)}]")
        success = run_single_training(task_seed, realization_seed, verbose=verbose,
                                      gradient_method=gradient_method)

        if success:
            success_count += 1
        else:
            failure_count += 1

        # Print periodic progress update
        if (idx + 1) % 10 == 0 or (idx + 1) == len(jobs):
            elapsed = time.time() - start_time_overall
            avg_time_per_job = elapsed / (idx + 1)
            remaining_jobs = len(jobs) - (idx + 1)
            estimated_remaining = avg_time_per_job * remaining_jobs

            print(f"\n{'~'*80}")
            print(f"Progress: {idx+1}/{len(jobs)} jobs completed")
            print(f"Success: {success_count}, Failed: {failure_count}")
            print(f"Average time per job: {avg_time_per_job/60:.2f} minutes")
            print(f"Estimated time remaining: {estimated_remaining/3600:.2f} hours")
            print(f"{'~'*80}\n")

    total_elapsed = time.time() - start_time_overall

    print(f"\n{'#'*80}")
    print(f"# ENSEMBLE TRAINING COMPLETE")
    print(f"# Successful: {success_count}/{len(jobs)}")
    print(f"# Failed: {failure_count}/{len(jobs)}")
    print(f"# Total time: {total_elapsed/3600:.2f} hours")
    print(f"{'#'*80}\n")

    # Print final progress summary
    print_progress_summary()


def main():
    """Main entry point for ensemble training."""
    parser = argparse.ArgumentParser(
        description='Ensemble training for auxetic networks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single job (debugging)
  python ensemble_runner.py --mode single --task 0 --realization 0 --verbose

  # Run all jobs sequentially
  python ensemble_runner.py --mode sequential

  # Resume incomplete jobs
  python ensemble_runner.py --mode sequential --resume

  # Check progress without running
  python ensemble_runner.py --mode status
        """
    )

    parser.add_argument(
        '--mode',
        choices=['sequential', 'single', 'status'],
        default='sequential',
        help='Execution mode'
    )
    parser.add_argument(
        '--task',
        type=int,
        help='Task seed for single mode'
    )
    parser.add_argument(
        '--realization',
        type=int,
        help='Realization seed for single mode'
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

    # Validate config before running
    validate_config()

    if args.mode == 'single':
        if args.task is None or args.realization is None:
            parser.error("--task and --realization required for single mode")
        if args.task < 0 or args.task >= N_TASKS:
            parser.error(f"--task must be between 0 and {N_TASKS-1}")
        if args.realization < 0 or args.realization >= N_REALIZATIONS:
            parser.error(f"--realization must be between 0 and {N_REALIZATIONS-1}")

        success = run_single_training(args.task, args.realization, verbose=args.verbose,
                                      gradient_method=args.gradient_method)
        sys.exit(0 if success else 1)

    elif args.mode == 'sequential':
        run_ensemble_sequential(resume=args.resume, verbose=args.verbose,
                                gradient_method=args.gradient_method)

    elif args.mode == 'status':
        print_progress_summary()


if __name__ == '__main__':
    main()
