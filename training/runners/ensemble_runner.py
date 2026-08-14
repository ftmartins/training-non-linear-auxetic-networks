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

_ROOT = Path(__file__).parent.parent.parent  # project root
_SRC  = Path(__file__).parent.parent / 'src'  # training/src/
for _p in [str(_ROOT), str(_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import config and modules
from base.config import *
from base.config import get_n_nodes, get_n_strain_steps
from base.network_utils import create_auxetic_network
from training.src.task_generator import (
    generate_task_config,
    generate_realization_stiffnesses,
    compute_target_extensions
)
from training.src.good_realizations import get_realization_seed
from training.src.checkpoint_manager import (
    is_training_complete,
    save_training_results,
    get_incomplete_jobs,
    print_progress_summary,
    save_checkpoint,
    load_checkpoint,
    has_checkpoint,
    remove_checkpoint,
    results_dir_for_gradient_method,
    save_run_metadata,
    save_run_code_snapshot,
    job_has_critical_mismatch,
    reset_job,
)

# Code archived alongside each job's results (see save_run_code_snapshot) so
# the exact code that produced it is recoverable without relying on git
# history/dirty state alone.
_CODE_SNAPSHOT_FILES = [
    Path(__file__),
    _SRC / 'task_generator.py',
    _SRC / 'checkpoint_manager.py',
    _SRC / 'training_functions.py',
    _SRC / 'run_provenance.py',
    _ROOT / 'base' / 'config.py',
    _ROOT / 'base' / 'network_utils.py',
]

# Import training functions
try:
    from training.src.training_functions import (
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
                        gradient_method='parallel', network_type=NETWORK_TYPE, overwrite=False):
    """
    Run a single training job with checkpoint support.

    Args:
        task_seed: Task index (0 to N_TASKS-1)
        realization_seed: Realization index (0 to N_REALIZATIONS-1)
        verbose: Print detailed progress
        use_checkpoint: Whether to use checkpointing (default: True)
        gradient_method: 'newton' (IFT), 'newton_fd' (Newton loss + finite-difference
            gradient), 'jax' (autodiff), or 'fire'/'parallel' (Cython FIRE + finite-difference)
        network_type: 'jammed' or 'lattice' (see create_auxetic_network)
        overwrite: If False (default) and this job's recorded solver/gradient_method/
            learning_rate/k_min/k_max/force_tol conflict with the current run's values,
            raises HyperparameterMismatchError instead of continuing (n_steps is exempt —
            resuming to train longer is fine). If True, the job's saved state (checkpoint,
            results, meta) is wiped and restarted from scratch under the new hyperparameters
            — never resumed under a relabeled meta, which would make training_meta.json
            describe a trajectory that never actually happened.

    Returns:
        success: Boolean indicating success
    """
    print(f"\n{'='*80}")
    print(f"Starting Task {task_seed}, Realization {realization_seed}")
    print(f"{'='*80}")

    # Results are partitioned by gradient_method (parallel/jax write to the
    # same task/realization otherwise, silently overwriting each other's
    # checkpoints and results).
    results_dir = results_dir_for_gradient_method(RESULTS_DIR, gradient_method)

    # realization_seed (0..N_REALIZATIONS-1) is a serial index into the
    # screened-good seeds for this task, not a literal RNG seed — see
    # training/src/good_realizations.py / docs/realization_screening.md.
    # Computed here (before critical_hparams below) so a mismatch between
    # this run's screened seed and whatever seed actually produced a resumed
    # checkpoint is caught the same way as any other critical key, instead of
    # silently resuming a trajectory screening never selected.
    screened_seed = get_realization_seed('auxetic_general', task_seed, realization_seed)

    # Guard against silently mixing incompatible hyperparameters into one
    # saved trajectory. Must run BEFORE any checkpoint/completion check —
    # resuming a checkpoint trained under OLD hyperparameters and merely
    # relabeling training_meta.json with the NEW ones would make it describe
    # a run that never actually happened.
    optimizer = 'fire' if (gradient_method == 'jax' and USE_OPT_FIRE) else 'gradient_descent'
    critical_hparams = {
        'learning_rate': LEARNING_RATE,
        'k_min': VMIN,
        'k_max': VMAX,
        'force_tol': FORCE_TOL,
        'gradient_method': gradient_method,
        'optimizer': optimizer,
        'opt_fire_finc': OPT_FIRE_FINC,
        'realization_seed': screened_seed,
    }
    if job_has_critical_mismatch(task_seed, realization_seed, critical_hparams,
                                 results_dir=results_dir, network_type=network_type):
        if not overwrite:
            print(f"\n{'='*80}")
            print(f"ERROR: Task {task_seed}, Realization {realization_seed}")
            print(f"Recorded solver/gradient_method/learning_rate/k_min/k_max/force_tol "
                  f"differ from this run's. Pass overwrite=True (--overwrite) to wipe and "
                  f"restart from scratch under the new hyperparameters.")
            print(f"{'='*80}\n")
            return False
        print(f"  overwrite=True: recorded hyperparameters differ from this run's — wiping "
              f"saved state for task {task_seed}, realization {realization_seed} and "
              f"restarting from scratch under the new hyperparameters.")
        reset_job(task_seed, realization_seed, results_dir=results_dir, network_type=network_type)

    # Check if already complete
    if is_training_complete(task_seed, realization_seed, results_dir=results_dir, network_type=network_type):
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
            checkpoint = load_checkpoint(task_seed, realization_seed, results_dir=results_dir, network_type=network_type)
            if checkpoint is not None:
                print(f"Found checkpoint at step {checkpoint['current_step']}")
                print(f"Resuming from checkpoint...")

        # 1. Generate task configuration (or load from checkpoint)
        if checkpoint is not None:
            print(checkpoint)
            try:
                task_config = checkpoint['task_config']
                assert task_config is not None
            except (KeyError, AssertionError) as e:
                task_config = generate_task_config(task_seed)
                print(f'Regenerated task config due to checkpoint not having task config ({e!r})')
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
                boundary_margin=BOUNDARY_MARGIN,
                central_force=PACKING_PARAMS['central'],
                network_type=network_type,
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
                boundary_margin=BOUNDARY_MARGIN,
                central_force=PACKING_PARAMS['central'],
                network_type=network_type,
            )
            print(f"  Network created: {len(network.positions)} nodes, {len(network.edges)} edges")

            # 3. Initialize stiffnesses (unique per realization)
            if verbose:
                print("Step 3: Initializing random stiffnesses...")
            n_edges = len(network.edges)
            # screened_seed computed earlier, up with critical_hparams.
            initial_stiffnesses = generate_realization_stiffnesses(
                task_seed,
                screened_seed,
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

        # Record the hyperparameters actually driving this job (the nominal
        # LEARNING_RATE constant, unmodified) + which code version (git
        # commit) invoked it, so a saved run's provenance is recoverable
        # later even if these module-level constants change. Raises
        # HyperparameterMismatchError (caught below) if this job was
        # previously recorded with different solver/gradient_method/
        # learning_rate/k_min/k_max/force_tol and overwrite=False.
        save_run_metadata(
            task_seed, realization_seed,
            hyperparams={
                'learning_rate': LEARNING_RATE,
                'k_min': VMIN,
                'k_max': VMAX,
                'n_steps': N_STEPS,
                'force_tol': FORCE_TOL,
                'gradient_method': gradient_method,
                'optimizer': optimizer,
                'opt_fire_finc': OPT_FIRE_FINC,
                'realization_seed': screened_seed,
            },
            results_dir=results_dir,
            network_type=network_type,
            overwrite=overwrite,
        )
        save_run_code_snapshot(task_seed, realization_seed, _CODE_SNAPSHOT_FILES,
                               results_dir=results_dir, network_type=network_type)

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
            if gradient_method == 'jax':
                train_fn = finish_training_GD_auxetic_batch_jax
                method_kwarg = {'opt_fire': USE_OPT_FIRE, 'opt_fire_finc': OPT_FIRE_FINC}
            else:
                train_fn = finish_training_GD_auxetic_batch
                method_kwarg = {'method': 'fire' if gradient_method in ('fire', 'parallel') else gradient_method}
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
                network_type=network_type,
                TARGETED_RESULTS_DIR=results_dir,
                **method_kwarg,
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
            task_config=generate_task_config(task_seed),
            boundary_dict=boundary_dict,
            results_dir=results_dir,
            network_type=network_type,
        )

        # Remove checkpoint file after successful completion
        if use_checkpoint:
            remove_checkpoint(task_seed, realization_seed, results_dir=results_dir, network_type=network_type)

        elapsed = time.time() - start_time
        final_loss = history['loss'][-1] if 'loss' in history and history['loss'] else float('nan')

        if not np.isfinite(final_loss):
            print(f"\n{'='*80}")
            print(f"FAILURE: Task {task_seed}, Realization {realization_seed}")
            print(f"Time elapsed: {elapsed/60:.2f} minutes")
            print(f"Final loss is not finite: {final_loss}")
            print(f"Training steps completed: {len(history.get('loss', []))}")
            print(f"{'='*80}\n")
            return False

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


def run_ensemble_sequential(resume=True, verbose=False, gradient_method='parallel',
                            network_type=NETWORK_TYPE, overwrite=False):
    """
    Run all ensemble jobs sequentially.

    Args:
        resume: Skip already completed jobs
        verbose: Print detailed progress
        gradient_method: 'newton' (IFT), 'newton_fd' (Newton loss + finite-difference
            gradient), 'jax' (autodiff), or 'fire'/'parallel' (Cython FIRE + finite-difference)
        network_type: 'jammed' or 'lattice' (see create_auxetic_network)
        overwrite: Passed through to run_single_training (see there).
    """
    print(f"\n{'#'*80}")
    print(f"# ENSEMBLE TRAINING: SEQUENTIAL MODE")
    print(f"# Total jobs: {N_TASKS} tasks × {N_REALIZATIONS} realizations = {N_TASKS * N_REALIZATIONS}")
    print(f"# Resume mode: {resume}")
    print(f"{'#'*80}\n")

    if resume:
        jobs = get_incomplete_jobs(results_dir=results_dir_for_gradient_method(RESULTS_DIR, gradient_method),
                                   network_type=network_type)
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
                                      gradient_method=gradient_method, network_type=network_type,
                                      overwrite=overwrite)

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
    print_progress_summary(results_dir=results_dir_for_gradient_method(RESULTS_DIR, gradient_method),
                           network_type=network_type)


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
        choices=['newton', 'newton_fd', 'jax', 'fire', 'parallel'],
        default='jax',
        help='Gradient computation method: newton (IFT), newton_fd (Newton loss + '
             'finite-difference gradient), jax (autodiff, default), or fire/parallel '
             '(Cython FIRE + finite-difference)'
    )
    parser.add_argument(
        '--network-type',
        choices=['jammed', 'lattice'],
        default=NETWORK_TYPE,
        help="Network generation method: 'jammed' (packing-derived, default) or "
             "'lattice' (perturbed triangular lattice square)"
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Allow a job to replace previously recorded solver/gradient_method/learning_rate/'
             'k_min/k_max/force_tol in training_meta.json instead of failing on mismatch'
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
                                      gradient_method=args.gradient_method,
                                      network_type=args.network_type,
                                      overwrite=args.overwrite)
        sys.exit(0 if success else 1)

    elif args.mode == 'sequential':
        run_ensemble_sequential(resume=args.resume, verbose=args.verbose,
                                gradient_method=args.gradient_method,
                                network_type=args.network_type,
                                overwrite=args.overwrite)

    elif args.mode == 'status':
        print_progress_summary(results_dir=results_dir_for_gradient_method(RESULTS_DIR, args.gradient_method),
                               network_type=args.network_type)


if __name__ == '__main__':
    main()
