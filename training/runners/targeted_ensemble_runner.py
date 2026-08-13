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
import os
_ROOT = Path(__file__).parent.parent.parent  # project root
_SRC  = Path(__file__).parent.parent / 'src'  # training/src/
for _p in [str(_ROOT), str(_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import shared config (not validate_config)
from base.config import (
    FORCE_TYPE, BOUNDARY_MARGIN,
    FORCE_TOL, PACKING_PARAMS, NETWORK_TYPE
)
VMIN = 1e-4
VMAX = 1e2


LEARNING_RATE = 3e-3  # starting lr / opt_fire dt_init (opt_fire hyperparameter grid sweep,
                       # tasks 3/15); decayed by training.src.lr_schedule when not using opt_fire
OPT_FIRE_FINC = 1.05   # opt_fire growth factor (same sweep) — never diverged in the grid
USE_OPT_FIRE = True    # FIRE-style adaptive optimizer vs plain gradient descent. Short A/B test
                       # (2026-08-13, tasks 3/15, 200-step budget, lr=3e-3 both): GD's early
                       # minima were competitive with (task 15: even better than) FIRE's, but GD
                       # could not sustain progress — it oscillated, then the stiffness update
                       # went NaN on BOTH tasks (step 5/12, physics solver needing 100-500x longer
                       # per step beforehand) well short of the step budget. FIRE completed both
                       # runs to full length with no NaN. Decisive for production use: a run that
                       # can't finish is worse than one with a middling final loss.
                       # 'optimizer' is a critical key in training_meta.json, so resuming under a
                       # different choice crashes rather than silently mixing trajectories.

# Import targeted config and task definitions
from training.src.targeted_task_generator import (
    get_targeted_task_config,
    get_all_targeted_task_configs,
    print_targeted_tasks_summary,
    N_TASKS, N_REALIZATIONS, N_STEPS, N_STRAIN_STEPS,
    TARGETED_RESULTS_DIR,
)

# Import shared utilities
from training.src.task_generator import generate_realization_stiffnesses, compute_target_extensions
from training.src.good_realizations import get_realization_seed
from base.network_utils import create_auxetic_network
from training.src.checkpoint_manager import (
    is_training_complete,
    save_training_results,
    get_incomplete_jobs,
    get_complete_jobs,
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
    _SRC / 'targeted_task_generator.py',
    _SRC / 'task_generator.py',
    _SRC / 'checkpoint_manager.py',
    _SRC / 'training_functions.py',
    _SRC / 'run_provenance.py',
    _ROOT / 'base' / 'config.py',
    _ROOT / 'base' / 'network_utils.py',
]

# Import training function
from training.src.training_functions import (
        finish_training_GD_auxetic_batch,
        finish_training_GD_auxetic_batch_jax,
    )
TRAINING_FUNCTIONS_AVAILABLE = True

def run_single_training(task_id, realization_seed=0, verbose=False, use_checkpoint=True,
                        gradient_method='newton', network_type=NETWORK_TYPE, overwrite=False):
    """
    Run a single targeted training job with checkpoint support.

    Args:
        task_id: Task index (0 to 4)
        realization_seed: Realization index (default: 0)
        network_type: 'jammed' or 'lattice' (see create_auxetic_network)
        verbose: Print detailed progress
        use_checkpoint: Whether to use checkpointing
        gradient_method: 'newton' (IFT, default), 'newton_fd' (Newton loss + finite-difference
            gradient), 'jax' (autodiff), or 'fire'/'parallel' (Cython FIRE + finite-difference)
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
    print(f"Starting Targeted Task {task_id}, Realization {realization_seed}")
    print(f"{'='*80}")
    from training.src.targeted_task_generator import TARGETED_RESULTS_DIR

    TARGETED_RESULTS_DIR_ = TARGETED_RESULTS_DIR / f'{gradient_method}'
    TARGETED_RESULTS_DIR = TARGETED_RESULTS_DIR_

    # Results are partitioned by gradient_method (newton/newton_fd/fire/parallel/jax
    # all write to the same task/realization otherwise, silently overwriting each
    # other's checkpoints and results).
    results_dir = results_dir_for_gradient_method(TARGETED_RESULTS_DIR, gradient_method)

    # Guard against silently mixing incompatible hyperparameters into one
    # saved trajectory. Must run BEFORE any NaN/checkpoint/completion check —
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
    }
    if job_has_critical_mismatch(task_id, realization_seed, critical_hparams,
                                 results_dir=results_dir, network_type=network_type):
        if not overwrite:
            print(f"\n{'='*80}")
            print(f"ERROR: Targeted Task {task_id}, Realization {realization_seed}")
            print(f"Recorded solver/gradient_method/learning_rate/k_min/k_max/force_tol "
                  f"differ from this run's. Pass overwrite=True (--overwrite) to wipe and "
                  f"restart from scratch under the new hyperparameters.")
            print(f"{'='*80}\n")
            return False
        print(f"  overwrite=True: recorded hyperparameters differ from this run's — wiping "
              f"saved state for task {task_id}, realization {realization_seed} and "
              f"restarting from scratch under the new hyperparameters.")
        reset_job(task_id, realization_seed, results_dir=results_dir, network_type=network_type)

    # If a previous attempt left NaN in this job's saved results, this run
    # will hit the same live NaN check inside the training loop and stop
    # early with whatever's salvageable — recovering means re-invoking with
    # a lower --learning-rate and --overwrite, which wipes and restarts
    # from scratch under the new (lower) nominal rate (job_has_critical_mismatch
    # above). No automatic retry-at-reduced-LR here anymore.
    if is_training_complete(task_id, realization_seed, results_dir=results_dir,
                            network_type=network_type):
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
                                         results_dir=results_dir, network_type=network_type)
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

        # 2. Create base network structure
        if verbose:
            print("Step 2: Creating network from packing...")
        network, boundary_dict = create_auxetic_network(
            n_nodes=task_config['n_nodes'],
            packing_seed=task_config['packing_seed'],
            force_type=FORCE_TYPE,
            boundary_margin=BOUNDARY_MARGIN,
            central_force=PACKING_PARAMS['central'],
            network_type=network_type,
        )
        print(f"  Network created: {len(network.positions)} nodes, {len(network.edges)} edges")

        # 3. Determine initial state: resume from checkpoint if present, else fresh start
        history = {}
        start_step = 0

        checkpoint = None
        if use_checkpoint:
            checkpoint = load_checkpoint(task_id, realization_seed,
                                         results_dir=results_dir, network_type=network_type)
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
                print("Step 3: Initializing random stiffnesses...")
            n_edges = len(network.edges)
            # realization_seed (0..N_REALIZATIONS-1) is a serial index into
            # the screened-good seeds for this task, not a literal RNG seed
            # — see training/src/good_realizations.py / docs/realization_screening.md.
            screened_seed = get_realization_seed('auxetic_targeted', task_id, realization_seed)
            initial_stiffnesses = generate_realization_stiffnesses(task_id, screened_seed, n_edges)
            network.stiffnesses = initial_stiffnesses
            network.save_original_parameters()
            print(f"  Stiffnesses initialized: range "
                  f"[{initial_stiffnesses.min():.2e}, {initial_stiffnesses.max():.2e}]")

        # 4. Prepare training parameters
        compression_strains = task_config['compression_strains']
        target_poisson_ratios = task_config['target_poisson_ratios']
        target_extensions = compute_target_extensions(compression_strains, target_poisson_ratios)
        if verbose:
            print(f"  Target extensions: {target_extensions}")

        remaining_steps = N_STEPS - start_step

        print(f"  Training parameters:")
        print(f"    Learning rate: {LEARNING_RATE} (starting; see training.src.lr_schedule)")
        print(f"    Steps remaining: {remaining_steps:,} / {N_STEPS:,}")
        print(f"    Strain steps: {N_STRAIN_STEPS}")
        print(f"    Force tolerance: {FORCE_TOL}")

        # Record the nominal learning rate + which code version (git commit)
        # invoked it. Raises HyperparameterMismatchError (caught below) if
        # this job was previously recorded with different solver/
        # gradient_method/learning_rate/k_min/k_max/force_tol and
        # overwrite=False.
        save_run_metadata(
            task_id, realization_seed,
            hyperparams={
                'learning_rate': LEARNING_RATE,
                'k_min': VMIN,
                'k_max': VMAX,
                'n_steps': N_STEPS,
                'force_tol': FORCE_TOL,
                'gradient_method': gradient_method,
                'optimizer': optimizer,
                'opt_fire_finc': OPT_FIRE_FINC,
            },
            results_dir=results_dir,
            network_type=network_type,
            overwrite=overwrite,
        )
        save_run_code_snapshot(task_id, realization_seed, _CODE_SNAPSHOT_FILES,
                               results_dir=results_dir, network_type=network_type)

        # 5. Run training
        if verbose:
            print("Step 4: Running training...")

        def _run_train(net, hist, lr, n_steps):
            if gradient_method == 'jax':
                train_fn = finish_training_GD_auxetic_batch_jax
                method_kwarg = {'opt_fire': USE_OPT_FIRE, 'opt_fire_finc': OPT_FIRE_FINC}
            else:
                train_fn = finish_training_GD_auxetic_batch
                method_kwarg = {'method': 'fire' if gradient_method in ('fire', 'parallel') else gradient_method}
            return train_fn(
                network=net,
                history=hist,
                learning_rate=lr,
                n_steps=n_steps,
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
                TARGETED_RESULTS_DIR=results_dir,
                network_type=network_type,
                **method_kwarg,
            )

        if remaining_steps > 0:
            history, trained_network = _run_train(network, history, LEARNING_RATE, remaining_steps)
        else:
            trained_network = network
            print("  Training already complete!")

        # 6. Save final results
        if verbose:
            print("Step 5: Saving results...")
        save_training_results(
            task_seed=task_id,
            realization_seed=realization_seed,
            history=history,
            network=trained_network,
            task_config=task_config,
            boundary_dict=boundary_dict,
            results_dir=results_dir,
            network_type=network_type,
        )

        # Remove checkpoint after success
        if use_checkpoint:
            remove_checkpoint(task_id, realization_seed, results_dir=results_dir,
                              network_type=network_type)

        elapsed = time.time() - start_time
        loss_list = history.get('loss', [])
        final_loss = loss_list[-1] if loss_list else float('nan')

        if not np.isfinite(final_loss):
            print(f"\n{'='*80}")
            print(f"FAILURE: Targeted Task {task_id}, Realization {realization_seed}")
            print(f"Time elapsed: {elapsed/60:.2f} minutes")
            print(f"Final loss is not finite: {final_loss}")
            print(f"Training steps completed: {len(loss_list)}")
            print(f"{'='*80}\n")
            return False

        print(f"\n{'='*80}")
        print(f"SUCCESS: Targeted Task {task_id}, Realization {realization_seed}")
        print(f"Time elapsed: {elapsed/60:.2f} minutes")
        print(f"Final loss: {final_loss:.4e}")
        print(f"Training steps completed: {len(loss_list)}")
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


def run_all_targeted(resume=True, verbose=False, gradient_method='newton', network_type=NETWORK_TYPE,
                     overwrite=False):
    """
    Run all 5 targeted training jobs sequentially.

    Args:
        resume: Skip already completed jobs
        verbose: Print detailed progress
        gradient_method: 'newton' (IFT, default), 'newton_fd' (Newton loss + finite-difference
            gradient), 'jax' (autodiff), or 'fire'/'parallel' (Cython FIRE + finite-difference)
        network_type: 'jammed' or 'lattice' (see create_auxetic_network)
        overwrite: Passed through to run_single_training (see there).
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
            results_dir=results_dir_for_gradient_method(TARGETED_RESULTS_DIR, gradient_method),
            network_type=network_type,
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
                                      gradient_method=gradient_method, network_type=network_type,
                                      overwrite=overwrite)

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

    print_targeted_progress(network_type=network_type, gradient_method=gradient_method)


def print_targeted_progress(network_type=NETWORK_TYPE, gradient_method='newton'):
    """Print progress summary for targeted tasks."""
    results_dir = results_dir_for_gradient_method(TARGETED_RESULTS_DIR, gradient_method)
    complete = get_complete_jobs(
        n_tasks=N_TASKS,
        n_realizations=N_REALIZATIONS,
        results_dir=results_dir,
        network_type=network_type,
    )
    incomplete = get_incomplete_jobs(
        n_tasks=N_TASKS,
        n_realizations=N_REALIZATIONS,
        results_dir=results_dir,
        network_type=network_type,
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
        help='Task ID for single mode (0 to N_TASKS-1)'
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
        choices=['newton', 'newton_fd', 'jax', 'fire', 'parallel'],
        default='newton',
        help='Gradient computation method: newton (IFT, default), newton_fd (Newton loss + '
             'finite-difference gradient), jax (autodiff), fire/parallel (Cython FIRE + finite-difference)'
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

    if args.mode == 'single':
        if args.task is None:
            parser.error("--task required for single mode")
        if args.task < 0 or args.task >= N_TASKS:
            parser.error(f"--task must be between 0 and {N_TASKS-1}")

        print_targeted_tasks_summary()
        success = run_single_training(args.task, args.realization, verbose=args.verbose,
                                      gradient_method=args.gradient_method,
                                      network_type=args.network_type,
                                      overwrite=args.overwrite)
        sys.exit(0 if success else 1)

    elif args.mode == 'sequential':
        run_all_targeted(resume=args.resume, verbose=args.verbose,
                         gradient_method=args.gradient_method,
                         network_type=args.network_type,
                         overwrite=args.overwrite)

    elif args.mode == 'status':
        print_targeted_tasks_summary()
        print_targeted_progress(network_type=args.network_type, gradient_method=args.gradient_method)


if __name__ == '__main__':
    main()

