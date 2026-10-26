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
from base.realization_record import RealizationRecord, Loss
from training.src import realization_lifecycle
from training.src.realization_lifecycle import TrainState, TrainResult, TrainingLossBelow
from training.src.run_provenance import DEFAULT_CRITICAL_KEYS, HyperparameterMismatchError
from training.src.checkpoint_manager import (
    is_training_complete,
    save_training_results,
    get_incomplete_jobs,
    print_progress_summary,
    results_dir_for_gradient_method,
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


class GeneralAuxeticTrainer:
    """The general-ensemble auxetic physics behind training.src.realization_lifecycle:
    generated task config, packed network (restored from the checkpoint on resume), the
    initial draw, the gradient-descent trainer, and the final save."""

    def __init__(self, *, task_id, realization_id, realization_seed, gradient_method,
                 network_type, learning_rate, n_steps, results_dir, verbose):
        self.task_id, self.realization_id = task_id, realization_id
        self.realization_seed = realization_seed
        self.gradient_method, self.network_type = gradient_method, network_type
        self.learning_rate, self.n_steps = learning_rate, n_steps
        self.results_dir, self.verbose = results_dir, verbose

    def prepare(self, record):
        resume = record.resume_state()
        if resume is not None:
            ckpt = resume.trainer_state
            print(f"Found checkpoint at step {ckpt['current_step']}")
            print(f"Resuming from checkpoint...")
            print(ckpt)
            try:
                self.task_config = ckpt['task_config']
                assert self.task_config is not None
            except (KeyError, AssertionError) as e:
                self.task_config = generate_task_config(self.task_id)
                print(f'Regenerated task config due to checkpoint not having task config ({e!r})')
        else:
            self.task_config = generate_task_config(self.task_id)
        print(f"  Compression strains: {self.task_config['compression_strains']}")
        print(f"  Target Poisson ratios: {self.task_config['target_poisson_ratios']}")
        self.network, self.boundary = create_auxetic_network(
            n_nodes=get_n_nodes(self.task_id),
            packing_seed=self.task_config['packing_seed'],
            force_type=FORCE_TYPE,
            boundary_margin=BOUNDARY_MARGIN,
            central_force=PACKING_PARAMS['central'],
            network_type=self.network_type,
        )

    def fresh_state(self):
        print(f"  Network created: {len(self.network.positions)} nodes, {len(self.network.edges)} edges")
        initial_stiffnesses = generate_realization_stiffnesses(
            self.task_id, self.realization_seed, len(self.network.edges))
        self.network.stiffnesses = initial_stiffnesses
        self.network.save_original_parameters()
        print(f"  Stiffnesses initialized: range [{initial_stiffnesses.min():.2e}, {initial_stiffnesses.max():.2e}]")
        return TrainState(0, initial_stiffnesses, Loss(training=np.empty(0)), trainer_state={})

    def resumed_state(self, resume):
        ckpt = resume.trainer_state
        self.network.positions = ckpt['network']['positions']
        self.network.stiffnesses = ckpt['network']['stiffnesses']
        self.network.rest_lengths = ckpt['network']['rest_lengths']
        self.network.edges = ckpt['network']['edges']
        print(f"  Network restored: {len(self.network.positions)} nodes, {len(self.network.edges)} edges")
        print(f"  Resuming from step {resume.step}/{self.n_steps}")
        return TrainState(resume.step, self.network.stiffnesses, resume.loss,
                          trainer_state=ckpt['history'])

    def train(self, state, n_steps, stop_rule, on_checkpoint):
        print(f"  Training parameters:")
        print(f"    Learning rate: {self.learning_rate}")
        print(f"    Steps: {self.n_steps:,}")
        print(f"    Strain steps: {get_n_strain_steps(self.task_id)}")
        print(f"    Force tolerance: {FORCE_TOL}")
        compression_strains = self.task_config['compression_strains']
        target_extensions = compute_target_extensions(
            compression_strains, self.task_config['target_poisson_ratios'])
        if self.gradient_method == 'jax':
            train_fn = finish_training_GD_auxetic_batch_jax
            method_kwarg = {'opt_fire': USE_OPT_FIRE, 'opt_fire_finc': OPT_FIRE_FINC}
        else:
            train_fn = finish_training_GD_auxetic_batch
            method_kwarg = {'method': 'fire' if self.gradient_method in ('fire', 'parallel')
                            else self.gradient_method}
        # The trainer saves results and checkpoints itself (through checkpoint_manager,
        # i.e. the record) every save_interval steps; on_checkpoint is not used.
        history, self.network = train_fn(
            network=self.network,
            history=state.trainer_state,
            learning_rate=self.learning_rate,
            n_steps=n_steps,
            top_nodes=self.boundary['top'],
            bottom_nodes=self.boundary['bottom'],
            left_nodes=self.boundary['left'],
            right_nodes=self.boundary['right'],
            force_type=FORCE_TYPE,
            n_strain_steps=get_n_strain_steps(self.task_id),
            source_compression_strain_list=compression_strains,
            desired_target_extension_list=target_extensions,
            force_tol=FORCE_TOL,
            vmin=VMIN,
            vmax=VMAX,
            task_seed=self.task_id,
            realization_seed=self.realization_id,
            save_interval=500,
            network_type=self.network_type,
            TARGETED_RESULTS_DIR=self.results_dir,
            **method_kwarg,
        )
        loss = np.asarray(history.get('loss', []), float)
        final = loss[-1] if len(loss) else np.nan
        if not np.isfinite(final):
            status = 'nan'
        elif len(loss) < self.n_steps and stop_rule.reached(loss[0], final):
            status = 'converged'
        else:
            status = 'steps_exhausted'
        return TrainResult(TrainState(len(loss), self.network.stiffnesses,
                                      Loss(training=loss), trainer_state=history), status)

    def save_final(self, record, state, status):
        """Final results (with the task config regenerated, not the checkpoint's); the
        checkpoint is removed whatever the outcome."""
        history = state.trainer_state
        save_training_results(
            task_seed=self.task_id, realization_seed=self.realization_id, history=history,
            network=self.network, task_config=generate_task_config(self.task_id),
            boundary_dict=self.boundary, results_dir=self.results_dir,
            network_type=self.network_type)
        record.drop_checkpoint()
        self.final_loss = (history['loss'][-1] if 'loss' in history and history['loss']
                           else float('nan'))
        self.n_steps_done = len(history.get('loss', []))


def run_single_training(task_seed, realization_seed, verbose=False, use_checkpoint=True,
                        gradient_method='parallel', network_type=NETWORK_TYPE, overwrite=False,
                        lr_override=None, steps_override=None):
    """
    Run a single training job with checkpoint support.

    Args:
        task_seed: Task index (0 to N_TASKS-1)
        realization_seed: Realization index (0 to N_REALIZATIONS-1)
        verbose: Print detailed progress
        use_checkpoint: Ignored; the lifecycle always resumes from a checkpoint when one
            exists (kept for callers)
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
    _LR = LEARNING_RATE if lr_override is None else lr_override
    _NSTEPS = N_STEPS if steps_override is None else steps_override

    # Results are partitioned by gradient_method (parallel/jax write to the
    # same task/realization otherwise, silently overwriting each other's
    # checkpoints and results).
    results_dir = results_dir_for_gradient_method(RESULTS_DIR, gradient_method)

    # realization_seed (0..N_REALIZATIONS-1) is a serial index into the
    # screened-good seeds for this task, not a literal RNG seed — see
    # training/src/good_realizations.py / docs/realization_screening.md.
    # Recorded as a critical key, so a mismatch between this run's screened seed and
    # whatever seed produced a resumed checkpoint is caught like any other.
    screened_seed = get_realization_seed('auxetic_general', task_seed, realization_seed)
    optimizer = 'fire' if (gradient_method == 'jax' and USE_OPT_FIRE) else 'gradient_descent'
    hparams = {
        'learning_rate': _LR,
        'k_min': VMIN,
        'k_max': VMAX,
        'n_steps': _NSTEPS,
        'force_tol': FORCE_TOL,
        'gradient_method': gradient_method,
        'optimizer': optimizer,
        'opt_fire_finc': OPT_FIRE_FINC,
        'realization_seed': screened_seed,
    }

    def refuse(record):
        raise HyperparameterMismatchError(
            f"{record.run_dir}: recorded critical hyperparameters (solver/gradient_method/"
            f"learning_rate/k_min/k_max/force_tol/optimizer/realization_seed) differ from this "
            f"run's. Resuming would mix incompatible settings into one training trajectory. "
            f"Pass overwrite=True (--overwrite) to wipe it and restart from scratch.")

    def done(record):
        return is_training_complete(task_seed, realization_seed, results_dir=results_dir,
                                    network_type=network_type)

    record = RealizationRecord.locate(results_dir, 'auxetic', task_seed, realization_seed,
                                      must_exist=False, network_type=network_type)
    trainer = GeneralAuxeticTrainer(
        task_id=task_seed, realization_id=realization_seed, realization_seed=screened_seed,
        gradient_method=gradient_method, network_type=network_type, learning_rate=_LR,
        n_steps=_NSTEPS, results_dir=results_dir, verbose=verbose)
    start_time = time.time()
    try:
        if not TRAINING_FUNCTIONS_AVAILABLE:
            raise ImportError("Training functions not available. Check imports.")
        outcome = realization_lifecycle.run(
            record, trainer, hparams=hparams, critical_keys=DEFAULT_CRITICAL_KEYS,
            code_files=_CODE_SNAPSHOT_FILES, target_steps=_NSTEPS, overwrite=overwrite,
            stop_rule=TrainingLossBelow(1e-6), on_mismatch=refuse, done=done,
            steps_key='n_steps')
    except HyperparameterMismatchError:
        raise
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
    if outcome.stage == 'complete' and outcome.state is None:
        print(f"{'='*80}\n")
        return True

    elapsed = time.time() - start_time
    failed = not np.isfinite(trainer.final_loss)
    print(f"\n{'='*80}")
    print(f"{'FAILURE' if failed else 'SUCCESS'}: Task {task_seed}, Realization {realization_seed}")
    print(f"Time elapsed: {elapsed/60:.2f} minutes")
    print(f"Final loss is not finite: {trainer.final_loss}" if failed
          else f"Final loss: {trainer.final_loss:.4e}")
    print(f"Training steps completed: {trainer.n_steps_done}")
    print(f"{'='*80}\n")
    return not failed


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

    parser.add_argument(
        '--learning-rate', type=float, default=None,
        help='Override the nominal LEARNING_RATE from config for this run '
             '(single mode). Use with --overwrite to restart a diverged run at a lower LR.'
    )
    parser.add_argument(
        '--steps', type=int, default=None,
        help='Override N_STEPS from config for this run (single mode). n_steps is exempt '
             'from the hyperparameter-mismatch guard, so this works on a plain resume.'
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
                                      overwrite=args.overwrite,
                                      lr_override=args.learning_rate,
                                      steps_override=args.steps)
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
