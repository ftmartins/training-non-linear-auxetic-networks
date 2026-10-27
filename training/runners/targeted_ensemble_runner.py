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
LOSS_TOL = 1e-6  # early-stop threshold; the trainers' default, passed explicitly so the
                 # runner can tell an early stop from a run that used all its steps


LEARNING_RATE = 3e-3  # starting lr / opt_fire dt_init (opt_fire hyperparameter grid sweep,
                       # tasks 3/15); decayed by training.src.lr_schedule when not using opt_fire
OPT_FIRE_FINC = 1.05   # opt_fire growth factor (same sweep) — never diverged in the grid
USE_OPT_FIRE = False   # FIRE-style adaptive optimizer vs plain gradient descent. Reverted
                       # 2026-08-15: production runs (targeted task_00/realization_02, general
                       # task idx 2) showed FIRE trapped in an exact repeating limit cycle —
                       # traced to 1-2 stiffness coordinates pinned at vmin with a genuinely
                       # ill-conditioned gradient there. Lowering opt_fire_dt_min 100x only
                       # relocated the same trap to a smaller floor (confirmed on a fresh,
                       # non-resumed rerun) — a real gradient singularity, not a step-size
                       # tuning issue. See base/config.py's USE_OPT_FIRE comment for the
                       # full writeup.
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
from base.stored_network import load_stored_network, network_fingerprint
from base.realization_record import RealizationRecord, Loss
from training.src import realization_lifecycle
from training.src.realization_lifecycle import TrainState, TrainResult, TrainingLossBelow
from training.src.run_provenance import DEFAULT_CRITICAL_KEYS, HyperparameterMismatchError
from training.src.checkpoint_manager import (
    is_training_complete,
    save_training_results,
    get_incomplete_jobs,
    get_complete_jobs,
    save_checkpoint,
    results_dir_for_gradient_method,
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
    _ROOT / 'base' / 'stored_network.py',
]

# Import training function
from training.src.training_functions import (
        finish_training_GD_auxetic_batch,
        finish_training_GD_auxetic_batch_jax,
    )
TRAINING_FUNCTIONS_AVAILABLE = True

class TargetedAuxeticTrainer:
    """The targeted auxetic physics behind training.src.realization_lifecycle: task config,
    network (stored or regenerated), initial draw, the gradient-descent trainer, and the
    final save with its checkpoint policy."""

    def __init__(self, *, task_id, realization_id, realization_seed, gradient_method,
                 network_type, learning_rate, n_steps, network_file, verbose):
        self.task_id, self.realization_id = task_id, realization_id
        self.realization_seed = realization_seed
        self.gradient_method, self.network_type = gradient_method, network_type
        self.learning_rate, self.n_steps = learning_rate, n_steps
        self.network_file, self.verbose = network_file, verbose

    def prepare(self, record):
        resume = record.resume_state()
        if resume is not None:
            print(f"Found checkpoint at step {resume.step}")
            print(f"Resuming from checkpoint...")
            self.task_config = resume.trainer_state['task_config']
        else:
            self.task_config = get_targeted_task_config(self.task_id)
        print(f"  Compression strains: {self.task_config['compression_strains']}")
        print(f"  Target Poisson ratios: {self.task_config['target_poisson_ratios']}")

        # The packing is hardware-dependent (docs/adr/0002-auxetic-ensemble-on-stored-
        # network-a.md), so runs that must share a network load it from network_file.
        if self.network_file is not None:
            self.network, self.boundary, _ = load_stored_network(self.network_file)
        else:
            self.network, self.boundary = create_auxetic_network(
                n_nodes=self.task_config['n_nodes'],
                packing_seed=self.task_config['packing_seed'],
                force_type=FORCE_TYPE,
                boundary_margin=BOUNDARY_MARGIN,
                central_force=PACKING_PARAMS['central'],
                network_type=self.network_type,
            )
        print(f"  Network created: {len(self.network.positions)} nodes, {len(self.network.edges)} edges")

    def fresh_state(self):
        n_edges = len(self.network.edges)
        initial_stiffnesses = generate_realization_stiffnesses(
            self.task_id, self.realization_seed, n_edges)
        self.network.stiffnesses = initial_stiffnesses
        self.network.save_original_parameters()
        print(f"  Stiffnesses initialized: range "
              f"[{initial_stiffnesses.min():.2e}, {initial_stiffnesses.max():.2e}]")
        return TrainState(0, initial_stiffnesses, Loss(training=np.empty(0)), trainer_state={})

    def resumed_state(self, resume):
        ckpt = resume.trainer_state
        print(f"Found checkpoint at step {ckpt['current_step']}")
        self.network.positions = ckpt['network']['positions']
        self.network.stiffnesses = ckpt['network']['stiffnesses']
        self.network.rest_lengths = ckpt['network']['rest_lengths']
        self.network.edges = ckpt['network']['edges']
        print(f"  Resuming from step {resume.step}/{self.n_steps}")
        return TrainState(resume.step, self.network.stiffnesses, resume.loss,
                          trainer_state=ckpt['history'])

    def extra_hparams(self, state):
        # Fingerprint of the network actually trained (after any checkpoint restore),
        # recorded as a critical key, so resuming onto a different network fails.
        fingerprint = network_fingerprint(self.network.positions, self.network.edges)
        print(f"  Network fingerprint: {fingerprint}")
        return {'network_fingerprint': fingerprint}

    def train(self, state, n_steps, stop_rule, on_checkpoint):
        print(f"  Training parameters:")
        print(f"    Learning rate: {self.learning_rate} (starting; see training.src.lr_schedule)")
        print(f"    Steps remaining: {n_steps:,} / {self.n_steps:,}")
        print(f"    Strain steps: {N_STRAIN_STEPS}")
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
            n_strain_steps=N_STRAIN_STEPS,
            source_compression_strain_list=compression_strains,
            desired_target_extension_list=target_extensions,
            force_tol=FORCE_TOL,
            vmin=VMIN,
            vmax=VMAX,
            task_seed=self.task_id,
            realization_seed=self.realization_id,
            save_interval=5,
            loss_tol=stop_rule.threshold,
            task_config=self.task_config,
            TARGETED_RESULTS_DIR=self.results_dir,
            network_type=self.network_type,
            **method_kwarg,
        )
        loss = np.asarray(history.get('loss', []), float)
        final = loss[-1] if len(loss) else np.nan
        if not np.isfinite(final) or np.isnan(np.asarray(self.network.stiffnesses)).any():
            status = 'nan'
        elif len(loss) < self.n_steps and stop_rule.reached(loss[0], final):
            status = 'converged'
        else:
            status = 'steps_exhausted'
        return TrainResult(TrainState(len(loss), self.network.stiffnesses,
                                      Loss(training=loss), trainer_state=history), status)

    def save_final(self, record, state, status):
        """Final results, and the completion marker only when training stopped because it
        converged. The checkpoint is the run's resume state: dropped only when training
        stopped because it converged, rewritten at the last step reached after a
        run that used all its steps (a later --steps extends from there), and left as the
        trainer's last periodic one after a NaN."""
        history = state.trainer_state
        save_training_results(
            task_seed=self.task_id, realization_seed=self.realization_id, history=history,
            network=self.network, task_config=self.task_config, boundary_dict=self.boundary,
            results_dir=self.results_dir, network_type=self.network_type)
        loss_list = history.get('loss', [])
        final_loss = loss_list[-1] if loss_list else float('nan')
        nan_stiffness = np.isnan(np.asarray(self.network.stiffnesses)).any()
        self.failed = not np.isfinite(final_loss) or nan_stiffness
        early_stop = not self.failed and len(loss_list) < self.n_steps and final_loss < LOSS_TOL
        if early_stop:
            record.drop_checkpoint()
            record.mark_complete()
        elif not self.failed:
            save_checkpoint(self.task_id, self.realization_id, history, self.network,
                            self.task_config, current_step=len(loss_list),
                            results_dir=self.results_dir, network_type=self.network_type)
        self.final_loss, self.n_steps_done = final_loss, len(loss_list)


def run_single_training(task_id, realization_id=0, verbose=False, use_checkpoint=True,
                        gradient_method='newton', network_type=NETWORK_TYPE, overwrite=False,
                        lr_override=None, steps_override=None, network_file=None,
                        results_root=None):
    """
    Run a single targeted training job with checkpoint support, through
    training.src.realization_lifecycle.

    Args:
        task_id: Task index (0 to 4)
        realization_id: Realization index (default: 0)
        network_type: 'jammed' or 'lattice' (see create_auxetic_network)
        verbose: Print detailed progress
        use_checkpoint: Ignored; the lifecycle always resumes from a checkpoint when one
            exists (kept for callers)
        gradient_method: 'newton' (IFT, default), 'newton_fd' (Newton loss + finite-difference
            gradient), 'jax' (autodiff), or 'fire'/'parallel' (Cython FIRE + finite-difference)
        overwrite: If False (default) and this job's recorded solver/gradient_method/
            learning_rate/k_min/k_max/force_tol conflict with the current run's values,
            raises HyperparameterMismatchError. If True, the job's saved state (checkpoint, results, meta) is
            wiped and restarted from scratch under the new hyperparameters — never resumed
            under a relabeled meta, which would make training_meta.json describe a
            trajectory that never actually happened.
        network_file: Stored-network file (base.stored_network) to train on instead
            of regenerating the packing; None (default) regenerates it.
        results_root: Replaces TARGETED_RESULTS_DIR from config (e.g. a scratch
            tree for a reproduction check); None (default) uses config.

    Returns:
        success: Boolean indicating success
    """
    _LR = LEARNING_RATE if lr_override is None else lr_override
    _NSTEPS = N_STEPS if steps_override is None else steps_override
    print(f"\n{'='*80}")
    print(f"Starting Targeted Task {task_id}, Realization {realization_id}")
    print(f"{'='*80}")
    root = TARGETED_RESULTS_DIR if results_root is None else Path(results_root)
    # Results are partitioned by gradient_method (newton/newton_fd/fire/parallel/jax
    # all write to the same task/realization otherwise, silently overwriting each
    # other's checkpoints and results).
    results_dir = results_dir_for_gradient_method(root / gradient_method, gradient_method)

    # realization_id (0..N_REALIZATIONS-1) is a serial index into the
    # screened-good seeds for this task, not a literal RNG seed — see
    # training/src/good_realizations.py / docs/realization_screening.md.
    # Recorded as a critical key, so a mismatch between this run's screened seed and
    # whatever seed produced a resumed checkpoint is caught like any other.
    screened_seed = get_realization_seed('auxetic_targeted', task_id, realization_id)
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
        return is_training_complete(task_id, realization_id, results_dir=results_dir,
                                    network_type=network_type)

    record = RealizationRecord.locate(results_dir, 'auxetic', task_id, realization_id,
                                      must_exist=False, network_type=network_type)
    trainer = TargetedAuxeticTrainer(
        task_id=task_id, realization_id=realization_id, realization_seed=screened_seed,
        gradient_method=gradient_method, network_type=network_type, learning_rate=_LR,
        n_steps=_NSTEPS, network_file=network_file, verbose=verbose)
    trainer.results_dir = results_dir
    start_time = time.time()
    try:
        if not TRAINING_FUNCTIONS_AVAILABLE:
            raise ImportError("Training functions not available. Check imports.")
        outcome = realization_lifecycle.run(
            record, trainer, hparams=hparams, critical_keys=DEFAULT_CRITICAL_KEYS,
            code_files=_CODE_SNAPSHOT_FILES, target_steps=_NSTEPS, overwrite=overwrite,
            stop_rule=TrainingLossBelow(LOSS_TOL), on_mismatch=refuse, done=done,
            steps_key='n_steps')
    except HyperparameterMismatchError:
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"ERROR: Targeted Task {task_id}, Realization {realization_id}")
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
    tag = 'FAILURE' if trainer.failed else 'SUCCESS'
    print(f"\n{'='*80}")
    print(f"{tag}: Targeted Task {task_id}, Realization {realization_id}")
    print(f"Time elapsed: {elapsed/60:.2f} minutes")
    if trainer.failed:
        print(f"Final loss is not finite: {trainer.final_loss}"
              if not np.isfinite(trainer.final_loss)
              else f"Final stiffnesses contain NaN (final loss {trainer.final_loss:.4e})")
    else:
        print(f"Final loss: {trainer.final_loss:.4e}")
    print(f"Training steps completed: {trainer.n_steps_done}")
    print(f"{'='*80}\n")
    return not trainer.failed


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

    for idx, (task_id, realization_id) in enumerate(jobs):
        print(f"\n[Job {idx+1}/{len(jobs)}]")
        success = run_single_training(task_id, realization_id, verbose=verbose,
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
    parser.add_argument(
        '--network-file', type=Path, default=None,
        help='Train on a stored network (e.g. base/networks/auxetic_network_A.npz) '
             'instead of regenerating the hardware-dependent packing (single mode).'
    )
    parser.add_argument(
        '--results-dir', type=Path, default=None,
        help='Replace TARGETED_RESULTS_DIR from config (single mode), e.g. a scratch '
             'tree for a reproduction check.'
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
                                      overwrite=args.overwrite,
                                      lr_override=args.learning_rate,
                                      steps_override=args.steps,
                                      network_file=args.network_file,
                                      results_root=args.results_dir)
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

