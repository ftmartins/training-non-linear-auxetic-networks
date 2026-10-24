#!/usr/bin/env python
"""
Post-training timestep-sweep analysis.

Run after a training job (targeted/ensemble auxetic, or allosteric) completes:
selects a log-spaced-by-loss subset of training steps, recomputes losses,
trajectories, and reduced-elastic/cost Hessian eigenpairs at those steps, and
saves everything to `timestep_sweep.npz` (+ `timestep_sweep_meta.json`) in the
same realization directory the training job wrote to.

See analysis/timestep_sweep.py for the underlying pipeline.

Usage:
    python post_training_sweep.py --task-type targeted   --task 3 --realization 2
    python post_training_sweep.py --task-type ensemble   --task 3 --realization 2
    python post_training_sweep.py --task-type allosteric --task 3 --realization 2 \
        [--geometry 0] [--targeted-ensemble]
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent.parent  # project root
_SRC  = Path(__file__).parent.parent / 'src'  # training/src/
for _p in (str(_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.timestep_sweep import (
    sweep_auxetic, sweep_allosteric, save_sweep_results,
)


def _run_auxetic(task_type, task, realization, results_dir, n_thresh_steps,
                 eps_min, n_traj_steps, k_eigs, force_type, network_type, gradient_method,
                 n_hessian_traj_steps):
    from base.elastic_network import ElasticNetwork
    from base.realization_record import RealizationRecord
    from training.src.checkpoint_manager import results_dir_for_gradient_method

    if results_dir is None:
        if task_type == 'targeted':
            from training.src.targeted_task_generator import TARGETED_RESULTS_DIR
            results_dir = TARGETED_RESULTS_DIR
        else:
            from base.config import RESULTS_DIR
            results_dir = RESULTS_DIR
    # Idempotent: safe even if results_dir (default or an explicit
    # --results-dir override) already points at a gradient_method
    # subdirectory — never produces a doubled-up .../newton/newton/... path.
    results_dir = results_dir_for_gradient_method(results_dir, gradient_method)
    record = RealizationRecord.locate(results_dir, 'auxetic', task, realization,
                                      must_exist=False, network_type=network_type)
    result_path = record.run_dir

    net = record.network
    network = ElasticNetwork(
        positions=net.positions, edges=net.edges,
        rest_lengths=net.rest_lengths, stiffnesses=record.current_stiffnesses,
    )
    boundary = record.boundary
    task_config = record.task.config

    loss_traj      = record.loss.training
    stiffness_traj = record.stiffness_trajectory.stiffnesses
    positions_traj = record.positions_trajectory

    if 'n_strain_steps' in task_config:
        n_strain_steps = task_config['n_strain_steps']
    else:
        from base.config import get_n_strain_steps
        n_strain_steps = get_n_strain_steps(task)
    from base.config import FORCE_TOL

    results = sweep_auxetic(
        network, task_config, boundary, stiffness_traj, positions_traj, loss_traj,
        n_thresh_steps=n_thresh_steps, eps_min=eps_min, n_traj_steps=n_traj_steps,
        n_hessian_traj_steps=n_hessian_traj_steps,
        k_eigs=k_eigs, force_type=force_type, n_strain_steps=n_strain_steps, tol=FORCE_TOL,
        gradient_method=gradient_method,
    )
    return result_path, results


def _run_allosteric(task, realization, geometry, targeted_ensemble, output_dir,
                    n_thresh_steps, eps_min, k_eigs, n_hessian_traj_steps,
                    cost_hessian_solver='match'):
    from training.runners.allosteric_trainer import (
        STRAIN_INPUT, STRAIN_INPUT2, NSTEPS_TASK1, NSTEPS_TASK2, TARGETED_ENSEMBLE,
    )

    from base.realization_record import RealizationRecord

    record = RealizationRecord.locate(
        output_dir, 'allosteric', ('targeted' if targeted_ensemble else geometry, task),
        realization, must_exist=False)
    result_path = record.run_dir

    network = record.network
    nodes, incidence_matrix, eq_lengths = (
        network.nodes, network.incidence_matrix, network.eq_lengths)
    stiffness_traj    = record.stiffness_trajectory.stiffnesses
    steps             = record.stiffness_trajectory.steps
    mse1, mse2        = (np.ascontiguousarray(c) for c in record.loss.per_subtask.T)

    # Evaluate the 'after' cost Hessian at the best-loss stiffnesses K* (ADR 0004), never
    # at the coarse checkpoint-grid argmin stiffness_traj[t_indices[-1]] -- they differ
    # for ~23% of converged realizations (best_stiffnesses_audit.md). The record raises
    # when K* cannot be determined, so no sweep is written for such a run.
    best_stiffnesses = record.best_loss_stiffnesses
    print(f"  cost_hessian_after: at K* (step {record.best_loss_step})", flush=True)

    strain_output, strain_output2 = record.task.strain_output, record.task.strain_output2
    tod  = (1 + strain_output)  * np.linalg.norm(nodes[3] - nodes[2])
    tod2 = (1 + strain_output2) * np.linalg.norm(nodes[3] - nodes[2])
    dinputdistance  = STRAIN_INPUT  * np.linalg.norm(nodes[0] - nodes[1])
    dinputdistance2 = STRAIN_INPUT2 * np.linalg.norm(nodes[0] - nodes[1])

    task_config = {
        'tod': tod, 'tod2': tod2,
        'dinputdistance': dinputdistance, 'dinputdistance2': dinputdistance2,
        'nsteps': NSTEPS_TASK1, 'nsteps2': NSTEPS_TASK2,
    }

    # Match training's actual physics backend for the stiffness-loss/elastic-
    # Hessian recompute (see sweep_allosteric's solver= docstring) instead of
    # silently falling back to whatever DEFAULT_SOLVER is today. Older
    # realizations trained before allosteric_trainer.py started writing this
    # file have no record of it -- fall back to None (evaluate_actuation's own
    # default) and warn, since that's the pre-existing (unverified) behavior.
    meta_path = result_path / 'training_meta.json'
    if record.has('config'):
        solver = record.config['solver']
    else:
        from training.runners.allosteric_trainer import DEFAULT_SOLVER
        solver = None
        print(f"  WARNING: {meta_path} not found (pre-dates training_meta.json); "
              f"recomputing with evaluate_actuation's default (currently "
              f"{DEFAULT_SOLVER!r}), which may not match what actually trained "
              f"this realization.")

    # Cost-Hessian backend. By default (`match`) the cost Hessian is computed
    # with the SAME physics the run trained with: a LAMMPS-trained realization
    # gets the finite-difference LAMMPS cost Hessian (_compute_cost_hessian_lammps),
    # not the JAX-FIRE autodiff one. This matters because near "best loss" the
    # LAMMPS and JAX-FIRE equilibria differ slightly, so the JAX-FIRE loss is not
    # stationary at a LAMMPS-trained K* — its cost-Hessian eigenvectors (esp. the
    # near-flat directions the figures care about) can misrepresent the true
    # LAMMPS landscape at that point. `jax_fire` forces the (cheap, autodiff)
    # JAX path regardless; `lammps` forces the FD-LAMMPS path regardless.
    # NOTE: the FD-LAMMPS cost Hessian is O(n_edges) LAMMPS actuations per
    # gradient, one gradient per Lanczos matvec -> hours per realization, and it
    # dominates the sweep's wall time. --k-eigs defaults to 4 and the sweep no
    # longer computes the combined (mean-over-subtasks) Hessian (only per-subtask
    # task1/task2) for this reason.
    import functools
    from analysis.timestep_sweep import _compute_cost_hessian_lammps

    if cost_hessian_solver == 'jax_fire':
        cost_hessian_fn = None                       # sweep default (_compute_cost_hessian_jax)
    elif cost_hessian_solver == 'lammps' or (cost_hessian_solver == 'match' and solver == 'lammps'):
        cost_hessian_fn = functools.partial(_compute_cost_hessian_lammps, solver='lammps')
    else:                                            # match & solver != 'lammps' (or None)
        cost_hessian_fn = None
    print(f"  cost-Hessian backend: "
          f"{'FD-LAMMPS (solver-matched)' if cost_hessian_fn is not None else 'JAX-FIRE autodiff'} "
          f"(--cost-hessian-solver={cost_hessian_solver}, training solver={solver!r})", flush=True)

    results = sweep_allosteric(
        nodes, incidence_matrix, eq_lengths, task_config,
        stiffness_traj, steps, mse1, mse2,
        n_thresh_steps=n_thresh_steps, eps_min=eps_min, k_eigs=k_eigs,
        n_hessian_traj_steps=n_hessian_traj_steps, solver=solver,
        cost_hessian_fn=cost_hessian_fn, best_stiffnesses=best_stiffnesses,
    )
    return result_path, results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task-type', choices=['targeted', 'ensemble', 'allosteric'], required=True)
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--realization', type=int, required=True)
    parser.add_argument('--geometry', type=int, default=0, help='allosteric only')
    parser.add_argument('--targeted-ensemble', action='store_true', help='allosteric only')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='auxetic only; override the default results directory')
    parser.add_argument('--output-dir', type=str,
                        default='/data2/shared/felipetm/allosteric_nets_aug',
                        help='allosteric only')
    parser.add_argument('--n-thresh-steps', type=int, default=50)
    parser.add_argument('--eps-min', type=float, default=1e-8)
    parser.add_argument('--n-traj-steps', type=int, default=100, help='auxetic only')
    parser.add_argument('--n-hessian-traj-steps', type=int, default=20,
                        help='number of linearly-spaced points along each recomputed '
                             'compression/actuation trajectory at which the elastic Hessian '
                             'spectrum is evaluated (auxetic and allosteric)')
    parser.add_argument('--k-eigs', type=int, default=4,
                        help="number of top (largest positive) cost-Hessian eigenpairs to "
                             "compute; does not affect the elastic Hessian, which always "
                             "returns its full spectrum. Kept small (default 4) because the "
                             "FD-LAMMPS cost Hessian costs one O(n_edges)-actuation gradient "
                             "per Lanczos matvec.")
    parser.add_argument('--cost-hessian-solver', choices=['match', 'jax_fire', 'lammps'],
                        default='match', help='allosteric only; which physics backend to build '
                             'the cost Hessian with. "match" (default): FD-LAMMPS when the run '
                             'trained with LAMMPS (per training_meta.json), JAX-FIRE autodiff '
                             'otherwise. "jax_fire"/"lammps": force that backend. FD-LAMMPS is '
                             'much slower but is stationary-consistent with a LAMMPS-trained K*.')
    parser.add_argument('--force-type', type=str, default='quadratic', help='auxetic only')
    parser.add_argument('--network-type', choices=['jammed', 'lattice'], default=None,
                        help="auxetic only; 'jammed' or 'lattice' (default: from config). "
                             "Must match the network_type the training job used.")
    parser.add_argument('--gradient-method', type=str, default='newton',
                        help="auxetic only; gradient method subdirectory the training job used "
                             "(e.g. 'newton', 'fire', 'parallel', 'jax')")
    args = parser.parse_args()

    if args.network_type is None:
        from base.config import NETWORK_TYPE
        args.network_type = NETWORK_TYPE

    #try:
    if args.task_type in ('targeted', 'ensemble'):
        result_path, results = _run_auxetic(
                args.task_type, args.task, args.realization, args.results_dir,
                args.n_thresh_steps, args.eps_min, args.n_traj_steps, args.k_eigs,
                args.force_type, args.network_type, args.gradient_method,
                args.n_hessian_traj_steps,
            )
    else:
        result_path, results = _run_allosteric(
                args.task, args.realization, args.geometry, args.targeted_ensemble,
                args.output_dir, args.n_thresh_steps, args.eps_min, args.k_eigs,
                args.n_hessian_traj_steps, cost_hessian_solver=args.cost_hessian_solver,
            )
    #except Exception as e:
    #    print(f"post_training_sweep FAILED: {e!r}", file=sys.stderr)
    #    return 1

    if args.task_type in ('targeted', 'ensemble'):
        from training.src.checkpoint_manager import _nt_filename
        sweep_filename = _nt_filename('timestep_sweep.npz', args.network_type)
        meta_filename = _nt_filename('timestep_sweep_meta.json', args.network_type)
    else:
        sweep_filename = 'timestep_sweep.npz'
        meta_filename = 'timestep_sweep_meta.json'

    out_path = result_path / sweep_filename
    save_sweep_results(out_path, **results)

    meta = {
        'task_type': args.task_type,
        'task': args.task,
        'realization': args.realization,
        'n_thresh_steps': args.n_thresh_steps,
        'eps_min': args.eps_min,
        'n_timesteps_selected': int(len(results['t_indices'])),
        'k_eigs': args.k_eigs,
        'cost_hessian_combined': False,   # combined mean-over-subtasks Hessian no longer computed
    }
    if args.task_type in ('targeted', 'ensemble'):
        meta['network_type'] = args.network_type
    else:
        meta['cost_hessian_solver'] = args.cost_hessian_solver
    with open(result_path / meta_filename, 'w') as fh:
        json.dump(meta, fh, indent=2)

    recomputed = np.asarray(results['recomputed_stiffness_loss'])
    if args.task_type == 'allosteric':
        # stored (mse1+mse2) is a sum over the two actuation tasks
        recomputed = recomputed.sum(axis=1)
    stored = np.asarray(results['stored_loss'])
    rel_err = np.abs(recomputed - stored) / np.maximum(np.abs(stored), 1e-300)
    print(f"Saved {out_path} ({meta['n_timesteps_selected']} timesteps).")
    print(f"  stiffness-vs-stored loss: mean rel. error = {rel_err.mean():.3e}, "
          f"max = {rel_err.max():.3e}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
