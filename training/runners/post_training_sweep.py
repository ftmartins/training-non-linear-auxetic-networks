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
    from training.src.checkpoint_manager import get_training_result_path, _nt_filename
    from training.src.data_loader import load_loss_trajectory, load_stiffness_trajectory

    if results_dir is None:
        if task_type == 'targeted':
            from training.src.targeted_task_generator import TARGETED_RESULTS_DIR
            results_dir = TARGETED_RESULTS_DIR
        else:
            from base.config import RESULTS_DIR
            results_dir = RESULTS_DIR
    # Results are partitioned by gradient_method (newton/fire/parallel/jax
    # write to separate subdirectories to avoid clobbering each other).
    results_dir = Path(results_dir) / gradient_method
    result_path = get_training_result_path(task, realization, results_dir)

    with open(result_path / _nt_filename('final_network.pkl', network_type), 'rb') as fh:
        net_dict = pickle.load(fh)
    network = ElasticNetwork(
        positions=net_dict['positions'], edges=net_dict['edges'],
        rest_lengths=net_dict['rest_lengths'], stiffnesses=net_dict['stiffnesses'],
    )

    with open(result_path / _nt_filename('boundary_nodes.json', network_type), 'r') as fh:
        boundary = {k: np.asarray(v, dtype=int) for k, v in json.load(fh).items()}

    with open(result_path / _nt_filename('task_config.json', network_type), 'r') as fh:
        task_config = json.load(fh)

    loss_traj      = load_loss_trajectory(task, realization, results_dir=results_dir, network_type=network_type)
    stiffness_traj = load_stiffness_trajectory(task, realization, results_dir=results_dir, network_type=network_type)

    if 'n_strain_steps' in task_config:
        n_strain_steps = task_config['n_strain_steps']
    else:
        from base.config import get_n_strain_steps
        n_strain_steps = get_n_strain_steps(task)
    from base.config import FORCE_TOL

    results = sweep_auxetic(
        network, task_config, boundary, stiffness_traj, loss_traj,
        n_thresh_steps=n_thresh_steps, eps_min=eps_min, n_traj_steps=n_traj_steps,
        n_hessian_traj_steps=n_hessian_traj_steps,
        k_eigs=k_eigs, force_type=force_type, n_strain_steps=n_strain_steps, tol=FORCE_TOL,
    )
    return result_path, results


def _run_allosteric(task, realization, geometry, targeted_ensemble, output_dir,
                    n_thresh_steps, eps_min, k_eigs, n_hessian_traj_steps):
    from training.runners.allosteric_trainer import (
        STRAIN_INPUT, STRAIN_INPUT2, NSTEPS_TASK1, NSTEPS_TASK2, TARGETED_ENSEMBLE,
    )

    geom_dir = 'geometry_targeted' if targeted_ensemble else f'geometry_{geometry}'
    result_path = Path(output_dir) / geom_dir / f'task_{task}' / f'realization_{realization}'

    nodes             = np.load(result_path / 'nodes.npy')
    incidence_matrix  = np.load(result_path / 'incidence_matrix.npy')
    eq_lengths        = np.load(result_path / 'eq_lengths.npy')
    stiffness_traj    = np.load(result_path / 'stiffnesses_traj.npy')
    steps             = np.load(result_path / 'stiffnesses_traj_steps.npy')
    mse1              = np.load(result_path / 'mse1.npy')
    mse2              = np.load(result_path / 'mse2.npy')

    gseed, strain_output2, strain_output = np.loadtxt(result_path / 'tasks.txt')
    tod  = (1 + strain_output)  * np.linalg.norm(nodes[3] - nodes[2])
    tod2 = (1 + strain_output2) * np.linalg.norm(nodes[3] - nodes[2])
    dinputdistance  = STRAIN_INPUT  * np.linalg.norm(nodes[0] - nodes[1])
    dinputdistance2 = STRAIN_INPUT2 * np.linalg.norm(nodes[0] - nodes[1])

    task_config = {
        'tod': tod, 'tod2': tod2,
        'dinputdistance': dinputdistance, 'dinputdistance2': dinputdistance2,
        'nsteps': NSTEPS_TASK1, 'nsteps2': NSTEPS_TASK2,
    }

    results = sweep_allosteric(
        nodes, incidence_matrix, eq_lengths, task_config,
        stiffness_traj, steps, mse1, mse2,
        n_thresh_steps=n_thresh_steps, eps_min=eps_min, k_eigs=k_eigs,
        n_hessian_traj_steps=n_hessian_traj_steps,
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
                        default='/data2/shared/felipetm/allosteric_nets',
                        help='allosteric only')
    parser.add_argument('--n-thresh-steps', type=int, default=50)
    parser.add_argument('--eps-min', type=float, default=1e-8)
    parser.add_argument('--n-traj-steps', type=int, default=100, help='auxetic only')
    parser.add_argument('--n-hessian-traj-steps', type=int, default=20,
                        help='number of linearly-spaced points along each recomputed '
                             'compression/actuation trajectory at which the elastic Hessian '
                             'spectrum is evaluated (auxetic and allosteric)')
    parser.add_argument('--k-eigs', type=int, default=10,
                        help="number of top (largest positive) cost-Hessian eigenpairs to "
                             "compute; does not affect the elastic Hessian, which always "
                             "returns its full spectrum")
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
                args.n_hessian_traj_steps,
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
    }
    if args.task_type in ('targeted', 'ensemble'):
        meta['network_type'] = args.network_type
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
