#!/usr/bin/env python
"""
Verify + plot loss reconstruction for a completed timestep sweep.

Loads `timestep_sweep.npz` (written by post_training_sweep.py), independently
re-reads the stiffness trajectory fresh from disk (a separate file read, not
reusing anything the sweep script held in memory) and recomputes loss a third
way at the same selected timesteps, to check serialization round-trip
fidelity. Produces a scatter plot comparing all three recomputed series
against the stored loss:
  1. recomputed_stiffness_loss  — from post_training_sweep.py (right after training)
  2. recomputed_trajectory_loss — from post_training_sweep.py (right after training)
  3. reloaded_stiffness_loss    — recomputed here, from stiffnesses reloaded fresh from disk

Usage:
    python verify_and_plot_loss.py --task-type targeted   --task 3 --realization 2
    python verify_and_plot_loss.py --task-type ensemble   --task 3 --realization 2
    python verify_and_plot_loss.py --task-type allosteric --task 3 --realization 2 \
        [--geometry 0] [--targeted-ensemble]
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = Path(__file__).parent.parent.parent
_SRC  = Path(__file__).parent.parent / 'src'
for _p in (str(_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.timestep_sweep import load_sweep_results


def _reload_auxetic_loss(task_type, task, realization, results_dir, t_indices, force_type):
    from base.elastic_network import ElasticNetwork
    from training.src.checkpoint_manager import get_training_result_path
    from training.src.training_functions import poisson_loss_batch_parallel

    if results_dir is None:
        if task_type == 'targeted':
            from training.src.targeted_task_generator import TARGETED_RESULTS_DIR
            results_dir = TARGETED_RESULTS_DIR
        else:
            from base.config import RESULTS_DIR
            results_dir = RESULTS_DIR
    result_path = get_training_result_path(task, realization, results_dir)

    with open(result_path / 'final_network.pkl', 'rb') as fh:
        net_dict = pickle.load(fh)
    network = ElasticNetwork(
        positions=net_dict['positions'], edges=net_dict['edges'],
        rest_lengths=net_dict['rest_lengths'], stiffnesses=net_dict['stiffnesses'],
    )
    with open(result_path / 'boundary_nodes.json', 'r') as fh:
        boundary = {k: np.asarray(v, dtype=int) for k, v in json.load(fh).items()}
    with open(result_path / 'task_config.json', 'r') as fh:
        task_config = json.load(fh)

    compression_strains   = list(task_config['compression_strains'])
    target_poisson_ratios = list(task_config['target_poisson_ratios'])
    if 'n_strain_steps' in task_config:
        n_strain_steps = task_config['n_strain_steps']
    else:
        from base.config import get_n_strain_steps
        n_strain_steps = get_n_strain_steps(task)
    from base.config import FORCE_TOL

    # Fresh re-read from disk, independent of post_training_sweep.py's in-memory state.
    stiffness_traj = np.load(result_path / 'stiffness_trajectory.npy')

    reloaded_loss = np.full(len(t_indices), np.nan)
    for i, t in enumerate(t_indices):
        network.stiffnesses = np.asarray(stiffness_traj[t], dtype=float)
        mse, _ = poisson_loss_batch_parallel(
            network, target_poisson_ratios,
            boundary['top'], boundary['bottom'], boundary['left'], boundary['right'],
            compression_strains, n_strain_steps=n_strain_steps,
            force_type=force_type, tol=FORCE_TOL,
        )
        reloaded_loss[i] = mse
    return result_path, reloaded_loss


def _reload_allosteric_loss(task, realization, geometry, targeted_ensemble, output_dir, t_indices):
    from training.runners.allosteric_trainer import (
        STRAIN_INPUT, STRAIN_INPUT2, NSTEPS_TASK1, NSTEPS_TASK2, evaluate_actuation,
    )

    geom_dir = 'geometry_targeted' if targeted_ensemble else f'geometry_{geometry}'
    result_path = Path(output_dir) / geom_dir / f'task_{task}' / f'realization_{realization}'

    nodes            = np.load(result_path / 'nodes.npy')
    incidence_matrix = np.load(result_path / 'incidence_matrix.npy')
    # Fresh re-read from disk, independent of post_training_sweep.py's in-memory state.
    stiffness_traj   = np.load(result_path / 'stiffnesses_traj.npy')
    gseed, strain_output2, strain_output = np.loadtxt(result_path / 'tasks.txt')

    tod  = (1 + strain_output)  * np.linalg.norm(nodes[3] - nodes[2])
    tod2 = (1 + strain_output2) * np.linalg.norm(nodes[3] - nodes[2])
    dx  = (STRAIN_INPUT  * np.linalg.norm(nodes[0] - nodes[1])) / NSTEPS_TASK1
    dx2 = (STRAIN_INPUT2 * np.linalg.norm(nodes[0] - nodes[1])) / NSTEPS_TASK2

    reloaded_loss = np.full((len(t_indices), 2), np.nan)
    for i, t in enumerate(t_indices):
        k_t = np.asarray(stiffness_traj[t], dtype=float)
        mse1, _, _ = evaluate_actuation(nodes, incidence_matrix, k_t, tod,  dx,  NSTEPS_TASK1)
        mse2, _, _ = evaluate_actuation(nodes, incidence_matrix, k_t, tod2, dx2, NSTEPS_TASK2)
        reloaded_loss[i] = (mse1, mse2)
    return result_path, reloaded_loss


def _make_scatter(stored, series_dict, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    lims = [max(min(stored.min(), *(np.nanmin(v) for v in series_dict.values())), 1e-300),
            max(stored.max(), *(np.nanmax(v) for v in series_dict.values()))]
    ax.plot(lims, lims, 'k--', linewidth=1, label='y = x', zorder=1)
    markers = ['o', 's', '^']
    for (label, values), marker in zip(series_dict.items(), markers):
        ax.scatter(stored, values, label=label, marker=marker, alpha=0.7, zorder=2)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('stored loss'); ax.set_ylabel('recomputed loss')
    ax.set_title('Loss reconstruction check')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task-type', choices=['targeted', 'ensemble', 'allosteric'], required=True)
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--realization', type=int, required=True)
    parser.add_argument('--geometry', type=int, default=0, help='allosteric only')
    parser.add_argument('--targeted-ensemble', action='store_true', help='allosteric only')
    parser.add_argument('--results-dir', type=str, default=None, help='auxetic only')
    parser.add_argument('--output-dir', type=str,
                        default='/data2/shared/felipetm/allosteric_nets', help='allosteric only')
    parser.add_argument('--force-type', type=str, default='quadratic', help='auxetic only')
    args = parser.parse_args()

    if args.task_type in ('targeted', 'ensemble'):
        from training.src.checkpoint_manager import get_training_result_path
        if args.results_dir is None:
            if args.task_type == 'targeted':
                from training.src.targeted_task_generator import TARGETED_RESULTS_DIR
                results_dir = TARGETED_RESULTS_DIR
            else:
                from base.config import RESULTS_DIR
                results_dir = RESULTS_DIR
        else:
            results_dir = args.results_dir
        result_path = get_training_result_path(args.task, args.realization, results_dir)
    else:
        geom_dir = 'geometry_targeted' if args.targeted_ensemble else f'geometry_{args.geometry}'
        result_path = (Path(args.output_dir) / geom_dir /
                       f'task_{args.task}' / f'realization_{args.realization}')

    sweep_path = result_path / 'timestep_sweep.npz'
    if not sweep_path.exists():
        print(f"verify_and_plot_loss: {sweep_path} not found "
              f"(run post_training_sweep.py first)", file=sys.stderr)
        return 1

    sweep = load_sweep_results(sweep_path)
    t_indices = sweep['t_indices']
    stored = np.asarray(sweep['stored_loss'])

    recomputed_stiffness  = np.asarray(sweep['recomputed_stiffness_loss'])
    recomputed_trajectory = np.asarray(sweep['recomputed_trajectory_loss'])

    if args.task_type in ('targeted', 'ensemble'):
        _, reloaded = _reload_auxetic_loss(
            args.task_type, args.task, args.realization, args.results_dir,
            t_indices, args.force_type)
    else:
        _, reloaded = _reload_allosteric_loss(
            args.task, args.realization, args.geometry, args.targeted_ensemble,
            args.output_dir, t_indices)
        # Combine per-task (mse1, mse2) series into the stored (mse1+mse2) convention.
        recomputed_stiffness  = recomputed_stiffness.sum(axis=1)
        recomputed_trajectory = recomputed_trajectory.sum(axis=1)
        reloaded               = reloaded.sum(axis=1)

    series = {
        'from stiffness (right after training)': recomputed_stiffness,
        'from trajectory (right after training)': recomputed_trajectory,
        'from stiffnesses loaded from file': reloaded,
    }

    print(f"{'series':<42}{'mean rel. err':>16}{'max rel. err':>16}")
    for label, values in series.items():
        rel_err = np.abs(values - stored) / np.maximum(np.abs(stored), 1e-300)
        print(f"{label:<42}{rel_err.mean():>16.3e}{rel_err.max():>16.3e}")

    out_path = result_path / 'loss_reconstruction_scatter.png'
    _make_scatter(stored, series, out_path)
    print(f"Saved {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
