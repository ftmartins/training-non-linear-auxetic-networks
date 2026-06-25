#!/usr/bin/env python3
"""
compute_trajectories.py — pre-compute and save quasistatic trajectories for all
trained networks in a given data directory.

Usage (local):
    python compute_trajectories.py --data_dir data/auxetic_nets/targeted_results_sqr
    python compute_trajectories.py --data_dir data/allosteric_nets --type allosteric

Usage (SLURM, via submit_figure_cache.sh):
    python compute_trajectories.py --task T --real R --data_dir <dir> [options]

Trajectories are saved as:
    <data_dir>/task_XX/realization_XX/trajectory.npz

For allosteric networks:
    <data_dir>/geometry_XX/task_XX/realization_XX/trajectory.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure project root on sys.path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.data_io import load_auxetic_network, load_allosteric_network
from analysis.trajectory import compute_auxetic_trajectory, save_trajectory
from base.config import BOUNDARY_MARGIN, FORCE_TYPE, get_n_strain_steps


def _process_auxetic(task_seed, real_seed, data_dir, n_steps, force_type, tol, overwrite):
    path = (Path(data_dir)
            / f'task_{task_seed:02d}'
            / f'realization_{real_seed:02d}')
    out_file = path / 'trajectory.npz'

    if out_file.exists() and not overwrite:
        print(f"  Skip task={task_seed} real={real_seed}: already exists", flush=True)
        return

    try:
        network, boundary = load_auxetic_network(task_seed, real_seed, data_dir)
    except FileNotFoundError:
        print(f"  WARN: no network at {path}", flush=True)
        return

    # Use the first compression strain from the task config for the trajectory
    from training.src.task_generator import generate_task_config
    task_cfg = generate_task_config(task_seed)
    compression_strains = task_cfg['compression_strains']

    for subtask_idx, cs in enumerate(compression_strains):
        out = path / f'trajectory_subtask{subtask_idx}.npz'
        if out.exists() and not overwrite:
            print(f"    Skip subtask {subtask_idx}", flush=True)
            continue
        print(f"  task={task_seed} real={real_seed} subtask={subtask_idx} cs={cs:.3f}", flush=True)
        traj = compute_auxetic_trajectory(network, cs, boundary,
                                          n_steps=n_steps, force_type=force_type, tol=tol)
        save_trajectory(traj, out)
        print(f"    Saved → {out}", flush=True)


def _discover_auxetic(data_dir):
    """Find all (task, real) pairs in data_dir."""
    data_dir = Path(data_dir)
    pairs = []
    for task_path in sorted(data_dir.glob('task_*')):
        task_seed = int(task_path.name.split('_')[1])
        for real_path in sorted(task_path.glob('realization_*')):
            real_seed = int(real_path.name.split('_')[1])
            pairs.append((task_seed, real_seed))
    return pairs


def main():
    parser = argparse.ArgumentParser(description='Pre-compute and save quasistatic trajectories.')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--type',     type=str, default='auxetic',
                        choices=['auxetic', 'allosteric'])
    parser.add_argument('--task',     type=int, default=None,
                        help='Single task index (cluster use)')
    parser.add_argument('--real',     type=int, default=None,
                        help='Single realization index (cluster use)')
    parser.add_argument('--n_steps',  type=int, default=None,
                        help='Trajectory steps (default: get_n_strain_steps(task))')
    parser.add_argument('--force_type', type=str, default=FORCE_TYPE)
    parser.add_argument('--tol',      type=float, default=1e-6)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    if args.type == 'auxetic':
        if args.task is not None and args.real is not None:
            n_steps = args.n_steps or get_n_strain_steps(args.task)
            _process_auxetic(args.task, args.real, args.data_dir,
                             n_steps, args.force_type, args.tol, args.overwrite)
        else:
            pairs = _discover_auxetic(args.data_dir)
            print(f"Found {len(pairs)} (task, real) pairs in {args.data_dir}")
            for task_seed, real_seed in pairs:
                n_steps = args.n_steps or get_n_strain_steps(task_seed)
                _process_auxetic(task_seed, real_seed, args.data_dir,
                                 n_steps, args.force_type, args.tol, args.overwrite)
    else:
        print("Allosteric trajectory computation not yet implemented in this script.")
        print("Use analysis/compute_actuation_modes.py for allosteric networks.")


if __name__ == '__main__':
    main()
