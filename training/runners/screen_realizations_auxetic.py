#!/usr/bin/env python
"""
Realization screening for auxetic training (see docs/realization_screening.md).

Runs a short training trial for one (task, candidate) pair, using the exact
same optimizer/hyperparameters the real targeted/general runner would use
(imported from that runner's own module, so screening always tracks whatever
the production default is). Writes one result JSON per candidate;
training/runners/screen_aggregate.py then picks the best N_KEEP candidates
per task and writes the good_realizations lookup.

Usage (SLURM array): SLURM_ARRAY_TASK_ID selects (task, candidate) from the
flattened grid for --kind {targeted,general}.

  python screen_realizations_auxetic.py --kind targeted --results-dir <dir>
  python screen_realizations_auxetic.py --kind general  --results-dir <dir>
"""
import argparse
import os
import sys
import json
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from base.config import FORCE_TYPE, BOUNDARY_MARGIN, FORCE_TOL, PACKING_PARAMS, NETWORK_TYPE
from base.network_utils import create_auxetic_network
from training.src.task_generator import (
    generate_task_config, generate_realization_stiffnesses, compute_target_extensions,
)
from training.src.targeted_task_generator import get_targeted_task_config, TARGETED_TASKS
from training.src.training_functions import finish_training_GD_auxetic_batch_jax
from training.src.good_realizations import SCREEN_SEED_BASE

N_STEPS = 150
POOL_SIZE = 15
VMIN, VMAX = 1e-4, 1e2

# 24 tasks for targeted (matches targeted_task_generator.py's TARGETED_TASKS);
# 30 tasks for general (matches base.config.N_TASKS).
N_TARGETED_TASKS = len(TARGETED_TASKS)


def build_grid(kind):
    if kind == 'targeted':
        return list(range(N_TARGETED_TASKS))
    elif kind == 'general':
        from base.config import N_TASKS
        return list(range(N_TASKS))
    raise ValueError(f"kind must be 'targeted' or 'general', got {kind!r}")


def _optimizer_kwargs(kind):
    """Import the production runner's own LEARNING_RATE/OPT_FIRE_FINC/USE_OPT_FIRE
    so screening always matches whatever the real run would use."""
    if kind == 'targeted':
        from training.runners import targeted_ensemble_runner as runner
    else:
        from training.runners import ensemble_runner as runner
    return runner.LEARNING_RATE, dict(opt_fire=runner.USE_OPT_FIRE, opt_fire_finc=runner.OPT_FIRE_FINC)


def run_one(kind, task_id, candidate_idx, results_dir):
    seed = SCREEN_SEED_BASE + task_id * 10_000 + candidate_idx
    learning_rate, opt_kwargs = _optimizer_kwargs(kind)

    if kind == 'targeted':
        task_config = get_targeted_task_config(task_id)
    else:
        task_config = generate_task_config(task_id)
        from base.config import get_n_nodes, get_n_strain_steps
        task_config = dict(task_config)
        task_config['n_nodes'] = get_n_nodes(task_id)
        task_config['n_strain_steps'] = get_n_strain_steps(task_id)

    network, boundary_dict = create_auxetic_network(
        n_nodes=task_config['n_nodes'], packing_seed=task_config['packing_seed'],
        force_type=FORCE_TYPE, boundary_margin=BOUNDARY_MARGIN,
        central_force=PACKING_PARAMS['central'], network_type=NETWORK_TYPE,
    )
    n_edges = len(network.edges)
    network.stiffnesses = generate_realization_stiffnesses(task_id, seed, n_edges)
    network.save_original_parameters()
    compression_strains = task_config['compression_strains']
    target_extensions = compute_target_extensions(compression_strains, task_config['target_poisson_ratios'])

    t0 = time.time()
    history, _ = finish_training_GD_auxetic_batch_jax(
        network=network, history={}, learning_rate=learning_rate, n_steps=N_STEPS,
        top_nodes=boundary_dict['top'], bottom_nodes=boundary_dict['bottom'],
        left_nodes=boundary_dict['left'], right_nodes=boundary_dict['right'],
        force_type=FORCE_TYPE, n_strain_steps=task_config['n_strain_steps'],
        source_compression_strain_list=compression_strains,
        desired_target_extension_list=target_extensions,
        force_tol=FORCE_TOL, vmin=VMIN, vmax=VMAX,
        task_seed=None, realization_seed=None, save_interval=N_STEPS + 1,
        **opt_kwargs,
    )
    dt = time.time() - t0

    losses = np.array(history['loss'])
    is_nan = bool(np.any(np.isnan(losses))) or len(losses) == 0
    result = {
        'kind': kind, 'task_id': task_id, 'candidate_idx': candidate_idx, 'seed': seed,
        'final_loss': None if is_nan else float(losses[-1]),
        'min_loss': None if is_nan else float(losses.min()),
        'n_steps': len(losses), 'time_s': dt, 'nan': is_nan,
    }
    result_dir = os.path.join(results_dir, 'trial_results')
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, f'{kind}_{task_id}_c{candidate_idx}.json'), 'w') as f:
        json.dump(result, f)
    print(f"[{kind} task={task_id} cand={candidate_idx} seed={seed}] "
          f"final={result['final_loss']} nan={is_nan} time={dt:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['targeted', 'general'], required=True)
    parser.add_argument('--results-dir', required=True)
    args = parser.parse_args()

    grid = build_grid(args.kind)
    idx = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
    task_pos, candidate_idx = divmod(idx, POOL_SIZE)
    if task_pos >= len(grid):
        print(f"idx={idx} -> task_pos={task_pos} out of range ({len(grid)} tasks); nothing to do.")
        return
    task_id = grid[task_pos]
    run_one(args.kind, task_id, candidate_idx, args.results_dir)


if __name__ == '__main__':
    main()
