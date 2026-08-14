#!/usr/bin/env python
"""
Realization screening for allosteric training (see docs/realization_screening.md).

Runs a short training trial for one (task[, geometry], candidate) triple,
reusing allosteric_trainer.py's own setup/training-loop code so screening
exercises exactly the same physics/update rule as a real run. Writes one
result JSON per candidate; training/runners/screen_aggregate.py then picks
the best N_KEEP candidates per task and writes the good_realizations lookup.

Usage (SLURM array): SLURM_ARRAY_TASK_ID selects (task[, geometry], candidate)
from the flattened grid for --kind {targeted,general}.

  python screen_realizations_allosteric.py --kind targeted --results-dir <dir>
  python screen_realizations_allosteric.py --kind general  --results-dir <dir>
"""
import argparse
import os
import sys
import json
import shutil
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from training.runners.allosteric_trainer import (
    K_MIN, K_MAX, ETA, LEARNING_RATE, DEFAULT_SOLVER,
    STRAIN_INPUT, STRAIN_INPUT2, NSTEPS_TASK1, NSTEPS_TASK2,
    TARGETED_ENSEMBLE, N_GEOMETRIES, N_TASKS,
    geometry_seed, task_rng, realization_rng,
    load_or_create_geometry, _run_training_loop, _TARGETED_GEOMETRY_SEED,
)
from training.src.good_realizations import SCREEN_SEED_BASE

N_STEPS = 150
POOL_SIZE = 15


def build_grid(kind):
    """List of (task_id, geometry_id_or_None) to screen."""
    if kind == 'targeted':
        return [(tid, None) for tid in range(len(TARGETED_ENSEMBLE))]
    elif kind == 'general':
        return [(tid, gid) for gid in range(N_GEOMETRIES) for tid in range(N_TASKS)]
    raise ValueError(f"kind must be 'targeted' or 'general', got {kind!r}")


def run_one(kind, task_id, geometry_id, candidate_idx, results_dir):
    targeted = (kind == 'targeted')
    seed = SCREEN_SEED_BASE + task_id * 10_000 + (geometry_id or 0) * 100 + candidate_idx

    geom_tag = 'targeted' if targeted else f'g{geometry_id}'
    # Private per-candidate dir, not shared across the pool: load_or_create_geometry's
    # check-then-create isn't atomic, so 15 parallel candidates racing on one shared
    # geom_dir intermittently crashed with EOFError reading a partially-written
    # incidence_matrix.npy. Regenerating per-candidate is cheap (geometry generation
    # is trivial next to the 150-step training loop) and matches how production
    # itself already does it — each real (task, realization) directory gets its own
    # geometry.npy, never shared/cached across parallel processes.
    geom_dir = os.path.join(results_dir, 'geometry_cache', f'{geom_tag}_t{task_id}_c{candidate_idx}')
    os.makedirs(geom_dir, exist_ok=True)

    work_dir = f"/tmp/screen_allosteric_{geom_tag}_t{task_id}_c{candidate_idx}_{os.getpid()}"
    os.makedirs(work_dir, exist_ok=True)
    original_dir = os.getcwd()
    os.chdir(work_dir)
    try:
        gseed = _TARGETED_GEOMETRY_SEED if targeted else geometry_seed(geometry_id)
        nodes, incidence_matrix, eq_lengths = load_or_create_geometry(geom_dir, gseed)

        if targeted:
            strain_output = TARGETED_ENSEMBLE[task_id]['strain_output']
            strain_output2 = TARGETED_ENSEMBLE[task_id]['strain_output2']
        else:
            trng = task_rng(task_id)
            soi1 = int(trng.randint(1, 11))
            soi2 = int(trng.randint(1, 11))
            strain_output = -0.1 * soi2
            strain_output2 = -0.1 * soi1

        tod = (1 + strain_output) * np.linalg.norm(nodes[3] - nodes[2])
        tod2 = (1 + strain_output2) * np.linalg.norm(nodes[3] - nodes[2])
        dinputdistance = STRAIN_INPUT * np.linalg.norm(nodes[0] - nodes[1])
        dinputdistance2 = STRAIN_INPUT2 * np.linalg.norm(nodes[0] - nodes[1])

        rrng = realization_rng(seed)
        stiffnesses = rrng.uniform(K_MIN, K_MAX, size=len(incidence_matrix))

        candidate_out = os.path.join(
            results_dir, 'candidates', geom_tag, f'task_{task_id}', f'cand_{candidate_idx}')
        os.makedirs(candidate_out, exist_ok=True)

        t0 = time.time()
        msearray, msearray2, final_stiffnesses, best_stiffnesses, best_combined_mse = _run_training_loop(
            nodes, incidence_matrix, eq_lengths, stiffnesses,
            LEARNING_RATE, tod, tod2, dinputdistance, dinputdistance2,
            NSTEPS_TASK1, NSTEPS_TASK2, N_STEPS, candidate_out,
            solver=DEFAULT_SOLVER,
        )
        dt = time.time() - t0
    finally:
        os.chdir(original_dir)
        shutil.rmtree(work_dir, ignore_errors=True)

    combined = (np.asarray(msearray) + np.asarray(msearray2)) / 2.0
    is_nan = bool(np.any(np.isnan(combined))) or len(combined) == 0
    result = {
        'kind': kind, 'task_id': task_id, 'geometry_id': geometry_id,
        'candidate_idx': candidate_idx, 'seed': seed,
        'final_loss': None if is_nan else float(combined[-1]),
        'min_loss': None if is_nan else float(np.min(combined)),
        'n_steps': len(combined), 'time_s': dt, 'nan': is_nan,
    }
    key = f"{geometry_id}_{task_id}" if geometry_id is not None else str(task_id)
    result_dir = os.path.join(results_dir, 'trial_results')
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, f'{kind}_{key}_c{candidate_idx}.json'), 'w') as f:
        json.dump(result, f)
    print(f"[{kind} task={task_id} geom={geometry_id} cand={candidate_idx} seed={seed}] "
          f"final={result['final_loss']} nan={is_nan} time={dt:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['targeted', 'general'], required=True)
    parser.add_argument('--results-dir', required=True)
    args = parser.parse_args()

    grid = build_grid(args.kind)
    idx = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
    grid_pos, candidate_idx = divmod(idx, POOL_SIZE)
    if grid_pos >= len(grid):
        print(f"idx={idx} -> grid_pos={grid_pos} out of range ({len(grid)} grid cells); nothing to do.")
        return
    task_id, geometry_id = grid[grid_pos]
    run_one(args.kind, task_id, geometry_id, candidate_idx, args.results_dir)


if __name__ == '__main__':
    main()
