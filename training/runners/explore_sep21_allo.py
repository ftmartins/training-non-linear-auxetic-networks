#!/usr/bin/env python3
"""
Allosteric coordination-number sweep (2026-09-21 explorations).

Thin wrapper around allosteric_trainer.py — the trainer itself is NOT modified. For one
case (see training/src/explore_sep21_cases.allo_z_cases) it
  1. builds the lattice geometry for (topology seed, target z): the standard
     allosteric_trainer.create_network(10, 0.15, 1.6) lattice, randomly bond-diluted to
     z = 2E/N = target (keeps the input bond 0-1 and output bond 2-3; every node keeps
     degree >= 3; among random dilution orders it keeps one with the fewest floppy modes),
  2. pre-places nodes.npy / incidence_matrix.npy / eq_lengths.npy in the realization dir
     (allosteric_trainer.load_or_create_geometry loads them instead of regenerating),
  3. patches get_realization_seed so the IC seed is ALLO_IC_SEED_BASE+ic (not the
     screened table, which was tuned for the native geometry), and
  4. calls allosteric_trainer.main() with --targeted-ensemble and a per-(z,topology)
     --output-dir under EXPLORE_ROOT, so nothing in allosteric_nets_aug is touched.

Usage:  python explore_sep21_allo.py --case-index N [--training-steps S] [--learning-rate LR]
        python explore_sep21_allo.py --list          # print case count
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_RUNNERS = str(Path(__file__).resolve().parent)
if _RUNNERS not in sys.path:
    sys.path.insert(0, _RUNNERS)

from training.src import explore_sep21_cases as EC


def floppy_modes(nodes, inc):
    """Non-trivial zero modes of the rigidity matrix (2N - 3 - rank)."""
    ends_i = np.argmin(inc, axis=1)     # -1 entry
    ends_j = np.argmax(inc, axis=1)     # +1 entry
    n = len(nodes)
    R = np.zeros((len(inc), 2 * n))
    u = nodes[ends_j] - nodes[ends_i]
    u /= np.linalg.norm(u, axis=1)[:, None]
    for e, (i, j) in enumerate(zip(ends_i, ends_j)):
        R[e, 2 * i:2 * i + 2] = -u[e]
        R[e, 2 * j:2 * j + 2] = u[e]
    return 2 * n - 3 - np.linalg.matrix_rank(R, tol=1e-9)


def build_geometry(geom_seed, z_target, n_attempts=300):
    """(nodes, incidence_matrix, eq_lengths, info) for one topology + target z."""
    import allosteric_trainer as at
    random.seed(geom_seed)
    nodes, inc, eq, _ = at.create_network(10, 0.15, 1.6)
    n_nodes, n_edges0 = len(nodes), len(inc)
    ends = np.stack([np.argmin(inc, axis=1), np.argmax(inc, axis=1)], axis=1)
    info = dict(n_nodes=n_nodes, n_edges_native=n_edges0, z_native=2 * n_edges0 / n_nodes)
    if z_target is None:
        info.update(z=info['z_native'], n_floppy=int(floppy_modes(nodes, inc)))
        return nodes, inc, eq, info

    protected = {i for i, (a, b) in enumerate(ends) if {a, b} in ({0, 1}, {2, 3})}
    n_target = int(round(z_target * n_nodes / 2))
    rng = np.random.RandomState((geom_seed * 7919 + int(round(z_target * 100))) % (2 ** 32))
    best = None
    for _ in range(n_attempts):
        deg = np.bincount(ends.ravel(), minlength=n_nodes)
        keep = np.ones(n_edges0, bool)
        n_keep = n_edges0
        for e in rng.permutation(n_edges0):
            if n_keep <= n_target:
                break
            if e in protected:
                continue
            a, b = ends[e]
            if deg[a] > 3 and deg[b] > 3:
                keep[e] = False
                deg[a] -= 1
                deg[b] -= 1
                n_keep -= 1
        if n_keep > n_target:          # could not reach the target with degree >= 3
            continue
        nf = int(floppy_modes(nodes, inc[keep]))
        if best is None or nf < best[0]:
            best = (nf, keep.copy())
        if nf == 0:
            break
    if best is None:
        raise RuntimeError(f"could not dilute seed {geom_seed} to z={z_target}")
    nf, keep = best
    inc_d = inc[keep]
    eq_d = np.linalg.norm(inc_d @ nodes, axis=1)
    info.update(z=2 * len(inc_d) / n_nodes, n_edges=int(len(inc_d)), n_floppy=nf)
    return nodes, inc_d, eq_d, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case-index', type=int)
    p.add_argument('--list', action='store_true')
    p.add_argument('--training-steps', type=int, default=EC.ALLO_TRAIN_STEPS)
    p.add_argument('--learning-rate', type=float, default=None)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--root', type=str, default=str(EC.EXPLORE_ROOT))
    p.add_argument('--geometry-only', action='store_true', help='write geometry files and exit')
    p.add_argument('--extra-cells', type=str, default=None,
                   help='e.g. "2:0,0:1,2:1" (task:z_idx pairs) -- use allo_z_extra_cases() instead '
                        'of the original 250-case list, for (task,z) cells short of enough converged '
                        'realizations. Combine with --extra-ic-start/--extra-ic-stop.')
    p.add_argument('--extra-ic-start', type=int, default=5)
    p.add_argument('--extra-ic-stop', type=int, default=20)
    a = p.parse_args()

    if a.extra_cells:
        cells = [tuple(map(int, pair.split(':'))) for pair in a.extra_cells.split(',')]
        cases = EC.allo_z_extra_cases(cells, a.extra_ic_start, a.extra_ic_stop)
    else:
        cases = EC.allo_z_cases()
    if a.list:
        print(len(cases))
        return
    c = cases[a.case_index]
    out_dir = EC.allo_z_dir(c, a.root)
    rid, tid = c['ic'], c['task']
    path = out_dir / 'geometry_targeted' / f'task_{tid}' / f'realization_{rid}'
    path.mkdir(parents=True, exist_ok=True)

    if not (path / 'nodes.npy').exists():
        nodes, inc, eq, info = build_geometry(c['geom_seed'], c['z_target'])
        np.save(path / 'nodes.npy', nodes)
        np.save(path / 'incidence_matrix.npy', inc)
        np.save(path / 'eq_lengths.npy', eq)
        with open(path / 'explore_case.json', 'w') as f:
            json.dump({**c, **info}, f, indent=1)
        print(f"geometry: {info}")
    if a.geometry_only:
        return

    import allosteric_trainer as at
    at.get_realization_seed = lambda kind, t, r, geometry_id=None: c['ic_seed']
    argv = ['allosteric_trainer.py', '--targeted-ensemble', '--task-id', str(tid),
            '--realization-id', str(rid), '--training-steps', str(a.training_steps),
            '--output-dir', str(out_dir)]
    if a.learning_rate is not None:
        argv += ['--learning-rate', str(a.learning_rate)]
    if a.overwrite:
        argv += ['--overwrite']
    sys.argv = argv
    at.main()


if __name__ == '__main__':
    main()
