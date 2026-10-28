#!/usr/bin/env python3
"""
Repair allosteric runs whose stiffness-trajectory snapshots were interleaved by a second job.

When two jobs append to one run directory, stiffnesses_traj_steps.npy goes backwards (4349, 49,
4399, 99, ...) and the record refuses to give the run's best-loss stiffnesses (K*, ADR 0004).
The surviving job is the chain that ends at the last snapshot; the other job's snapshots are
the ones that break the increasing order. This script drops them and keeps the originals as
*.interleaved.bak, but only when afterwards the trajectory fits the loss trajectory AND the
record can determine K* (the stored best loss and best stiffnesses agree with the snapshot at
the best step). Anything else is left alone and reported.

    python repair_interleaved_trajectories.py --root data/allosteric_nets_aug          # dry run
    python repair_interleaved_trajectories.py --root data/allosteric_nets_aug --apply
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from base.realization_record import (  # noqa: E402
    BestLossUndetermined, RealizationRecord, StiffnessTrajectory, _fits)


def surviving_chain(steps):
    """Indices of the snapshots that increase strictly up to the last one."""
    keep = [len(steps) - 1]
    for i in range(len(steps) - 2, -1, -1):
        if steps[i] < steps[keep[-1]]:
            keep.append(i)
    return keep[::-1]


def check(record):
    """(repair indices, reason) for one run: indices to keep, or None and why not."""
    if not (record.has('loss') and record.has('stiffness_trajectory')):
        return None, 'no loss or trajectory'
    traj, n_steps = record.stiffness_trajectory, record.loss.n_steps
    if _fits(traj, n_steps):
        return None, 'fine'
    keep = surviving_chain(traj.steps)
    fixed = StiffnessTrajectory(traj.steps[keep], traj.stiffnesses[keep])
    if not _fits(fixed, n_steps):
        return None, 'still does not fit after dropping out-of-order snapshots'
    record.__dict__['stiffness_trajectory'] = fixed
    record.__dict__.pop('_best_loss', None)
    try:
        record.best_loss_stiffnesses
    except BestLossUndetermined as e:
        return None, f'K* still undetermined: {e}'
    return keep, f'drops {len(traj.steps) - len(keep)} of {len(traj.steps)} snapshots'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True, help='allosteric data root')
    parser.add_argument('--apply', action='store_true', help='write the repair (default: dry run)')
    args = parser.parse_args()

    repaired = 0
    for record in RealizationRecord.iterate(args.root, 'allosteric'):
        keep, why = check(record)
        if why == 'fine':
            continue
        print(f'{record.cell} realization {record.realization_id}: '
              f'{"REPAIR" if keep is not None else "SKIP"} ({why})')
        if keep is None or not args.apply:
            continue
        run_dir = record.run_dir
        steps_path, traj_path = run_dir / 'stiffnesses_traj_steps.npy', run_dir / 'stiffnesses_traj.npy'
        original = RealizationRecord.open(run_dir, kind='allosteric').stiffness_trajectory
        for path in (steps_path, traj_path):
            shutil.copy2(path, path.with_name(path.name + '.interleaved.bak'))
        np.save(steps_path, original.steps[keep])
        np.save(traj_path, original.stiffnesses[keep])
        repaired += 1
    print(f'{repaired} run(s) repaired' if args.apply else 'dry run: nothing written')


if __name__ == '__main__':
    main()
