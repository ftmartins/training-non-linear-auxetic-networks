#!/usr/bin/env python3
"""
Compare a pilot rerun on the stored Network A against the original run.

Passes (exit 0) only if the pilot trained on Network A, with the same
realization seed, and its loss and stiffness trajectories are bit-identical
to the stored run's. See submit_netA_pilot.sh and docs/adr/0002.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from base.realization_record import RealizationRecord
from base.stored_network import NETWORK_A_FINGERPRINT, network_fingerprint


def _record(root, task, real):
    return RealizationRecord.locate(root, 'auxetic', task, real, network_type='jammed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--realization', type=int, required=True)
    parser.add_argument('--ref-root', type=Path, required=True)
    parser.add_argument('--pilot-root', type=Path, required=True)
    args = parser.parse_args()

    ref = _record(args.ref_root, args.task, args.realization)
    new = _record(args.pilot_root, args.task, args.realization)
    ok = True

    def check(name, passed, detail=''):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name} {detail}")

    fp_ref = network_fingerprint(ref.network.positions, ref.network.edges)
    fp_new = network_fingerprint(new.network.positions, new.network.edges)
    check('reference run is Network A', fp_ref == NETWORK_A_FINGERPRINT, fp_ref)
    check('pilot trained on Network A', fp_new == NETWORK_A_FINGERPRINT, fp_new)

    seed_ref, seed_new = ref.realization_seed, new.realization_seed
    check('same realization seed', seed_ref == seed_new, f'{seed_ref} vs {seed_new}')
    check('pilot meta records fingerprint', new.config.get('network_fingerprint') == fp_new)

    trajectories = {'loss_trajectory': (ref.loss.training, new.loss.training),
                    'stiffness_trajectory': (ref.stiffness_trajectory.stiffnesses,
                                             new.stiffness_trajectory.stiffnesses)}
    for stem, (a, b) in trajectories.items():
        same = a.shape == b.shape and np.array_equal(a, b)
        detail = f'shape {a.shape} vs {b.shape}'
        if a.shape == b.shape and not same:
            detail += f', max |diff| {np.nanmax(np.abs(a - b)):.3e}'
        check(f'{stem} bit-identical', same, detail)

    print('PILOT PASSED' if ok else 'PILOT FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
