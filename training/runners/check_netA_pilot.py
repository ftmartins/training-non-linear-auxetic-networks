#!/usr/bin/env python3
"""
Compare a pilot rerun on the stored Network A against the original run.

Passes (exit 0) only if the pilot trained on Network A, with the same
realization seed, and its loss and stiffness trajectories are bit-identical
to the stored run's. See submit_netA_pilot.sh and docs/adr/0002.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from base.stored_network import NETWORK_A_FINGERPRINT, network_fingerprint


def _run_dir(root, task, real):
    return Path(root) / f'task_{task:02d}' / f'realization_{real:02d}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--realization', type=int, required=True)
    parser.add_argument('--ref-root', type=Path, required=True)
    parser.add_argument('--pilot-root', type=Path, required=True)
    args = parser.parse_args()

    ref = _run_dir(args.ref_root, args.task, args.realization)
    new = _run_dir(args.pilot_root, args.task, args.realization)
    ok = True

    def check(name, passed, detail=''):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name} {detail}")

    with open(ref / 'final_network_jammed.pkl', 'rb') as f:
        ref_net = pickle.load(f)
    with open(new / 'final_network_jammed.pkl', 'rb') as f:
        new_net = pickle.load(f)
    fp_ref = network_fingerprint(ref_net['positions'], ref_net['edges'])
    fp_new = network_fingerprint(new_net['positions'], new_net['edges'])
    check('reference run is Network A', fp_ref == NETWORK_A_FINGERPRINT, fp_ref)
    check('pilot trained on Network A', fp_new == NETWORK_A_FINGERPRINT, fp_new)

    meta_ref = json.loads((ref / 'training_meta_jammed.json').read_text())
    meta_new = json.loads((new / 'training_meta_jammed.json').read_text())
    check('same realization seed', meta_ref['realization_seed'] == meta_new['realization_seed'],
          f"{meta_ref['realization_seed']} vs {meta_new['realization_seed']}")
    check('pilot meta records fingerprint', meta_new.get('network_fingerprint') == fp_new)

    for stem in ('loss_trajectory', 'stiffness_trajectory'):
        a = np.load(ref / f'{stem}_jammed.npy')
        b = np.load(new / f'{stem}_jammed.npy')
        same = a.shape == b.shape and np.array_equal(a, b)
        detail = f'shape {a.shape} vs {b.shape}'
        if a.shape == b.shape and not same:
            detail += f', max |diff| {np.nanmax(np.abs(a - b)):.3e}'
        check(f'{stem} bit-identical', same, detail)

    print('PILOT PASSED' if ok else 'PILOT FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
