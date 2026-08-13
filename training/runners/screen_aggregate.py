#!/usr/bin/env python
"""
Aggregate realization-screening trial results into the good_realizations
lookup table (see docs/realization_screening.md and training/src/good_realizations.py).

For each task (or geometry/task pair), ranks non-NaN screened candidates by
final loss (ascending) and keeps the best N_KEEP, storing their seeds in
serial realization_id order (0 = best candidate found).

Usage:
  python screen_aggregate.py --physics auxetic    --kind targeted --results-dir <dir>
  python screen_aggregate.py --physics auxetic    --kind general  --results-dir <dir>
  python screen_aggregate.py --physics allosteric --kind targeted --results-dir <dir>
  python screen_aggregate.py --physics allosteric --kind general  --results-dir <dir>
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from training.src.good_realizations import save_table

N_KEEP = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--physics', choices=['auxetic', 'allosteric'], required=True)
    parser.add_argument('--kind', choices=['targeted', 'general'], required=True)
    parser.add_argument('--results-dir', required=True)
    args = parser.parse_args()

    lookup_kind = f'{args.physics}_{args.kind}'
    pattern = os.path.join(args.results_dir, 'trial_results', f'{args.kind}_*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No trial result files found matching {pattern}")

    by_key = defaultdict(list)
    for path in files:
        with open(path) as f:
            r = json.load(f)
        if args.physics == 'allosteric':
            key = f"{r['geometry_id']}_{r['task_id']}" if r.get('geometry_id') is not None else str(r['task_id'])
        else:
            key = str(r['task_id'])
        by_key[key].append(r)

    table = {}
    report = []
    for key, candidates in sorted(by_key.items()):
        valid = [c for c in candidates if not c['nan']]
        n_nan = len(candidates) - len(valid)
        valid.sort(key=lambda c: c['final_loss'])
        kept = valid[:N_KEEP]
        if len(kept) < N_KEEP:
            print(f"  WARNING: {lookup_kind}/{key} only has {len(kept)} valid candidates "
                  f"(< {N_KEEP} requested) out of {len(candidates)} screened ({n_nan} NaN).")
        table[key] = [c['seed'] for c in kept]
        report.append((key, len(candidates), n_nan, len(kept),
                       kept[0]['final_loss'] if kept else None,
                       kept[-1]['final_loss'] if kept else None))

    path = save_table(lookup_kind, table)
    print(f"\nWrote {path}")
    print(f"{'key':>10} {'screened':>9} {'nan':>4} {'kept':>5} {'best_loss':>12} {'worst_kept':>12}")
    for key, n_screened, n_nan, n_kept, best, worst in report:
        best_s = f"{best:.4e}" if best is not None else "n/a"
        worst_s = f"{worst:.4e}" if worst is not None else "n/a"
        print(f"{key:>10} {n_screened:>9} {n_nan:>4} {n_kept:>5} {best_s:>12} {worst_s:>12}")


if __name__ == '__main__':
    main()
