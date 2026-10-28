#!/usr/bin/env python3
"""
Write the canonical figure ensemble to manifests/canonical_ensemble_{allosteric,auxetic}.tsv.

The ensemble comes from aug26_common.figure_ensemble_allosteric / figure_ensemble_auxetic:
best-trained converged realizations with distinct realization seeds per cell (ADR 0001), auxetic
on the stored Network A only (ADR 0002), runs whose best-loss stiffnesses (K*) cannot be
determined left out (ADR 0004). Same columns as the replication repo's manifests, so the two
can be diffed.

    python analysis/export_canonical_ensemble.py [--out manifests]
"""
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'analysis' / 'notebooks' / 'figures'))

import aug26_common as C  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=_ROOT / 'manifests')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    allosteric = C.figure_ensemble_allosteric()
    with open(args.out / 'canonical_ensemble_allosteric.tsv', 'w') as f:
        f.write('geometry\ttask\trealization\n')
        f.writelines(f'{g}\t{t}\t{r}\n' for g, t, r in allosteric)
    auxetic = C.figure_ensemble_auxetic()
    with open(args.out / 'canonical_ensemble_auxetic.tsv', 'w') as f:
        f.write('task\trealization\n')
        f.writelines(f'{t}\t{r}\n' for _family, t, r in auxetic)
    print(f'{len(allosteric)} allosteric and {len(auxetic)} auxetic members -> {args.out}')


if __name__ == '__main__':
    main()
