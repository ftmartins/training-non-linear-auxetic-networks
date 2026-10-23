"""
List the allosteric realizations whose post-training sweep cannot be read (ADR 0004): the
sweep's after cost Hessian is not at K* (stale), or K* itself is undetermined. Stale ones
need post_training_sweep.py rerun; undetermined ones need a decision about the run itself.

    python training/runners/list_unreadable_allosteric_sweeps.py <data root, e.g. allosteric_nets_aug>

Prints one tab-separated line per realization: geometry, task, realization, reason.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from base.realization_record import (  # noqa: E402
    RealizationRecord, BestLossUndetermined, MissingRecordPart, StaleSweep)


def unreadable(root):
    for d in sorted(Path(root).glob('geometry_*/task_*/realization_*')):
        if not (d / 'timestep_sweep.npz').exists():
            continue
        try:
            RealizationRecord.open(d, kind='allosteric').sweep
        except StaleSweep:
            reason = 'stale_sweep'
        except (BestLossUndetermined, MissingRecordPart) as e:
            reason = f'k_star_undetermined: {str(e).split(": ", 1)[-1]}'
        else:
            continue
        yield (d.parent.parent.name.removeprefix('geometry_'), d.parent.name.removeprefix('task_'),
               d.name.removeprefix('realization_'), reason)


if __name__ == '__main__':
    for row in unreadable(sys.argv[1]):
        print('\t'.join(row))
