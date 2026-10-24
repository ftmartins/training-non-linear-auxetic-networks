"""
List the realizations whose post-training sweep cannot be read (ADR 0004): the sweep's
after cost Hessian is not at K* (stale), or K* itself is undetermined. Stale ones need
post_training_sweep.py rerun; undetermined ones need a decision about the run itself.

    python training/runners/list_unreadable_sweeps.py allosteric <root, e.g. allosteric_nets_aug>
    python training/runners/list_unreadable_sweeps.py auxetic <root, e.g. auxetic_nets_aug/targeted>

Prints one tab-separated line per realization: cell (allosteric: geometry, task; auxetic:
task), realization, reason.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from base.realization_record import (  # noqa: E402
    RealizationRecord, BestLossUndetermined, MissingRecordPart, StaleSweep)

_PATTERN = {'allosteric': 'geometry_*/task_*/realization_*', 'auxetic': 'task_*/realization_*'}


def _cell(kind, d):
    task, real = d.parent.name.removeprefix('task_'), d.name.removeprefix('realization_')
    if kind == 'allosteric':
        return d.parent.parent.name.removeprefix('geometry_'), task, real
    return task, real


def unreadable(kind, root):
    for d in sorted(Path(root).glob(_PATTERN[kind])):
        rec = RealizationRecord.open(d, kind=kind)
        try:
            if not rec.has('sweep'):
                continue
        except StaleSweep:
            reason = 'stale_sweep'
        except (BestLossUndetermined, MissingRecordPart) as e:
            reason = f'k_star_undetermined: {str(e).split(": ", 1)[-1]}'
        else:
            continue
        yield _cell(kind, d) + (reason,)


if __name__ == '__main__':
    for row in unreadable(sys.argv[1], sys.argv[2]):
        print('\t'.join(row))
