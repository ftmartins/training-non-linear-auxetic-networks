"""
Record the golden digests from today's code. Run only to create the goldens or after
an intended behaviour change, and name that change in the commit that updates them.

    python -m tests.golden.record            # write + read
    python -m tests.golden.record write      # only the data-free write scenarios
"""
import json

from tests.golden import EXPECTED_DIR
from tests.golden import sample


def _write(kind, name, value):
    path = EXPECTED_DIR / kind / f'{name}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + '\n')
    print(f'wrote {path.relative_to(EXPECTED_DIR)}')


def main(kinds=('write', 'read')):
    if 'write' in kinds:
        from tests.golden import write_scenarios as W
        for name in W.ALLOSTERIC_SCENARIOS:
            _write('write', f'allo_{name}', W.allosteric_scenario(name))
        for name in W.TARGETED_SCENARIOS:
            _write('write', f'targeted_{name}', W.targeted_scenario(name))
    if 'read' in kinds:
        from tests.golden.readers import read_allosteric
        for geometry, task, real, _why in sample.ALLOSTERIC:
            _write('read', sample.allosteric_name(geometry, task, real),
                   read_allosteric(geometry, task, real))


if __name__ == '__main__':
    import sys
    main(tuple(sys.argv[1:]) or ('write', 'read'))
