"""
Record the golden digests from today's code. Run only to create the goldens or after
an intended behaviour change, and name that change in the commit that updates them.

    python -m tests.golden.record            # write + read + discovery
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


def main(kinds=('write', 'read', 'discovery')):
    if 'write' in kinds:
        from tests.golden import write_scenarios as W
        for name in W.ALLOSTERIC_SCENARIOS:
            _write('write', f'allo_{name}', W.allosteric_scenario(name))
        for name in W.TARGETED_SCENARIOS:
            _write('write', f'targeted_{name}', W.targeted_scenario(name))
        for name in W.GENERAL_SCENARIOS:
            _write('write', f'general_{name}', W.general_scenario(name))
    if 'discovery' in kinds:
        from tests.golden.discovery import discover
        path = EXPECTED_DIR / 'discovery.json'
        path.write_text(json.dumps(discover(), indent=1, sort_keys=True) + '\n')
        print(f'wrote {path.relative_to(EXPECTED_DIR)}')
    if 'read' in kinds:
        from tests.golden.readers import read_allosteric, read_auxetic, read_auxetic_legacy
        for geometry, task, real, _why in sample.ALLOSTERIC:
            _write('read', sample.allosteric_name(geometry, task, real),
                   read_allosteric(geometry, task, real))
        for family, task, real, _why in sample.AUXETIC:
            _write('read', sample.auxetic_name(family, task, real),
                   read_auxetic(family, task, real))
        for data_dir, task, real, network_type, _why in sample.LEGACY:
            _write('read', sample.legacy_name(data_dir, task, real, network_type),
                   read_auxetic_legacy(data_dir, task, real, network_type))


if __name__ == '__main__':
    import sys
    main(tuple(sys.argv[1:]) or ('write', 'read', 'discovery'))
