"""
Write side and realization lifecycle: every scenario in write_scenarios leaves exactly
the recorded files, contents and outcomes. Needs no data (fake physics, tmp dirs), so it
runs with the default test suite (~35 s).

Run: python -m unittest tests.golden.test_write
"""
import json
import unittest

from tests.golden import EXPECTED_DIR
from tests.golden import write_scenarios as W


def _expected(name):
    path = EXPECTED_DIR / 'write' / f'{name}.json'
    if not path.exists():
        raise AssertionError(f'no golden recorded for {name} (python -m tests.golden.record)')
    return json.loads(path.read_text())


class TestWriteScenarios(unittest.TestCase):
    def _check(self, got, want):
        self.assertEqual(got['outcomes'], want['outcomes'])
        self.assertEqual(sorted(got['dir']), sorted(want['dir']))
        for path in want['dir']:
            self.assertEqual(got['dir'][path], want['dir'][path], path)
        for key in set(want) - {'outcomes', 'dir'}:
            self.assertEqual(got[key], want[key], key)

    def test_allosteric_trainer(self):
        for name in W.ALLOSTERIC_SCENARIOS:
            with self.subTest(name):
                self._check(json.loads(json.dumps(W.allosteric_scenario(name))),
                            _expected(f'allo_{name}'))

    def test_targeted_runner(self):
        for name in W.TARGETED_SCENARIOS:
            with self.subTest(name):
                self._check(json.loads(json.dumps(W.targeted_scenario(name))),
                            _expected(f'targeted_{name}'))

    def test_general_runner(self):
        for name in W.GENERAL_SCENARIOS:
            with self.subTest(name):
                self._check(json.loads(json.dumps(W.general_scenario(name))),
                            _expected(f'general_{name}'))


if __name__ == '__main__':
    unittest.main()
