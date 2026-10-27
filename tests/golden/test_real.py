"""
Real-physics tier (see real_physics). Opt-in: GOLDEN_REAL=1 python -m unittest tests.golden.test_real
"""
import json
import os
import unittest

from tests.golden import EXPECTED_DIR
from tests.golden import real_physics as P


@unittest.skipUnless(os.environ.get('GOLDEN_REAL') == '1', 'real-physics tier needs GOLDEN_REAL=1')
class TestRealPhysics(unittest.TestCase):
    def test_scenarios(self):
        for name, run in P.SCENARIOS.items():
            with self.subTest(name):
                want = json.loads((EXPECTED_DIR / 'real' / f'{name}.json').read_text())
                got = json.loads(json.dumps(run()))
                self.assertEqual(P.compare(got, want), [])


if __name__ == '__main__':
    unittest.main()
