"""
Read side: today's loaders, K* helpers and initial-stiffness regeneration return
exactly the recorded digests for every sample realization.

Run: GOLDEN=1 python -m unittest tests.golden.test_read
"""
import json
import unittest

from tests.golden import EXPECTED_DIR, skip_unless_golden
from tests.golden import sample


def _expected(name):
    path = EXPECTED_DIR / 'read' / f'{name}.json'
    if not path.exists():
        raise AssertionError(f'no golden recorded for {name} (python -m tests.golden.record)')
    return json.loads(path.read_text())


@skip_unless_golden
class TestAllostericRead(unittest.TestCase):
    def test_every_sample_reads_as_recorded(self):
        from tests.golden.readers import read_allosteric
        for geometry, task, real, why in sample.ALLOSTERIC:
            name = sample.allosteric_name(geometry, task, real)
            with self.subTest(name, why=why):
                got = read_allosteric(geometry, task, real)
                want = _expected(name)
                self.assertEqual(sorted(got), sorted(want))
                for call in want:
                    self.assertEqual(got[call], want[call], call)


if __name__ == '__main__':
    unittest.main()
