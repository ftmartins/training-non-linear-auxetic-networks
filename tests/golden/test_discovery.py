"""
Discovery and completion: discover_allosteric / discover_auxetic, is_training_complete
(with its marker side effect) and get_(in)complete_jobs return exactly the recorded
results on a scratch copy of the sample runs.

Run: GOLDEN=1 python -m unittest tests.golden.test_discovery
"""
import json
import unittest

from tests.golden import EXPECTED_DIR, skip_unless_golden


@skip_unless_golden
class TestDiscovery(unittest.TestCase):
    def test_discovery_as_recorded(self):
        from tests.golden.discovery import discover
        got = discover()
        want = json.loads((EXPECTED_DIR / 'discovery.json').read_text())
        self.assertEqual(sorted(got), sorted(want))
        for call in want:
            self.assertEqual(got[call], want[call], call)


if __name__ == '__main__':
    unittest.main()
