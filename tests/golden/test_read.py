"""
Read side: today's loaders, K* helpers and initial-stiffness regeneration return
exactly the recorded digests for every sample realization.

Run: GOLDEN=1 python -m unittest tests.golden.test_read
"""
import json
import unittest

import numpy as np

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


# Samples whose stored data do not determine K* (ADR 0004), with the reason.
ALLOSTERIC_K_STAR_UNDETERMINED = {
    (4, 2, 1): 'stiffness trajectory steps interleaved by two jobs writing one directory',
    (4, 4, 2): 'stiffness trajectory steps interleaved by two jobs writing one directory',
}


@skip_unless_golden
class TestAllostericBestLossStiffnesses(unittest.TestCase):
    def test_record_k_star_is_todays_best_stiffnesses(self):
        from base.realization_record import RealizationRecord, BestLossUndetermined
        from tests.golden import DATA_ROOT
        from tests.golden.digest import digest
        for geometry, task, real, why in sample.ALLOSTERIC:
            name = sample.allosteric_name(geometry, task, real)
            with self.subTest(name, why=why):
                rec = RealizationRecord.locate(DATA_ROOT / 'allosteric_nets_aug', 'allosteric',
                                               (geometry, task), real)
                if (geometry, task, real) in ALLOSTERIC_K_STAR_UNDETERMINED:
                    with self.assertRaises(BestLossUndetermined):
                        rec.best_loss_stiffnesses
                    continue
                self.assertEqual(digest(rec.best_loss_stiffnesses),
                                 digest(np.load(rec.run_dir / 'best_stiffnesses.npy')))


@skip_unless_golden
class TestAuxeticBestLossStiffnesses(unittest.TestCase):
    def test_record_k_star_is_todays_per_step_pairing(self):
        """K* = stiffness_trajectory[nanargmin(loss_trajectory)], as the Aug26 helpers and
        the post-training sweep pair them; runs without a stiffness trajectory have none."""
        from base.realization_record import RealizationRecord, MissingRecordPart
        from tests.golden import DATA_ROOT
        for family, task, real, why in sample.AUXETIC:
            with self.subTest(sample.auxetic_name(family, task, real), why=why):
                rec = RealizationRecord.locate(DATA_ROOT / 'auxetic_nets_aug' / family, 'auxetic',
                                               task, real)
                if not rec.has('stiffness_trajectory'):
                    with self.assertRaises(MissingRecordPart):
                        rec.best_loss_stiffnesses
                    continue
                loss = np.load(rec.run_dir / 'loss_trajectory_jammed.npy')
                k = np.load(rec.run_dir / 'stiffness_trajectory_jammed.npy')
                b = int(np.nanargmin(loss))
                self.assertEqual(rec.best_loss_step, b)
                np.testing.assert_array_equal(rec.best_loss_stiffnesses, k[b])


def _check(case, name, got):
    want = _expected(name)
    case.assertEqual(sorted(got), sorted(want))
    for call in want:
        case.assertEqual(got[call], want[call], call)


@skip_unless_golden
class TestAuxeticRead(unittest.TestCase):
    def test_every_sample_reads_as_recorded(self):
        from tests.golden.readers import read_auxetic
        for family, task, real, why in sample.AUXETIC:
            with self.subTest(sample.auxetic_name(family, task, real), why=why):
                _check(self, sample.auxetic_name(family, task, real),
                       read_auxetic(family, task, real))

    def test_every_legacy_sample_reads_as_recorded(self):
        from tests.golden.readers import read_auxetic_legacy
        for data_dir, task, real, network_type, why in sample.LEGACY:
            name = sample.legacy_name(data_dir, task, real, network_type)
            with self.subTest(name, why=why):
                _check(self, name, read_auxetic_legacy(data_dir, task, real, network_type))


if __name__ == '__main__':
    unittest.main()
