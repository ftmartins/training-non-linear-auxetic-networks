"""
How the Aug26 auxetic best-loss helper behaves when K* (ADR 0004) cannot be determined:
each fallback to the final network is pinned here, and its test flips when the fallback is
removed. Synthetic run directories in tmp dirs.

Run: python -m unittest tests.test_auxetic_k_star_consumers
"""
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / 'analysis' / 'notebooks' / 'figures'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import aug26_common as A  # noqa: E402
from base.realization_record import BestLossUndetermined, MissingRecordPart  # noqa: E402
from tests.test_realization_record_auxetic import write_auxetic_run, N_STEPS, N_EDGES  # noqa: E402

TASK, REAL = 3, 1


class AuxeticConsumerTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        patcher = mock.patch.object(A, 'AUX_DATA_DIR', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.d = write_auxetic_run(self.root / f'task_{TASK:02d}' / f'realization_{REAL:02d}')

    def tearDown(self):
        self._tmp.cleanup()

    def best_state(self, **kw):
        aa = A.load_auxetic(TASK, REAL)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return aa, A.aux_best_loss_state(aa, **kw)


class TestAuxBestLossState(AuxeticConsumerTestCase):
    def test_is_k_star_with_its_positions(self):
        np.save(self.d / 'loss_trajectory_jammed.npy', np.array([3.0, 2.0, 0.5, 1, 1, 1, 1.0]))
        aa, (pos, k, b) = self.best_state()
        self.assertEqual(b, 2)
        np.testing.assert_array_equal(k, np.load(self.d / 'stiffness_trajectory_jammed.npy')[2])

    def test_without_stiffness_trajectory_raises(self):
        (self.d / 'stiffness_trajectory_jammed.npy').unlink()
        with self.assertRaises(MissingRecordPart):
            self.best_state()

    def test_all_nan_loss_raises(self):
        np.save(self.d / 'loss_trajectory_jammed.npy', np.full(N_STEPS, np.nan))
        with self.assertRaises(BestLossUndetermined):
            self.best_state()

    def test_stiffness_trajectory_of_another_length_raises(self):
        np.save(self.d / 'loss_trajectory_jammed.npy', np.array([3.0, 2, 1, 1, 1, 1, 0.5]))
        np.save(self.d / 'stiffness_trajectory_jammed.npy', np.ones((3, N_EDGES)))
        with self.assertRaises(BestLossUndetermined):
            self.best_state()


if __name__ == '__main__':
    unittest.main()
