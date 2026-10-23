"""
How the allosteric consumers of the best-loss stiffnesses (K*, ADR 0004) behave when K* is
missing or its cost Hessian is not available at K*: the Aug26 figure helpers, the
post-training sweep and the z-sweep susceptibility cache. Each fallback that used to
substitute something else is pinned here, and its test flips when that fallback is removed.

Run on synthetic run directories in tmp dirs:
    python -m unittest tests.test_allosteric_k_star_consumers
"""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / 'training' / 'runners', _ROOT / 'analysis' / 'notebooks' / 'figures'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import aug26_common as A  # noqa: E402
import compute_z_sweep_cache as Z  # noqa: E402
from training.runners import post_training_sweep as P  # noqa: E402
from base.realization_record import BestLossUndetermined, MissingRecordPart  # noqa: E402
from tests.test_realization_record import write_allosteric_run, TRAJ_STEPS, BEST_STEP  # noqa: E402

TASK, REAL = 0, 3


class ConsumerTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        patcher = mock.patch.object(A, 'ALLO_DATA_DIR', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self._tmp.cleanup()

    def run_dir(self, **kw):
        return write_allosteric_run(
            self.root / 'geometry_targeted' / f'task_{TASK}' / f'realization_{REAL}', **kw)

    def loaded(self, **kw):
        self.run_dir(**kw)
        return A.load_allosteric('targeted', TASK, REAL)

    def k_star(self):
        return np.load(self.root / 'geometry_targeted' / f'task_{TASK}' / f'realization_{REAL}'
                       / 'stiffnesses_traj.npy')[list(TRAJ_STEPS).index(BEST_STEP)]


class TestAug26BestLossStiffness(ConsumerTestCase):
    def test_is_k_star(self):
        np.testing.assert_array_equal(A.allo_best_loss_stiffness(self.loaded()), self.k_star())

    def test_without_stored_best_raises(self):
        with self.assertRaises(MissingRecordPart):
            A.allo_best_loss_stiffness(self.loaded(best=False))

    def test_undetermined_k_star_raises(self):
        d = self.loaded(started='2026-08-01T00:00:00')
        with self.assertRaises(BestLossUndetermined):
            A.allo_best_loss_stiffness(d)

    def test_operating_point_without_sweep_is_k_star(self):
        k, tag = A.allo_operating_point(self.loaded(), 'best')
        self.assertEqual(tag, 'best')
        np.testing.assert_array_equal(k, self.k_star())

    def test_operating_point_without_sweep_or_stored_best_raises(self):
        with self.assertRaises(MissingRecordPart):
            A.allo_operating_point(self.loaded(best=False), 'best')


class TestZSweepOperatingStiffness(ConsumerTestCase):
    def test_is_k_star(self):
        np.testing.assert_array_equal(Z._allo_operating_k(self.loaded()), self.k_star())

    def test_without_stored_best_falls_back_to_the_final_snapshot(self):
        d = self.loaded(best=False)
        np.testing.assert_array_equal(Z._allo_operating_k(d), d['stiff_traj'][-1])


class TestPostTrainingSweep(ConsumerTestCase):
    def sweep_kwargs(self):
        got = {}
        with mock.patch.object(P, 'sweep_allosteric', lambda *a, **k: got.update(k)), \
                contextlib.redirect_stdout(io.StringIO()):
            P._run_allosteric(TASK, REAL, 0, True, str(self.root), 50, 1e-8, 4, 6, 'jax_fire')
        return got

    def test_after_cost_hessian_at_k_star(self):
        self.run_dir()
        np.testing.assert_array_equal(self.sweep_kwargs()['best_stiffnesses'], self.k_star())

    def test_without_stored_best_sweeps_at_the_checkpoint_grid(self):
        self.run_dir(best=False)
        self.assertIsNone(self.sweep_kwargs()['best_stiffnesses'])


class TestAug26CostHessianAtKStar(ConsumerTestCase):
    def test_without_sweep_recomputes_with_jax_fire_for_a_lammps_run(self):
        d = self.loaded()
        d['solver'] = 'lammps'
        with mock.patch.object(A, 'allo_per_task_cost_hessian',
                               return_value='recomputed') as recompute:
            self.assertEqual(A.allo_cost_hessian_at(d, A.allo_best_loss_stiffness(d)),
                             'recomputed')
        recompute.assert_called_once()


class TestAug26StaleSweep(ConsumerTestCase):
    def test_sweep_without_k_star_after_hessian_is_read(self):
        d = self.loaded(sweep=True)
        self.assertEqual(sorted(d['sweep']), ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
