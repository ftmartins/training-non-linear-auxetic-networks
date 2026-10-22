"""base.realization_record, through its public interface, on allosteric run dirs in tmp dirs."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from base.realization_record import (
    RealizationRecord, MissingRecordPart, NotARealization)

N_NODES, N_EDGES = 5, 4


def write_allosteric_run(d, n_steps=6, meta=True, nan_current=False, ckpt_step=None,
                         best=True, sweep=False, loss=True):
    """A minimal allosteric _aug run directory, as allosteric_trainer writes it."""
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    np.save(d / 'nodes.npy', rng.rand(N_NODES, 2))
    inc = np.zeros((N_EDGES, N_NODES))
    for e in range(N_EDGES):
        inc[e, e], inc[e, e + 1] = -1.0, 1.0
    np.save(d / 'incidence_matrix.npy', inc)
    np.save(d / 'eq_lengths.npy', rng.rand(N_EDGES))
    np.savetxt(d / 'tasks.txt', [1000100.0, -0.6, -0.8])
    if loss:
        np.save(d / 'mse1.npy', rng.rand(n_steps))
        np.save(d / 'mse2.npy', rng.rand(n_steps))
    k = rng.rand(N_EDGES)
    if nan_current:
        k[1] = np.nan
    np.save(d / 'stiffnesses.npy', k)
    np.save(d / 'stiffnesses_traj.npy', rng.rand(2, N_EDGES))
    np.save(d / 'stiffnesses_traj_steps.npy', np.array([2, 5]))
    if ckpt_step is not None:
        np.save(d / 'stiffnesses_ckpt.npy', rng.rand(N_EDGES))
        np.savetxt(d / 'ckpt_step.txt', [ckpt_step], fmt='%d')
    if best:
        np.save(d / 'best_stiffnesses.npy', rng.rand(N_EDGES))
        np.savetxt(d / 'best_combined_mse.txt', [0.125])
    if meta:
        (d / 'training_meta.json').write_text(json.dumps(
            {'solver': 'jax_fire', 'k_min': 1e-4, 'k_max': 10.0, 'realization_seed': 1000032}))
    if sweep:
        np.savez(d / 'timestep_sweep.npz', a=np.arange(3), b=np.ones(2))
    return d


class RecordTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def run_dir(self, **kw):
        return write_allosteric_run(self.root / 'geometry_targeted' / 'task_0' / 'realization_3', **kw)


class TestOpen(RecordTestCase):
    def test_detects_allosteric_from_its_files(self):
        self.assertEqual(RealizationRecord.open(self.run_dir()).kind, 'allosteric')

    def test_directory_without_a_known_layout_raises(self):
        (self.root / 'empty').mkdir()
        with self.assertRaises(NotARealization):
            RealizationRecord.open(self.root / 'empty')

    def test_missing_directory_raises(self):
        with self.assertRaises(NotARealization):
            RealizationRecord.open(self.root / 'nope')

    def test_given_kind_opens_a_directory_with_nothing_written_yet(self):
        (self.root / 'fresh').mkdir()
        rec = RealizationRecord.open(self.root / 'fresh', kind='allosteric')
        self.assertIsNone(rec.resume_state())

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            RealizationRecord.open(self.run_dir(), kind='poisson')

    def test_locate_targeted_and_numbered_geometries(self):
        d = self.run_dir()
        rec = RealizationRecord.locate(self.root, 'allosteric', ('targeted', 0), 3)
        self.assertEqual(rec.run_dir, d)
        self.assertEqual(
            RealizationRecord.locate(self.root, 'allosteric', (2, 7), 11, must_exist=False).run_dir,
            self.root / 'geometry_2' / 'task_7' / 'realization_11')

    def test_locate_missing_realization_raises(self):
        with self.assertRaises(NotARealization):
            RealizationRecord.locate(self.root, 'allosteric', (2, 7), 11)


class TestParts(RecordTestCase):
    def test_network(self):
        d = self.run_dir()
        net = RealizationRecord.open(d).network
        np.testing.assert_array_equal(net.nodes, np.load(d / 'nodes.npy'))
        np.testing.assert_array_equal(net.incidence_matrix, np.load(d / 'incidence_matrix.npy'))
        np.testing.assert_array_equal(net.eq_lengths, np.load(d / 'eq_lengths.npy'))
        np.testing.assert_array_equal(net.edges, [[0, 1], [1, 2], [2, 3], [3, 4]])

    def test_task_reads_tasks_txt_in_its_stored_order(self):
        task = RealizationRecord.open(self.run_dir()).task
        self.assertEqual((task.geometry_seed, task.strain_output, task.strain_output2),
                         (1000100.0, -0.8, -0.6))

    def test_loss_per_subtask_and_training_mean(self):
        d = self.run_dir()
        loss = RealizationRecord.open(d).loss
        mse1, mse2 = np.load(d / 'mse1.npy'), np.load(d / 'mse2.npy')
        np.testing.assert_array_equal(loss.per_subtask, np.column_stack([mse1, mse2]))
        np.testing.assert_array_equal(loss.training, 0.5 * (mse1 + mse2))
        np.testing.assert_array_equal(loss.training, loss.per_subtask.mean(axis=1))
        self.assertEqual(loss.n_steps, 6)

    def test_loss_subtask_lengths_must_agree(self):
        d = self.run_dir()
        np.save(d / 'mse2.npy', np.ones(4))
        with self.assertRaises(ValueError):
            RealizationRecord.open(d).loss

    def test_config_and_seed_from_training_meta(self):
        rec = RealizationRecord.open(self.run_dir())
        self.assertEqual(rec.config['solver'], 'jax_fire')
        self.assertEqual(rec.realization_seed, 1000032)
        self.assertIsInstance(rec.realization_seed, int)

    def test_missing_parts_raise_and_has_says_so(self):
        rec = RealizationRecord.open(self.run_dir(meta=False, best=False))
        for part in ('config', 'realization_seed', 'best_so_far', 'sweep'):
            self.assertFalse(rec.has(part), part)
            with self.assertRaises(MissingRecordPart):
                getattr(rec, part)
        for part in ('network', 'task', 'loss', 'stiffness_trajectory', 'current_stiffnesses'):
            self.assertTrue(rec.has(part), part)

    def test_missing_part_is_a_file_not_found_error(self):
        with self.assertRaises(FileNotFoundError):
            RealizationRecord.open(self.run_dir(meta=False)).config

    def test_has_rejects_unknown_parts(self):
        with self.assertRaises(ValueError):
            RealizationRecord.open(self.run_dir()).has('k_star')

    def test_stiffness_trajectory_and_current_stiffnesses(self):
        d = self.run_dir()
        rec = RealizationRecord.open(d)
        np.testing.assert_array_equal(rec.stiffness_trajectory.steps, [2, 5])
        np.testing.assert_array_equal(rec.stiffness_trajectory.stiffnesses,
                                      np.load(d / 'stiffnesses_traj.npy'))
        np.testing.assert_array_equal(rec.current_stiffnesses, np.load(d / 'stiffnesses.npy'))

    def test_best_so_far(self):
        d = self.run_dir()
        best = RealizationRecord.open(d).best_so_far
        np.testing.assert_array_equal(best.stiffnesses, np.load(d / 'best_stiffnesses.npy'))
        self.assertEqual(best.training_loss, 0.125)

    def test_best_so_far_needs_both_files(self):
        d = self.run_dir()
        (d / 'best_combined_mse.txt').unlink()
        with self.assertRaises(MissingRecordPart):
            RealizationRecord.open(d).best_so_far

    def test_sweep(self):
        sweep = RealizationRecord.open(self.run_dir(sweep=True)).sweep
        self.assertEqual(sorted(sweep), ['a', 'b'])
        np.testing.assert_array_equal(sweep['a'], np.arange(3))


class TestResumeState(RecordTestCase):
    def test_clean_current_stiffnesses_resume_at_loss_length(self):
        d = self.run_dir()
        state = RealizationRecord.open(d).resume_state()
        self.assertEqual((state.step, state.source), (6, 'current'))
        np.testing.assert_array_equal(state.stiffnesses, np.load(d / 'stiffnesses.npy'))
        np.testing.assert_array_equal(state.loss.per_subtask[:, 0], np.load(d / 'mse1.npy'))

    def test_nan_current_rolls_back_to_checkpoint_and_truncates_loss(self):
        d = self.run_dir(nan_current=True, ckpt_step=4)
        state = RealizationRecord.open(d).resume_state()
        self.assertEqual((state.step, state.source), (4, 'checkpoint'))
        np.testing.assert_array_equal(state.stiffnesses, np.load(d / 'stiffnesses_ckpt.npy'))
        np.testing.assert_array_equal(state.loss.per_subtask[:, 1], np.load(d / 'mse2.npy')[:4])

    def test_nan_current_without_checkpoint_has_no_resume_state(self):
        self.assertIsNone(RealizationRecord.open(self.run_dir(nan_current=True)).resume_state())

    def test_no_current_stiffnesses_has_no_resume_state(self):
        d = self.run_dir()
        (d / 'stiffnesses.npy').unlink()
        self.assertIsNone(RealizationRecord.open(d).resume_state())

    def test_stiffnesses_without_loss_resume_at_step_zero(self):
        state = RealizationRecord.open(self.run_dir(loss=False)).resume_state()
        self.assertEqual(state.step, 0)
        self.assertEqual(state.loss.per_subtask.shape, (0, 2))


if __name__ == '__main__':
    unittest.main()
