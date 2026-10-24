"""base.realization_record on auxetic run directories in tmp dirs: suffixed and plain file
names, network types, padded and unpadded task directories, and the checkpoint as resume
state.

Run: python -m unittest tests.test_realization_record_auxetic
"""
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from base.realization_record import RealizationRecord, MissingRecordPart, BestLossUndetermined

N_NODES, N_EDGES, N_STEPS = 6, 5, 7
TASK_CONFIG = {'task_seed': 1, 'packing_seed': 42, 'n_nodes': N_NODES, 'n_strain_steps': 400,
               'compression_strains': [-0.2, -0.1], 'target_poisson_ratios': [-0.8, -0.8]}


def write_auxetic_run(d, suffix='_jammed', checkpoint_step=4, task_config=TASK_CONFIG,
                      seed=1010001):
    """A minimal auxetic run as the ensemble runners write it; suffix '' for plain names."""
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    edges = np.array([[i, i + 1] for i in range(N_EDGES)])
    net = {'positions': rng.rand(N_NODES, 2), 'edges': edges, 'stiffnesses': rng.rand(N_EDGES),
           'rest_lengths': rng.rand(N_EDGES)}
    with open(d / f'final_network{suffix}.pkl', 'wb') as f:
        pickle.dump(net, f)
    (d / f'boundary_nodes{suffix}.json').write_text(json.dumps(
        {'top': [0], 'bottom': [1], 'left': [2], 'right': [3]}))
    (d / f'task_config{suffix}.json').write_text(json.dumps(task_config))
    loss = np.linspace(1.0, 0.1, N_STEPS)
    np.save(d / f'loss_trajectory{suffix}.npy', loss)
    np.save(d / f'stiffness_trajectory{suffix}.npy', rng.rand(N_STEPS, N_EDGES))
    with open(d / f'history{suffix}.pkl', 'wb') as f:
        pickle.dump({'positions': [rng.rand(N_NODES, 2) for _ in range(N_STEPS)]}, f)
    (d / f'training_meta{suffix}.json').write_text(json.dumps({'realization_seed': seed}))
    if checkpoint_step is not None:
        with open(d / f'checkpoint{suffix}.pkl', 'wb') as f:
            pickle.dump({'current_step': checkpoint_step, 'network': dict(net),
                         'history': {'loss': list(loss[:checkpoint_step])},
                         'task_config': task_config}, f)
    return d


class AuxeticTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def run_dir(self, task=3, real=1, **kw):
        return write_auxetic_run(self.root / f'task_{task:02d}' / f'realization_{real:02d}', **kw)


class TestLayout(AuxeticTestCase):
    def test_detects_auxetic(self):
        self.assertEqual(RealizationRecord.open(self.run_dir()).kind, 'auxetic')

    def test_locate_padded_and_unpadded_task_dirs(self):
        d = self.run_dir()
        self.assertEqual(RealizationRecord.locate(self.root, 'auxetic', 3, 1).run_dir, d)
        u = write_auxetic_run(self.root / 'task_7' / 'realization_02')
        self.assertEqual(RealizationRecord.locate(self.root, 'auxetic', 7, 2).run_dir, u)

    def test_jammed_falls_back_to_plain_names(self):
        rec = RealizationRecord.open(self.run_dir(suffix=''), kind='auxetic')
        self.assertEqual(rec.realization_seed, 1010001)
        self.assertEqual(rec.loss.n_steps, N_STEPS)

    def test_suffixed_names_win_over_plain(self):
        d = self.run_dir(seed=1)
        write_auxetic_run(d, suffix='', seed=2)
        self.assertEqual(RealizationRecord.open(d, kind='auxetic').realization_seed, 1)

    def test_lattice_never_reads_plain_or_jammed_files(self):
        d = self.run_dir()
        write_auxetic_run(d, suffix='')
        rec = RealizationRecord.open(d, kind='auxetic', network_type='lattice')
        self.assertFalse(rec.has('config'))
        write_auxetic_run(d, suffix='_lattice', seed=5)
        rec = RealizationRecord.open(d, kind='auxetic', network_type='lattice')
        self.assertEqual(rec.realization_seed, 5)

    def test_network_type_none_reads_plain_names_only(self):
        rec = RealizationRecord.open(self.run_dir(), kind='auxetic', network_type=None)
        self.assertFalse(rec.has('network'))


class TestParts(AuxeticTestCase):
    def test_network_boundary_task(self):
        rec = RealizationRecord.open(self.run_dir())
        self.assertEqual(rec.network.edges.shape, (N_EDGES, 2))
        np.testing.assert_array_equal(rec.boundary['top'], [0])
        self.assertEqual(rec.task.config, TASK_CONFIG)
        self.assertEqual(rec.task.compression_strains, [-0.2, -0.1])

    def test_null_task_config_is_missing(self):
        rec = RealizationRecord.open(self.run_dir(task_config=None))
        self.assertFalse(rec.has('task'))

    def test_loss_has_training_only(self):
        loss = RealizationRecord.open(self.run_dir()).loss
        np.testing.assert_array_equal(loss.training, np.linspace(1.0, 0.1, N_STEPS))
        with self.assertRaises(MissingRecordPart):
            loss.per_subtask

    def test_trajectories_pair_one_row_per_step(self):
        rec = RealizationRecord.open(self.run_dir())
        np.testing.assert_array_equal(rec.stiffness_trajectory.steps, np.arange(N_STEPS))
        self.assertEqual(len(rec.positions_trajectory), N_STEPS)

    def test_current_stiffnesses_are_the_final_network(self):
        d = self.run_dir()
        with open(d / 'final_network_jammed.pkl', 'rb') as f:
            final = pickle.load(f)
        np.testing.assert_array_equal(RealizationRecord.open(d).current_stiffnesses,
                                      final['stiffnesses'])

    def test_no_best_so_far(self):
        self.assertFalse(RealizationRecord.open(self.run_dir()).has('best_so_far'))


class TestBestLossStiffnesses(AuxeticTestCase):
    def test_k_star_is_the_trajectory_row_at_the_lowest_loss(self):
        d = self.run_dir()
        np.save(d / 'loss_trajectory_jammed.npy', np.array([3.0, 2.0, 0.5, 1.0, 0.7, 0.9, 2.0]))
        rec = RealizationRecord.open(d)
        self.assertEqual(rec.best_loss_step, 2)
        np.testing.assert_array_equal(rec.best_loss_stiffnesses,
                                      np.load(d / 'stiffness_trajectory_jammed.npy')[2])

    def test_ties_go_to_the_first_step_and_nan_is_excluded(self):
        d = self.run_dir()
        np.save(d / 'loss_trajectory_jammed.npy', np.array([np.nan, 2.0, 0.5, 0.5, 1, 1, 1.0]))
        self.assertEqual(RealizationRecord.open(d).best_loss_step, 2)

    def test_all_nan_loss_raises(self):
        d = self.run_dir()
        np.save(d / 'loss_trajectory_jammed.npy', np.full(N_STEPS, np.nan))
        with self.assertRaises(BestLossUndetermined):
            RealizationRecord.open(d).best_loss_stiffnesses

    def test_trajectories_of_different_lengths_raise(self):
        d = self.run_dir()
        np.save(d / 'stiffness_trajectory_jammed.npy', np.ones((N_STEPS - 1, N_EDGES)))
        with self.assertRaises(BestLossUndetermined):
            RealizationRecord.open(d).best_loss_stiffnesses

    def test_missing_stiffness_trajectory_raises_missing_part(self):
        d = self.run_dir()
        (d / 'stiffness_trajectory_jammed.npy').unlink()
        rec = RealizationRecord.open(d)
        with self.assertRaises(MissingRecordPart):
            rec.best_loss_stiffnesses
        self.assertFalse(rec.has('best_loss_stiffnesses'))


class TestResumeState(AuxeticTestCase):
    def test_checkpoint_is_the_resume_state(self):
        state = RealizationRecord.open(self.run_dir(checkpoint_step=4)).resume_state()
        self.assertEqual((state.step, state.source), (4, 'checkpoint'))
        self.assertEqual(state.loss.n_steps, 4)
        self.assertEqual(state.trainer_state['current_step'], 4)

    def test_no_checkpoint_no_resume_state(self):
        self.assertIsNone(RealizationRecord.open(self.run_dir(checkpoint_step=None)).resume_state())

    def test_unreadable_checkpoint_no_resume_state(self):
        d = self.run_dir()
        (d / 'checkpoint_jammed.pkl').write_bytes(b'truncated')
        self.assertIsNone(RealizationRecord.open(d).resume_state())

    def test_plain_checkpoint_is_read_for_jammed(self):
        state = RealizationRecord.open(self.run_dir(suffix=''), kind='auxetic').resume_state()
        self.assertEqual(state.step, 4)


if __name__ == '__main__':
    unittest.main()
