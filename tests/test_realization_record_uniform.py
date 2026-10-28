"""The uniform layout (ADR 0006) holds the same record content as the legacy layouts.

The same sequence of record writes goes into a legacy and a uniform run directory; every part
read back through the record must be bit-identical. Also: a run keeps the layout it started
in, and a killed write leaves the previous complete file.

Run: python -m unittest tests.test_realization_record_uniform
"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from base.realization_record import (
    PARTS, LAYOUT_ENV, AllostericTask, BestSoFar, Loss, MissingRecordPart, NotARealization,
    RealizationRecord)

N_NODES, N_EDGES = 6, 5


def allosteric_network():
    rng = np.random.RandomState(1)
    inc = np.zeros((N_EDGES, N_NODES))
    for e in range(N_EDGES):
        inc[e, e], inc[e, e + 1] = -1.0, 1.0
    return SimpleNamespace(nodes=rng.rand(N_NODES, 2), incidence_matrix=inc,
                           eq_lengths=rng.rand(N_EDGES))


def play_allosteric(record, steps=6, stop_after=None, nan_at=None):
    """What allosteric_trainer does to a record: network and task, a checkpoint per step, a
    final save. nan_at: the step whose post-update stiffnesses hold NaN."""
    rng = np.random.RandomState(2)
    record.start()
    record.write_network(allosteric_network())
    record.write_task(AllostericTask(1000100.0, -0.8, -0.6))
    (record.run_dir / 'training_meta.json').write_text(json.dumps({'realization_seed': 1000032}))
    (record.run_dir / 'run_provenance.json').write_text(json.dumps([{'timestamp': '2026-10-01T00:00:00'}]))
    mse = []
    best = None
    for step in range(1, (stop_after or steps) + 1):
        measured = rng.rand(N_EDGES)
        current = rng.rand(N_EDGES)
        if step == nan_at:
            current[0] = np.nan
        mse.append((1.0 / step, 0.5 / step))
        loss = Loss(np.array(mse))
        if best is None or loss.training[-1] < best.training_loss:
            best = BestSoFar(measured, float(loss.training[-1]))
            changed = best
        else:
            changed = None
        record.save_checkpoint(step, current, loss, measured, changed)
    record.save_final(current, loss, best)


def auxetic_network():
    rng = np.random.RandomState(3)
    return SimpleNamespace(positions=rng.rand(N_NODES, 2), edges=np.array([[i, i + 1] for i in range(N_EDGES)]),
                           stiffnesses=rng.rand(N_EDGES), rest_lengths=rng.rand(N_EDGES))


def play_auxetic(record, steps=5, checkpoint_at=None):
    rng = np.random.RandomState(4)
    record.start()
    (record.run_dir / record.meta_file_name('training_meta')).write_text(json.dumps({'realization_seed': 7}))
    task_config = {'task_seed': 0, 'compression_strains': [-0.2], 'target_poisson_ratios': [-0.8]}
    history = {'stiffnesses': [], 'loss': [], 'positions': [], 'freetraj': []}
    network = auxetic_network()
    for step in range(1, steps + 1):
        history['stiffnesses'].append(rng.rand(N_EDGES))
        history['loss'].append(1.0 / step)
        history['positions'].append(rng.rand(N_NODES, 2))
        network.stiffnesses = rng.rand(N_EDGES)
        if step == checkpoint_at:
            record.save_training_checkpoint(step, history, network, task_config, 0, 7)
    boundary = {'top': [0], 'bottom': [1], 'left': [2], 'right': [3]}
    record.save_results(history, network, task_config, boundary)


def content(record):
    """Every part the record can give, as plain comparable data (or the exception name)."""
    out = {}
    for part in PARTS:
        if part == 'sweep':
            continue
        try:
            value = getattr(record, part)
        except Exception as e:  # noqa: BLE001
            out[part] = type(e).__name__
            continue
        out[part] = _plain(value)
    resume = record.resume_state()
    out['resume'] = None if resume is None else _plain(
        {'step': resume.step, 'k': resume.stiffnesses, 'loss': resume.loss.training,
         'source': resume.source,
         'trainer_step': (resume.trainer_state or {}).get('current_step')})
    return out


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if hasattr(x, '__dict__') and not isinstance(x, np.ndarray):
        return _plain({k: v for k, v in vars(x).items() if not k.startswith('_')})
    if isinstance(x, np.ndarray):
        return (x.dtype.str, x.shape, x.tobytes())
    return x


class UniformTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def record(self, name, kind, layout):
        with mock.patch.dict('os.environ', {LAYOUT_ENV: layout}):
            return RealizationRecord.open(self.root / name, kind=kind)

    def assertSameContent(self, play, kind, **kw):
        legacy, uniform = self.record('legacy', kind, 'legacy'), self.record('uniform', kind, 'uniform')
        play(legacy, **kw)
        play(uniform, **kw)
        self.assertTrue((uniform.run_dir / 'record.json').exists())
        self.assertFalse((legacy.run_dir / 'record.json').exists())
        a, b = content(RealizationRecord.open(legacy.run_dir)), content(RealizationRecord.open(uniform.run_dir))
        self.assertEqual(a, b)
        return a


class TestAllosteric(UniformTestCase):
    def test_finished_run(self):
        got = self.assertSameContent(play_allosteric, 'allosteric')
        self.assertNotIn('MissingRecordPart', got['best_loss_stiffnesses'])

    def test_killed_run_resumes_from_the_same_state(self):
        got = self.assertSameContent(play_allosteric, 'allosteric', stop_after=3, nan_at=None)
        self.assertEqual(got['resume']['step'] if isinstance(got['resume'], dict) else None, 3)

    def test_nan_run_rolls_back_to_the_last_clean_checkpoint(self):
        got = self.assertSameContent(play_allosteric, 'allosteric', nan_at=4, stop_after=4)
        self.assertEqual(got['resume']['source'], 'checkpoint')
        self.assertEqual(got['resume']['step'], 3)

    def test_detection_without_a_kind(self):
        rec = self.record('uniform', 'allosteric', 'uniform')
        play_allosteric(rec)
        self.assertEqual(RealizationRecord.open(rec.run_dir).kind, 'allosteric')


class TestAuxetic(UniformTestCase):
    def test_finished_run(self):
        got = self.assertSameContent(play_auxetic, 'auxetic')
        self.assertNotIn('MissingRecordPart', got['positions_trajectory'])

    def test_checkpoint_resume_state(self):
        got = self.assertSameContent(play_auxetic, 'auxetic', checkpoint_at=3)
        self.assertEqual(got['resume']['step'], 3)
        self.assertEqual(got['resume']['trainer_step'], 3)

    def test_drop_checkpoint(self):
        rec = self.record('uniform', 'auxetic', 'uniform')
        play_auxetic(rec, checkpoint_at=3)
        rec.drop_checkpoint()
        self.assertIsNone(rec.resume_state())

    def test_markers(self):
        rec = self.record('uniform', 'auxetic', 'uniform')
        play_auxetic(rec)
        rec.mark_complete()
        rec.mark_complete_small_loss(1e8)
        self.assertTrue(rec.has_marker('training_complete'))
        self.assertTrue(rec.has_marker('training_complete_small_loss'))


class TestLayoutChoice(UniformTestCase):
    def test_default_for_a_fresh_directory_is_legacy(self):
        with mock.patch.dict('os.environ', clear=False):
            import os
            os.environ.pop(LAYOUT_ENV, None)
            rec = RealizationRecord.open(self.root / 'fresh', kind='allosteric')
            play_allosteric(rec)
        self.assertFalse((rec.run_dir / 'record.json').exists())
        self.assertTrue((rec.run_dir / 'mse1.npy').exists())

    def test_a_run_keeps_the_layout_it_started_in(self):
        legacy = self.record('old', 'allosteric', 'legacy')
        play_allosteric(legacy, stop_after=3)
        uniform = self.record('new', 'allosteric', 'uniform')
        play_allosteric(uniform, stop_after=3)
        for name, layout in (('old', 'uniform'), ('new', 'legacy')):   # flipped setting
            reopened = self.record(name, 'allosteric', layout)
            reopened.save_final(np.ones(N_EDGES), Loss(np.ones((3, 2))), None)
        self.assertTrue((legacy.run_dir / 'mse1.npy').exists())
        self.assertFalse((legacy.run_dir / 'record.json').exists())
        self.assertTrue((uniform.run_dir / 'loss.npy').exists())
        self.assertFalse((uniform.run_dir / 'mse1.npy').exists())

    def test_iterate_yields_both_layouts(self):
        for r, layout in ((0, 'legacy'), (1, 'uniform')):
            rec = RealizationRecord.open(self.root / 'task_00' / f'realization_{r:02d}', kind='auxetic') \
                if layout == 'legacy' else None
            if rec is None:
                with mock.patch.dict('os.environ', {LAYOUT_ENV: 'uniform'}):
                    rec = RealizationRecord.open(self.root / 'task_00' / f'realization_{r:02d}', kind='auxetic')
            play_auxetic(rec)
        recs = list(RealizationRecord.iterate(self.root, 'auxetic'))
        self.assertEqual([(r.realization_id, (r.run_dir / 'record.json').exists()) for r in recs],
                         [(0, False), (1, True)])
        self.assertEqual(content(recs[0])['loss'], content(recs[1])['loss'])


class TestSweepFiles(UniformTestCase):
    def test_file_names_per_layout(self):
        names = lambda rec: tuple(p.name for p in rec.sweep_files())
        self.assertEqual(names(self.record('a', 'allosteric', 'legacy')),
                         ('timestep_sweep.npz', 'timestep_sweep_meta.json'))
        self.assertEqual(names(RealizationRecord.open(self.root / 'x', kind='auxetic')),
                         ('timestep_sweep_jammed.npz', 'timestep_sweep_meta_jammed.json'))
        self.assertEqual(names(RealizationRecord.open(self.root / 'y', kind='auxetic',
                                                      network_type='lattice')),
                         ('timestep_sweep_lattice.npz', 'timestep_sweep_meta_lattice.json'))
        self.assertEqual(names(RealizationRecord.open(self.root / 'z', kind='auxetic',
                                                      network_type=None)),
                         ('timestep_sweep.npz', 'timestep_sweep_meta.json'))
        self.assertEqual(names(self.record('b', 'allosteric', 'uniform')),
                         ('sweep.npz', 'sweep_meta.json'))

    def test_a_sweep_written_where_sweep_files_says_is_read_back(self):
        for kind, play, layout in (('allosteric', play_allosteric, 'uniform'),
                                   ('allosteric', play_allosteric, 'legacy')):
            rec = self.record(f'{kind}-{layout}', kind, layout)
            play(rec)
            path, _ = rec.sweep_files()
            np.savez(path, cost_hessian_after_stiffness=rec.best_loss_stiffnesses, t=np.arange(3))
            rec.refresh()
            self.assertEqual(sorted(rec.sweep), ['cost_hessian_after_stiffness', 't'])


class TestAtomicWrites(UniformTestCase):
    def test_a_killed_write_leaves_the_previous_file(self):
        rec = self.record('uniform', 'allosteric', 'uniform')
        play_allosteric(rec, stop_after=2)
        before = (rec.run_dir / 'loss.npy').read_bytes()
        with mock.patch('numpy.save', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                rec.save_final(np.ones(N_EDGES), Loss(np.ones((9, 2))), None)
        self.assertEqual((rec.run_dir / 'loss.npy').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
