"""
training.src.realization_lifecycle through run(), on tmp run directories with a fake trainer:
every stage (fresh, resume, extend, overwrite, complete), the mismatch refusal, and how
the outcome's status follows the trainer.

Run: python -m unittest tests.test_realization_lifecycle
"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from base.realization_record import (
    RealizationRecord, Network, AllostericTask, Loss, BestSoFar)
from training.src.realization_lifecycle import (
    run, SubtaskLossRatioBelow, TrainState, TrainResult)
from training.src.run_provenance import HyperparameterMismatchError

N_EDGES = 4
CRITICAL = frozenset({'learning_rate'})
STOP = SubtaskLossRatioBelow(1e-5)


class FakeTrainer:
    """Loss at step s is (1/(s+1), 2/(s+1)) per subtask, or tiny from `converge_at` on;
    stiffnesses shrink by 1% per step; a checkpoint every `every` steps; dies with
    KeyboardInterrupt at step `die_at` (a killed job)."""

    def __init__(self, converge_at=None, every=5, die_at=None):
        self.converge_at, self.every, self.die_at = converge_at, every, die_at
        self.prepared = self.trained = 0

    def prepare(self, record):
        self.prepared += 1
        if not record.has('network'):
            inc = np.zeros((N_EDGES, N_EDGES + 1))
            for e in range(N_EDGES):
                inc[e, e], inc[e, e + 1] = -1.0, 1.0
            record.write_network(Network(np.arange(10.0).reshape(5, 2), inc, np.ones(N_EDGES)))
        record.write_task(AllostericTask(1.0, -0.8, -0.6))

    def fresh_state(self):
        return TrainState(0, np.ones(N_EDGES), Loss(np.empty((0, 2))))

    def loss_at(self, s):
        if self.converge_at is not None and s >= self.converge_at:
            return np.array([1e-9, 1e-9])
        return np.array([1.0, 2.0]) / (s + 1)

    def train(self, state, n_steps, stop_rule, on_checkpoint):
        self.trained += 1
        k, rows, best = state.stiffnesses, list(state.loss.per_subtask), state.best
        status = 'steps_exhausted'
        for s in range(state.step, state.step + n_steps):
            if s == self.die_at:
                raise KeyboardInterrupt('killed')
            measured, k = k, k * 0.99
            rows.append(self.loss_at(s))
            if best is None or rows[-1].mean() < best.training_loss:
                best = BestSoFar(measured.copy(), rows[-1].mean())
            loss = Loss(np.array(rows))
            if (s + 1) % self.every == 0:
                on_checkpoint(s + 1, k, loss, measured, best)
            if stop_rule.reached(rows[0], rows[-1]):
                on_checkpoint(s + 1, k, loss, measured, best)
                status = 'converged'
                break
        return TrainResult(TrainState(len(rows), k, Loss(np.array(rows)), best), status)


class LifecycleTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.record = RealizationRecord.open(Path(self._tmp.name) / 'realization_0',
                                             kind='allosteric')

    def tearDown(self):
        self._tmp.cleanup()

    def invoke(self, trainer=None, steps=20, lr=1.0, overwrite=False, on_mismatch=None):
        def refuse(record):
            raise HyperparameterMismatchError('refused')
        with contextlib.redirect_stdout(io.StringIO()):
            return run(self.record, trainer or FakeTrainer(),
                       hparams={'learning_rate': lr, 'n_training_steps': steps},
                       critical_keys=CRITICAL, code_files=[], target_steps=steps,
                       overwrite=overwrite, stop_rule=STOP, on_mismatch=on_mismatch or refuse)

    def stored_steps(self):
        self.record.refresh()
        return self.record.loss.n_steps


class TestStages(LifecycleTestCase):
    def test_fresh_trains_to_the_target(self):
        out = self.invoke(steps=20)
        self.assertEqual((out.stage, out.status, out.state.step), ('fresh', 'steps_exhausted', 20))
        self.assertEqual(self.stored_steps(), 20)
        self.assertEqual(self.record.config['n_training_steps'], 20)

    def test_killed_run_resumes_from_its_last_checkpoint(self):
        with self.assertRaises(KeyboardInterrupt):
            self.invoke(FakeTrainer(die_at=13), steps=20)
        self.assertEqual(self.stored_steps(), 10)
        out = self.invoke(steps=20)
        self.assertEqual((out.stage, out.state.step), ('resume', 20))

    def test_resumed_run_ends_as_a_straight_run(self):
        with self.assertRaises(KeyboardInterrupt):
            self.invoke(FakeTrainer(die_at=13), steps=20)
        self.invoke(steps=20)
        resumed = self.record.loss.per_subtask
        self.record.wipe()
        self.invoke(steps=20)
        self.record.refresh()
        np.testing.assert_array_equal(self.record.loss.per_subtask, resumed)

    def test_larger_target_extends(self):
        self.invoke(steps=20)
        out = self.invoke(steps=30)
        self.assertEqual((out.stage, out.state.step), ('extend', 30))
        self.assertEqual(self.record.config['n_training_steps'], 30)

    def test_reached_target_is_complete_and_does_not_train(self):
        self.invoke(steps=20)
        trainer = FakeTrainer()
        out = self.invoke(trainer, steps=20)
        self.assertEqual((out.stage, out.status, trainer.trained), ('complete', None, 0))
        self.assertEqual(trainer.prepared, 1)

    def test_converged_run_stops_early(self):
        out = self.invoke(FakeTrainer(converge_at=7), steps=20)
        self.assertEqual((out.status, out.state.step), ('converged', 8))
        self.assertEqual(self.stored_steps(), 8)


class TestHyperparameterMismatch(LifecycleTestCase):
    def test_without_overwrite_on_mismatch_decides(self):
        self.invoke(steps=20, lr=1.0)
        with self.assertRaises(HyperparameterMismatchError):
            self.invoke(steps=20, lr=2.0)
        self.assertEqual(self.record.config['learning_rate'], 1.0)

    def test_a_returning_on_mismatch_refuses_without_writing(self):
        self.invoke(steps=20, lr=1.0)
        trainer = FakeTrainer()
        out = self.invoke(trainer, steps=20, lr=2.0, on_mismatch=lambda record: False)
        self.assertEqual((out.stage, out.refusal, trainer.prepared), ('refused', False, 0))

    def test_overwrite_wipes_and_starts_fresh(self):
        self.invoke(steps=20, lr=1.0)
        (self.record.run_dir / 'stray.txt').write_text('x')
        out = self.invoke(steps=12, lr=2.0, overwrite=True)
        self.assertEqual((out.stage, out.state.step), ('overwrite', 12))
        self.assertFalse((self.record.run_dir / 'stray.txt').exists())
        self.record.refresh()
        self.assertEqual(self.record.config['learning_rate'], 2.0)

    def test_overwrite_without_mismatch_keeps_the_run(self):
        self.invoke(steps=20)
        out = self.invoke(steps=30, overwrite=True)
        self.assertEqual((out.stage, out.state.step), ('extend', 30))


class TestFinalState(LifecycleTestCase):
    def test_best_so_far_is_carried_across_invocations(self):
        self.invoke(steps=20)
        self.invoke(steps=30)
        self.record.refresh()
        self.assertEqual(self.record.best_so_far.training_loss, 1.5 / 30)

    def test_stop_rule_compares_every_subtask_with_its_first_loss(self):
        self.assertTrue(STOP.reached((1.0, 2.0), (9e-6, 1.9e-5)))
        self.assertFalse(STOP.reached((1.0, 2.0), (9e-6, 2e-5)))
        self.assertFalse(STOP.reached((1.0, 2.0), (np.nan, 0.0)))


if __name__ == '__main__':
    unittest.main()
