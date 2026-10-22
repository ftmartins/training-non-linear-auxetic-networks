"""
Checkpoint retention and exit status of the targeted auxetic runner, per training outcome:
the checkpoint is removed only on early stop, kept (at the last step reached) when all steps
ran so a later --steps extends from there, kept on NaN and on exceptions; NaN in the final
loss or stiffnesses fails the run.

The real trainer is replaced by a fast deterministic fake that writes periodic checkpoints the
same way (every save_interval steps, current_step = len(history['loss'])).

Run: python -m unittest tests.test_targeted_runner_checkpoint
"""
import contextlib
import io
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / 'training' / 'src', _ROOT / 'training' / 'runners'):
    sys.path.insert(0, str(_p))

import targeted_ensemble_runner as runner
from base.stored_network import NETWORK_A_FILE
from training.src.checkpoint_manager import save_checkpoint

TASK, REAL = 0, 0


def _fake_trainer(loss_at=lambda step: 1.0 / (step + 1), nan_stiffness_at=None, raise_at=None):
    """Deterministic stand-in for finish_training_GD_auxetic_batch_jax.

    Loss is a function of the global step only, stiffnesses shrink by 1% per step, so a
    straight run and a resumed run must give identical trajectories.
    """
    calls = []

    def train(network, history, learning_rate, n_steps, task_seed, realization_seed,
              save_interval, task_config, TARGETED_RESULTS_DIR, network_type,
              loss_tol=1e-6, **_):
        calls.append({'start_step': len(history.get('loss', [])), 'n_steps': n_steps})
        for key in ('stiffnesses', 'loss', 'positions'):
            history.setdefault(key, [])
        for step in range(n_steps):
            global_step = len(history['loss'])
            if raise_at is not None and global_step == raise_at:
                raise RuntimeError('fake solver failure')
            loss = loss_at(global_step)
            history['stiffnesses'].append(np.copy(network.stiffnesses))
            history['loss'].append(loss)
            history['positions'].append(np.copy(network.positions))
            network.stiffnesses = np.asarray(network.stiffnesses) * 0.99
            if nan_stiffness_at is not None and global_step == nan_stiffness_at:
                network.stiffnesses = np.full_like(network.stiffnesses, np.nan)
            if np.isnan(network.stiffnesses).any() or np.isnan(loss):
                break
            if step % save_interval == 0:
                save_checkpoint(task_seed, realization_seed, history, network, task_config,
                                current_step=len(history['loss']),
                                results_dir=TARGETED_RESULTS_DIR, network_type=network_type)
            if loss < loss_tol:
                break
        return history, network

    train.calls = calls
    return train


class TestTargetedRunnerCheckpoint(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.run_dir = self.root / 'jax' / f'task_{TASK:02d}' / f'realization_{REAL:02d}'
        self.ckpt = self.run_dir / 'checkpoint_jammed.pkl'

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, trainer, steps, root=None):
        with mock.patch.object(runner, 'finish_training_GD_auxetic_batch_jax', trainer), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return runner.run_single_training(
                TASK, REAL, gradient_method='jax', network_type='jammed',
                steps_override=steps, network_file=NETWORK_A_FILE,
                results_root=root or self.root)

    def _ckpt_step(self):
        with open(self.ckpt, 'rb') as f:
            return pickle.load(f)['current_step']

    def test_all_steps_keeps_checkpoint_at_last_step_and_succeeds(self):
        self.assertTrue(self._run(_fake_trainer(), steps=12))
        self.assertTrue(self.ckpt.exists())
        self.assertEqual(self._ckpt_step(), 12)

    def test_early_stop_removes_checkpoint_and_succeeds(self):
        trainer = _fake_trainer(loss_at=lambda s: 1e-7 if s >= 7 else 1.0)
        self.assertTrue(self._run(trainer, steps=50))
        self.assertEqual(len(np.load(self.run_dir / 'loss_trajectory_jammed.npy')), 8)
        self.assertFalse(self.ckpt.exists())

    def test_nan_loss_keeps_last_periodic_checkpoint_and_fails(self):
        trainer = _fake_trainer(loss_at=lambda s: np.nan if s == 8 else 1.0)
        self.assertFalse(self._run(trainer, steps=50))
        self.assertTrue(self.ckpt.exists())
        self.assertEqual(self._ckpt_step(), 6)

    def test_nan_stiffness_with_finite_loss_keeps_checkpoint_and_fails(self):
        trainer = _fake_trainer(nan_stiffness_at=8)
        self.assertFalse(self._run(trainer, steps=50))
        self.assertTrue(self.ckpt.exists())
        self.assertEqual(self._ckpt_step(), 6)

    def test_exception_keeps_checkpoint_and_fails(self):
        self.assertFalse(self._run(_fake_trainer(raise_at=8), steps=50))
        self.assertTrue(self.ckpt.exists())
        self.assertEqual(self._ckpt_step(), 6)

    def test_extend_resumes_from_last_step_and_matches_straight_run(self):
        trainer = _fake_trainer()
        self.assertTrue(self._run(trainer, steps=12))
        self.assertTrue(self._run(trainer, steps=24))
        self.assertEqual([c['start_step'] for c in trainer.calls], [0, 12])

        straight = self.root / 'straight'
        self.assertTrue(self._run(_fake_trainer(), steps=24, root=straight))
        straight_dir = straight / 'jax' / f'task_{TASK:02d}' / f'realization_{REAL:02d}'
        for name in ('loss_trajectory_jammed.npy', 'stiffness_trajectory_jammed.npy'):
            self.assertEqual((self.run_dir / name).read_bytes(),
                             (straight_dir / name).read_bytes(), name)


if __name__ == '__main__':
    unittest.main()
