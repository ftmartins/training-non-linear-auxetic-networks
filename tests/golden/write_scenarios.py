"""
Write side and realization lifecycle: run today's trainers end to end on tmp dirs,
with the physics replaced by a fast deterministic fake, through fresh / kill + resume
/ extend / early stop / NaN / overwrite sequences, and digest every file they leave.

Only the physics is faked: geometry, seeds, checkpoints, resume, markers, metadata and
every write are today's code. Each scenario returns the outcome of each invocation and
a digest of the run directory (dir_digest), which test_write compares to expected/write.
"""
import contextlib
import io
import json
import pickle
import re
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

from tests.golden import REPO_ROOT
from tests.golden.digest import digest

for _p in (REPO_ROOT, REPO_ROOT / 'training' / 'runners'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Per-invocation provenance that legitimately differs between runs of the same code.
_VOLATILE_PROVENANCE = ('timestamp', 'git_commit', 'git_dirty', 'argv')
_ISO_TIME = re.compile(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?')


def _file_digest(path):
    name = path.name
    if name.endswith('.gz'):
        return 'code snapshot'   # its content is the code under refactor
    if name.endswith('.json'):
        data = json.loads(path.read_text())
        if name.startswith('run_provenance'):
            data = [{k: v for k, v in inv.items() if k not in _VOLATILE_PROVENANCE}
                    for inv in (data if isinstance(data, list) else [data])]
        return {'json': data}
    if name.endswith('.npy'):
        return {'npy_sha256': digest(np.frombuffer(path.read_bytes(), np.uint8))['sha256'],
                'array': digest(np.load(path))}
    if name.endswith('.pkl'):
        # Content, not bytes: a resumed array keeps the dtype object unpickled from the
        # checkpoint, so the pickle memo can differ while the content is identical.
        with open(path, 'rb') as f:
            return {'pickle': digest(pickle.load(f))}
    if name.endswith('.npz'):
        with np.load(path, allow_pickle=True) as z:
            return {'npz': digest(z)}
    if name.endswith('.txt'):
        return {'text': _ISO_TIME.sub('<time>', path.read_text())}
    return {'sha256': digest(np.frombuffer(path.read_bytes(), np.uint8))['sha256']}


def dir_digest(root):
    root = Path(root)
    return {str(p.relative_to(root)): _file_digest(p)
            for p in sorted(root.rglob('*')) if p.is_file()}


def _quiet():
    stack = contextlib.ExitStack()
    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
    return stack


# ── Allosteric trainer ────────────────────────────────────────────────────────

class FakeAllostericPhysics:
    """Stand-in for allosteric_trainer.evaluate_actuation.

    Moves only output node 3, along the node-2 -> node-3 direction: the free state sits
    at distance tod * (1 + 0.5 * mean stiffness) from node 2, the clamped state at tod.
    Calls are counted (two per training step: task 1, task 2); from call `converge_at`
    on the free state hits tod exactly (mse 0, early stop), call `nan_at` returns NaN
    positions, call `raise_at` raises (a killed job).
    """

    def __init__(self, converge_at=None, nan_at=None, raise_at=None):
        self.calls = 0
        self.converge_at, self.nan_at, self.raise_at = converge_at, nan_at, raise_at

    def __call__(self, nodes, incidence_matrix, stiffnesses, tod, dx, nsteps, **_):
        call, self.calls = self.calls, self.calls + 1
        if self.raise_at is not None and call == self.raise_at:
            raise KeyboardInterrupt('fake job killed')
        free = np.array(nodes, dtype=float)
        u = (free[3] - free[2]) / np.linalg.norm(free[3] - free[2])
        clamped = free.copy()
        clamped[3] = free[2] + u * tod
        if self.converge_at is not None and call >= self.converge_at:
            free[3] = clamped[3]
        else:
            free[3] = free[2] + u * tod * (1 + 0.5 * float(np.mean(stiffnesses)))
        if self.nan_at is not None and call == self.nan_at:
            free[:] = np.nan
        return None, free, clamped


ALLOSTERIC_SEED = 1_000_032   # stands in for the screened-seed table, untracked for allosteric


def run_allosteric(output_dir, steps, physics, learning_rate=None, overwrite=False,
                   task=0, realization=0):
    """One allosteric_trainer.main() invocation (targeted ensemble). Returns 'ok' or the
    exception it ends with."""
    import allosteric_trainer as T
    argv = ['allosteric_trainer.py', '--targeted-ensemble', '--task-id', str(task),
            '--realization-id', str(realization), '--training-steps', str(steps),
            '--output-dir', str(output_dir), '--solver', 'jax_fire']
    if learning_rate is not None:
        argv += ['--learning-rate', repr(learning_rate)]
    if overwrite:
        argv += ['--overwrite']
    with mock.patch.object(T, 'evaluate_actuation', physics), \
            mock.patch.object(T, 'get_realization_seed', lambda *a, **k: ALLOSTERIC_SEED), \
            mock.patch.object(sys, 'argv', argv), _quiet():
        try:
            T.main()
            return 'ok'
        except BaseException as e:  # noqa: BLE001 - a killed job is a scenario
            return f'{type(e).__name__}: {e}'.replace(str(output_dir), '<out>')


ALLOSTERIC_SCENARIOS = {
    # name: list of invocations (steps, physics kwargs, extra run_allosteric kwargs)
    'fresh_60': [(60, {}, {})],
    'early_stop_at_step_7': [(60, {'converge_at': 14}, {})],
    'extend_60_to_120': [(60, {}, {}), (120, {}, {})],
    'killed_at_step_55_then_resumed_to_120': [(120, {'raise_at': 110}, {}), (120, {}, {})],
    'nan_at_step_52_then_resumed_to_60': [(60, {'nan_at': 104}, {}), (60, {}, {})],
    'nan_at_step_52': [(60, {'nan_at': 104}, {})],
    'lr_change_without_overwrite': [(60, {}, {}), (60, {}, {'learning_rate': 5.0})],
    'lr_change_with_overwrite': [(60, {}, {}),
                                 (60, {}, {'learning_rate': 5.0, 'overwrite': True})],
}


def allosteric_scenario(name):
    with tempfile.TemporaryDirectory() as tmp:
        outcomes = [run_allosteric(tmp, steps, FakeAllostericPhysics(**phys), **kw)
                    for steps, phys, kw in ALLOSTERIC_SCENARIOS[name]]
        return {'outcomes': outcomes, 'dir': dir_digest(tmp)}


# ── Targeted auxetic runner (stored Network A) ───────────────────────────────

def run_targeted(root, steps, trainer_kwargs, lr=None, overwrite=False, task=0, realization=0):
    """One targeted_ensemble_runner.run_single_training() invocation with the JAX trainer
    replaced by the fake from tests.test_targeted_runner_checkpoint. Returns its return
    value or the exception it raises."""
    import targeted_ensemble_runner as R
    from base.stored_network import NETWORK_A_FILE
    from tests.test_targeted_runner_checkpoint import _fake_trainer
    loss_at = trainer_kwargs.pop('early_stop_at', None)
    if loss_at is not None:
        trainer_kwargs['loss_at'] = lambda s, k=loss_at: 1e-7 if s >= k else 1.0 / (s + 1)
    if 'nan_loss_at' in trainer_kwargs:
        k = trainer_kwargs.pop('nan_loss_at')
        trainer_kwargs['loss_at'] = lambda s: np.nan if s == k else 1.0 / (s + 1)
    with mock.patch.object(R, 'finish_training_GD_auxetic_batch_jax',
                           _fake_trainer(**trainer_kwargs)), _quiet():
        try:
            return R.run_single_training(
                task, realization, gradient_method='jax', network_type='jammed',
                overwrite=overwrite, lr_override=lr, steps_override=steps,
                network_file=NETWORK_A_FILE, results_root=root)
        except Exception as e:  # noqa: BLE001
            return f'{type(e).__name__}: {e}'.replace(str(root), '<root>')


TARGETED_SCENARIOS = {
    'all_steps_12': [(12, {}, {})],
    'early_stop_at_step_7': [(50, {'early_stop_at': 7}, {})],
    'extend_12_to_24': [(12, {}, {}), (24, {}, {})],
    'nan_loss_at_step_8_then_resumed_to_24': [(24, {'nan_loss_at': 8}, {}), (24, {}, {})],
    'nan_stiffness_at_step_8': [(24, {'nan_stiffness_at': 8}, {})],
    'exception_at_step_8_then_resumed_to_24': [(24, {'raise_at': 8}, {}), (24, {}, {})],
    'lr_change_without_overwrite': [(12, {}, {}), (12, {}, {'lr': 1e-3})],
    'lr_change_with_overwrite': [(12, {}, {}), (12, {}, {'lr': 1e-3, 'overwrite': True})],
}


def targeted_scenario(name, task=0, realization=0):
    """Outcomes of the invocations, then of is_training_complete (which may write a
    marker), then the digest of the whole results tree."""
    from training.src.checkpoint_manager import is_training_complete
    with tempfile.TemporaryDirectory() as tmp:
        outcomes = [digest(run_targeted(tmp, steps, dict(tk), **kw))
                    for steps, tk, kw in TARGETED_SCENARIOS[name]]
        with _quiet():
            complete = is_training_complete(task, realization, results_dir=Path(tmp) / 'jax',
                                            network_type='jammed')
        return {'outcomes': outcomes, 'is_training_complete': complete, 'dir': dir_digest(tmp)}


# ── General auxetic runner (ensemble_runner) ─────────────────────────────────

GENERAL_SEED = 2_000_017   # stands in for the screened-seed table, untracked for auxetic_general


def run_general(root, steps, trainer_kwargs, lr=None, overwrite=False, task=0, realization=0):
    """One ensemble_runner.run_single_training() invocation. The JAX trainer is the same
    fake as for the targeted runner; network creation (a ~30 s packing) is replaced by
    the stored Network A. Returns its return value or the exception it raises."""
    import ensemble_runner as R
    from base.stored_network import NETWORK_A_FILE, load_stored_network
    from tests.test_targeted_runner_checkpoint import _fake_trainer
    from training.src.task_generator import generate_task_config
    trainer_kwargs = dict(trainer_kwargs)
    k = trainer_kwargs.pop('early_stop_at', None)
    if k is not None:
        trainer_kwargs['loss_at'] = lambda s: 1e-7 if s >= k else 1.0 / (s + 1)
    if 'nan_loss_at' in trainer_kwargs:
        k = trainer_kwargs.pop('nan_loss_at')
        trainer_kwargs['loss_at'] = lambda s: np.nan if s == k else 1.0 / (s + 1)
    fake = _fake_trainer(**trainer_kwargs)

    def train(**kw):   # ensemble_runner passes no task_config; the fake wants one
        return fake(task_config=generate_task_config(kw['task_seed']), **kw)

    def stored_network(**_):
        network, boundary, _info = load_stored_network(NETWORK_A_FILE)
        return network, boundary

    with mock.patch.object(R, 'finish_training_GD_auxetic_batch_jax', train), \
            mock.patch.object(R, 'create_auxetic_network', stored_network), \
            mock.patch.object(R, 'get_realization_seed', lambda *a, **k: GENERAL_SEED), \
            mock.patch.object(R, 'RESULTS_DIR', Path(root)), _quiet():
        try:
            return R.run_single_training(
                task, realization, gradient_method='jax', network_type='jammed',
                overwrite=overwrite, lr_override=lr, steps_override=steps)
        except Exception as e:  # noqa: BLE001
            return f'{type(e).__name__}: {e}'.replace(str(root), '<root>')


GENERAL_SCENARIOS = {
    'all_steps_12': [(12, {}, {})],
    'early_stop_at_step_7': [(50, {'early_stop_at': 7}, {})],
    'extend_12_to_24': [(12, {}, {}), (24, {}, {})],
    'nan_loss_at_step_8_then_resumed_to_24': [(24, {'nan_loss_at': 8}, {}), (24, {}, {})],
    'exception_at_step_8_then_resumed_to_24': [(24, {'raise_at': 8}, {}), (24, {}, {})],
    'lr_change_without_overwrite': [(12, {}, {}), (12, {}, {'lr': 1e-3})],
    'lr_change_with_overwrite': [(12, {}, {}), (12, {}, {'lr': 1e-3, 'overwrite': True})],
}


def general_scenario(name, task=0, realization=0):
    from training.src.checkpoint_manager import is_training_complete
    with tempfile.TemporaryDirectory() as tmp:
        outcomes = [digest(run_general(tmp, steps, tk, **kw))
                    for steps, tk, kw in GENERAL_SCENARIOS[name]]
        with _quiet():
            complete = is_training_complete(task, realization, results_dir=Path(tmp) / 'jax',
                                            network_type='jammed')
        return {'outcomes': outcomes, 'is_training_complete': complete, 'dir': dir_digest(tmp)}
