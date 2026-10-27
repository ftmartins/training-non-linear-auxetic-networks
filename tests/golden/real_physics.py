"""
Real-physics tier: tiny runs of today's trainers with the real JAX solver (3 steps for
the targeted auxetic runner on stored Network A, 2 for the allosteric trainer with
--solver jax_fire), compared to expected/real/ with a float tolerance, because bitwise
equality of solver output is not portable across machines. ~4 minutes, opt-in
(GOLDEN_REAL=1), meant for the refactor's end-to-end gate rather than every edit.
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

from tests.golden import REPO_ROOT
from tests.golden.write_scenarios import ALLOSTERIC_SEED, _quiet, _file_digest

for _p in (REPO_ROOT, REPO_ROOT / 'training' / 'runners'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Numeric files compared to a tolerance; everything else through _file_digest (exact).
_ARRAYS = {
    'targeted': ('loss_trajectory_jammed.npy', 'stiffness_trajectory_jammed.npy'),
    'allosteric': ('stiffnesses.npy', 'best_stiffnesses.npy', 'mse1.npy', 'mse2.npy',
                   'stiffnesses_traj.npy', 'stiffnesses_ckpt.npy'),
}


def _summarize(root, arrays):
    root = Path(root)
    out = {'files': {}, 'arrays': {}}
    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        if p.name in arrays:
            out['arrays'][rel] = np.load(p).tolist()
        elif p.suffix in ('.json', '.txt'):
            out['files'][rel] = _file_digest(p)
        else:
            out['files'][rel] = 'present'   # pickles/npys of solver state: existence only
    return out


def targeted_real():
    import targeted_ensemble_runner as R
    from base.stored_network import NETWORK_A_FILE
    with tempfile.TemporaryDirectory() as tmp, _quiet():
        ret = R.run_single_training(0, 0, gradient_method='jax', network_type='jammed',
                                    steps_override=3, network_file=NETWORK_A_FILE,
                                    results_root=tmp)
        out = _summarize(tmp, _ARRAYS['targeted'])
    out['return'] = ret
    return out


def allosteric_real():
    import allosteric_trainer as T
    with tempfile.TemporaryDirectory() as tmp:
        argv = ['allosteric_trainer.py', '--targeted-ensemble', '--task-id', '0',
                '--realization-id', '0', '--training-steps', '2', '--output-dir', tmp,
                '--solver', 'jax_fire']
        with mock.patch.object(T, 'get_realization_seed', lambda *a, **k: ALLOSTERIC_SEED), \
                mock.patch.object(sys, 'argv', argv), _quiet():
            T.main()
        return _summarize(tmp, _ARRAYS['allosteric'])


SCENARIOS = {'targeted_real': targeted_real, 'allosteric_real': allosteric_real}


def compare(got, want, rtol=1e-6):
    """Mismatches between two summaries, as strings (empty if equal)."""
    bad = []
    if sorted(got['files']) != sorted(want['files']):
        bad.append(f"file set differs: {sorted(set(got['files']) ^ set(want['files']))}")
    bad += [f'{k}' for k in want['files'] if k in got['files'] and got['files'][k] != want['files'][k]]
    for k, v in want['arrays'].items():
        g = got['arrays'].get(k)
        if g is None or np.shape(g) != np.shape(v) or not np.allclose(g, v, rtol=rtol, atol=1e-12):
            bad.append(f'array {k}')
    if got.get('return') != want.get('return'):
        bad.append('return value')
    return bad
