"""
Today's public read-side calls for one sample realization, run the way their callers
run them. Each call's outcome is kept as {'value'} or {'raises'}, plus any warnings,
so a fallback that warns today, or a read that raises, is pinned as well.
"""
import contextlib
import io
import sys
import warnings

from tests.golden import REPO_ROOT, DATA_ROOT
from tests.golden.digest import digest, portable

for _p in (REPO_ROOT, REPO_ROOT / 'training' / 'runners',
           REPO_ROOT / 'analysis' / 'notebooks' / 'figures'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import aug26_common as A  # noqa: E402
import allosteric_trainer  # noqa: E402
from analysis import data_io  # noqa: E402
from training.src import checkpoint_manager  # noqa: E402

def outcome(fn, *args, **kwargs):
    """Digest of calling fn: its value or the exception it raises, and its warnings."""
    with warnings.catch_warnings(record=True) as caught, \
            contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('always')
        try:
            out = {'value': digest(fn(*args, **kwargs))}
        except Exception as e:  # noqa: BLE001 - the exception is the pinned behaviour
            out = {'raises': portable(f'{type(e).__name__}: {e}')}
    msgs = sorted({portable(f'{w.category.__name__}: {w.message}') for w in caught
                   if not issubclass(w.category, DeprecationWarning)})
    if msgs:
        out['warnings'] = msgs
    return out


def _value(fn, *args, **kwargs):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        try:
            return fn(*args, **kwargs)
        except Exception:  # noqa: BLE001
            return None


def read_allosteric(geometry, task, real):
    run_dir = A._allo_geom_dir(geometry) / f'task_{task}' / f'realization_{real}'
    out = {
        'load_allosteric': outcome(A.load_allosteric, geometry, task, real),
        'allo_realization_seed': outcome(A.allo_realization_seed, geometry, task, real),
        'load_resume_state': outcome(allosteric_trainer.load_resume_state, str(run_dir)),
    }
    d = _value(A.load_allosteric, geometry, task, real)
    if d is not None:
        out['allo_best_loss_stiffness'] = outcome(A.allo_best_loss_stiffness, d)
        out['allo_operating_point.best'] = outcome(A.allo_operating_point, d, 'best')
        out['allo_operating_point.before'] = outcome(A.allo_operating_point, d, 'before')
        out['allo_initial_stiffnesses'] = outcome(A.allo_initial_stiffnesses, d)
    return out


def read_auxetic(family, task, real):
    """A run in data/auxetic_nets_aug/<family>, read through aug26_common and data_io."""
    data_dir = DATA_ROOT / 'auxetic_nets_aug' / family
    out = {
        'load_auxetic': outcome(A.load_auxetic, task, real, family),
        'aux_realization_seed': outcome(A.aux_realization_seed, task, real, family),
        'aux_network_fingerprint': outcome(A.aux_network_fingerprint, task, real, family),
        'data_io.load_auxetic_network': outcome(
            data_io.load_auxetic_network, task, real, data_dir=data_dir, network_type='jammed'),
        'load_checkpoint': outcome(
            checkpoint_manager.load_checkpoint, task, real, results_dir=data_dir,
            network_type='jammed'),
    }
    aa = _value(A.load_auxetic, task, real, family)
    if aa is not None:
        out['aux_best_loss_state'] = outcome(A.aux_best_loss_state, aa)
        out['aux_best_loss_state.no_positions'] = outcome(
            A.aux_best_loss_state, aa, load_positions=False)
    return out


def read_auxetic_legacy(data_dir, task, real, network_type):
    """A run in an older auxetic tree (data/auxetic_nets/...), which only data_io and
    checkpoint_manager read; network_type None means the plain, unsuffixed names."""
    data_dir = DATA_ROOT / data_dir
    return {
        'data_io.load_auxetic_network': outcome(
            data_io.load_auxetic_network, task, real, data_dir=data_dir,
            network_type=network_type),
        'load_checkpoint': outcome(
            checkpoint_manager.load_checkpoint, task, real, results_dir=data_dir,
            network_type=network_type or 'jammed'),
    }
