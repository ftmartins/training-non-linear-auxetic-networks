"""
Today's discovery and completion calls, run on a scratch copy of the sample runs so the
marker files they write never reach data/. Small files are copied, large ones
symlinked (nothing writes through a file symlink, only new files are created).
"""
import contextlib
import io
import shutil
import tempfile
from pathlib import Path

from tests.golden import DATA_ROOT, sample
from tests.golden.digest import digest
from tests.golden.readers import A, outcome
from training.src import checkpoint_manager

_COPY_LIMIT = 1_000_000


def _clone(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            if f.stat().st_size <= _COPY_LIMIT:
                shutil.copy2(f, dst / f.name)
            else:
                (dst / f.name).symlink_to(f.resolve())


def sample_dirs():
    """(path under data/) for every sample run."""
    for geometry, task, real, _ in sample.ALLOSTERIC:
        g = 'geometry_targeted' if geometry == 'targeted' else f'geometry_{geometry}'
        yield f'allosteric_nets_aug/{g}/task_{task}/realization_{real}'
    for family, task, real, _ in sample.AUXETIC:
        yield f'auxetic_nets_aug/{family}/task_{task:02d}/realization_{real:02d}'
    for data_dir, task, real, _nt, _ in sample.LEGACY:
        yield f'{data_dir}/task_{task:02d}/realization_{real:02d}'


@contextlib.contextmanager
def scratch_tree():
    """Scratch `data/` with the sample runs, and the loaders' roots pointed at it."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'data'
        for rel in dict.fromkeys(sample_dirs()):
            _clone(DATA_ROOT / rel, root / rel)
        old = (A.REPO_ROOT, A.ALLO_DATA_DIR, A.AUX_DATA_DIR)
        A.REPO_ROOT = Path(tmp)
        A.ALLO_DATA_DIR = root / 'allosteric_nets_aug'
        A.AUX_DATA_DIR = root / 'auxetic_nets_aug' / A.AUX_FAMILY
        try:
            yield root
        finally:
            A.REPO_ROOT, A.ALLO_DATA_DIR, A.AUX_DATA_DIR = old


def _listing(root: Path):
    return sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink())


def discover():
    out = {}
    with scratch_tree() as root:
        for pool in ('targeted', 'general', 'all'):
            for conv in (True, False):
                out[f'discover_allosteric.{pool}.converged={conv}'] = outcome(
                    A.discover_allosteric, pool, None, conv)
        out['discover_allosteric.max1'] = outcome(A.discover_allosteric, 'all', 1, False)
        for family in ('targeted', 'general'):
            for conv in (True, False):
                out[f'discover_auxetic.{family}.converged={conv}'] = outcome(
                    A.discover_auxetic, None, conv, 1e-4, family)
            out[f'discover_auxetic.{family}.ratio1e-2'] = outcome(
                A.discover_auxetic, None, True, 1e-2, family)
        out['discover_auxetic.max1'] = outcome(A.discover_auxetic, 1, False, 1e-4, 'targeted')
        before = set(_listing(root))
        checks = {}
        for data_dir, task, real, nt, _ in sample.LEGACY:
            key = f'{data_dir}/t{task}/r{real}/{nt or "plain"}'
            checks[key] = outcome(checkpoint_manager.is_training_complete, task, real,
                                  results_dir=root / data_dir, network_type=nt or 'jammed')
        for family, task, real, _ in sample.AUXETIC:
            key = f'aug/{family}/t{task}/r{real}'
            checks[key] = outcome(checkpoint_manager.is_training_complete, task, real,
                                  results_dir=root / 'auxetic_nets_aug' / family,
                                  network_type='jammed')
        # Synthetic dirs (no sample run has a True) for each completion path.
        import numpy as np
        syn = root / 'synthetic'
        for name, loss, marker in (('marker', [1.0, 0.5], True), ('drop_1e8', [1.0, 1e-8], False),
                                   ('drop_1e6', [1.0, 1e-6], False), ('no_loss', None, False)):
            d = syn / 'task_00' / f'realization_{len(checks) % 90:02d}'
            d.mkdir(parents=True)
            if loss is not None:
                np.save(d / 'loss_trajectory_jammed.npy', np.array(loss))
            if marker:
                (d / 'training_complete_small_loss_jammed.txt').write_text('x')
            checks[f'synthetic/{name}'] = outcome(
                checkpoint_manager.is_training_complete, 0, int(d.name.split('_')[1]),
                results_dir=syn, network_type='jammed')
        out['is_training_complete'] = checks
        out['markers_written'] = sorted(set(_listing(root)) - before)
        for family in ('targeted', 'general'):
            d = root / 'auxetic_nets_aug' / family
            for name, fn in (('incomplete', checkpoint_manager.get_incomplete_jobs),
                             ('complete', checkpoint_manager.get_complete_jobs)):
                out[f'{name}_jobs.{family}'] = outcome(
                    fn, 30, 20, results_dir=d, network_type='jammed')
    return out
