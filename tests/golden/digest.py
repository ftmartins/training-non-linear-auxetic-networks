"""
Exact, JSON-serialisable digest of whatever a reader returns.

Arrays become (dtype, shape, sha256 of the C-contiguous bytes), floats their repr,
objects their class name plus the digest of their attributes. Two values with equal
digests are bit-identical as far as any caller can observe; this is what the golden
tests compare, so the stored expectations stay small and never contain data.
Paths inside the checkout are written relative to <repo>, so the goldens hold in any
checkout or worktree.
"""
import hashlib
from pathlib import Path

import numpy as np

from tests.golden import REPO_ROOT


def portable(text):
    """text with this checkout's absolute path replaced by <repo>."""
    return text.replace(str(REPO_ROOT), '<repo>')


def _array(a):
    a = np.asarray(a)
    if a.dtype == object:
        return {'__object_array__': list(a.shape), 'items': [digest(x) for x in a.ravel()]}
    return {'__nd__': a.dtype.str, 'shape': list(a.shape),
            'sha256': hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}


def digest(x):
    if x is None or isinstance(x, bool):
        return x
    if isinstance(x, str):
        return portable(x)
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return {'__int__': type(x).__name__, 'v': int(x)}
    if isinstance(x, (float, np.floating)):
        return {'__float__': type(x).__name__, 'v': repr(float(x))}
    if isinstance(x, np.ndarray):
        return _array(x)
    if isinstance(x, Path):
        return {'__path__': portable(str(x))}
    if isinstance(x, dict):
        return {'__dict__': {str(k): digest(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}}
    if isinstance(x, (list, tuple)):
        return {'__' + type(x).__name__ + '__': [digest(v) for v in x]}
    if isinstance(x, np.lib.npyio.NpzFile):
        return {'__npz__': {k: digest(x[k]) for k in sorted(x.files)}}
    if hasattr(x, '__array__') and hasattr(x, 'dtype'):   # jax arrays
        return _array(np.asarray(x))
    if hasattr(x, '__dict__'):
        return {'__obj__': type(x).__module__ + '.' + type(x).__qualname__,
                'attrs': digest(dict(vars(x)))}
    raise TypeError(f'no digest for {type(x)!r}')
