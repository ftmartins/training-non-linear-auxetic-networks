"""
Golden characterization tests for the realization record refactor.

They pin what today's readers, K* helpers, initial-condition draws and discovery
helpers return for a fixed sample of real run directories (sample.py), as exact
digests (digest.py) stored under expected/. Nothing here holds data: the runs are
read from the local mirror, `<repo>/data` (in a worktree, a symlink to the main
checkout's data/).

Opt-in, because they read a few GB from the Box-backed mirror:
    GOLDEN=1 python -m unittest tests.golden.test_read
Re-record after an intended behaviour change (and say so in the commit):
    python -m tests.golden.record
"""
import os
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / 'data'
EXPECTED_DIR = Path(__file__).parent / 'expected'


def golden_enabled():
    return os.environ.get('GOLDEN') == '1'


skip_unless_golden = unittest.skipUnless(golden_enabled(), 'golden tests need GOLDEN=1 and data/')
