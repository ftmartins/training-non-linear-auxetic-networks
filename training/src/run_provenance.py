"""
Run provenance: git commit/dirty state + argv, a quick-check snapshot of the
hyperparameters (learning rate, stiffness bounds, step count, ...) that
actually produced a given training run's results, and a gzip snapshot of the
actual code (in case git history/dirty state alone isn't enough to recover
exactly what ran).

Files written per run directory:
  - training_meta.json    : single dict, written once. Critical keys
                             (solver/gradient_method/learning_rate/k_min/
                             k_max/force_tol/optimizer/opt_fire_finc/
                             realization_seed by
                             default) are enforced to match on every
                             subsequent call, raising
                             HyperparameterMismatchError on conflict unless
                             overwrite=True — resuming a job under different
                             physics/optimizer settings would otherwise
                             silently corrupt the saved trajectory. All other
                             keys (e.g. n_steps — training longer is fine)
                             always track the current value.
  - run_provenance.json   : list, appended to on every invocation, so a job
                             that gets resumed under a different git commit
                             keeps the full sequence of code versions that
                             touched it.
  - <name>.py.gz (per file passed to save_code_snapshot): gzip copy of the
                             actual script/module source, written once.
"""
import gzip
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CRITICAL_KEYS = frozenset({
    'solver', 'gradient_method', 'learning_rate', 'k_min', 'k_max', 'force_tol',
    'optimizer', 'opt_fire_finc', 'realization_seed',
})


class HyperparameterMismatchError(RuntimeError):
    """Raised when a resumed run's critical hyperparameters conflict with
    what's already recorded in training_meta.json and overwrite=False."""


def has_critical_mismatch(output_path, meta, filename='training_meta.json',
                           critical_keys=DEFAULT_CRITICAL_KEYS):
    """
    Read-only check: do `meta`'s critical_keys conflict with what's already
    recorded in output_path/filename? Returns False if the file doesn't
    exist yet (nothing to conflict with).

    Meant to be called BEFORE any checkpoint is loaded or expensive setup
    happens, so a caller can decide to wipe and restart from scratch (see
    module docstring / checkpoint_manager.reset_job) rather than resume
    stale state under a new, inaccurate label. Resuming a checkpoint trained
    under old hyperparameters and then merely overwriting training_meta.json
    with the new ones would make the recorded metadata describe a trajectory
    that was never actually trained end-to-end under those settings — the
    whole point of tracking this is reproducibility, so that's the one thing
    this module must never do.

    Args:
        output_path: Directory the meta file lives in.
        meta: Dict of hyperparameters for this run.
        filename: Meta filename (default: 'training_meta.json').
        critical_keys: Set of `meta` keys checked for conflict (default:
            DEFAULT_CRITICAL_KEYS).

    Returns:
        True if any critical key present in both `meta` and the recorded
        file has a different value.
    """
    meta_file = Path(output_path) / filename
    if not meta_file.exists():
        return False
    with open(meta_file) as f:
        existing = json.load(f)
    return any(k in critical_keys and k in existing and existing.get(k) != v
               for k, v in meta.items())


def get_git_info(repo_root=None):
    """Return {'git_commit': <sha or None>, 'git_dirty': <bool or None>}.

    None values mean git info could not be determined (e.g. not a git
    checkout, git not installed) rather than a genuine "no commit" state.
    """
    repo_root = repo_root or _REPO_ROOT
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=repo_root,
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ['git', 'status', '--porcelain'], cwd=repo_root,
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout
        return {'git_commit': commit, 'git_dirty': bool(status.strip())}
    except Exception:
        return {'git_commit': None, 'git_dirty': None}


def save_run_provenance(output_path, extra=None, filename='run_provenance.json'):
    """
    Append one provenance entry (git state, argv, timestamp, extra) to
    output_path/filename.

    Appends rather than overwrites so that if a job is resumed after the
    code changed (e.g. a bugfix between SLURM resubmissions), the full
    sequence of commits that touched this run's results is recoverable.

    Args:
        output_path: Directory the provenance file lives in.
        extra: Optional dict merged into the entry (e.g. hyperparameters).
        filename: Provenance filename (default: 'run_provenance.json').

    Returns:
        Path to the provenance file.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    provenance_file = output_path / filename

    entries = []
    if provenance_file.exists():
        try:
            with open(provenance_file) as f:
                entries = json.load(f)
        except (json.JSONDecodeError, OSError):
            entries = []

    entry = {
        'timestamp': datetime.now().isoformat(),
        'argv': sys.argv,
        **get_git_info(),
    }
    if extra:
        entry.update(extra)
    entries.append(entry)

    with open(provenance_file, 'w') as f:
        json.dump(entries, f, indent=2)

    return provenance_file


def save_training_meta(output_path, meta, filename='training_meta.json',
                        overwrite=False, critical_keys=DEFAULT_CRITICAL_KEYS):
    """
    Write a quick-check hyperparameter snapshot, once, and guard against
    silently resuming a run under different physics/optimizer settings.

    On the first call for a given output_path/filename, `meta` is written
    as-is. On every later call:
      - Keys in `critical_keys` that conflict with what's recorded raise
        HyperparameterMismatchError unless overwrite=True.
      - All other keys always track the current value (no error, no
        warning) — e.g. n_steps, where just wanting to train longer is
        expected and fine.
      - Keys present in `meta` but absent from an existing file are
        backfilled, so older runs recorded before a given field existed
        still pick it up.

    NOTE: overwrite=True here only relabels the JSON file — it does NOT
    touch any checkpoint/results on disk. If those were produced under the
    OLD (conflicting) hyperparameters, calling this with overwrite=True and
    then resuming training would silently mix two different settings into
    one saved trajectory while training_meta.json claims only the new one
    was used. Callers that let users pass --overwrite should check
    has_critical_mismatch() BEFORE loading any checkpoint and, if it's
    True, wipe the job's saved state (checkpoint_manager.reset_job) and
    restart from scratch — only then is calling this function with
    overwrite=True actually accurate.

    Args:
        output_path: Directory the meta file lives in.
        meta: Dict of hyperparameters for this run.
        filename: Meta filename (default: 'training_meta.json').
        overwrite: If True, replace conflicting critical values instead of
            raising.
        critical_keys: Set of `meta` keys that must match the recorded
            value unless overwrite=True (default: DEFAULT_CRITICAL_KEYS).

    Returns:
        The recorded dict (existing+merged, or newly written).

    Raises:
        HyperparameterMismatchError: A critical key conflicts with the
            recorded value and overwrite=False.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    meta_file = output_path / filename

    if not meta_file.exists():
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)
        return meta

    with open(meta_file) as f:
        existing = json.load(f)

    critical_mismatches = {
        k: (existing.get(k), v) for k, v in meta.items()
        if k in critical_keys and k in existing and existing.get(k) != v
    }
    if critical_mismatches and not overwrite:
        raise HyperparameterMismatchError(
            f"{meta_file} already recorded different values for "
            f"{sorted(critical_mismatches)} (recorded, current): {critical_mismatches}. "
            f"Resuming with mismatched hyperparameters would mix incompatible settings "
            f"into one training trajectory. Pass overwrite=True (e.g. --overwrite) if "
            f"this is intentional."
        )
    if critical_mismatches:
        print(f"  WARNING: overwrite=True; replacing recorded values in {meta_file} for "
              f"{sorted(critical_mismatches)} (recorded, current): {critical_mismatches}")

    merged = dict(existing)
    for k, v in meta.items():
        if k in critical_keys and k in existing and not overwrite:
            continue  # verified equal above; keep the recorded value
        merged[k] = v

    if merged != existing:
        with open(meta_file, 'w') as f:
            json.dump(merged, f, indent=2)
    return merged


def save_code_snapshot(output_path, files):
    """
    Gzip-archive each file in `files` into output_path, so the exact code
    that produced a run's results is recoverable even without git access
    (e.g. inspecting results long after the commit is gone locally, or a
    dirty working tree at run time).

    Skips any file that already has a `<name>.gz` counterpart in
    output_path, so this is cheap to call on every invocation — only the
    first invocation for a given run directory actually archives anything.

    Args:
        output_path: Directory to archive into.
        files: Iterable of file paths (str or Path) to archive. Missing
            files are silently skipped.

    Returns:
        List of archive destination paths (newly written or already present).
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    archived = []
    for src in files:
        src = Path(src)
        if not src.exists():
            continue
        dst = output_path / (src.name + '.gz')
        if not dst.exists():
            with open(src, 'rb') as fi, gzip.open(dst, 'wb') as fo:
                shutil.copyfileobj(fi, fo)
        archived.append(dst)
    return archived
