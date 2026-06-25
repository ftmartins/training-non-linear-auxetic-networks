#!/usr/bin/env python3
"""
Regenerate missing or null task_config.json files in ensemble result directories.

For each realization directory under RESULTS_ROOT, if task_config.json is absent
or contains null, this script regenerates it deterministically from the task seed
using the same generate_task_config() that the ensemble runner uses.

Usage:
    python analysis/regenerate_task_configs.py [--dry-run] [--results-dir PATH]
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.src.task_generator import generate_task_config

DEFAULT_RESULTS_DIR = REPO_ROOT / 'data' / 'auxetic_nets' / 'results'


def _needs_regeneration(json_path: Path) -> bool:
    """Return True if the file is absent or contains null/invalid config."""
    if not json_path.exists():
        return True
    try:
        data = json.loads(json_path.read_text())
    except (json.JSONDecodeError, OSError):
        return True
    return data is None or not isinstance(data, dict) or 'compression_strains' not in data


def regenerate(results_dir: Path, dry_run: bool) -> None:
    task_dirs = sorted(p for p in results_dir.glob('task_*') if p.is_dir())
    if not task_dirs:
        print(f"No task_* directories found under {results_dir}")
        return

    written = skipped = 0

    for task_dir in task_dirs:
        task_token = task_dir.name.split('_')[1]
        try:
            task_id = int(task_token)
        except ValueError:
            print(f"  Skipping unrecognised directory: {task_dir.name}")
            continue

        config = generate_task_config(task_id)

        for real_dir in sorted(p for p in task_dir.glob('realization_*') if p.is_dir()):
            json_path = real_dir / 'task_config.json'
            if not _needs_regeneration(json_path):
                skipped += 1
                continue

            status = '[dry-run] would write' if dry_run else 'writing'
            print(f"  {status}: {json_path.relative_to(results_dir)}")

            if not dry_run:
                json_path.write_text(json.dumps(config, indent=2))

            written += 1

    action = 'Would write' if dry_run else 'Wrote'
    print(f"\nDone. {action} {written} file(s), skipped {skipped} already-valid file(s).")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be written without writing anything')
    parser.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS_DIR,
                        help=f'Path to results root (default: {DEFAULT_RESULTS_DIR})')
    args = parser.parse_args()

    print(f"Results dir : {args.results_dir}")
    print(f"Dry run     : {args.dry_run}\n")

    regenerate(args.results_dir, args.dry_run)


if __name__ == '__main__':
    main()
