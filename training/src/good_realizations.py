"""
Lookup for screened "good" realization seeds.

Instead of using a training run's realization_id (0..N_REALIZATIONS-1) directly
as the random seed for initial stiffnesses, each task is first screened over a
larger pool of candidate seeds with a short training run (see
training/runners/screen_realizations_auxetic.py and
screen_realizations_allosteric.py), scored by final loss, and the best
N_REALIZATIONS are kept. Their seeds are stored here, indexed serially by
realization_id, so realization_id=0 always means "the best candidate found",
regardless of which raw seed that turned out to be.

See docs/realization_screening.md for the full procedure.

Files (one per kind, written by the corresponding screen_aggregate step):
  training/src/good_realizations/auxetic_targeted.json
  training/src/good_realizations/auxetic_general.json
  training/src/good_realizations/allosteric_targeted.json
  training/src/good_realizations/allosteric_general.json
  training/src/good_realizations/auxetic_cl_targeted.json
  training/src/good_realizations/auxetic_cl_general.json

Each file: {"<key>": [seed_0, seed_1, ...]} where key is "<task_id>" for
targeted ensembles (shared/fixed geometry) and auxetic-general/auxetic-cl-*, or
"<geometry_id>_<task_id>" for allosteric-general.

`auxetic_cl_*` is the coupled-learning (LAMMPS free/clamped) auxetic
variant — same task/geometry definitions as `auxetic_*` (task_generator.py /
targeted_task_generator.py, unchanged), a separate screened-seed table
because the training dynamics (and therefore which realizations screen well)
differ from the gradient-based `auxetic_*` pipeline.
"""
import json
from pathlib import Path

GOOD_REALIZATIONS_DIR = Path(__file__).parent / 'good_realizations'

KINDS = ('auxetic_targeted', 'auxetic_general', 'allosteric_targeted', 'allosteric_general',
         'auxetic_cl_targeted', 'auxetic_cl_general')

# Screening candidate seeds live in a disjoint range from any real/legacy
# seed value, so a screening run can never collide with a production one.
SCREEN_SEED_BASE = 1_000_000


def _table_path(kind):
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return GOOD_REALIZATIONS_DIR / f'{kind}.json'


def _key(task_id, geometry_id=None):
    return f"{geometry_id}_{task_id}" if geometry_id is not None else str(task_id)


def load_table(kind):
    path = _table_path(kind)
    if not path.exists():
        raise FileNotFoundError(
            f"No screened realizations at {path}. Run the {kind} screening + "
            f"aggregation step first — see docs/realization_screening.md."
        )
    with open(path) as f:
        return json.load(f)


def save_table(kind, table):
    path = _table_path(kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(table, f, indent=2, sort_keys=True)
    return path


def get_realization_seed(kind, task_id, realization_id, geometry_id=None):
    """
    Look up the actual stiffness-generation seed for (task_id, realization_id)
    [, geometry_id for allosteric-general], as chosen by screening.

    Raises FileNotFoundError / KeyError / IndexError with an actionable
    message if screening hasn't been run for this kind/task yet, rather than
    silently falling back to realization_id itself — an un-screened seed
    should never be mistaken for a screened one.
    """
    table = load_table(kind)
    key = _key(task_id, geometry_id)
    if key not in table:
        raise KeyError(f"No screened realizations for {kind}/{key} in {_table_path(kind)}.")
    seeds = table[key]
    if realization_id < 0 or realization_id >= len(seeds):
        raise IndexError(
            f"realization_id={realization_id} out of range for {kind}/{key} "
            f"({len(seeds)} screened realizations stored)."
        )
    return seeds[realization_id]
