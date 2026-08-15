"""
Configuration constants for ensemble training of auxetic networks.

This module contains all hyperparameters, paths, and configuration
settings for the ensemble training system.
"""

import numpy as np
from pathlib import Path

# ============================================================================
# Network Parameters
# ============================================================================

# Legacy scalar — use get_n_nodes(task_seed) for task-aware code.
# Kept for backward compatibility with runners that have fixed network sizes.
N_NODES = 100  # Number of nodes in packing (default / tasks < 10 and >= 20)
PACKING_DIM = 2  # Spatial dimension
FORCE_TYPE = 'quadratic'  # Force law: 'quadratic' or 'quartic'

# ============================================================================
# Task Configuration
# ============================================================================

N_TASKS = 30  # Number of distinct training tasks
N_REALIZATIONS = 5  # Screened realizations per task — see training/src/good_realizations.py

# ---------------------------------------------------------------------------
# Pool definitions — indexed by task range
#
#   task <  10  : Pool 0  (large compressions, extreme Poisson ratios)
#   10 <= task < 20 : Pool 1  (moderate compressions / Poisson)
#   20 <= task < 30 : Pool 2  (hybrid subset drawn from pools 0 and 1)
# ---------------------------------------------------------------------------

# Pool 0: tasks 0–9
_COMPRESSION_POOL_0 = (-np.arange(0.05, 0.31, 0.05)).tolist()   # 6 values: -0.05 … -0.30
_POISSON_POOL_0     = [-0.1, -0.25, -0.3, -0.5, -0.8, -1.0]

# Pool 1: tasks 10–19
_COMPRESSION_POOL_1 = (-np.arange(0.025, 0.16, 0.025)).tolist()  # 6 values: -0.025 … -0.150
_POISSON_POOL_1     = [-0.05, -0.1, -0.15, -0.2, -0.3, -0.4]

# Pool 2: tasks 20–29 — 6 values drawn from _POOL_0 ∪ _POOL_1
_COMPRESSION_POOL_2 = [-0.05, -0.10, -0.125, -0.15, -0.20, -0.25]  # spans both pools
_POISSON_POOL_2     = [-0.10, -0.15, -0.25, -0.30, -0.40, -0.50]   # spans both pools

# Legacy scalar aliases — point to pool 1 (current active range).
# Prefer get_compression_pool(task_seed) / get_poisson_pool(task_seed).
COMPRESSION_POOL = _COMPRESSION_POOL_1
POISSON_POOL     = _POISSON_POOL_1


def get_compression_pool(task_seed: int) -> list:
    """Return the compression strain pool for the given task seed."""
    if task_seed < 10:
        return _COMPRESSION_POOL_0
    elif task_seed < 20:
        return _COMPRESSION_POOL_1
    elif task_seed < 30:
        return _COMPRESSION_POOL_2
    else:
        raise ValueError(f"No compression pool defined for task_seed={task_seed} (>= 30).")


def get_poisson_pool(task_seed: int) -> list:
    """Return the Poisson ratio pool for the given task seed."""
    if task_seed < 10:
        return _POISSON_POOL_0
    elif task_seed < 20:
        return _POISSON_POOL_1
    elif task_seed < 30:
        return _POISSON_POOL_2
    else:
        raise ValueError(f"No Poisson pool defined for task_seed={task_seed} (>= 30).")


def get_n_strain_steps(task_seed: int) -> int:
    """Return the quasistatic trajectory step count for the given task seed."""
    return 400
#used to be 100 if task_seed < 20 before the may realizations

def get_n_nodes(task_seed: int) -> int:
    """Return the network node count for the given task seed."""
    if task_seed < 10:
        return 100
    elif task_seed < 20:
        return 100
    else:
        return 100


# ============================================================================
# Training Hyperparameters
# ============================================================================

LEARNING_RATE = 3e-3  # starting lr / opt_fire dt_init (opt_fire hyperparameter grid sweep,
                       # tasks 3/15); decayed by training.src.lr_schedule when not using opt_fire
OPT_FIRE_FINC = 1.05   # opt_fire growth factor (same sweep) — never diverged in the grid
USE_OPT_FIRE = False   # FIRE-style adaptive optimizer vs plain gradient descent. Reverted
                       # 2026-08-15: production runs (targeted task_00/realization_02, general
                       # task idx 2) showed FIRE trapped in an exact repeating limit cycle —
                       # traced to 1-2 stiffness coordinates pinned at vmin with a genuinely
                       # ill-conditioned gradient there (grad_norm swings 1e2->1e7 from a ~3e-6
                       # move in a single coordinate). Lowering opt_fire_dt_min 100x only
                       # relocated the same trap to a smaller floor — confirmed on a fresh
                       # (non-resumed) rerun, so this is a real gradient singularity, not a
                       # step-size tuning issue. Plain GD sidesteps it because lr_schedule's
                       # halving-on-overshoot backs off from the same blowup instead of trying
                       # to grow through it. (Earlier 2026-08-13 A/B test favored FIRE because
                       # short GD trials NaN'd — that risk is being accepted for now while FIRE's
                       # failure mode is worse: FIRE finishes the full step budget but produces
                       # zero progress, silently, which is harder to catch than a NaN.)
                       # 'optimizer' is a critical key in training_meta.json, so resuming under a
                       # different choice crashes rather than silently mixing trajectories.
N_STEPS = 5_000  # Number of training iterations
# Legacy scalar — use get_n_strain_steps(task_seed) for task-aware code.
N_STRAIN_STEPS = 200  # Number of steps in quasistatic trajectory (tasks < 20)
FORCE_TOL = 1e-6  # Force convergence tolerance
VMIN = 1e-4  # Minimum stiffness value
VMAX = 1e2  # Maximum stiffness value
ETA = 0.1  # Coupling factor (from notebook)

# ============================================================================
# Stiffness Initialization
# ============================================================================

STIFFNESS_LOG_MIN = np.log(1e-3)  # log(minimum stiffness)
STIFFNESS_LOG_MAX = np.log(1.0)   # log(maximum stiffness)

# ============================================================================
# Parallelization
# ============================================================================

N_JOBS_OUTER = 4  # Parallel jobs for gradient computation across edges
N_JOBS_INNER = 1  # Parallel jobs for Poisson ratio computation across strains

# ============================================================================
# Paths
# ============================================================================

# Project root (parent of base/)
BASE_DIR = Path(__file__).parent.parent.resolve()

# Module paths (kept for compatibility)
INSTRUMENTS_PATH = BASE_DIR
PRODUCTION_PATH = BASE_DIR
PACKING_PATH = BASE_DIR.parent.parent / 'cl_mech_repo' / 'physical_learning'

# ── Cluster data paths (used when running on HPC) ───────────────────────────
_CLUSTER_DATA_ROOT = Path('/data2/shared/felipetm/auxetic_networks')
CLUSTER_ENSEMBLE_DIR  = _CLUSTER_DATA_ROOT / 'ensemble_training_new_sqr'
CLUSTER_TARGETED_DIR  = _CLUSTER_DATA_ROOT / 'targeted_results_sqr'
CLUSTER_ALLOSTERIC_DIR = _CLUSTER_DATA_ROOT / 'allosteric_nets'

# ── Local data paths (relative to project root) ──────────────────────────────
LOCAL_DATA_ROOT     = BASE_DIR / 'data'
LOCAL_AUXETIC_DIR   = LOCAL_DATA_ROOT / 'auxetic_nets'
LOCAL_ALLOSTERIC_DIR = LOCAL_DATA_ROOT / 'allosteric_nets'

# Aliases used throughout training and analysis code.
# Training runners on the cluster should override these via CLI arguments.
DATA_DIR          = LOCAL_AUXETIC_DIR / 'results'
TARGETED_DATA_DIR = LOCAL_AUXETIC_DIR / 'targeted_results_sqr'
ALLOSTERIC_DATA_DIR = LOCAL_ALLOSTERIC_DIR

# Legacy cluster-style aliases kept for backward compatibility with runners
ENSEMBLE_DIR  = CLUSTER_ENSEMBLE_DIR
# '_aug' keeps the raw-gradient/lr-schedule runs (this branch) from writing
# into the pre-existing normalized-gradient results — resume/checkpoint
# logic is unchanged, it just now resumes from whatever's under this new path.
RESULTS_DIR   = CLUSTER_ENSEMBLE_DIR / 'results_new_sqr_aug'
CHECKPOINT_DIR = CLUSTER_ENSEMBLE_DIR / 'checkpoints_new_sqr'

# ============================================================================
# Network Creation Parameters
# ============================================================================

BOUNDARY_MARGIN = 0.02  # Margin for identifying boundary nodes
PACKING_PARAMS = {
    'central': 0.00005,  # Reduced from 0.0005 for less hexagonal, more disordered packings
    'drag': 0.05,
    'contact': 0.1
}
PACKING_DURATION = 1000.0
PACKING_FRAMES = 200

# Network generation method: 'jammed' (packing-derived) or 'lattice'
# (perturbed triangular lattice square). See create_auxetic_network().
NETWORK_TYPE = 'jammed'
LATTICE_JITTER = 0.15    # Jitter amplitude for lattice node positions
LATTICE_CUTOFF = 1.6     # Distance cutoff for lattice bond formation
LATTICE_DILUTION = 0.05  # Fraction of lattice bonds randomly removed

# ============================================================================
# Save/Checkpoint Parameters
# ============================================================================

CHECKPOINT_INTERVAL = 100  # Save checkpoint every N steps
SAVE_FULL_HISTORY = True  # Save full training history (stiffnesses at each step)
USE_CHECKPOINTING = True  # Enable checkpoint/resume functionality

# ============================================================================
# Validation
# ============================================================================

def validate_config():
    """Validate configuration settings."""
    for pool_name, pool in [
        ('_COMPRESSION_POOL_0', _COMPRESSION_POOL_0),
        ('_COMPRESSION_POOL_1', _COMPRESSION_POOL_1),
        ('_COMPRESSION_POOL_2', _COMPRESSION_POOL_2),
    ]:
        assert len(pool) == 6, f"Expected 6 compressions in {pool_name}, got {len(pool)}"
    for pool_name, pool in [
        ('_POISSON_POOL_0', _POISSON_POOL_0),
        ('_POISSON_POOL_1', _POISSON_POOL_1),
        ('_POISSON_POOL_2', _POISSON_POOL_2),
    ]:
        assert len(pool) == 6, f"Expected 6 Poisson ratios in {pool_name}, got {len(pool)}"

    # Check each pool has enough combinations for N_TASKS
    for task_seed in [0, 10, 20]:
        cp = get_compression_pool(task_seed)
        pp = get_poisson_pool(task_seed)
        max_tasks = (len(cp) * (len(cp) - 1) // 2) * (len(pp) * (len(pp) - 1) // 2)
        assert N_TASKS <= max_tasks, (
            f"N_TASKS={N_TASKS} too large for pool at task_seed={task_seed} "
            f"(max {max_tasks} combinations)"
        )

    assert LEARNING_RATE > 0, "Learning rate must be positive"
    assert VMIN < VMAX, "VMIN must be less than VMAX"
    assert N_STEPS > 0, "N_STEPS must be positive"

    # Check that paths exist
    assert INSTRUMENTS_PATH.exists(), f"Instruments path not found: {INSTRUMENTS_PATH}"
    assert PRODUCTION_PATH.exists(), f"Production path not found: {PRODUCTION_PATH}"

    print("Configuration validation passed!")

if __name__ == '__main__':
    # Validate config when run as script
    validate_config()

    # Print summary
    print("\n" + "="*80)
    print("ENSEMBLE TRAINING CONFIGURATION")
    print("="*80)
    print(f"Network: {N_NODES} nodes, {FORCE_TYPE} force law")
    print(f"Tasks: {N_TASKS} tasks × {N_REALIZATIONS} realizations = {N_TASKS * N_REALIZATIONS} total jobs")
    print(f"Compression pool: {len(COMPRESSION_POOL)} options")
    print(f"Poisson pool: {len(POISSON_POOL)} options")
    print(f"\nTraining:")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Steps: {N_STEPS:,}")
    print(f"  Strain steps: {N_STRAIN_STEPS}")
    print(f"  Stiffness bounds: [{VMIN}, {VMAX}]")
    print(f"\nPaths:")
    print(f"  Base: {BASE_DIR}")
    print(f"  Results: {RESULTS_DIR}")
    print("="*80 + "\n")
