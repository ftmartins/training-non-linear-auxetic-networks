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

N_TASKS = 10  # Number of distinct training tasks
N_REALIZATIONS = 20  # Number of realizations per task

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
    return 100 if task_seed < 20 else 400


def get_n_nodes(task_seed: int) -> int:
    """Return the network node count for the given task seed."""
    if task_seed < 10:
        return 100
    elif task_seed < 20:
        return 300
    else:
        return 100


# ============================================================================
# Training Hyperparameters
# ============================================================================

LEARNING_RATE = 1e-3
N_STEPS = 10_000  # Number of training iterations
# Legacy scalar — use get_n_strain_steps(task_seed) for task-aware code.
N_STRAIN_STEPS = 100  # Number of steps in quasistatic trajectory (tasks < 20)
FORCE_TOL = 1e-8  # Force convergence tolerance for FIRE
VMIN = 1e-3  # Minimum stiffness value
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

# Base directory (relative to this file)
BASE_DIR = Path(__file__).parent.parent.resolve()

# Module paths
INSTRUMENTS_PATH = BASE_DIR #/ 'instruments'
PRODUCTION_PATH = BASE_DIR #/ 'production'
PACKING_PATH = BASE_DIR.parent.parent / 'cl_mech_repo' / 'physical_learning'

# Data paths
DATA_DIR = Path('/data2/shared/felipetm/auxetic_networks')  # Change as needed
ENSEMBLE_DIR = DATA_DIR / 'ensemble_training_new/'
RESULTS_DIR = ENSEMBLE_DIR / 'results_new/'
CHECKPOINT_DIR = ENSEMBLE_DIR / 'checkpoints_new/'

# ============================================================================
# Network Creation Parameters
# ============================================================================

BOUNDARY_MARGIN = 0.05  # Margin for identifying boundary nodes
PACKING_PARAMS = {
    'central': 0.0005,
    'drag': 0.05,
    'contact': 0.1
}
PACKING_DURATION = 1000.0
PACKING_FRAMES = 200

# ============================================================================
# Save/Checkpoint Parameters
# ============================================================================

CHECKPOINT_INTERVAL = 50  # Save checkpoint every N steps
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
