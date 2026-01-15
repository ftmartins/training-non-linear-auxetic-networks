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

N_NODES = 100  # Number of nodes in packing
PACKING_DIM = 2  # Spatial dimension
FORCE_TYPE = 'quadratic'  # Force law: 'quadratic' or 'quartic'

# ============================================================================
# Task Configuration
# ============================================================================

N_TASKS = 10  # Number of distinct training tasks
N_REALIZATIONS = 50  # Number of realizations per task

# Compression strain pool (9 options)
COMPRESSION_POOL = (-np.arange(0.01, 0.10, 0.01)).tolist()  # [-0.01, -0.02, ..., -0.09]

# Poisson ratio pool (6 options)
POISSON_POOL = [-0.1, -0.25, -0.3, -0.5, -0.8, -1.0]

# ============================================================================
# Training Hyperparameters
# ============================================================================

LEARNING_RATE = 1e-2
N_STEPS = 100_000  # Number of training iterations
N_STRAIN_STEPS = 20  # Number of steps in quasistatic trajectory
FORCE_TOL = 1e-6  # Force convergence tolerance for FIRE
VMIN = 1e-3  # Minimum stiffness value
VMAX = 1e2  # Maximum stiffness value
ETA = 0.1  # Coupling factor (from notebook)

# ============================================================================
# Stiffness Initialization
# ============================================================================

STIFFNESS_LOG_MIN = np.log(1e-6)  # log(minimum stiffness)
STIFFNESS_LOG_MAX = np.log(1.0)   # log(maximum stiffness)

# ============================================================================
# Parallelization
# ============================================================================

N_JOBS_OUTER = 4  # Parallel jobs for gradient computation across edges
N_JOBS_INNER = 2  # Parallel jobs for Poisson ratio computation across strains

# ============================================================================
# Paths
# ============================================================================

# Base directory (relative to this file)
BASE_DIR = Path(__file__).parent.parent.resolve()

# Module paths
INSTRUMENTS_PATH = BASE_DIR / 'instruments'
PRODUCTION_PATH = BASE_DIR / 'production'
PACKING_PATH = BASE_DIR.parent.parent / 'cl_mech_repo' / 'physical_learning'

# Data paths
ENSEMBLE_DIR = BASE_DIR / 'ensemble_training'
RESULTS_DIR = ENSEMBLE_DIR / 'results'
CHECKPOINT_DIR = ENSEMBLE_DIR / 'checkpoints'

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

CHECKPOINT_INTERVAL = 10_000  # Save checkpoint every N steps
SAVE_FULL_HISTORY = True  # Save full training history (stiffnesses at each step)
USE_CHECKPOINTING = True  # Enable checkpoint/resume functionality

# ============================================================================
# Validation
# ============================================================================

def validate_config():
    """Validate configuration settings."""
    assert len(COMPRESSION_POOL) == 9, f"Expected 9 compressions, got {len(COMPRESSION_POOL)}"
    assert len(POISSON_POOL) == 6, f"Expected 6 Poisson ratios, got {len(POISSON_POOL)}"
    assert N_TASKS <= (len(COMPRESSION_POOL) * (len(COMPRESSION_POOL) - 1) // 2) * \
                       (len(POISSON_POOL) * (len(POISSON_POOL) - 1) // 2), \
           f"N_TASKS={N_TASKS} too large for available pool combinations"
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

N_NODES = 100  # Number of nodes in packing
PACKING_DIM = 2  # Spatial dimension
FORCE_TYPE = 'quadratic'  # Force law: 'quadratic' or 'quartic'

# ============================================================================
# Task Configuration
# ============================================================================

N_TASKS = 10  # Number of distinct training tasks
N_REALIZATIONS = 20  # Number of realizations per task

# Compression strain pool (9 options)
COMPRESSION_POOL = (-np.arange(0.01, 0.10, 0.01)).tolist()  # [-0.01, -0.02, ..., -0.09]

# Poisson ratio pool (6 options)
POISSON_POOL = [-0.1, -0.25, -0.3, -0.5, -0.8, -1.0]

# ============================================================================
# Training Hyperparameters
# ============================================================================

LEARNING_RATE = 1e-2
N_STEPS = 3_000  # Number of training iterations
N_STRAIN_STEPS = 20  # Number of steps in quasistatic trajectory
FORCE_TOL = 1e-6  # Force convergence tolerance for FIRE
VMIN = 1e-3  # Minimum stiffness value
VMAX = 1e2  # Maximum stiffness value
ETA = 0.1  # Coupling factor (from notebook)

# ============================================================================
# Stiffness Initialization
# ============================================================================

STIFFNESS_LOG_MIN = np.log(1e-6)  # log(minimum stiffness)
STIFFNESS_LOG_MAX = np.log(1.0)   # log(maximum stiffness)

# ============================================================================
# Parallelization
# ============================================================================

N_JOBS_OUTER = 4  # Parallel jobs for gradient computation across edges
N_JOBS_INNER = 2  # Parallel jobs for Poisson ratio computation across strains

# ============================================================================
# Paths
# ============================================================================

# Base directory (relative to this file)
BASE_DIR = Path(__file__).parent.parent.resolve()

# Module paths
INSTRUMENTS_PATH = BASE_DIR #/ 'instruments'
PRODUCTION_PATH = BASE_DIR #/ 'production'
PACKING_PATH = BASE_DIR.parent.parent #/ 'cl_mech_repo' / 'physical_learning'

# Data paths
ENSEMBLE_DIR = BASE_DIR / 'ensemble_training'
RESULTS_DIR = ENSEMBLE_DIR / 'results'
CHECKPOINT_DIR = ENSEMBLE_DIR / 'checkpoints'

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

CHECKPOINT_INTERVAL = 500  # Save checkpoint every N steps
SAVE_FULL_HISTORY = True  # Save full training history (stiffnesses at each step)
USE_CHECKPOINTING = True  # Enable checkpoint/resume functionality

# ============================================================================
# Validation
# ============================================================================

def validate_config():
    """Validate configuration settings."""
    assert len(COMPRESSION_POOL) == 9, f"Expected 9 compressions, got {len(COMPRESSION_POOL)}"
    assert len(POISSON_POOL) == 6, f"Expected 6 Poisson ratios, got {len(POISSON_POOL)}"
    assert N_TASKS <= (len(COMPRESSION_POOL) * (len(COMPRESSION_POOL) - 1) // 2) * \
                       (len(POISSON_POOL) * (len(POISSON_POOL) - 1) // 2), \
           f"N_TASKS={N_TASKS} too large for available pool combinations"
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
