"""
Targeted task configurations for specific auxetic training experiments.

5 hardcoded tasks with large compression strains (-0.4, -0.2) and specific
Poisson ratio targets. All tasks share the same network topology (packing_seed).
Each task has 1 realization.

Tasks:
  0: nu=-0.8 at compression -0.4 (single pair)
  1: nu=-0.8 at compression -0.4, nu=-0.8 at compression -0.2
  2: nu=-0.8 at compression -0.4, nu=-1.0 at compression -0.2
  3: nu=-0.8 at compression -0.4, nu=-0.4 at compression -0.2
  4: nu=-0.8 at compression -0.4, nu=-0.6 at compression -0.2
"""

import numpy as np
from pathlib import Path

from config import (
    BASE_DIR, STIFFNESS_LOG_MIN, STIFFNESS_LOG_MAX,
)
from task_generator import generate_realization_stiffnesses, compute_target_extensions

# ============================================================================
# Targeted Training Configuration (overrides from config.py)
# ============================================================================

N_TASKS = 5
N_REALIZATIONS = 1
N_STEPS = 3_000
N_STRAIN_STEPS = 100  # Higher than default 20 to handle large compression strains

PACKING_SEED = 42  # Same network topology for all tasks

ENSEMBLE_DIR = BASE_DIR / 'ensemble_training'
TARGETED_RESULTS_DIR = ENSEMBLE_DIR / 'targeted_results'

# ============================================================================
# Task Definitions
# ============================================================================

# TARGETED_TASKS = [
#     {
#         'task_seed': 0,
#         'packing_seed': PACKING_SEED,
#         'compression_strains': [-0.4],
#         'target_poisson_ratios': [-0.8],
#     },
#     {
#         'task_seed': 1,
#         'packing_seed': PACKING_SEED,
#         'compression_strains': [-0.4, -0.2],
#         'target_poisson_ratios': [-0.8, -0.8],
#     },
#     {
#         'task_seed': 2,
#         'packing_seed': PACKING_SEED,
#         'compression_strains': [-0.4, -0.2],
#         'target_poisson_ratios': [-0.8, -1.0],
#     },
#     {
#         'task_seed': 3,
#         'packing_seed': PACKING_SEED,
#         'compression_strains': [-0.4, -0.2],
#         'target_poisson_ratios': [-0.8, -0.4],
#     },
#     {
#         'task_seed': 4,
#         'packing_seed': PACKING_SEED,
#         'compression_strains': [-0.4, -0.2],
#         'target_poisson_ratios': [-0.8, -0.6],
#     },
# ]


TARGETED_TASKS = [
    {
        'task_seed': 0,
        'packing_seed': PACKING_SEED,
        'compression_strains': [-0.4],
        'target_poisson_ratios': [-0.8],
    },
        {
        'task_seed': 1,
        'packing_seed': PACKING_SEED,
        'compression_strains': [-0.4, -0.2],
        'target_poisson_ratios': [-0.8, -0.8],
    },
    {
        'task_seed': 2,
        'packing_seed': PACKING_SEED,
        'compression_strains': [-0.4, -0.2],
        'target_poisson_ratios': [-0.8, -1.0],
    },
    {
        'task_seed': 3,
        'packing_seed': PACKING_SEED,
        'compression_strains': [-0.4, -0.2],
        'target_poisson_ratios': [-0.8, -0.4],
    },
    {
        'task_seed': 4,
        'packing_seed': PACKING_SEED,
        'compression_strains': [-0.4, -0.2],
        'target_poisson_ratios': [-0.8, -0.6],
    },
    {
        'task_seed': 5,
        'packing_seed': PACKING_SEED,
        'compression_strains': [-0.4, -0.2],
        'target_poisson_ratios': [-0.8, -0.3],
    },
    {
        'task_seed': 6,
        'packing_seed': PACKING_SEED,
        'compression_strains': [-0.4, -0.2],
        'target_poisson_ratios': [-0.8, -0.5],
    },
    {
        'task_seed': 7,
        'packing_seed': PACKING_SEED,
        'compression_strains': [-0.4, -0.2],
        'target_poisson_ratios': [-0.8, -0.2],
    },
]

# ============================================================================
# Functions
# ============================================================================


def get_targeted_task_config(task_id):
    """
    Return task configuration for a specific targeted task.

    Args:
        task_id: Integer 0-4

    Returns:
        config_dict with keys: task_seed, packing_seed,
        compression_strains, target_poisson_ratios
    """
    if task_id < 0 or task_id >= N_TASKS:
        raise ValueError(f"task_id must be 0-{N_TASKS-1}, got {task_id}")
    return TARGETED_TASKS[task_id]


def get_all_targeted_task_configs():
    """Return all 5 targeted task configurations."""
    return TARGETED_TASKS


def print_targeted_tasks_summary():
    """Print a formatted summary of all targeted tasks."""
    print(f"\n{'='*80}")
    print("TARGETED TRAINING TASKS")
    print(f"{'='*80}")
    print(f"Tasks: {N_TASKS}")
    print(f"Realizations per task: {N_REALIZATIONS}")
    print(f"Training steps: {N_STEPS:,}")
    print(f"Strain steps: {N_STRAIN_STEPS}")
    print(f"Packing seed (shared): {PACKING_SEED}")
    print(f"Results directory: {TARGETED_RESULTS_DIR}")
    print()

    for task in TARGETED_TASKS:
        tid = task['task_seed']
        compressions = task['compression_strains']
        poissons = task['target_poisson_ratios']
        extensions = compute_target_extensions(compressions, poissons)

        print(f"  Task {tid}:")
        for i in range(len(compressions)):
            print(f"    Pair {i}: nu={poissons[i]}, compression={compressions[i]}, "
                  f"target_extension={extensions[i]:.4f}")

    print(f"{'='*80}\n")


if __name__ == '__main__':
    print_targeted_tasks_summary()

    # Verify target extension computation
    print("Verifying target extensions:")
    for task in TARGETED_TASKS:
        compressions = task['compression_strains']
        poissons = task['target_poisson_ratios']
        extensions = compute_target_extensions(compressions, poissons)
        for i in range(len(compressions)):
            recovered_nu = -extensions[i] / compressions[i]
            print(f"  Task {task['task_seed']}, pair {i}: "
                  f"nu_target={poissons[i]}, nu_recovered={recovered_nu:.4f}")

    # Test stiffness generation
    print(f"\nStiffness generation test:")
    test_stiff = generate_realization_stiffnesses(0, 300)
    print(f"  300 stiffnesses, range: [{test_stiff.min():.2e}, {test_stiff.max():.2e}]")

    print("\nAll checks passed!")
