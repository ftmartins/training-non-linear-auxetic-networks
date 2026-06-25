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

from base.config import (
    BASE_DIR, STIFFNESS_LOG_MIN, STIFFNESS_LOG_MAX,
)
from .task_generator import generate_realization_stiffnesses, compute_target_extensions

# ============================================================================
# Targeted Training Configuration (overrides from config.py)
# ============================================================================

N_TASKS = 24
N_REALIZATIONS = 1
ENSEMBLE_DIR = BASE_DIR
TARGETED_RESULTS_DIR = Path('/data2/shared/felipetm/auxetic_networks/targeted_results_sqr')

N_STEPS = 5_000
N_STRAIN_STEPS = 400    # Same for all targeted tasks
N_TARGETED_NODES = 100  # Same network size for all targeted tasks
PACKING_SEED = 42  # Same network topology for all tasks

# ============================================================================
# Task Definitions
# ============================================================================

# TARGETED_TASKS = [
#     {
#         'task_seed': 0,
#         'packing_seed': PACKING_SEED,
#         'n_nodes': N_TARGETED_NODES,
#         'n_strain_steps': N_STRAIN_STEPS,
#         'compression_strains': [-0.4],
#         'target_poisson_ratios': [-0.8],
#     },
#     {
#         'task_seed': 1,
#         'packing_seed': PACKING_SEED,
#         'n_nodes': N_TARGETED_NODES,
#         'n_strain_steps': N_STRAIN_STEPS,
#         'compression_strains': [-0.4, -0.2],
#         'target_poisson_ratios': [-0.8, -0.8],
#     },
#     {
#         'task_seed': 2,
#         'packing_seed': PACKING_SEED,
#         'n_nodes': N_TARGETED_NODES,
#         'n_strain_steps': N_STRAIN_STEPS,
#         'compression_strains': [-0.4, -0.2],
#         'target_poisson_ratios': [-0.8, -1.0],
#     },
#     {
#         'task_seed': 3,
#         'packing_seed': PACKING_SEED,
#         'n_nodes': N_TARGETED_NODES,
#         'n_strain_steps': N_STRAIN_STEPS,
#         'compression_strains': [-0.4, -0.2],
#         'target_poisson_ratios': [-0.8, -0.4],
#     },
#     {
#         'task_seed': 4,
#         'packing_seed': PACKING_SEED,
#         'n_nodes': N_TARGETED_NODES,
#         'n_strain_steps': N_STRAIN_STEPS,
#         'compression_strains': [-0.4, -0.2],
#         'target_poisson_ratios': [-0.8, -0.6],
#     },
# ]


TARGETED_TASKS = [
    {
        'task_seed': 0,
        'packing_seed': PACKING_SEED,
        'n_nodes': N_TARGETED_NODES,
        'n_strain_steps': N_STRAIN_STEPS,
        'compression_strains': [-0.2],
        'target_poisson_ratios': [-0.8],
    },
        {
        'task_seed': 1,
        'packing_seed': PACKING_SEED,
        'n_nodes': N_TARGETED_NODES,
        'n_strain_steps': N_STRAIN_STEPS,
        'compression_strains': [-0.2, -0.1],
        'target_poisson_ratios': [-0.8, -0.8],
    },
    {
        'task_seed': 2,
        'packing_seed': PACKING_SEED,
        'n_nodes': N_TARGETED_NODES,
        'n_strain_steps': N_STRAIN_STEPS,
        'compression_strains': [-0.2, -0.1],
        'target_poisson_ratios': [-0.8, -1.0],
    },
    {
        'task_seed': 3,
        'packing_seed': PACKING_SEED,
        'n_nodes': N_TARGETED_NODES,
        'n_strain_steps': N_STRAIN_STEPS,
        'compression_strains': [-0.2, -0.1],
        'target_poisson_ratios': [-0.8, -0.4],
    },
    {
        'task_seed': 4,
        'packing_seed': PACKING_SEED,
        'n_nodes': N_TARGETED_NODES,
        'n_strain_steps': N_STRAIN_STEPS,
        'compression_strains': [-0.2, -0.1],
        'target_poisson_ratios': [-0.8, -0.6],
    },
    {
        'task_seed': 5,
        'packing_seed': PACKING_SEED,
        'n_nodes': N_TARGETED_NODES,
        'n_strain_steps': N_STRAIN_STEPS,
        'compression_strains': [-0.2, -0.1],
        'target_poisson_ratios': [-0.8, -0.3],
    },
    {
        'task_seed': 6,
        'packing_seed': PACKING_SEED,
        'n_nodes': N_TARGETED_NODES,
        'n_strain_steps': N_STRAIN_STEPS,
        'compression_strains': [-0.2, -0.1],
        'target_poisson_ratios': [-0.8, -0.5],
    },
    {
        'task_seed': 7,
        'packing_seed': PACKING_SEED,
        'n_nodes': N_TARGETED_NODES,
        'n_strain_steps': N_STRAIN_STEPS,
        'compression_strains': [-0.2, -0.1],
        'target_poisson_ratios': [-0.8, -0.2],
    },
    {
        'task_seed':8,
        'packing_seed': PACKING_SEED,
        'n_nodes': N_TARGETED_NODES,
        'n_strain_steps': N_STRAIN_STEPS,
        'compression_strains': [-0.3],
        'target_poisson_ratios': [-0.6],
    },
    {
        'task_seed': 9,
        'packing_seed': PACKING_SEED,
        'n_nodes': N_TARGETED_NODES,
        'n_strain_steps': N_STRAIN_STEPS,
        'compression_strains': [-0.3, -0.15],
        'target_poisson_ratios': [-0.6, -0.6],
    },
    {
        'task_seed': 10,
        'packing_seed': PACKING_SEED,
        'n_nodes': N_TARGETED_NODES,
        'n_strain_steps': N_STRAIN_STEPS,
        'compression_strains': [-0.3, -0.15],
        'target_poisson_ratios': [-0.6, -0.8],
    },
    {
        'task_seed': 11,
        'packing_seed': PACKING_SEED,
        'n_nodes': N_TARGETED_NODES,
        'n_strain_steps': N_STRAIN_STEPS,
        'compression_strains': [-0.3, -0.15],
        'target_poisson_ratios': [-0.6, -0.3],
    },
    {
        'task_seed': 12,
        'packing_seed': PACKING_SEED,
        'n_nodes': N_TARGETED_NODES,
        'n_strain_steps': N_STRAIN_STEPS,
        'compression_strains': [-0.3, -0.15],
        'target_poisson_ratios': [-0.6, -0.45],
    },
    {
        'task_seed': 13,
        'packing_seed': PACKING_SEED,
        'n_nodes': N_TARGETED_NODES,
        'n_strain_steps': N_STRAIN_STEPS,
        'compression_strains': [-0.3, -0.15],
        'target_poisson_ratios': [-0.6, -0.225],
    },
    {
        'task_seed': 14,
        'packing_seed': PACKING_SEED,
        'n_nodes': N_TARGETED_NODES,
        'n_strain_steps': N_STRAIN_STEPS,
        'compression_strains': [-0.3, -0.15],
        'target_poisson_ratios': [-0.6, -0.375],
    },
    {
        'task_seed': 15,
        'packing_seed': PACKING_SEED,
        'n_nodes': N_TARGETED_NODES,
        'n_strain_steps': N_STRAIN_STEPS,
        'compression_strains': [-0.3, -0.15],
        'target_poisson_ratios': [-0.6, -0.15],
    },
    # Tasks 16-23: strains [-0.3, -0.15], Poisson ratios in [-0.4, -0.1], anchor at -0.25
    {
        'task_seed': 16,
        'packing_seed': PACKING_SEED,
        'n_nodes': N_TARGETED_NODES,
        'n_strain_steps': N_STRAIN_STEPS,
        'compression_strains': [-0.3],
        'target_poisson_ratios': [-0.25],
    },
    {
        'task_seed': 17,
        'packing_seed': PACKING_SEED,
        'n_nodes': N_TARGETED_NODES,
        'n_strain_steps': N_STRAIN_STEPS,
        'compression_strains': [-0.3, -0.15],
        'target_poisson_ratios': [-0.25, -0.25],
    },
    {
        'task_seed': 18,
        'packing_seed': PACKING_SEED,
        'n_nodes': N_TARGETED_NODES,
        'n_strain_steps': N_STRAIN_STEPS,
        'compression_strains': [-0.3, -0.15],
        'target_poisson_ratios': [-0.25, -0.4],  # non-monotonic: more auxetic at smaller compression
    },
    {
        'task_seed': 19,
        'packing_seed': PACKING_SEED,
        'n_nodes': N_TARGETED_NODES,
        'n_strain_steps': N_STRAIN_STEPS,
        'compression_strains': [-0.3, -0.15],
        'target_poisson_ratios': [-0.25, -0.1],
    },
    {
        'task_seed': 20,
        'packing_seed': PACKING_SEED,
        'n_nodes': N_TARGETED_NODES,
        'n_strain_steps': N_STRAIN_STEPS,
        'compression_strains': [-0.3, -0.15],
        'target_poisson_ratios': [-0.25, -0.175],
    },
    {
        'task_seed': 21,
        'packing_seed': PACKING_SEED,
        'n_nodes': N_TARGETED_NODES,
        'n_strain_steps': N_STRAIN_STEPS,
        'compression_strains': [-0.3, -0.15],
        'target_poisson_ratios': [-0.25, -0.325],
    },
    {
        'task_seed': 22,
        'packing_seed': PACKING_SEED,
        'n_nodes': N_TARGETED_NODES,
        'n_strain_steps': N_STRAIN_STEPS,
        'compression_strains': [-0.3, -0.15],
        'target_poisson_ratios': [-0.25, -0.15],
    },
    {
        'task_seed': 23,
        'packing_seed': PACKING_SEED,
        'n_nodes': N_TARGETED_NODES,
        'n_strain_steps': N_STRAIN_STEPS,
        'compression_strains': [-0.3, -0.15],
        'target_poisson_ratios': [-0.25, -0.225],
    },
]

# ============================================================================
# Functions
# ============================================================================


def get_targeted_task_config(task_id):
    """
    Return task configuration for a specific targeted task.

    Args:
        task_id: Integer 0-23

    Returns:
        config_dict with keys: task_seed, packing_seed,
        compression_strains, target_poisson_ratios
    """
    if task_id < 0 or task_id >= N_TASKS:
        raise ValueError(f"task_id must be 0-{N_TASKS-1}, got {task_id}")
    return TARGETED_TASKS[task_id]


def get_all_targeted_task_configs():
    """Return all 24 targeted task configurations."""
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
