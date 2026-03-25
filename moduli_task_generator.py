"""
Moduli-based task configurations for elastic network training.

18 hardcoded tasks across 6 categories, all sharing the same network topology
(PACKING_SEED = 42). Target values are multipliers of the natural moduli
computed from the uniform-stiffness (k=1) reference network.

Categories:
  1: Single modulus (B or G) at -0.2 compression
  2: B and G at -0.2 compression
  3: B and G at -0.2 and -0.4 compression
  4: C_ij triplet at -0.2 compression
  5: C_ij triplet at -0.2 and -0.4 compression
  6: C_ij triplet at -0.2, -0.3, -0.4 compression

Tasks are organized in 6 threads (A-F), each spanning multiple categories.
Higher-category tasks strictly contain the lower-category targets at -0.2,
then add targets at additional compression strains:

  Thread A (tasks 0,3,6):  B×0.5 → +G×1.3 → +strain -0.4
  Thread B (tasks 1,4,7):  B×1.5 → +G×0.7 → +strain -0.4
  Thread C (tasks 2,5,8):  G×0.5 → +B×1.5 → +strain -0.4
  Thread D (tasks 9,12,15): (C11,C22,C12) at -0.2 → +strain -0.4 → +strain -0.3
  Thread E (tasks 10,13,16): (C11,C33,C12) at -0.2 → +strain -0.4 → +strain -0.3
  Thread F (tasks 11,14,17): (C22,C33,C23) at -0.2 → +strain -0.4 → +strain -0.3
"""

import numpy as np
from pathlib import Path

import jax.numpy as jnp

from config import BASE_DIR
from task_generator import generate_realization_stiffnesses

# ============================================================================
# Moduli Training Configuration
# ============================================================================

N_TASKS = 18
N_REALIZATIONS = 3
N_STEPS = 10_000
N_STRAIN_STEPS = 100
PACKING_SEED = 42

ENSEMBLE_DIR = Path('/data2/shared/felipetm/auxetic_networks/') / 'ensemble_training'
MODULI_RESULTS_DIR = ENSEMBLE_DIR / 'moduli_results'

# All compression strains used across categories
ALL_COMPRESSION_STRAINS = [-0.1, -0.15, -0.2]

# ============================================================================
# Task Definitions (multiplier-based)
# ============================================================================

MODULI_TASKS = [
    # ==================================================================
    # Thread A: B×0.5 → +G×1.3 → +strain -0.4
    # ==================================================================

    # Category 1: Single modulus at -0.2
    {
        'task_seed': 0,
        'packing_seed': PACKING_SEED,
        'category': 1,
        'category_name': 'single_modulus_1strain',
        'training_goals_multipliers': {
            -0.1: {'B': 0.5},
        },
    },
    # Category 2: B and G at -0.2 (adds G×1.3 to task 0's B×0.5)
    {
        'task_seed': 3,
        'packing_seed': PACKING_SEED,
        'category': 2,
        'category_name': 'BG_1strain',
        'training_goals_multipliers': {
            -0.1: {'B': 0.5, 'G': 1.3},
        },
    },
    # Category 3: B and G at -0.2 and -0.4 (adds strain -0.4 to task 3)
    {
        'task_seed': 6,
        'packing_seed': PACKING_SEED,
        'category': 3,
        'category_name': 'BG_2strains',
        'training_goals_multipliers': {
            -0.1: {'B': 0.5, 'G': 1.3},
            -0.2: {'B': 0.7, 'G': 1.5},
        },
    },

    # ==================================================================
    # Thread B: B×1.5 → +G×0.7 → +strain -0.4
    # ==================================================================

    # Category 1
    {
        'task_seed': 1,
        'packing_seed': PACKING_SEED,
        'category': 1,
        'category_name': 'single_modulus_1strain',
        'training_goals_multipliers': {
            -0.1: {'B': 1.5},
        },
    },
    # Category 2 (adds G×0.7 to task 1's B×1.5)
    {
        'task_seed': 4,
        'packing_seed': PACKING_SEED,
        'category': 2,
        'category_name': 'BG_1strain',
        'training_goals_multipliers': {
            -0.1: {'B': 1.5, 'G': 0.7},
        },
    },
    # Category 3 (adds strain -0.4 to task 4)
    {
        'task_seed': 7,
        'packing_seed': PACKING_SEED,
        'category': 3,
        'category_name': 'BG_2strains',
        'training_goals_multipliers': {
            -0.1: {'B': 1.5, 'G': 0.7},
            -0.2: {'B': 1.3, 'G': 0.5},
        },
    },

    # ==================================================================
    # Thread C: G×0.5 → +B×1.5 → +strain -0.4
    # ==================================================================

    # Category 1
    {
        'task_seed': 2,
        'packing_seed': PACKING_SEED,
        'category': 1,
        'category_name': 'single_modulus_1strain',
        'training_goals_multipliers': {
            -0.1: {'G': 0.5},
        },
    },
    # Category 2 (adds B×1.5 to task 2's G×0.5)
    {
        'task_seed': 5,
        'packing_seed': PACKING_SEED,
        'category': 2,
        'category_name': 'BG_1strain',
        'training_goals_multipliers': {
            -0.1: {'B': 1.5, 'G': 0.5},
        },
    },
    # Category 3 (adds strain -0.4 to task 5)
    {
        'task_seed': 8,
        'packing_seed': PACKING_SEED,
        'category': 3,
        'category_name': 'BG_2strains',
        'training_goals_multipliers': {
            -0.1: {'B': 1.5, 'G': 0.5},
            -0.2: {'B': 1.3, 'G': 0.7},
        },
    },

    # ==================================================================
    # Thread D: (C11,C22,C12) at -0.2 → +strain -0.4 → +strain -0.3
    # ==================================================================

    # Category 4: C_ij triplet at -0.2
    {
        'task_seed': 9,
        'packing_seed': PACKING_SEED,
        'category': 4,
        'category_name': 'Cij_triplet_1strain',
        'training_goals_multipliers': {
            -0.1: {'C_11': 0.7, 'C_22': 1.3, 'C_12': 0.5},
        },
    },
    # Category 5: adds strain -0.4 (same keys, different multipliers)
    {
        'task_seed': 12,
        'packing_seed': PACKING_SEED,
        'category': 5,
        'category_name': 'Cij_triplet_2strains',
        'training_goals_multipliers': {
            -0.1: {'C_11': 0.7, 'C_22': 1.3, 'C_12': 0.5},
            -0.2: {'C_11': 1.3, 'C_22': 0.7, 'C_12': 1.5},
        },
    },
    # Category 6: adds strain -0.3
    {
        'task_seed': 15,
        'packing_seed': PACKING_SEED,
        'category': 6,
        'category_name': 'Cij_triplet_3strains',
        'training_goals_multipliers': {
            -0.1: {'C_11': 0.7, 'C_22': 1.3, 'C_12': 0.5},
            -0.15: {'C_11': 1.0, 'C_22': 1.0, 'C_12': 0.7},
            -0.2: {'C_11': 1.3, 'C_22': 0.7, 'C_12': 1.5},
        },
    },

    # ==================================================================
    # Thread E: (C11,C33,C12) at -0.2 → +strain -0.4 → +strain -0.3
    # ==================================================================

    # Category 4
    {
        'task_seed': 10,
        'packing_seed': PACKING_SEED,
        'category': 4,
        'category_name': 'Cij_triplet_1strain',
        'training_goals_multipliers': {
            -0.1: {'C_11': 1.3, 'C_33': 0.7, 'C_12': 1.5},
        },
    },
    # Category 5 (adds strain -0.4, same -0.2 targets as task 10)
    {
        'task_seed': 13,
        'packing_seed': PACKING_SEED,
        'category': 5,
        'category_name': 'Cij_triplet_2strains',
        'training_goals_multipliers': {
            -0.1: {'C_11': 1.3, 'C_33': 0.7, 'C_12': 1.5},
            -0.2: {'C_11': 0.5, 'C_33': 1.3, 'C_12': 0.7},
        },
    },
    # Category 6 (adds strain -0.3, same -0.2 and -0.4 as task 13)
    {
        'task_seed': 16,
        'packing_seed': PACKING_SEED,
        'category': 6,
        'category_name': 'Cij_triplet_3strains',
        'training_goals_multipliers': {
            -0.1: {'C_11': 1.3, 'C_33': 0.7, 'C_12': 1.5},
            -0.15: {'C_11': 0.7, 'C_33': 1.3, 'C_12': 0.5},
            -0.2: {'C_11': 0.5, 'C_33': 1.3, 'C_12': 0.7},
        },
    },

    # ==================================================================
    # Thread F: (C22,C33,C23) at -0.2 → +strain -0.4 → +strain -0.3
    # ==================================================================

    # Category 4
    {
        'task_seed': 11,
        'packing_seed': PACKING_SEED,
        'category': 4,
        'category_name': 'Cij_triplet_1strain',
        'training_goals_multipliers': {
            -0.1: {'C_22': 0.5, 'C_33': 1.5, 'C_23': 0.7},
        },
    },
    # Category 5 (adds strain -0.4, same -0.2 targets as task 11)
    {
        'task_seed': 14,
        'packing_seed': PACKING_SEED,
        'category': 5,
        'category_name': 'Cij_triplet_2strains',
        'training_goals_multipliers': {
            -0.1: {'C_22': 0.5, 'C_33': 1.5, 'C_23': 0.7},
            -0.2: {'C_22': 1.5, 'C_33': 0.5, 'C_23': 1.3},
        },
    },
    # Category 6 (adds strain -0.3, same -0.2 and -0.4 as task 14)
    {
        'task_seed': 17,
        'packing_seed': PACKING_SEED,
        'category': 6,
        'category_name': 'Cij_triplet_3strains',
        'training_goals_multipliers': {
            -0.1: {'C_22': 0.5, 'C_33': 1.5, 'C_23': 0.7},
            -0.15: {'C_22': 1.3, 'C_33': 0.7, 'C_23': 1.5},
            -0.2: {'C_22': 1.5, 'C_33': 0.5, 'C_23': 1.3},
        },
    },
]


# ============================================================================
# Reference Moduli Computation
# ============================================================================


def compute_reference_moduli(network, boundary_dict, compression_strains,
                              n_strain_steps=100, force_type='quadratic',
                              fire_max_steps=100_000, fire_tol=1e-6):
    """
    Compute B, G, nu, and full C_voigt at each compression strain
    for the reference network (uniform stiffnesses) using the JAX
    quasistatic trajectory.

    Called ONCE at runner startup to resolve multiplier-based targets
    into absolute values.

    Parameters
    ----------
    network : ElasticNetwork
        Reference network with uniform stiffnesses (k=1).
    boundary_dict : dict
        Boundary node indices with keys 'top', 'bottom', 'left', 'right'.
    compression_strains : list of float
        Compression strains at which to evaluate moduli (e.g., [-0.2, -0.3, -0.4]).
    n_strain_steps : int
        Number of quasistatic compression steps.
    force_type : str
        'quadratic' or 'quartic'.
    fire_max_steps : int
        Max iterations for JAX FIRE solver.
    fire_tol : float
        Force tolerance for JAX FIRE solver.

    Returns
    -------
    ref : dict
        Maps compression_strain (float) -> {
            'B': float, 'G': float, 'nu': float, 'C_voigt': (3, 3) array
        }
    """
    from training_functions_with_toggle import make_compute_response_fire
    from training_functions_with_toggle import compute_quasistatic_trajectory_auxetic_jax
    from elasticity_tensor import (
        compute_elasticity_tensor_2d,
        extract_moduli_2d,
        precompute_dof_indices,
    )

    d = 2

    # Build JAX FIRE solver
    crf = make_compute_response_fire(
        d=d, force_type=force_type,
        max_steps=fire_max_steps, tol=fire_tol,
    )

    # Prepare arrays
    edges_jax = jnp.asarray(np.array(network.edges, dtype=np.int32))
    rest_lengths_jax = jnp.asarray(np.array(network.rest_lengths, dtype=np.float64))
    stiffnesses_jax = jnp.asarray(np.array(network.stiffnesses, dtype=np.float64))
    positions_flat_jax = jnp.asarray(network.positions.flatten())
    top_jax = jnp.asarray(boundary_dict['top'])
    bottom_jax = jnp.asarray(boundary_dict['bottom'])

    # Precompute DOF indices for elasticity tensor
    boundary_nodes_all = np.unique(np.concatenate([
        boundary_dict['top'], boundary_dict['bottom'],
        boundary_dict['left'], boundary_dict['right'],
    ]))
    n_nodes = len(network.positions)
    boundary_dofs, interior_dofs = precompute_dof_indices(
        boundary_nodes_all, n_nodes, d=d,
    )

    ref = {}
    for strain in compression_strains:
        print(f"  Computing reference moduli at compression strain {strain}...")

        # Quasistatic compression
        strained_pos_flat = compute_quasistatic_trajectory_auxetic_jax(
            crf, stiffnesses_jax, edges_jax, rest_lengths_jax,
            positions_flat_jax,
            top_jax, bottom_jax,
            strain, n_strain_steps, d=d,
        )
        strained_pos_2d = jnp.reshape(strained_pos_flat, (-1, d))

        # Elasticity tensor
        C_voigt = compute_elasticity_tensor_2d(
            strained_pos_2d, edges_jax, stiffnesses_jax,
            rest_lengths_jax, boundary_dofs, interior_dofs,
            force_type=force_type,
        )

        moduli = extract_moduli_2d(C_voigt)
        ref[strain] = {
            'B': float(moduli['B']),
            'G': float(moduli['G']),
            'nu': float(moduli['nu']),
            'C_voigt': np.array(C_voigt),
        }
        print(f"    B={ref[strain]['B']:.4f}, G={ref[strain]['G']:.4f}, "
              f"nu={ref[strain]['nu']:.4f}")

    return ref


# ============================================================================
# Resolve Multipliers to Absolute Targets
# ============================================================================

# Mapping from target key to how to look up the reference value
_MODULUS_KEYS = {'B', 'G', 'nu'}


def _get_reference_value(key, ref_at_strain):
    """
    Look up the reference value for a target key.

    For 'B', 'G', 'nu': direct lookup.
    For 'C_ij': index into C_voigt (1-indexed Voigt).
    """
    if key in _MODULUS_KEYS:
        return ref_at_strain[key]
    # C_ij key
    if key.startswith('C_') and len(key) == 4:
        i = int(key[2]) - 1
        j = int(key[3]) - 1
        return float(ref_at_strain['C_voigt'][i, j])
    raise ValueError(f"Unknown target key: {key}")


def resolve_training_goals(task_config, reference_moduli):
    """
    Convert multiplier-based goals to absolute target values.

    Parameters
    ----------
    task_config : dict
        Task configuration with 'training_goals_multipliers'.
    reference_moduli : dict
        Output of compute_reference_moduli().

    Returns
    -------
    training_goals : dict
        Maps compression_strain -> {key: absolute_value}, suitable for
        run_moduli_training().
    """
    training_goals = {}
    for strain, targets in task_config['training_goals_multipliers'].items():
        ref_at_strain = reference_moduli[strain]
        resolved = {}
        for key, multiplier in targets.items():
            ref_val = _get_reference_value(key, ref_at_strain)
            resolved[key] = multiplier * ref_val
        training_goals[strain] = resolved
    return training_goals


# ============================================================================
# Accessor Functions
# ============================================================================


def get_moduli_task_config(task_id):
    """Return task configuration for a specific moduli task."""
    if task_id < 0 or task_id >= N_TASKS:
        raise ValueError(f"task_id must be 0-{N_TASKS-1}, got {task_id}")
    return MODULI_TASKS[task_id]


def get_all_moduli_task_configs():
    """Return all 18 moduli task configurations."""
    return MODULI_TASKS


def print_moduli_tasks_summary(reference_moduli=None):
    """Print a formatted summary of all moduli tasks."""
    print(f"\n{'='*80}")
    print("MODULI TRAINING TASKS")
    print(f"{'='*80}")
    print(f"Tasks: {N_TASKS}")
    print(f"Realizations per task: {N_REALIZATIONS}")
    print(f"Training steps: {N_STEPS:,}")
    print(f"Strain steps: {N_STRAIN_STEPS}")
    print(f"Packing seed (shared): {PACKING_SEED}")
    print(f"Results directory: {MODULI_RESULTS_DIR}")
    print()

    current_cat = None
    for task in MODULI_TASKS:
        tid = task['task_seed']
        cat = task['category']
        cat_name = task['category_name']

        if cat != current_cat:
            current_cat = cat
            print(f"  --- Category {cat}: {cat_name} ---")

        goals = task['training_goals_multipliers']
        parts = []
        for strain, targets in goals.items():
            target_strs = [f"{k}x{v}" for k, v in targets.items()]
            parts.append(f"strain={strain}: {', '.join(target_strs)}")

            if reference_moduli is not None:
                abs_targets = resolve_training_goals(task, reference_moduli)
                abs_strs = [f"{k}={v:.4f}" for k, v in abs_targets[strain].items()]
                parts[-1] += f"  => {', '.join(abs_strs)}"

        print(f"    Task {tid:2d}: {' | '.join(parts)}")

    print(f"{'='*80}\n")


if __name__ == '__main__':
    print_moduli_tasks_summary()

    # Verify task structure
    print("Verifying task structure:")
    for task in MODULI_TASKS:
        tid = task['task_seed']
        cat = task['category']
        n_strains = len(task['training_goals_multipliers'])
        n_targets = sum(len(t) for t in task['training_goals_multipliers'].values())
        print(f"  Task {tid:2d}: category={cat}, strains={n_strains}, targets={n_targets}")

    # Test stiffness generation
    print(f"\nStiffness generation test:")
    test_stiff = generate_realization_stiffnesses(0, 300)
    print(f"  300 stiffnesses, range: [{test_stiff.min():.2e}, {test_stiff.max():.2e}]")

    print("\nAll checks passed!")
