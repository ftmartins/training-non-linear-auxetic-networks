"""
Task configuration and seed management for ensemble training.

This module handles the generation of task configurations and stiffness
initializations with proper seed management for reproducibility.
"""

import numpy as np
from config import (
    COMPRESSION_POOL, POISSON_POOL, N_TASKS, N_REALIZATIONS,
    STIFFNESS_LOG_MIN, STIFFNESS_LOG_MAX
)


def generate_task_config(task_seed):
    """
    Generate a reproducible task configuration from a task seed.

    Uses the task_seed to deterministically select 2 compression strains
    and 2 Poisson ratios from the respective pools.

    Args:
        task_seed: Integer seed (0 to N_TASKS-1)

    Returns:
        config_dict: Dictionary containing:
            - 'task_seed': int
            - 'packing_seed': int (same as task_seed)
            - 'compression_strains': list of 2 floats
            - 'target_poisson_ratios': list of 2 floats
    """
    # Create random state for reproducibility
    rng = np.random.RandomState(task_seed)

    # Select 2 compressions (without replacement)
    compression_strains = rng.choice(
        COMPRESSION_POOL,
        size=2,
        replace=False
    ).tolist()

    # Select 2 Poisson ratios (without replacement)
    target_poisson_ratios = rng.choice(
        POISSON_POOL,
        size=2,
        replace=False
    ).tolist()

    # Use task_seed as packing_seed (one unique network topology per task)
    packing_seed = task_seed

    return {
        'task_seed': task_seed,
        'packing_seed': packing_seed,
        'compression_strains': compression_strains,
        'target_poisson_ratios': target_poisson_ratios
    }


def generate_realization_stiffnesses(realization_seed, n_edges):
    """
    Generate random initial stiffnesses from log-uniform distribution.

    Args:
        realization_seed: Integer seed (0 to N_REALIZATIONS-1)
        n_edges: Number of edges in the network

    Returns:
        stiffnesses: Array of shape (n_edges,) with log-uniform values
    """
    rng = np.random.RandomState(realization_seed)
    log_stiffnesses = rng.uniform(
        STIFFNESS_LOG_MIN,
        STIFFNESS_LOG_MAX,
        size=n_edges
    )
    return np.exp(log_stiffnesses)


def get_all_task_configs(n_tasks=N_TASKS):
    """
    Generate all task configurations for the ensemble.

    Args:
        n_tasks: Number of tasks

    Returns:
        configs: List of task configuration dictionaries
    """
    return [generate_task_config(i) for i in range(n_tasks)]


def compute_target_extensions(compression_strains, target_poisson_ratios):
    """
    Compute target lateral extensions from compressions and Poisson ratios.

    For auxetic behavior, Poisson ratio ν = -(lateral_strain / vertical_strain).
    Given vertical compression strain (negative) and desired ν, we need:
    lateral_strain = -ν * vertical_strain

    Args:
        compression_strains: List of vertical compression strains (negative values)
        target_poisson_ratios: List of target Poisson ratios (negative for auxetic)

    Returns:
        target_extensions: List of target lateral strain values
    """
    return [
        -poisson * compression
        for poisson, compression in zip(target_poisson_ratios, compression_strains)
    ]


if __name__ == '__main__':
    # Test task generation reproducibility
    print("Testing task configuration generation...")

    # Test 1: Reproducibility
    print("\n1. Testing reproducibility:")
    config1_a = generate_task_config(0)
    config1_b = generate_task_config(0)
    print(f"   Config 0 (first call):  {config1_a}")
    print(f"   Config 0 (second call): {config1_b}")
    print(f"   Identical: {config1_a == config1_b}")

    # Test 2: Uniqueness
    print("\n2. Testing uniqueness across tasks:")
    for i in range(min(5, N_TASKS)):
        config = generate_task_config(i)
        print(f"   Task {i}:")
        print(f"     Compressions: {config['compression_strains']}")
        print(f"     Poisson ratios: {config['target_poisson_ratios']}")

    # Test 3: Stiffness generation reproducibility
    print("\n3. Testing stiffness generation reproducibility:")
    n_test_edges = 300
    stiff_a = generate_realization_stiffnesses(0, n_test_edges)
    stiff_b = generate_realization_stiffnesses(0, n_test_edges)
    print(f"   Generated {n_test_edges} stiffnesses")
    print(f"   Identical: {np.allclose(stiff_a, stiff_b)}")
    print(f"   Range: [{stiff_a.min():.2e}, {stiff_a.max():.2e}]")
    print(f"   Mean (log): {np.mean(np.log(stiff_a)):.2f}")

    # Test 4: Target extension computation
    print("\n4. Testing target extension computation:")
    test_compressions = [-0.04, -0.08]
    test_poissons = [-0.5, -1.0]
    test_extensions = compute_target_extensions(test_compressions, test_poissons)
    print(f"   Compressions: {test_compressions}")
    print(f"   Poisson ratios: {test_poissons}")
    print(f"   Target extensions: {test_extensions}")
    for i in range(len(test_compressions)):
        computed_poisson = -test_extensions[i] / test_compressions[i]
        print(f"   Verification: ν{i} = -{test_extensions[i]:.3f}/{test_compressions[i]:.3f} = {computed_poisson:.3f}")

    # Test 5: Generate all configs
    print("\n5. Generating all task configs:")
    all_configs = get_all_task_configs()
    print(f"   Total configs generated: {len(all_configs)}")

    # Check for duplicates
    compression_sets = [tuple(sorted(c['compression_strains'])) for c in all_configs]
    poisson_sets = [tuple(sorted(c['target_poisson_ratios'])) for c in all_configs]
    combined = list(zip(compression_sets, poisson_sets))
    unique_combined = set(combined)
    print(f"   Unique configurations: {len(unique_combined)}")
    if len(unique_combined) < len(all_configs):
        print(f"   WARNING: {len(all_configs) - len(unique_combined)} duplicate configurations found!")
    else:
        print("   All configurations are unique ✓")

    print("\nTask generator tests complete!")
