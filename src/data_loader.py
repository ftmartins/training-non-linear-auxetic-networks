"""
Data loading and analysis utilities for ensemble training results.

This module provides functions to load training results and compute
summary statistics across the ensemble.
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from config import RESULTS_DIR, N_TASKS, N_REALIZATIONS
from checkpoint_manager import get_training_result_path


def load_loss_trajectory(task_seed, realization_seed, results_dir=None):
    """
    Load only the loss trajectory (fast, without loading full history).

    Args:
        task_seed: Task index
        realization_seed: Realization index
        results_dir: Results directory (default: from config)

    Returns:
        loss: (n_steps,) numpy array of loss values

    Example:
        >>> loss = load_loss_trajectory(task_seed=0, realization_seed=5)
        >>> print(f"Final loss: {loss[-1]:.4e}")
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    result_path = get_training_result_path(task_seed, realization_seed, results_dir)
    loss_file = result_path / "loss_trajectory.npy"

    if not loss_file.exists():
        raise FileNotFoundError(f"Loss trajectory not found: {loss_file}")

    return np.load(loss_file)


def load_stiffness_trajectory(task_seed, realization_seed, results_dir=None):
    """
    Load only the stiffness trajectory (fast, without loading full history).

    Args:
        task_seed: Task index
        realization_seed: Realization index
        results_dir: Results directory (default: from config)

    Returns:
        stiffnesses: (n_steps, n_edges) numpy array of stiffness evolution

    Example:
        >>> stiffnesses = load_stiffness_trajectory(task_seed=0, realization_seed=5)
        >>> initial = stiffnesses[0]
        >>> final = stiffnesses[-1]
        >>> print(f"Stiffness range: [{final.min():.2e}, {final.max():.2e}]")
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    result_path = get_training_result_path(task_seed, realization_seed, results_dir)
    stiffness_file = result_path / "stiffness_trajectory.npy"

    if not stiffness_file.exists():
        raise FileNotFoundError(f"Stiffness trajectory not found: {stiffness_file}")

    return np.load(stiffness_file)


def load_training_result(task_seed, realization_seed, results_dir=None):
    """
    Load results for a single training run.

    Args:
        task_seed: Task index
        realization_seed: Realization index
        results_dir: Results directory (default: from config)

    Returns:
        result_dict: Dictionary containing:
            - 'history': Training history dict with full trajectory:
                * 'stiffnesses': (n_steps, n_edges) array of stiffness evolution
                * 'loss': (n_steps,) array of loss values
                * 'positions': List of position arrays at each step
                * 'freetraj': Optional free trajectory data
            - 'network': Final network state dict
            - 'task_config': Task configuration dict

    Example:
        >>> result = load_training_result(task_seed=0, realization_seed=5)
        >>> stiffness_trajectory = result['history']['stiffnesses']  # (n_steps, n_edges)
        >>> loss_trajectory = result['history']['loss']               # (n_steps,)
        >>> final_stiffnesses = stiffness_trajectory[-1]              # (n_edges,)
        >>> print(f"Training converged from {loss_trajectory[0]:.2e} to {loss_trajectory[-1]:.2e}")

    Note:
        For faster access to just loss or stiffness trajectories, use:
        - load_loss_trajectory(task_seed, realization_seed)
        - load_stiffness_trajectory(task_seed, realization_seed)
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    result_path = get_training_result_path(task_seed, realization_seed, results_dir)

    if not result_path.exists():
        raise FileNotFoundError(f"Results not found: {result_path}")

    # Load history
    with open(result_path / "history.pkl", "rb") as f:
        history = pickle.load(f)

    # Load network
    with open(result_path / "final_network.pkl", "rb") as f:
        network = pickle.load(f)

    # Load config
    with open(result_path / "task_config.json", "r") as f:
        task_config = json.load(f)

    return {
        'history': history,
        'network': network,
        'task_config': task_config
    }


def load_all_results(results_dir=None):
    """
    Load all ensemble results.

    Args:
        results_dir: Results directory (default: from config)

    Returns:
        results: Dict mapping (task_seed, realization_seed) -> result_dict
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    results = {}
    missing = []

    for task in range(N_TASKS):
        for real in range(N_REALIZATIONS):
            try:
                results[(task, real)] = load_training_result(task, real, results_dir)
            except FileNotFoundError:
                missing.append((task, real))

    if missing:
        print(f"Warning: {len(missing)} results missing")
        print(f"  Missing jobs: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    return results


def create_ensemble_dataframe(results_dir=None):
    """
    Create a pandas DataFrame summarizing all ensemble results.

    Args:
        results_dir: Results directory (default: from config)

    Returns:
        df: DataFrame with columns:
            - task_seed
            - realization_seed
            - compression_strain_1
            - compression_strain_2
            - target_poisson_1
            - target_poisson_2
            - final_loss
            - min_loss
            - n_edges
            - n_steps_completed
            - converged (final_loss < threshold)
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    rows = []

    for task in range(N_TASKS):
        for real in range(N_REALIZATIONS):
            try:
                result = load_training_result(task, real, results_dir)
                history = result['history']
                config = result['task_config']
                network = result['network']

                row = {
                    'task_seed': task,
                    'realization_seed': real,
                    'compression_strain_1': config['compression_strains'][0],
                    'compression_strain_2': config['compression_strains'][1],
                    'target_poisson_1': config['target_poisson_ratios'][0],
                    'target_poisson_2': config['target_poisson_ratios'][1],
                    'final_loss': history['loss'][-1] if 'loss' in history and history['loss'] else np.nan,
                    'min_loss': np.min(history['loss']) if 'loss' in history and history['loss'] else np.nan,
                    'n_edges': len(network['edges']),
                    'n_steps_completed': len(history.get('loss', []))
                }

                # Check convergence (arbitrary threshold)
                row['converged'] = row['final_loss'] < 1e-4 if not np.isnan(row['final_loss']) else False

                rows.append(row)

            except FileNotFoundError:
                pass

    df = pd.DataFrame(rows)
    return df


def print_ensemble_summary(results_dir=None):
    """
    Print summary statistics for ensemble.

    Args:
        results_dir: Results directory (default: from config)
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    try:
        df = create_ensemble_dataframe(results_dir)
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        return

    if len(df) == 0:
        print("No results found!")
        return

    print(f"\n{'='*80}")
    print(f"ENSEMBLE SUMMARY")
    print(f"{'='*80}")
    print(f"Total completed jobs: {len(df)} / {N_TASKS * N_REALIZATIONS}")

    print(f"\nFinal Loss Statistics:")
    print(df['final_loss'].describe())

    print(f"\nMinimum Loss Statistics:")
    print(df['min_loss'].describe())

    print(f"\nNetwork Size Statistics:")
    print(df['n_edges'].describe())

    print(f"\nConvergence:")
    n_converged = df['converged'].sum()
    print(f"  Converged (loss < 1e-4): {n_converged} / {len(df)} ({100*n_converged/len(df):.1f}%)")

    print(f"\nTraining Steps:")
    print(df['n_steps_completed'].describe())

    print(f"\n{'='*80}\n")

    # Summary by task
    print("Summary by task:")
    task_summary = df.groupby('task_seed').agg({
        'final_loss': ['mean', 'std', 'min', 'max'],
        'converged': 'sum',
        'realization_seed': 'count'
    })
    print(task_summary)

    print(f"\n{'='*80}\n")


def export_results_to_csv(output_file=None, results_dir=None):
    """
    Export ensemble results to CSV file.

    Args:
        output_file: Output CSV file path (default: results/ensemble_summary.csv)
        results_dir: Results directory (default: from config)
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    if output_file is None:
        output_file = Path(results_dir) / "ensemble_summary.csv"

    df = create_ensemble_dataframe(results_dir)
    df.to_csv(output_file, index=False)
    print(f"Exported {len(df)} results to {output_file}")


if __name__ == '__main__':
    # Test data loader
    print("Testing data loader...")

    # Try to load results (will fail if none exist yet)
    try:
        print_ensemble_summary()
    except Exception as e:
        print(f"\nNo results to load yet (expected): {e}")
        print("\nTo generate test results, run:")
        print("  python ensemble_runner.py --mode single --task 0 --realization 0")

    print("\nData loader module loaded successfully!")
