"""
Example script showing how to load and analyze training trajectories.

This demonstrates how to access the full stiffness and loss evolution
for each trained network.
"""

import numpy as np
import matplotlib.pyplot as plt
from data_loader import (
    load_training_result,
    load_loss_trajectory,
    load_stiffness_trajectory,
    load_all_results
)
from config import N_TASKS, N_REALIZATIONS

# ============================================================================
# Example 1: Load single training trajectory
# ============================================================================

def example_single_trajectory():
    """Load and plot a single training trajectory."""
    print("Example 1: Loading single trajectory")
    print("="*60)

    # Load results for task 0, realization 0
    result = load_training_result(task_seed=0, realization_seed=0)

    # Extract trajectory data
    history = result['history']
    stiffness_trajectory = history['stiffnesses']  # Shape: (n_steps, n_edges)
    loss_trajectory = history['loss']              # Shape: (n_steps,)

    print(f"Training steps: {len(loss_trajectory)}")
    print(f"Number of edges: {stiffness_trajectory.shape[1]}")
    print(f"Initial loss: {loss_trajectory[0]:.6e}")
    print(f"Final loss: {loss_trajectory[-1]:.6e}")
    print(f"Loss reduction: {loss_trajectory[0]/loss_trajectory[-1]:.1f}x")

    # Plot loss evolution
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.semilogy(loss_trajectory)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Loss Evolution')
    plt.grid(True, alpha=0.3)

    # Plot stiffness evolution for first 10 edges
    plt.subplot(1, 2, 2)
    for i in range(min(10, stiffness_trajectory.shape[1])):
        plt.plot(stiffness_trajectory[:, i], alpha=0.5, label=f'Edge {i}')
    plt.xlabel('Training Step')
    plt.ylabel('Stiffness')
    plt.title('Stiffness Evolution (first 10 edges)')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('trajectory_example.pdf', bbox_inches='tight')
    print(f"\nPlot saved to: trajectory_example.pdf")
    print()


# ============================================================================
# Example 2: Compare initial vs final stiffnesses
# ============================================================================

def example_stiffness_comparison():
    """Compare initial and final stiffness distributions."""
    print("Example 2: Stiffness distribution comparison")
    print("="*60)

    result = load_training_result(task_seed=0, realization_seed=0)
    stiffness_trajectory = result['history']['stiffnesses']

    initial_stiffnesses = stiffness_trajectory[0]
    final_stiffnesses = stiffness_trajectory[-1]

    print(f"Initial stiffnesses:")
    print(f"  Mean: {initial_stiffnesses.mean():.4e}")
    print(f"  Std:  {initial_stiffnesses.std():.4e}")
    print(f"  Min:  {initial_stiffnesses.min():.4e}")
    print(f"  Max:  {initial_stiffnesses.max():.4e}")

    print(f"\nFinal stiffnesses:")
    print(f"  Mean: {final_stiffnesses.mean():.4e}")
    print(f"  Std:  {final_stiffnesses.std():.4e}")
    print(f"  Min:  {final_stiffnesses.min():.4e}")
    print(f"  Max:  {final_stiffnesses.max():.4e}")

    # Plot distributions
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(np.log10(initial_stiffnesses), bins=30, alpha=0.7, label='Initial')
    plt.hist(np.log10(final_stiffnesses), bins=30, alpha=0.7, label='Final')
    plt.xlabel('log10(Stiffness)')
    plt.ylabel('Count')
    plt.title('Stiffness Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(initial_stiffnesses, final_stiffnesses, alpha=0.5)
    plt.plot([1e-6, 1e2], [1e-6, 1e2], 'r--', label='No change')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Initial Stiffness')
    plt.ylabel('Final Stiffness')
    plt.title('Stiffness Changes')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('stiffness_comparison.pdf', bbox_inches='tight')
    print(f"\nPlot saved to: stiffness_comparison.pdf")
    print()


# ============================================================================
# Example 3: Analyze convergence across realizations
# ============================================================================

def example_ensemble_convergence():
    """Analyze convergence patterns across ensemble."""
    print("Example 3: Ensemble convergence analysis")
    print("="*60)

    # Try to load first few realizations of task 0
    task_seed = 0
    n_realizations_to_load = min(5, N_REALIZATIONS)

    final_losses = []
    convergence_steps = []

    for real in range(n_realizations_to_load):
        try:
            result = load_training_result(task_seed=task_seed, realization_seed=real)
            loss_traj = result['history']['loss']

            final_losses.append(loss_traj[-1])

            # Find step where loss drops below 10% of initial
            threshold = 0.1 * loss_traj[0]
            converged_idx = np.where(loss_traj < threshold)[0]
            if len(converged_idx) > 0:
                convergence_steps.append(converged_idx[0])
            else:
                convergence_steps.append(len(loss_traj))

        except FileNotFoundError:
            print(f"  Realization {real} not found (not yet trained)")
            continue

    if len(final_losses) > 0:
        print(f"\nAnalyzed {len(final_losses)} realizations of Task {task_seed}:")
        print(f"  Mean final loss: {np.mean(final_losses):.6e} ± {np.std(final_losses):.6e}")
        print(f"  Mean convergence step: {np.mean(convergence_steps):.0f} ± {np.std(convergence_steps):.0f}")

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.bar(range(len(final_losses)), final_losses)
        plt.xlabel('Realization')
        plt.ylabel('Final Loss')
        plt.title(f'Final Loss Across Realizations (Task {task_seed})')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.bar(range(len(convergence_steps)), convergence_steps)
        plt.xlabel('Realization')
        plt.ylabel('Steps to 10% of Initial Loss')
        plt.title(f'Convergence Speed (Task {task_seed})')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('ensemble_convergence.pdf', bbox_inches='tight')
        print(f"\nPlot saved to: ensemble_convergence.pdf")
    else:
        print("No training results found yet.")

    print()


# ============================================================================
# Example 4: Fast loading with separate numpy files
# ============================================================================

def example_fast_loading():
    """Demonstrate fast loading using separate numpy files."""
    print("Example 4: Fast loading with separate numpy files")
    print("="*60)

    # Method 1: Load from separate numpy files (FAST - no pickle loading)
    print("Method 1: Load directly from numpy files (recommended for speed)")
    loss_traj = load_loss_trajectory(task_seed=0, realization_seed=0)
    stiffness_traj = load_stiffness_trajectory(task_seed=0, realization_seed=0)

    print(f"  Loss trajectory shape: {loss_traj.shape}")
    print(f"  Stiffness trajectory shape: {stiffness_traj.shape}")
    print(f"  Final loss: {loss_traj[-1]:.4e}")

    # Method 2: Load from full history pickle (slower but includes positions)
    print("\nMethod 2: Load from full history (slower, includes all data)")
    result = load_training_result(task_seed=0, realization_seed=0)
    loss_from_pkl = result['history']['loss']
    stiffness_from_pkl = result['history']['stiffnesses']

    print(f"  Loss trajectory shape: {loss_from_pkl.shape}")
    print(f"  Stiffness trajectory shape: {stiffness_from_pkl.shape}")

    # Verify they match
    print(f"\nVerification:")
    print(f"  Loss arrays match: {np.allclose(loss_traj, loss_from_pkl)}")
    print(f"  Stiffness arrays match: {np.allclose(stiffness_traj, stiffness_from_pkl)}")

    print()


# ============================================================================
# Example 5: Export trajectory data
# ============================================================================

def example_export_trajectory():
    """Export trajectory data to custom numpy files."""
    print("Example 5: Export trajectory data to custom location")
    print("="*60)

    # Note: Trajectories are already saved as:
    #   results/task_XX/realization_YY/loss_trajectory.npy
    #   results/task_XX/realization_YY/stiffness_trajectory.npy

    # But you can also export to custom location
    stiffness_trajectory = load_stiffness_trajectory(task_seed=0, realization_seed=0)
    loss_trajectory = load_loss_trajectory(task_seed=0, realization_seed=0)

    # Save to custom location
    np.save('custom_stiffness_trajectory.npy', stiffness_trajectory)
    np.save('custom_loss_trajectory.npy', loss_trajectory)

    print(f"Saved to custom location:")
    print(f"  custom_stiffness_trajectory.npy  (shape: {stiffness_trajectory.shape})")
    print(f"  custom_loss_trajectory.npy       (shape: {loss_trajectory.shape})")
    print()

    # Can reload with:
    # stiff = np.load('custom_stiffness_trajectory.npy')
    # loss = np.load('custom_loss_trajectory.npy')


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("TRAINING TRAJECTORY ANALYSIS EXAMPLES")
    print("="*60 + "\n")

    try:
        example_single_trajectory()
        example_stiffness_comparison()
        example_ensemble_convergence()
        example_fast_loading()
        example_export_trajectory()

        print("="*60)
        print("All examples completed successfully!")
        print("="*60 + "\n")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nNo training results found yet. Run training first:")
        print("  python ensemble_runner.py --mode single --task 0 --realization 0")
        print()
