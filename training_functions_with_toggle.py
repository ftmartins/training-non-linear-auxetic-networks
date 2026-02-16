"""
Training functions for quartic spring networks using Cython FIRE minimization.

This module provides all necessary functions for training elastic networks
with optimized Cython FIRE minimization.
"""

import numpy as np
import copy
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from fire_minimize_memview_cy import fire_minimize_dof
from checkpoint_manager import (
    save_training_results,
)


# ============================================================================
# FIRE MINIMIZATION
# ============================================================================

def fire_minimize_network(network, constrained_dof_idx=None, force_type='quartic',
                         tol=1e-6, max_steps=500_000, deltaT=1e-2):
    """
    Minimize network using Cython FIRE.

    Args:
        network: ElasticNetwork object
        constrained_dof_idx: List of DOF indices to constrain or None
        force_type: 'quartic' or 'quadratic'
        tol: Convergence tolerance
        max_steps: Maximum number of minimization steps
        deltaT: FIRE timestep (default 1e-2 for fast convergence)

    Returns:
        min_positions: Minimized positions (N, d) array
        force_norm: Final force norm
    """
    if constrained_dof_idx is None:
        constrained_dof_idx = []

    min_pos, force_norm, _ = fire_minimize_dof(
        network.positions,
        np.array(network.edges, dtype=np.int32),
        np.array(network.rest_lengths, dtype=np.float64),
        np.array(network.stiffnesses, dtype=np.float64),
        deltaT,
        max_steps,
        tol,
        constrained_dof_idx,
        1 if force_type == 'quartic' else 0
    )
    return min_pos, force_norm


# ============================================================================
# TRAJECTORY COMPUTATION
# ============================================================================

def compute_quasistatic_trajectory_auxetic(network, compression_strain, top_nodes, bottom_nodes,
                                          n_steps=100, verbose=False, force_type='quartic',
                                          tol=1e-6, d=2):
    """
    Apply a quasistatic compression-extension trajectory to the network.

    Args:
        network: ElasticNetwork object
        compression_strain: Target strain (e.g., 0.08 for 8%)
        top_nodes: Indices of top boundary nodes
        bottom_nodes: Indices of bottom boundary nodes
        n_steps: Number of quasistatic steps
        verbose: Print progress
        force_type: 'quartic' or 'quadratic'
        tol: Convergence tolerance
        d: Dimensionality (2 or 3)

    Returns:
        traj: List of positions arrays at each step
    """
    positions = np.copy(network.positions)
    y_top = positions[top_nodes, 1]
    y_bottom = positions[bottom_nodes, 1]
    initial_height = y_top.mean() - y_bottom.mean()
    target_height = initial_height * (1 + compression_strain)

    traj = [np.copy(positions)]

    # Prepare DOF constraints (y-direction only for top and bottom nodes)
    constrained_idx_dof = []
    for i in np.concatenate([top_nodes, bottom_nodes]):
        constrained_idx_dof.append(i * d + 1)  # y DOF only

    for step in range(n_steps):
        frac = step / (n_steps - 1)
        height_to_impose = initial_height - frac * (initial_height - target_height)
        y_top_new = y_bottom.mean() + height_to_impose

        # Set constrained positions
        positions_step = np.copy(positions)
        positions_step[top_nodes, 1] = y_top_new + (positions[top_nodes, 1] - positions[top_nodes, 1].mean())
        positions_step[bottom_nodes, 1] = y_bottom

        # Direct Cython call for better performance
        min_pos, force_norm, _ = fire_minimize_dof(
            positions_step,
            np.array(network.edges, dtype=np.int32),
            np.array(network.rest_lengths, dtype=np.float64),
            np.array(network.stiffnesses, dtype=np.float64),
            1e-2,  # deltaT for trajectory (smaller for accuracy)
            500_000,
            tol,
            constrained_idx_dof,
            1 if force_type == 'quartic' else 0
        )

        positions = min_pos
        traj.append(np.copy(min_pos))

    return traj


def compute_quasistatic_trajectory_full_cycle(network, amp, top_nodes, bottom_nodes,
                                              n_steps_per_phase=100, verbose=False,
                                              force_type='quartic', tol=1e-6, d=2):
    """
    Apply a full compression-extension-relaxation cycle to the network.

    Cycle phases:
    1. Compress from 0 to +amp (compression)
    2. Extend from +amp to -amp (extension through initial state)
    3. Relax from -amp back to 0 (return to initial)

    Args:
        network: ElasticNetwork object
        amp: Amplitude of strain (e.g., 0.08 for 8%)
        top_nodes: Indices of top boundary nodes
        bottom_nodes: Indices of bottom boundary nodes
        n_steps_per_phase: Number of quasistatic steps per phase
        verbose: Print progress
        force_type: 'quartic' or 'quadratic'
        tol: Convergence tolerance
        d: Dimensionality (2 or 3)

    Returns:
        traj: List of positions arrays at each step (length = 3*n_steps_per_phase + 1)
    """
    positions = np.copy(network.positions)
    y_top = positions[top_nodes, 1]
    y_bottom = positions[bottom_nodes, 1]
    initial_height = y_top.mean() - y_bottom.mean()

    traj = [np.copy(positions)]

    # Prepare DOF constraints (y-direction only for top and bottom nodes)
    constrained_idx_dof = []
    for i in np.concatenate([top_nodes, bottom_nodes]):
        constrained_idx_dof.append(i * d + 1)  # y DOF only

    # Phase 1: Compress from 0 to +amp
    if verbose:
        print(f"Phase 1: Compressing to +{amp}")
    for step in range(n_steps_per_phase):
        frac = (step + 1) / n_steps_per_phase
        current_strain = frac * amp
        height_to_impose = initial_height * (1 + current_strain)
        y_top_new = y_bottom.mean() + height_to_impose

        # Set constrained positions
        positions_step = np.copy(positions)
        positions_step[top_nodes, 1] = y_top_new + (positions[top_nodes, 1] - positions[top_nodes, 1].mean())
        positions_step[bottom_nodes, 1] = y_bottom

        # Direct FIRE call
        min_pos, force_norm, _ = fire_minimize_dof(
            positions_step,
            np.array(network.edges, dtype=np.int32),
            np.array(network.rest_lengths, dtype=np.float64),
            np.array(network.stiffnesses, dtype=np.float64),
            1e-3,
            500_000,
            tol,
            constrained_idx_dof,
            1 if force_type == 'quartic' else 0
        )

        positions = min_pos
        traj.append(np.copy(min_pos))

    # Phase 2: Extend from +amp to -amp
    if verbose:
        print(f"Phase 2: Extending from +{amp} to -{amp}")
    for step in range(n_steps_per_phase):
        frac = (step + 1) / n_steps_per_phase
        current_strain = amp - frac * (2 * amp)  # Goes from +amp to -amp
        height_to_impose = initial_height * (1 + current_strain)
        y_top_new = y_bottom.mean() + height_to_impose

        # Set constrained positions
        positions_step = np.copy(positions)
        positions_step[top_nodes, 1] = y_top_new + (positions[top_nodes, 1] - positions[top_nodes, 1].mean())
        positions_step[bottom_nodes, 1] = y_bottom

        # Direct FIRE call
        min_pos, force_norm, _ = fire_minimize_dof(
            positions_step,
            np.array(network.edges, dtype=np.int32),
            np.array(network.rest_lengths, dtype=np.float64),
            np.array(network.stiffnesses, dtype=np.float64),
            1e-3,
            500_000,
            tol,
            constrained_idx_dof,
            1 if force_type == 'quartic' else 0
        )

        positions = min_pos
        traj.append(np.copy(min_pos))

    # Phase 3: Relax from -amp back to 0
    if verbose:
        print(f"Phase 3: Relaxing from -{amp} to 0")
    for step in range(n_steps_per_phase):
        frac = (step + 1) / n_steps_per_phase
        current_strain = -amp + frac * amp  # Goes from -amp to 0
        height_to_impose = initial_height * (1 + current_strain)
        y_top_new = y_bottom.mean() + height_to_impose

        # Set constrained positions
        positions_step = np.copy(positions)
        positions_step[top_nodes, 1] = y_top_new + (positions[top_nodes, 1] - positions[top_nodes, 1].mean())
        positions_step[bottom_nodes, 1] = y_bottom

        # Direct FIRE call
        min_pos, force_norm, _ = fire_minimize_dof(
            positions_step,
            np.array(network.edges, dtype=np.int32),
            np.array(network.rest_lengths, dtype=np.float64),
            np.array(network.stiffnesses, dtype=np.float64),
            1e-3,
            500_000,
            tol,
            constrained_idx_dof,
            1 if force_type == 'quartic' else 0
        )

        positions = min_pos
        traj.append(np.copy(min_pos))

    return traj


# ============================================================================
# TRAINING HELPER FUNCTIONS
# ============================================================================

def compute_poisson_ratio_single(network, top_nodes, bottom_nodes, left_nodes, right_nodes,
                                 compression_strain, n_strain_steps=100, force_type='quartic'):
    """
    Compute Poisson ratio for a single compression strain.
    """
    traj = compute_quasistatic_trajectory_auxetic(
        network,
        compression_strain,
        top_nodes,
        bottom_nodes,
        n_steps=n_strain_steps,
        verbose=False,
        force_type=force_type,
        tol=1e-6
    )

    positions_free_final = traj[-1]
    positions_free_initial = traj[0]

    left_x1 = positions_free_final[left_nodes, 0].mean()
    right_x1 = positions_free_final[right_nodes, 0].mean()
    left_x2 = positions_free_initial[left_nodes, 0].mean()
    right_x2 = positions_free_initial[right_nodes, 0].mean()

    width_free_final = right_x1 - left_x1
    width_free_initial = right_x2 - left_x2
    lateral_strain = (width_free_final - width_free_initial) / width_free_initial
    poisson_ratio = -(lateral_strain / compression_strain)

    return poisson_ratio


def poisson_loss_batch_parallel(network, target_poisson_list, top_nodes, bottom_nodes,
                                left_nodes, right_nodes, compression_strain_list,
                                n_strain_steps=100, n_jobs_inner=4, force_type='quartic'):
    """
    Compute MSE loss across multiple compression-Poisson pairs in parallel.
    """
    computed_poisson_ratios = Parallel(n_jobs=n_jobs_inner)(
        delayed(compute_poisson_ratio_single)(
            network, top_nodes, bottom_nodes, left_nodes, right_nodes,
            cs, n_strain_steps, force_type=force_type
        )
        for cs in compression_strain_list
    )

    computed_poisson_ratios = np.array(computed_poisson_ratios)
    mse_loss = np.mean((computed_poisson_ratios - np.array(target_poisson_list))**2)

    return mse_loss, computed_poisson_ratios


def compute_gradient_entry_batch(i, network, target_poisson_list, top_nodes, bottom_nodes,
                                 left_nodes, right_nodes, compression_strain_list,
                                 epsilon, n_strain_steps, n_jobs_inner=4, force_type='quartic'):
    """
    Compute gradient for a single stiffness using finite differences.
    """
    orig = network.stiffnesses[i]

    # Perturb up
    network.stiffnesses[i] = orig + epsilon
    loss_plus, _ = poisson_loss_batch_parallel(
        network, target_poisson_list, top_nodes, bottom_nodes,
        left_nodes, right_nodes, compression_strain_list,
        n_strain_steps, n_jobs_inner, force_type=force_type
    )

    # Perturb down
    network.stiffnesses[i] = orig - epsilon
    loss_minus, _ = poisson_loss_batch_parallel(
        network, target_poisson_list, top_nodes, bottom_nodes,
        left_nodes, right_nodes, compression_strain_list,
        n_strain_steps, n_jobs_inner, force_type=force_type
    )

    # Restore
    network.stiffnesses[i] = orig

    # Gradient
    value = (loss_plus - loss_minus) / (2 * epsilon)

    return (i, value)


def finite_difference_gradient_parallel_batch(network, target_poisson_list, top_nodes, bottom_nodes,
                                             left_nodes, right_nodes, compression_strain_list,
                                             epsilon=1e-8, n_jobs_outer=4, n_jobs_inner=2,
                                             n_strain_steps=100, force_type='quartic'):
    """
    Compute gradient across all stiffnesses using parallel finite differences.

    - Outer parallelization: across stiffness indices
    - Inner parallelization: across compression strains
    """
    n_edges = len(network.stiffnesses)

    results = Parallel(n_jobs=n_jobs_outer)(
        delayed(compute_gradient_entry_batch)(
            i, network, target_poisson_list,
            top_nodes, bottom_nodes, left_nodes, right_nodes,
            compression_strain_list, epsilon, n_strain_steps,
            n_jobs_inner, force_type=force_type
        )
        for i in range(n_edges)
    )

    grad = np.zeros(n_edges)
    for i, value in results:
        grad[i] = value

    return grad


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def finish_training_GD_auxetic_batch(
    network, history, learning_rate, n_steps,
    top_nodes, bottom_nodes, left_nodes, right_nodes,
    force_type='quartic', n_strain_steps=100,
    source_compression_strain_list=[0.2], desired_target_extension_list=[0.2],
    verbose=False, stiffnesses_filename=None, force_tol=1e-6,
    vmin=1e-3, vmax=1e3,
    task_seed=None, realization_seed=None, save_interval=500, task_config=None, TARGETED_RESULTS_DIR=None, loss_tol = 1e-5
):
    """
    Train the network for auxetic response using gradient descent.

    Trains the network to achieve specific Poisson ratios at given compression strains
    by adjusting spring stiffnesses.

    Args:
        network: ElasticNetwork object to train
        history: Dictionary to store training history
        learning_rate: Gradient descent learning rate
        n_steps: Number of training steps
        top_nodes, bottom_nodes, left_nodes, right_nodes: Boundary node indices
        force_type: 'quartic' or 'quadratic'
        n_strain_steps: Steps per quasistatic trajectory
        source_compression_strain_list: List of compression strains (e.g., [0.04, 0.08])
        desired_target_extension_list: List of target lateral extensions (e.g., [-0.02, -0.02])
        verbose: Print detailed progress
        stiffnesses_filename: Optional file to save stiffnesses
        force_tol: Convergence tolerance for FIRE minimization
        vmin, vmax: Bounds for stiffness values
        task_seed: Task index (for saving intermediate results)
        realization_seed: Realization index (for saving intermediate results)
        save_interval: Save intermediate trajectories every N steps (default: 500)

    Returns:
        history: Updated history dictionary with training results
    """
    import copy

    network = copy.copy(network)
    last_relaxed_positions = np.copy(network.positions)
    loss = np.inf
    min_loss = np.inf

    # Initialize history if needed
    if 'stiffnesses' not in history:
        history['stiffnesses'] = []
    if 'loss' not in history:
        history['loss'] = []
    if 'positions' not in history:
        history['positions'] = []
    if 'freetraj' not in history:
        history['freetraj'] = []

    # Import checkpoint saving functions if task/realization provided
    save_intermediate = (task_seed is not None and realization_seed is not None)
    if save_intermediate:
        from pathlib import Path
        # Import save functions (avoid circular import by importing here)
        import sys
        from pathlib import Path
        ensemble_dir = Path(__file__).parent
        if str(ensemble_dir) not in sys.path:
            sys.path.insert(0, str(ensemble_dir))
        from checkpoint_manager import get_training_result_path

    # Convert extensions to target Poisson ratios
    desired_poisson_list = [
        -(desired_target_extension / source_compression_strain)
        for source_compression_strain, desired_target_extension
        in zip(source_compression_strain_list, desired_target_extension_list)
    ]

    pbar = tqdm(range(n_steps), desc=f'(loss = {loss:.4e}, min loss={min_loss:.4e})')

    for step in pbar:
        # --- Minimize positions ---
        network.update_positions(last_relaxed_positions)
        min_pos, force_norm = fire_minimize_network(
            network,
            constrained_dof_idx=None,  # Free minimization
            force_type=force_type,
            tol=force_tol
        )

        # Check convergence (Cython only)
        if force_norm is not None:
            assert force_norm < force_tol, f"FIRE did not converge: {force_norm:.3e} > {force_tol:.3e}"

        last_relaxed_positions = min_pos
        network.update_positions(min_pos)

        # --- Gradient update ---
        update = -finite_difference_gradient_parallel_batch(
            copy.deepcopy(network),
            target_poisson_list=desired_poisson_list,
            top_nodes=top_nodes,
            bottom_nodes=bottom_nodes,
            left_nodes=left_nodes,
            right_nodes=right_nodes,
            compression_strain_list=source_compression_strain_list,
            epsilon=1e-8,
            n_strain_steps=n_strain_steps,
            n_jobs_outer=4,
            n_jobs_inner=2,
            force_type=force_type
        )

        # --- Update stiffnesses ---
        network.stiffnesses = np.array(network.stiffnesses) + learning_rate * np.array(update)
        network.stiffnesses = np.clip(network.stiffnesses, vmin, vmax)

        # Check for NaN in stiffnesses
        if np.isnan(network.stiffnesses).any():
            print(f"\n{'='*60}")
            print(f"WARNING: Stiffnesses contain NaN at step {step}")
            print("Stopping training and saving current results.")
            print(f"{'='*60}")
            # Save intermediate results before breaking
            if save_intermediate:
                save_training_results(
                    task_seed=task_seed,
                    realization_seed=realization_seed,
                    history=history,
                    network=network,
                    task_config=task_config,
                    results_dir=TARGETED_RESULTS_DIR,
                )
            break

        # --- Loss computation ---
        network.update_positions(min_pos)
        loss, computed_poisson_ratios = poisson_loss_batch_parallel(
            network,
            target_poisson_list=desired_poisson_list,
            top_nodes=top_nodes,
            bottom_nodes=bottom_nodes,
            left_nodes=left_nodes,
            right_nodes=right_nodes,
            compression_strain_list=source_compression_strain_list,
            force_type=force_type,
        )

        # --- Update history ---
        history['stiffnesses'].append(np.copy(network.stiffnesses))
        history['loss'].append(loss)
        history['positions'].append(np.copy(min_pos))

        # Check for NaN in loss
        if np.isnan(loss):
            print(f"\n{'='*60}")
            print(f"WARNING: Loss is NaN at step {step}")
            print("Stopping training and saving current results.")
            print(f"{'='*60}")
            # Save intermediate results before breaking
            if save_intermediate:
                save_training_results(
                    task_seed=task_seed,
                    realization_seed=realization_seed,
                    history=history,
                    network=network,
                    task_config=task_config,
                    results_dir=TARGETED_RESULTS_DIR,
                )
            break

        # Track minimum loss
        if loss < min_loss:
            min_loss = loss

        # Update progress bar
        pbar.set_description(f'(loss = {loss:.4e}, min loss={min_loss:.4e}), grad_norm = {np.linalg.norm(update):.4e}')

        # Verbose output
        if verbose and step % 100 == 0:
            print(f"\nStep {step}:")
            print(f"  Loss: {loss:.6e}")
            print(f"  Computed Poisson ratios: {computed_poisson_ratios}")
            print(f"  Target Poisson ratios: {desired_poisson_list}")

        # Periodically save intermediate trajectories
        if save_intermediate and (step) % save_interval == 0:
            save_training_results(
                task_seed=task_seed,
                realization_seed=realization_seed,
                history=history,
                network=network,
                task_config=task_config,
                results_dir=TARGETED_RESULTS_DIR,
            )
        if loss < loss_tol:
            break

    # Final summary
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Final loss: {loss:.6e}")
    print(f"  Minimum loss: {min_loss:.6e}")
    print(f"{'='*60}")

    trained_network = network
    trained_network.update_positions(last_relaxed_positions)
    trained_network.stiffnesses = np.array(network.stiffnesses)

    return history, trained_network 
