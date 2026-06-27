"""
Training functions for spring networks.

Physics/simulation primitives (elastic_energy, FIRE minimizer, quasistatic
trajectories, JAX differentiable solver) live in base.simulate and are
re-exported here for convenience. Only training-specific logic (loss, gradient,
GD loops, checkpointing) is defined in this module.
"""

import sys
import copy
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from base.config import FORCE_TOL
from base.simulate import (
    elastic_energy,
    fire_minimize_network,
    compute_quasistatic_trajectory_auxetic,
    compute_quasistatic_trajectory_full_cycle,
    compute_ift_gradient,
    make_compute_response_fire,
    crf,
    compute_quasistatic_trajectory_auxetic_jax,
    compute_poisson_ratio_single_jax,
)
from .checkpoint_manager import (
    save_training_results,
    save_checkpoint,
)



# ============================================================================
def poisson_loss_batch_jax(crf, stiffnesses, edges, rest_lengths, positions_flat,
                           top_nodes, bottom_nodes, left_nodes, right_nodes,
                           compression_strain_list, target_poisson_list, n_steps, d=2):
    """
    JAX-differentiable MSE loss across multiple compression-Poisson pairs.

    Returns:
        mse_loss: scalar = mean((computed_nu - target_nu)^2)
    """
    total_loss = 0.0
    n_pairs = len(compression_strain_list)
    for cs, target_nu in zip(compression_strain_list, target_poisson_list):
        computed_nu = compute_poisson_ratio_single_jax(
            crf, stiffnesses, edges, rest_lengths, positions_flat,
            top_nodes, bottom_nodes, left_nodes, right_nodes,
            cs, n_steps, d=d
        )
        total_loss = total_loss + (computed_nu - target_nu) ** 2
    return total_loss / n_pairs


# ============================================================================
# TRAINING HELPER FUNCTIONS (Cython-based, original)
# ============================================================================

def compute_poisson_ratio_single(network, top_nodes, bottom_nodes, left_nodes, right_nodes,
                                 compression_strain, n_strain_steps=100, tol=FORCE_TOL, force_type='quadratic'):
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
        tol=tol,
        method='fire',
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
                                n_strain_steps=100, n_jobs_inner=4, force_type='quadratic', tol=FORCE_TOL):
    """
    Compute MSE loss across multiple compression-Poisson pairs in parallel.
    """
    computed_poisson_ratios = Parallel(n_jobs=n_jobs_inner)(
        delayed(compute_poisson_ratio_single)(
            network, top_nodes, bottom_nodes, left_nodes, right_nodes,
            cs, n_strain_steps, force_type=force_type, tol=tol
        )
        for cs in compression_strain_list
    )

    computed_poisson_ratios = np.array(computed_poisson_ratios)
    mse_loss = np.mean((computed_poisson_ratios - np.array(target_poisson_list))**2)

    return mse_loss, computed_poisson_ratios


def compute_gradient_entry_batch(i, network, target_poisson_list, top_nodes, bottom_nodes,
                                 left_nodes, right_nodes, compression_strain_list,
                                 epsilon, n_strain_steps, n_jobs_inner=4, force_type='quadratic', tol=FORCE_TOL):
    """
    Compute gradient for a single stiffness using finite differences.
    """
    orig = network.stiffnesses[i]

    # Perturb up
    network.stiffnesses[i] = orig + epsilon
    loss_plus, _ = poisson_loss_batch_parallel(
        network, target_poisson_list, top_nodes, bottom_nodes,
        left_nodes, right_nodes, compression_strain_list,
        n_strain_steps, n_jobs_inner, force_type=force_type, tol=tol
    )

    # Perturb down
    network.stiffnesses[i] = orig - epsilon
    loss_minus, _ = poisson_loss_batch_parallel(
        network, target_poisson_list, top_nodes, bottom_nodes,
        left_nodes, right_nodes, compression_strain_list,
        n_strain_steps, n_jobs_inner, force_type=force_type, tol=tol
    )

    # Restore
    network.stiffnesses[i] = orig

    # Gradient
    value = (loss_plus - loss_minus) / (2 * epsilon)

    return (i, value)


def finite_difference_gradient_parallel_batch(network, target_poisson_list, top_nodes, bottom_nodes,
                                             left_nodes, right_nodes, compression_strain_list,
                                             epsilon=1e-8, n_jobs_outer=4, n_jobs_inner=2,
                                             n_strain_steps=100, force_type='quadratic', tol=FORCE_TOL):
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
            n_jobs_inner, force_type=force_type, tol=tol
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
    force_type='quadratic', n_strain_steps=100,
    source_compression_strain_list=[0.2], desired_target_extension_list=[0.2],
    verbose=False, stiffnesses_filename=None, force_tol=1e-6,
    vmin=1e-3, vmax=1e3,
    task_seed=None, realization_seed=None, save_interval=500, task_config=None, TARGETED_RESULTS_DIR=None, loss_tol=1e-5,
    method='newton',
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
        force_tol: Convergence tolerance for minimization
        vmin, vmax: Bounds for stiffness values
        task_seed: Task index (for saving intermediate results)
        realization_seed: Realization index (for saving intermediate results)
        save_interval: Save intermediate trajectories every N steps (default: 500)
        method: 'newton' (default) — Newton quasistatic + IFT gradient (fast);
                'fire'   — Cython FIRE quasistatic + finite-difference gradient (original).

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
        if method == 'newton':
            _, grad = compute_ift_gradient(
                network,
                compression_strains=source_compression_strain_list,
                target_poissons=desired_poisson_list,
                top_nodes=top_nodes,
                bottom_nodes=bottom_nodes,
                left_nodes=left_nodes,
                right_nodes=right_nodes,
                tol=force_tol,
                n_strain_steps=n_strain_steps,
            )
            update = -grad
        else:
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
                force_type=force_type,
                tol=force_tol,
            )

        # --- Update stiffnesses ---
        update_norm = np.linalg.norm(update)
        if update_norm > 0:
            network.stiffnesses = np.array(network.stiffnesses) + learning_rate * np.array(update) / update_norm
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

        # --- Loss computation (post-update, both methods) ---
        # history['loss'][i] and history['stiffnesses'][i] are now aligned:
        # both reflect the network state AFTER the stiffness update at step i.
        network.update_positions(min_pos)
        if method == 'newton':
            loss, _ = compute_ift_gradient(
                network,
                compression_strains=source_compression_strain_list,
                target_poissons=desired_poisson_list,
                top_nodes=top_nodes,
                bottom_nodes=bottom_nodes,
                left_nodes=left_nodes,
                right_nodes=right_nodes,
                tol=force_tol,
                n_strain_steps=n_strain_steps,
            )
        else:
            loss, computed_poisson_ratios = poisson_loss_batch_parallel(
                network,
                target_poisson_list=desired_poisson_list,
                top_nodes=top_nodes,
                bottom_nodes=bottom_nodes,
                left_nodes=left_nodes,
                right_nodes=right_nodes,
                compression_strain_list=source_compression_strain_list,
                n_strain_steps=n_strain_steps,
                force_type=force_type,
                tol=force_tol,
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
            print(f"  Target Poisson ratios: {desired_poisson_list}")

        # Periodically save intermediate trajectories and checkpoint
        if save_intermediate and step % save_interval == 0:
            save_training_results(
                task_seed=task_seed,
                realization_seed=realization_seed,
                history=history,
                network=network,
                task_config=task_config,
                results_dir=TARGETED_RESULTS_DIR,
            )

            save_checkpoint(
                task_seed=task_seed,
                realization_seed=realization_seed,
                history=history,
                network=network,
                task_config=task_config,
                current_step=step,
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


# ============================================================================
# JAX-BASED TRAINING FUNCTION (autodiff gradients)
# ============================================================================

def finish_training_GD_auxetic_batch_jax(
    network, history, learning_rate, n_steps,
    top_nodes, bottom_nodes, left_nodes, right_nodes,
    force_type='quadratic', n_strain_steps=100,
    source_compression_strain_list=[0.2], desired_target_extension_list=[0.2],
    verbose=False, force_tol=1e-6,
    vmin=1e-6, vmax=1e3,
    task_seed=None, realization_seed=None, save_interval=10,
    task_config=None, TARGETED_RESULTS_DIR=None,
    fire_max_steps=100_000, fire_tol=FORCE_TOL
):
    """
    Train the network for auxetic response using JAX autodiff gradients.

    Same goal as finish_training_GD_auxetic_batch but replaces finite-difference
    gradients with jax.grad through a differentiable FIRE solver (custom VJP).
    The gradient function is JIT-compiled once at the start.

    Args:
        network: ElasticNetwork object to train
        history: Dictionary to store training history
        learning_rate: Gradient descent learning rate
        n_steps: Number of training steps
        top_nodes, bottom_nodes, left_nodes, right_nodes: Boundary node indices
        force_type: 'quadratic' or 'quartic'
        n_strain_steps: Steps per quasistatic trajectory
        source_compression_strain_list: List of compression strains
        desired_target_extension_list: List of target lateral extensions
        verbose: Print detailed progress
        force_tol: Convergence tolerance for free FIRE minimization
        vmin, vmax: Bounds for stiffness values
        task_seed, realization_seed: For saving intermediate results
        save_interval: Save every N steps
        task_config, TARGETED_RESULTS_DIR: For checkpoint saving
        fire_max_steps: Max steps for JAX FIRE solver
        fire_tol: Convergence tolerance for JAX FIRE solver

    Returns:
        (history, trained_network)
    """
    network = copy.copy(network)
    last_relaxed_positions = np.copy(network.positions)
    loss = np.inf
    min_loss = np.inf

    # Initialize history
    for key in ('stiffnesses', 'loss', 'positions'):
        if key not in history:
            history[key] = []

    save_intermediate = (task_seed is not None and realization_seed is not None)

    # Convert extensions to target Poisson ratios
    desired_poisson_list = [
        -(ext / cs)
        for cs, ext in zip(source_compression_strain_list, desired_target_extension_list)
    ]

    # Build differentiable FIRE solver
    crf_local = make_compute_response_fire(
        d=2, force_type=force_type,
        max_steps=fire_max_steps, tol=fire_tol
    )

    # Pre-convert static arrays to JAX
    edges_jax = jnp.asarray(np.array(network.edges, dtype=np.int32), dtype=jnp.int32)
    rest_lengths_jax = jnp.asarray(np.array(network.rest_lengths, dtype=np.float64), dtype=jnp.float64)
    top_nodes_jax = jnp.asarray(top_nodes, dtype=jnp.int32)
    bottom_nodes_jax = jnp.asarray(bottom_nodes, dtype=jnp.int32)
    left_nodes_jax = jnp.asarray(left_nodes, dtype=jnp.int32)
    right_nodes_jax = jnp.asarray(right_nodes, dtype=jnp.int32)

    # Build and JIT the loss+grad function
    # positions_flat is passed as argument (changes each step after free relaxation)
    def loss_fn(stiffnesses_jax, positions_flat_jax):
        return poisson_loss_batch_jax(
            crf_local, stiffnesses_jax, edges_jax, rest_lengths_jax, positions_flat_jax,
            top_nodes_jax, bottom_nodes_jax, left_nodes_jax, right_nodes_jax,
            source_compression_strain_list, desired_poisson_list, n_strain_steps, d=2
        )

    # jax.value_and_grad: returns (loss, grad) in one forward+backward pass
    loss_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=0))

    pbar = tqdm(range(n_steps), desc=f'(loss = {loss:.4e}, min loss={min_loss:.4e})')

    for step in pbar:
        # --- Free-minimize positions at current stiffnesses (Cython FIRE) ---
        network.update_positions(last_relaxed_positions)
        min_pos, force_norm = fire_minimize_network(
            network,
            constrained_dof_idx=None,
            force_type=force_type,
            tol=force_tol
        )

        if force_norm is not None:
            assert force_norm < force_tol, f"FIRE did not converge: {force_norm:.3e} > {force_tol:.3e}"

        last_relaxed_positions = min_pos
        network.update_positions(min_pos)

        # --- JAX autodiff gradient ---
        stiffnesses_jax = jnp.asarray(np.array(network.stiffnesses, dtype=np.float64), dtype=jnp.float64)
        positions_flat_jax = jnp.asarray(min_pos.flatten(), dtype=jnp.float64)

        loss_val, grad = loss_and_grad_fn(stiffnesses_jax, positions_flat_jax)
        loss = float(loss_val)
        grad_np = np.array(grad)

        # --- Update history (before stiffness update, so loss and stiffnesses match) ---
        history['stiffnesses'].append(np.copy(network.stiffnesses))
        history['loss'].append(loss)
        history['positions'].append(np.copy(min_pos))

        # --- Update stiffnesses ---
        grad_norm = np.linalg.norm(grad_np)
        if grad_norm > 0:
            network.stiffnesses = np.array(network.stiffnesses) - learning_rate * grad_np / grad_norm
        network.stiffnesses = np.clip(network.stiffnesses, vmin, vmax)

        # Check for NaN
        if np.isnan(network.stiffnesses).any() or np.isnan(loss):
            label = "stiffnesses" if np.isnan(network.stiffnesses).any() else "loss"
            print(f"\n{'='*60}")
            print(f"WARNING: {label} contain NaN at step {step}")
            print("Stopping training and saving current results.")
            print(f"{'='*60}")
            if save_intermediate:
                save_training_results(
                    task_seed=task_seed, realization_seed=realization_seed,
                    history=history, network=network,
                    task_config=task_config, results_dir=TARGETED_RESULTS_DIR,
                )
            break

        if loss < min_loss:
            min_loss = loss

        pbar.set_description(f'(loss = {loss:.4e}, min loss={min_loss:.4e}, init loss = {history["loss"][0]:.4e}, log mean update ={np.mean(np.log10(np.abs(grad_np / grad_norm + 1e-12))):.2f})')

        if verbose and step % save_interval == 0:
            print(f"\nStep {step}:")
            print(f"  Loss: {loss:.6e}")
            print(f"  Target Poisson ratios: {desired_poisson_list}")
            print(f"  Grad norm: {grad_norm:.6e}")

        if save_intermediate and step % save_interval == 0:
            save_training_results(
                task_seed=task_seed, realization_seed=realization_seed,
                history=history, network=network,
                task_config=task_config, results_dir=TARGETED_RESULTS_DIR,
            )
            save_checkpoint(
                task_seed=task_seed, realization_seed=realization_seed,
                history=history, network=network,
                task_config=task_config, current_step=step,
                results_dir=TARGETED_RESULTS_DIR,
            )

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

