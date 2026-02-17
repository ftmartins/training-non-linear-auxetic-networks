"""
Moduli-based training for elastic networks.

Provides:
  - make_moduli_loss_fn: Factory for JAX-differentiable loss functions targeting
    elastic moduli (B, G, nu) or specific Voigt tensor elements (C_ij).
  - finish_training_GD_general_jax: General-purpose gradient descent training loop
    that accepts any loss_fn(stiffnesses, positions_flat) -> scalar.
  - run_moduli_training: High-level entry point combining the two.
"""

import numpy as np
import copy
from tqdm import tqdm

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from training_functions_with_toggle import (
    make_compute_response_fire,
    compute_quasistatic_trajectory_auxetic_jax,
    fire_minimize_network,
    elastic_energy,
)
from elasticity_tensor import (
    compute_elasticity_tensor_2d,
    extract_moduli_2d,
    precompute_dof_indices,
)
from checkpoint_manager import save_training_results


def _parse_voigt_key(key):
    """
    Parse a Voigt tensor element key like 'C_12' -> (0, 1).

    Accepts 1-indexed Voigt indices: i, j in {1, 2, 3}.
    Returns 0-indexed tuple for array access.
    """
    if not key.startswith('C_') or len(key) != 4:
        raise ValueError(
            f"Invalid tensor element key '{key}'. "
            "Expected format 'C_ij' with i,j in {{1,2,3}}, e.g. 'C_12'."
        )
    i = int(key[2]) - 1
    j = int(key[3]) - 1
    if not (0 <= i <= 2 and 0 <= j <= 2):
        raise ValueError(
            f"Voigt indices in '{key}' out of range. "
            "Expected i,j in {{1,2,3}}."
        )
    return (i, j)


def make_moduli_loss_fn(crf, edges, rest_lengths,
                        top_nodes, bottom_nodes, left_nodes, right_nodes,
                        training_goals, n_strain_steps,
                        force_type='quadratic', d=2):
    """
    Build a JAX-differentiable loss function for elastic moduli training.

    Parameters
    ----------
    crf : callable
        Differentiable FIRE solver from make_compute_response_fire.
    edges : array (E, 2)
        Edge connectivity.
    rest_lengths : array (E,)
        Rest lengths.
    top_nodes, bottom_nodes, left_nodes, right_nodes : arrays
        Boundary node indices.
    training_goals : dict
        Maps compression_strain (float) -> target dict, e.g.:
            {
                -0.2: {'B': 5.0, 'G': 2.0},
                -0.4: {'nu': -0.8, 'C_12': 1.5, 'C_33': 0.3},
            }
        Supported keys: 'B', 'G', 'nu', 'C_ij' (1-indexed Voigt: i,j in {1,2,3}).
        All targets are weighted equally in the MSE loss.
    n_strain_steps : int
        Number of quasistatic compression steps.
    force_type : str
        'quadratic' or 'quartic'.
    d : int
        Spatial dimension (2).

    Returns
    -------
    loss_fn : callable
        loss_fn(stiffnesses_jax, positions_flat_jax) -> scalar loss.
        Differentiable w.r.t. stiffnesses_jax (first argument).
    """
    # Convert to JAX arrays (captured by closure)
    edges_jax = jnp.asarray(np.array(edges, dtype=np.int32))
    rest_lengths_jax = jnp.asarray(np.array(rest_lengths, dtype=np.float64))
    top_jax = jnp.asarray(top_nodes)
    bottom_jax = jnp.asarray(bottom_nodes)
    left_jax = jnp.asarray(left_nodes)
    right_jax = jnp.asarray(right_nodes)

    # Precompute DOF indices (concrete numpy, outside JAX tracing)
    boundary_nodes_all = np.unique(np.concatenate([
        top_nodes, bottom_nodes, left_nodes, right_nodes
    ]))
    # n_nodes must be inferred from edges
    n_nodes = int(np.max(edges)) + 1
    boundary_dofs_np, interior_dofs_np = precompute_dof_indices(
        boundary_nodes_all, n_nodes, d=d
    )

    # Pre-parse training goals: list of (strain, [(key, target_val, voigt_idx_or_None), ...])
    parsed_goals = []
    total_n_targets = 0
    for strain, targets in training_goals.items():
        entries = []
        for key, target_val in targets.items():
            if key in ('B', 'G', 'nu'):
                entries.append((key, float(target_val), None))
            else:
                voigt_idx = _parse_voigt_key(key)
                entries.append(('C', float(target_val), voigt_idx))
            total_n_targets += 1
        parsed_goals.append((float(strain), entries))

    def loss_fn(stiffnesses_jax, positions_flat_jax):
        total_loss = 0.0

        for compression_strain, entries in parsed_goals:
            # 1. Compress to target strain
            strained_pos_flat = compute_quasistatic_trajectory_auxetic_jax(
                crf, stiffnesses_jax, edges_jax, rest_lengths_jax,
                positions_flat_jax,
                top_jax, bottom_jax,
                compression_strain, n_strain_steps, d=d
            )

            # 2. Compute elasticity tensor at strained configuration
            strained_pos_2d = jnp.reshape(strained_pos_flat, (-1, d))
            C_voigt = compute_elasticity_tensor_2d(
                strained_pos_2d, edges_jax, stiffnesses_jax,
                rest_lengths_jax, boundary_dofs_np, interior_dofs_np,
                force_type=force_type
            )

            # 3. Extract moduli
            C11 = C_voigt[0, 0]
            C22 = C_voigt[1, 1]
            C12 = C_voigt[0, 1]
            B = (C11 + C22 + 2.0 * C12) / 4.0
            G = (C11 + C22 - 2.0 * C12) / 4.0
            nu = (B - G) / (B + G)

            # 4. Accumulate MSE
            for key, target_val, voigt_idx in entries:
                if key == 'B':
                    total_loss = total_loss + (B - target_val) ** 2
                elif key == 'G':
                    total_loss = total_loss + (G - target_val) ** 2
                elif key == 'nu':
                    total_loss = total_loss + (nu - target_val) ** 2
                elif key == 'C':
                    i, j = voigt_idx
                    total_loss = total_loss + (C_voigt[i, j] - target_val) ** 2

        return total_loss / total_n_targets

    return loss_fn


def finish_training_GD_general_jax(
    network, loss_fn, history, learning_rate, n_steps,
    force_type='quadratic', force_tol=1e-6,
    vmin=1e-6, vmax=1e3,
    task_seed=None, realization_seed=None, save_interval=10,
    task_config=None, TARGETED_RESULTS_DIR=None,
):
    """
    General-purpose JAX gradient descent training loop.

    Refactored from finish_training_GD_auxetic_batch_jax to accept any
    loss_fn(stiffnesses_jax, positions_flat_jax) -> scalar.

    The loop:
      1. Free-relax positions (Cython FIRE, non-differentiable)
      2. Compute loss + grad via jax.value_and_grad (JIT-compiled)
      3. Normalized gradient descent: k -= lr * grad / ||grad||
      4. Clip stiffnesses to [vmin, vmax]
      5. Checkpoint at save_interval

    Parameters
    ----------
    network : ElasticNetwork
        Network to train (stiffnesses are modified in-place).
    loss_fn : callable
        loss_fn(stiffnesses_jax, positions_flat_jax) -> scalar.
        Must be differentiable w.r.t. first argument.
    history : dict
        Storage for 'stiffnesses', 'loss', 'positions' lists.
    learning_rate : float
        Gradient descent step size.
    n_steps : int
        Number of training iterations.
    force_type : str
        'quadratic' or 'quartic'.
    force_tol : float
        FIRE convergence tolerance for free relaxation.
    vmin, vmax : float
        Stiffness bounds.
    task_seed, realization_seed : int or None
        For checkpoint saving.
    save_interval : int
        Save every N steps.
    task_config, TARGETED_RESULTS_DIR : optional
        Passed to save_training_results.

    Returns
    -------
    history : dict
    trained_network : ElasticNetwork
    """
    network = copy.copy(network)
    last_relaxed_positions = np.copy(network.positions)
    loss = np.inf
    min_loss = np.inf

    for key in ('stiffnesses', 'loss', 'positions'):
        if key not in history:
            history[key] = []

    save_intermediate = (task_seed is not None and realization_seed is not None)

    # JIT-compile loss + grad
    loss_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=0))

    pbar = tqdm(range(n_steps), desc=f'(loss = {loss:.4e}, min loss={min_loss:.4e})')

    for step in pbar:
        # --- Free-minimize positions (Cython FIRE, non-differentiable) ---
        network.update_positions(last_relaxed_positions)
        min_pos, force_norm = fire_minimize_network(
            network,
            constrained_dof_idx=None,
            force_type=force_type,
            tol=force_tol
        )

        if force_norm is not None:
            assert force_norm < force_tol, (
                f"FIRE did not converge: {force_norm:.3e} > {force_tol:.3e}"
            )

        last_relaxed_positions = min_pos
        network.update_positions(min_pos)

        # --- JAX autodiff gradient ---
        stiffnesses_jax = jnp.asarray(
            np.array(network.stiffnesses, dtype=np.float64)
        )
        positions_flat_jax = jnp.asarray(min_pos.flatten())

        loss_val, grad = loss_and_grad_fn(stiffnesses_jax, positions_flat_jax)
        loss = float(loss_val)
        grad_np = np.array(grad)

        # --- Update history ---
        history['stiffnesses'].append(np.copy(network.stiffnesses))
        history['loss'].append(loss)
        history['positions'].append(np.copy(min_pos))

        # --- Update stiffnesses ---
        grad_norm = np.linalg.norm(grad_np)
        if grad_norm > 0:
            network.stiffnesses = (
                np.array(network.stiffnesses)
                - learning_rate * grad_np / grad_norm
            )
        network.stiffnesses = np.clip(network.stiffnesses, vmin, vmax)

        # NaN check
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

        pbar.set_description(
            f'(loss = {loss:.4e}, min loss={min_loss:.4e}, '
            f'log mean update = {np.mean(np.log10(np.abs(grad_np / grad_norm + 1e-12))):.2f})'
        )

        if save_intermediate and step % save_interval == 0:
            save_training_results(
                task_seed=task_seed, realization_seed=realization_seed,
                history=history, network=network,
                task_config=task_config, results_dir=TARGETED_RESULTS_DIR,
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


def run_moduli_training(network, training_goals, learning_rate, n_steps,
                        top_nodes, bottom_nodes, left_nodes, right_nodes,
                        force_type='quadratic', n_strain_steps=100,
                        fire_max_steps=100_000, fire_tol=1e-6,
                        history=None, **kwargs):
    """
    High-level entry point for moduli training.

    Builds the differentiable FIRE solver, constructs the moduli loss function,
    and runs the generalized training loop.

    Parameters
    ----------
    network : ElasticNetwork
        Network to train.
    training_goals : dict
        Maps compression_strain -> target dict, e.g.:
            {-0.2: {'B': 5.0, 'G': 2.0}}
    learning_rate : float
        GD step size.
    n_steps : int
        Number of training steps.
    top_nodes, bottom_nodes, left_nodes, right_nodes : arrays
        Boundary node indices.
    force_type : str
        'quadratic' or 'quartic'.
    n_strain_steps : int
        Steps per quasistatic trajectory.
    fire_max_steps : int
        Max steps for JAX FIRE solver.
    fire_tol : float
        Convergence tolerance for JAX FIRE solver.
    history : dict or None
        Training history. Created if None.
    **kwargs
        Passed to finish_training_GD_general_jax (e.g. vmin, vmax, save_interval).

    Returns
    -------
    history : dict
    trained_network : ElasticNetwork
    """
    if history is None:
        history = {}

    # Build differentiable FIRE solver
    crf = make_compute_response_fire(
        d=2, force_type=force_type,
        max_steps=fire_max_steps, tol=fire_tol
    )

    # Build loss function
    loss_fn = make_moduli_loss_fn(
        crf=crf,
        edges=network.edges,
        rest_lengths=network.rest_lengths,
        top_nodes=top_nodes,
        bottom_nodes=bottom_nodes,
        left_nodes=left_nodes,
        right_nodes=right_nodes,
        training_goals=training_goals,
        n_strain_steps=n_strain_steps,
        force_type=force_type,
    )

    return finish_training_GD_general_jax(
        network=network,
        loss_fn=loss_fn,
        history=history,
        learning_rate=learning_rate,
        n_steps=n_steps,
        force_type=force_type,
        **kwargs,
    )
