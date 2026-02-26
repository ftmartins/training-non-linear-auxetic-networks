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
    save_checkpoint,
)
from config import FORCE_TOL

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg


# ============================================================================
# JAX ELASTIC ENERGY
# ============================================================================

def elastic_energy(flat_positions, edges, rest_lengths, stiffnesses, *, d=2, force_type="quadratic"):
    """
    JAX-differentiable elastic energy.

    flat_positions: (N*d,)
    edges: (E, 2)
    rest_lengths: (E,)
    stiffnesses: (E,)
    """
    pos = jnp.reshape(flat_positions, (-1, d))
    edges = jnp.asarray(edges)
    k = jnp.asarray(stiffnesses)
    L0 = jnp.asarray(rest_lengths)

    i = edges[:, 0]
    j = edges[:, 1]
    dists = jnp.linalg.norm(pos[j] - pos[i], axis=1)
    delta = dists - L0

    if force_type == "quadratic":
        return 0.5 * jnp.sum(k * delta**2)
    elif force_type == "quartic":
        avg_rest = jnp.mean(L0)
        scale = 36.0 / (avg_rest**2 + 1e-12)
        return 0.25 * jnp.sum(k * (scale * delta**4 - delta**2))
    else:
        raise ValueError(f"Unknown force_type: {force_type}")


# ============================================================================
# FIRE MINIMIZATION
# ============================================================================

def fire_minimize_network(network, constrained_dof_idx=None, force_type='quadratic',
                         tol=1e-6, max_steps=500_000, deltaT=1e-2,
                         retry_steps_1=500_000, retry_steps_2=500_000):
    """
    Minimize network using Cython FIRE, with up to two retry passes.

    If not converged after ``max_steps``, retries from the last positions with
    ``retry_steps_1`` additional steps, then ``retry_steps_2`` more.  Mirrors
    the retry pattern of the JAX ``make_compute_response_fire`` solver.
    The caller is still responsible for checking convergence (force_norm vs tol).

    Args:
        network: ElasticNetwork object
        constrained_dof_idx: List of DOF indices to constrain or None
        force_type: 'quartic' or 'quadratic'
        tol: Convergence tolerance
        max_steps: Maximum number of minimization steps (first attempt)
        deltaT: FIRE timestep (default 1e-2 for fast convergence)
        retry_steps_1: Extra steps for the first retry (default 500_000)
        retry_steps_2: Extra steps for the second retry (default 500_000)

    Returns:
        min_positions: Minimized positions (N, d) array
        force_norm: Final force norm (after all retry attempts)
    """
    if constrained_dof_idx is None:
        constrained_dof_idx = []

    edges_i32 = np.array(network.edges, dtype=np.int32)
    rest_lengths_f64 = np.array(network.rest_lengths, dtype=np.float64)
    stiffnesses_f64 = np.array(network.stiffnesses, dtype=np.float64)
    force_type_int = 1 if force_type == 'quartic' else 0

    min_pos, force_norm, _ = fire_minimize_dof(
        network.positions,
        edges_i32, rest_lengths_f64, stiffnesses_f64,
        deltaT, max_steps, tol, constrained_dof_idx, force_type_int,
    )

    # Retry 1: continue from last positions if not yet converged
    if force_norm is not None and force_norm >= tol:
        print(f"  FIRE retry 1 ({retry_steps_1:,} steps, force_norm={force_norm:.3e})...")
        min_pos, force_norm, _ = fire_minimize_dof(
            min_pos,
            edges_i32, rest_lengths_f64, stiffnesses_f64,
            deltaT, retry_steps_1, tol, constrained_dof_idx, force_type_int,
        )

    # Retry 2: continue from last positions if still not converged
    if force_norm is not None and force_norm >= tol:
        print(f"  FIRE retry 2 ({retry_steps_2:,} steps, force_norm={force_norm:.3e})...")
        min_pos, force_norm, _ = fire_minimize_dof(
            min_pos,
            edges_i32, rest_lengths_f64, stiffnesses_f64,
            deltaT, retry_steps_2, tol, constrained_dof_idx, force_type_int,
        )

    return min_pos, force_norm


# ============================================================================
# TRAJECTORY COMPUTATION
# ============================================================================

def compute_quasistatic_trajectory_auxetic(network, compression_strain, top_nodes, bottom_nodes,
                                          n_steps=100, verbose=False, force_type='quadratic',
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

    x_top_init = positions[top_nodes, 0]
    x_bottom_init = positions[bottom_nodes, 0]

    # Precompute constrained DOF indices (y-component of top + bottom nodes)
    all_boundary = jnp.concatenate([jnp.asarray(top_nodes), jnp.asarray(bottom_nodes)])
    constrained_idx_dof = jnp.concatenate([all_boundary * d, all_boundary * d + 1])  # x- and y-DOF

    traj = [np.copy(positions)]

    for step in range(n_steps):
        frac = step / (n_steps - 1)
        height_to_impose = initial_height - frac * (initial_height - target_height)
        y_top_new = y_bottom.mean() + height_to_impose

        # Set constrained positions
        positions_step = np.copy(positions)
        positions_step[top_nodes, 1] = y_top_new + (positions[top_nodes, 1] - positions[top_nodes, 1].mean())
        positions_step[bottom_nodes, 1] = y_bottom
        positions_step[top_nodes, 0] = x_top_init  # hold x fixed for top nodes
        positions_step[bottom_nodes, 0] = x_bottom_init  # hold x fixed for bottom nodes

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
                                              force_type='quadratic', tol=1e-6, d=2):
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

def make_compute_response_fire(*, d=2, dt_init=1e-2, dt_max=1e-1, dt_min=1e-4, alpha_start=0.1,
                               finc=1.1, fdec=0.5, falpha=0.99, max_steps=1_000_00, tol=1e-6,
                               force_type="quadratic"):
    """Returns a differentiable FIRE solver with ONLY 6 inputs:
       (stiffnesses, edges, rest_lengths, positions0, source_nodes_dof, imposed_positions).
       All hyperparameters above are captured by closure and NOT passed to JAX.
    """

    def _fire_forward(stiffnesses, edges, rest_lengths, positions0, source_nodes_dof, imposed_positions):
        positions0 = jnp.asarray(positions0).flatten()
        imposed_positions = jnp.asarray(imposed_positions).flatten()
        n_dof = positions0.shape[0]
        mask = jnp.ones(n_dof, dtype=bool)
        mask = mask.at[jnp.asarray(source_nodes_dof)].set(False)

        def energy_fn_positions(pos_flat, k):
            return elastic_energy(pos_flat, edges, rest_lengths, k, d=d, force_type=force_type)

        def forces(pos_flat, k):
            # ∂E/∂x (grad wrt positions)
            return jax.grad(energy_fn_positions, argnums=0)(pos_flat, k)

        vel0 = jnp.zeros_like(positions0)
        dt0, alpha0, npos0 = dt_init, alpha_start, 0

        pos0 = positions0.at[source_nodes_dof].set(imposed_positions)
        grads0 = forces(pos0, stiffnesses) * mask
        gnorm0 = jnp.linalg.norm(jnp.where(mask, grads0, 0.0))
        active0 = 1.0
        state0 = (pos0, vel0, dt0, alpha0, npos0, gnorm0, active0)

        def body_fn(i, state):
            pos, vel, dt, alpha, npos, gnorm, active = state

            def do_update(inputs):
                pos, vel, dt, alpha, npos, gnorm = inputs
                g = forces(pos, stiffnesses) * mask
                f = -g
                P = jnp.vdot(vel, f)

                # velocity-Verlet
                pos = pos + vel * dt + 0.5 * f * (dt * dt)
                vel = vel + 0.5 * dt * f

                g = forces(pos, stiffnesses) * mask
                f = -g
                vel = vel + 0.5 * dt * f

                def pos_power(args):
                    pos, vel, dt, alpha, npos = args
                    vnorm = jnp.linalg.norm(vel)
                    fnorm = jnp.linalg.norm(f)
                    vel = (1 - alpha) * vel + alpha * f * (vnorm / (fnorm + 1e-12))
                    pos = pos.at[source_nodes_dof].set(imposed_positions)
                    npos += 1
                    dt = jnp.minimum(dt * finc, dt_max)
                    alpha = alpha * falpha
                    return pos, vel, dt, alpha, npos

                def neg_power(args):
                    pos, vel, dt, alpha, npos = args
                    vel = jnp.zeros_like(vel)
                    dt = jnp.maximum(dt * fdec, dt_min)
                    alpha = alpha_start
                    npos = 0
                    return pos, vel, dt, alpha, npos

                pos, vel, dt, alpha, npos = jax.lax.cond(P >= 0, pos_power, neg_power, (pos, vel, dt, alpha, npos))
                pos = pos.at[source_nodes_dof].set(imposed_positions)
                vel = vel * mask

                gnorm = jnp.linalg.norm(jnp.where(mask, g, 0.0))
                active_new = jnp.where(gnorm / n_dof < tol, 0.0, 1.0)
                return (pos, vel, dt, alpha, npos, gnorm, active_new)

            def skip_update(inputs):
                pos, vel, dt, alpha, npos, gnorm = inputs
                return (pos, vel, dt, alpha, npos, gnorm, active)

            return jax.lax.cond(active > 0.5, do_update, skip_update, (pos, vel, dt, alpha, npos, gnorm))

        final_state = jax.lax.fori_loop(0, max_steps, body_fn, state0)
        final_pos_flat, *_ = final_state
        return final_pos_flat  # (N*d,)

    @jax.custom_vjp
    def compute_response_fire(stiffnesses, edges, rest_lengths, positions0, source_nodes_dof, imposed_positions):
        return _fire_forward(stiffnesses, edges, rest_lengths, positions0, source_nodes_dof, imposed_positions)

    def crf_fwd(stiffnesses, edges, rest_lengths, positions0, source_nodes_dof, imposed_positions):
        pos_final = _fire_forward(stiffnesses, edges, rest_lengths, positions0, source_nodes_dof, imposed_positions)
        saved = (pos_final, stiffnesses, edges, rest_lengths, positions0, source_nodes_dof, imposed_positions)
        return pos_final, saved

    def crf_bwd(saved, cot_pos_flat):
        pos_final, stiffnesses, edges, rest_lengths, positions0, source_nodes_dof, imposed_positions = saved
        n_dof = pos_final.shape[0]
        mask = jnp.ones((n_dof,), dtype=bool).at[jnp.asarray(source_nodes_dof)].set(False)

        def energy_positions_flat(p_flat, k):
            return elastic_energy(p_flat, edges, rest_lengths, k, d=d, force_type=force_type)

        def R_of_pos_and_k(p_flat, k):
            # Residual: masked ∇E = 0 at equilibrium
            grad_p = jax.grad(energy_positions_flat, argnums=0)(p_flat, k)
            return jnp.where(mask, grad_p, 0.0)

        # Jacobians at equilibrium
        J = jax.jacobian(R_of_pos_and_k, argnums=0)(pos_final, stiffnesses)      # (n_dof, n_dof)
        dR_dk = jax.jacobian(R_of_pos_and_k, argnums=1)(pos_final, stiffnesses)  # (n_dof, n_k)

        # Solve (J^T) w = cot_pos
        reg = 1e-8
        J_reg = J + reg * jnp.eye(J.shape[0], dtype=J.dtype)
        w = jsp_linalg.solve(J_reg.T, cot_pos_flat)

        # dL/dk = - (∂R/∂k)^T w
        cot_k = - jnp.dot(dR_dk.T, w)

        # Return one cotangent per *input* to compute_response_fire (6 total)
        return (cot_k, None, None, None, None, None)

    compute_response_fire.defvjp(crf_fwd, crf_bwd)
    return compute_response_fire


# ---------- Build the solver instance (capture hyperparams by closure) ----------
crf = make_compute_response_fire(d=2, max_steps=500_000, tol=FORCE_TOL, force_type="quadratic")


# ============================================================================
# JAX-DIFFERENTIABLE TRAJECTORY / POISSON / LOSS
# ============================================================================

def compute_quasistatic_trajectory_auxetic_jax(crf, stiffnesses, edges, rest_lengths,
                                               positions_flat, top_nodes, bottom_nodes,
                                               compression_strain, n_steps, d=2):
    """
    JAX-differentiable quasistatic compression trajectory.

    Ramps strain from 0 to compression_strain over n_steps increments,
    calling the differentiable FIRE solver (crf) at each step.

    Args:
        crf: Differentiable FIRE solver from make_compute_response_fire
        stiffnesses: (E,) JAX array — differentiable
        edges: (E, 2) array
        rest_lengths: (E,) array
        positions_flat: (N*d,) JAX array — initial (relaxed) positions
        top_nodes: node indices on top boundary
        bottom_nodes: node indices on bottom boundary
        compression_strain: target strain (negative = compression)
        n_steps: number of quasistatic increments
        d: spatial dimension (default 2)

    Returns:
        final_pos_flat: (N*d,) equilibrium positions at target strain
    """
    positions_flat = jnp.asarray(positions_flat).flatten()
    edges = jnp.asarray(edges)
    rest_lengths = jnp.asarray(rest_lengths)

    top_nodes = jnp.asarray(np.array(top_nodes))
    bottom_nodes = jnp.asarray(np.array(bottom_nodes))

    pos_2d = jnp.reshape(positions_flat, (-1, d))
    top_nodes = jnp.asarray(top_nodes, dtype=jnp.int32)
    bottom_nodes = jnp.asarray(bottom_nodes, dtype=jnp.int32)
    y_top_init = pos_2d[top_nodes, 1]
    y_bottom_init = pos_2d[bottom_nodes, 1]
    y_bottom_mean = jnp.mean(y_bottom_init)
    initial_height = jnp.mean(y_top_init) - y_bottom_mean

    x_top_init = pos_2d[top_nodes, 0]
    x_bottom_init = pos_2d[bottom_nodes, 0]

    # Precompute constrained DOF indices (y-component of top + bottom nodes)
    all_boundary = jnp.concatenate([jnp.asarray(top_nodes), jnp.asarray(bottom_nodes)])
    source_nodes_dof = jnp.concatenate([all_boundary * d, all_boundary * d + 1])  # x- and y-DOF

    # Offsets of individual top/bottom nodes from their mean y
    top_offsets = y_top_init - jnp.mean(y_top_init)
    bottom_y_values = y_bottom_init  # held fixed

    current_pos = positions_flat
    for step in range(n_steps):
        frac = step / (n_steps - 1)
        target_height = initial_height * (1 + compression_strain * frac)
        y_top_new = y_bottom_mean + target_height

        # Imposed y-values: top nodes (shifted) then bottom nodes (fixed)
        imposed_y_top = y_top_new + top_offsets
        imposed_y_bottom = bottom_y_values
        imposed_x_top = x_top_init  # hold x fixed for top nodes
        imposed_x_bottom = x_bottom_init  # hold x fixed for bottom nodes
        imposed_positions = jnp.concatenate([imposed_x_top, imposed_x_bottom, imposed_y_top, imposed_y_bottom])

        current_pos = crf(stiffnesses, edges, rest_lengths,
                          current_pos, source_nodes_dof, imposed_positions)

    return current_pos


def compute_poisson_ratio_single_jax(crf, stiffnesses, edges, rest_lengths, positions_flat,
                                     top_nodes, bottom_nodes, left_nodes, right_nodes,
                                     compression_strain, n_steps, d=2):
    """
    JAX-differentiable Poisson ratio computation for a single compression strain.

    Returns:
        poisson_ratio: scalar = -(lateral_strain / compression_strain)
    """
    positions_flat = jnp.asarray(positions_flat).flatten()
    left_nodes = jnp.asarray(left_nodes, dtype=jnp.int32)
    right_nodes = jnp.asarray(right_nodes, dtype=jnp.int32)

    final_pos_flat = compute_quasistatic_trajectory_auxetic_jax(
        crf, stiffnesses, edges, rest_lengths, positions_flat,
        top_nodes, bottom_nodes, compression_strain, n_steps, d=d
    )

    initial_pos_2d = jnp.reshape(positions_flat, (-1, d))
    final_pos_2d = jnp.reshape(final_pos_flat, (-1, d))

    width_initial = jnp.mean(initial_pos_2d[right_nodes, 0]) - jnp.mean(initial_pos_2d[left_nodes, 0])
    width_final = jnp.mean(final_pos_2d[right_nodes, 0]) - jnp.mean(final_pos_2d[left_nodes, 0])

    lateral_strain = (width_final - width_initial) / width_initial
    poisson_ratio = -(lateral_strain / compression_strain)

    return poisson_ratio


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
        tol=tol
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
            cs, n_strain_steps, force_type=force_type, tol=FORCE_TOL
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
            n_jobs_inner, force_type=force_type, tol=FORCE_TOL
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
            force_type=force_type,
            tol=force_tol
        )

        # --- Update stiffnesses ---
        network.stiffnesses = np.array(network.stiffnesses) + learning_rate * np.array(update)/ np.linalg.norm(update)
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
<<<<<<< HEAD
        if loss < loss_tol:
            break
=======
            save_checkpoint(
                task_seed=task_seed,
                realization_seed=realization_seed,
                history=history,
                network=network,
                task_config=task_config,
                current_step=step,
                results_dir=TARGETED_RESULTS_DIR,
            )
>>>>>>> 430816fbb7f3a5d7ebbd9d0bca61659e3dd7477d

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
    edges_jax = jnp.asarray(np.array(network.edges, dtype=np.int32))
    rest_lengths_jax = jnp.asarray(np.array(network.rest_lengths, dtype=np.float64))
    top_nodes_jax = jnp.asarray(top_nodes)
    bottom_nodes_jax = jnp.asarray(bottom_nodes)
    left_nodes_jax = jnp.asarray(left_nodes)
    right_nodes_jax = jnp.asarray(right_nodes)

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
        stiffnesses_jax = jnp.asarray(np.array(network.stiffnesses, dtype=np.float64))
        positions_flat_jax = jnp.asarray(min_pos.flatten())

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
