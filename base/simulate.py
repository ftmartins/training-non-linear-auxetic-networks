"""
Physics simulation functions shared by training and analysis.

Includes: JAX elastic energy, Cython FIRE wrapper, quasistatic trajectory
computation (both Cython-FIRE and JAX-differentiable variants), and the
JAX-differentiable Poisson-ratio observable.
"""

import sys
import warnings
import numpy as np
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

# Cython .so lives at project root
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fire_minimize_memview_cy import fire_minimize_dof
from .config import FORCE_TOL


# ============================================================================
# JAX ELASTIC ENERGY
# ============================================================================

def elastic_energy(flat_positions, edges, rest_lengths, stiffnesses, *, d=2, force_type="quadratic"):
    """JAX-differentiable elastic energy. flat_positions: (N*d,)."""
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
# CYTHON FIRE WRAPPER
# ============================================================================

def fire_minimize_network(network, constrained_dof_idx=None, force_type='quadratic',
                          tol=1e-6, max_steps=500_000, deltaT=1e-2,
                          retry_steps_1=500_000, retry_steps_2=500_000):
    """
    Minimize network using Cython FIRE, with up to two retry passes.

    Returns:
        min_positions: (N, d) minimized positions
        force_norm: final force norm
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

    if force_norm is not None and force_norm >= tol:
        min_pos, force_norm, _ = fire_minimize_dof(
            min_pos, edges_i32, rest_lengths_f64, stiffnesses_f64,
            deltaT, retry_steps_1, tol, constrained_dof_idx, force_type_int,
        )

    if force_norm is not None and force_norm >= tol:
        min_pos, force_norm, _ = fire_minimize_dof(
            min_pos, edges_i32, rest_lengths_f64, stiffnesses_f64,
            deltaT, retry_steps_2, tol, constrained_dof_idx, force_type_int,
        )

    return min_pos, force_norm


# ============================================================================
# QUASISTATIC TRAJECTORY — CYTHON FIRE
# ============================================================================

def compute_quasistatic_trajectory_auxetic(network, compression_strain, top_nodes, bottom_nodes,
                                           n_steps=100, verbose=False, force_type='quadratic',
                                           tol=1e-6, d=2):
    """
    Quasistatic compression trajectory via Cython FIRE.

    Ramps strain from 0 to compression_strain over n_steps increments.

    Returns:
        traj: list of (N, d) position arrays, length n_steps
    """
    positions = np.copy(network.positions)
    y_top = positions[top_nodes, 1]
    y_bottom = positions[bottom_nodes, 1]
    initial_height = y_top.mean() - y_bottom.mean()
    target_height = initial_height * (1 + compression_strain)

    x_top_init = positions[top_nodes, 0]
    x_bottom_init = positions[bottom_nodes, 0]

    all_boundary = np.concatenate([np.asarray(top_nodes), np.asarray(bottom_nodes)])
    constrained_idx_dof = np.concatenate([all_boundary * d, all_boundary * d + 1])

    traj = [np.copy(positions)]

    edges_i32 = np.array(network.edges, dtype=np.int32)
    rest_i64 = np.array(network.rest_lengths, dtype=np.float64)
    ft_int = 1 if force_type == 'quartic' else 0

    for step in range(1, n_steps):
        frac = step / (n_steps - 1)
        height_to_impose = initial_height - frac * (initial_height - target_height)
        y_top_new = y_bottom.mean() + height_to_impose

        positions_step = np.copy(positions)
        positions_step[top_nodes, 1] = y_top_new + (positions[top_nodes, 1] - positions[top_nodes, 1].mean())
        positions_step[bottom_nodes, 1] = y_bottom
        positions_step[top_nodes, 0] = x_top_init
        positions_step[bottom_nodes, 0] = x_bottom_init

        min_pos, force_norm, _ = fire_minimize_dof(
            positions_step, edges_i32, rest_i64,
            np.array(network.stiffnesses, dtype=np.float64),
            1e-2, 1_000_000, tol, constrained_idx_dof, ft_int,
        )
        assert force_norm <= tol, (
            f"FIRE did not converge at step {step}: force_norm={force_norm:.3e}"
        )

        positions = min_pos
        traj.append(np.copy(min_pos))

    return traj


def compute_quasistatic_trajectory_full_cycle(network, amp, top_nodes, bottom_nodes,
                                              n_steps_per_phase=100, verbose=False,
                                              force_type='quadratic', tol=1e-6, d=2):
    """
    Full compression–extension–relaxation cycle via Cython FIRE.

    Returns:
        traj: list of (N, d) arrays, length 3*n_steps_per_phase + 1
    """
    positions = np.copy(network.positions)
    y_top = positions[top_nodes, 1]
    y_bottom = positions[bottom_nodes, 1]
    initial_height = y_top.mean() - y_bottom.mean()

    traj = [np.copy(positions)]

    constrained_idx_dof = []
    for i in np.concatenate([top_nodes, bottom_nodes]):
        constrained_idx_dof.append(i * d + 1)

    edges_i32 = np.array(network.edges, dtype=np.int32)
    rest_i64 = np.array(network.rest_lengths, dtype=np.float64)
    ft_int = 1 if force_type == 'quartic' else 0

    def _step(strain, dt, max_s):
        nonlocal positions
        height_to_impose = initial_height * (1 + strain)
        y_top_new = y_bottom.mean() + height_to_impose
        pos_step = np.copy(positions)
        pos_step[top_nodes, 1] = y_top_new + (positions[top_nodes, 1] - positions[top_nodes, 1].mean())
        pos_step[bottom_nodes, 1] = y_bottom
        min_pos, _, _ = fire_minimize_dof(
            pos_step, edges_i32, rest_i64,
            np.array(network.stiffnesses, dtype=np.float64),
            dt, max_s, tol, constrained_idx_dof, ft_int,
        )
        positions = min_pos
        traj.append(np.copy(min_pos))

    for s in range(n_steps_per_phase):
        _step((s + 1) / n_steps_per_phase * amp, 1e-3, 800_000)
    for s in range(n_steps_per_phase):
        _step(amp - (s + 1) / n_steps_per_phase * 2 * amp, 1e-3, 500_000)
    for s in range(n_steps_per_phase):
        _step(-amp + (s + 1) / n_steps_per_phase * amp, 1e-3, 500_000)

    return traj


# ============================================================================
# JAX-DIFFERENTIABLE FIRE SOLVER
# ============================================================================

def make_compute_response_fire(*, d=2, dt_init=1e-2, dt_max=1e-1, dt_min=1e-4,
                                alpha_start=0.1, finc=1.1, fdec=0.5, falpha=0.99,
                                max_steps=1_000_000, tol=1e-6, force_type="quadratic"):
    """
    Returns a JAX-differentiable FIRE solver (custom VJP).

    Signature: (stiffnesses, edges, rest_lengths, positions0,
                source_nodes_dof, imposed_positions) → final_positions
    """

    def _fire_forward(stiffnesses, edges, rest_lengths, positions0,
                      source_nodes_dof, imposed_positions):
        positions0 = jnp.asarray(positions0).flatten()
        imposed_positions = jnp.asarray(imposed_positions).flatten()
        n_dof = positions0.shape[0]
        mask = jnp.ones(n_dof, dtype=bool).at[jnp.asarray(source_nodes_dof)].set(False)

        def energy_fn(pos_flat, k):
            return elastic_energy(pos_flat, edges, rest_lengths, k, d=d, force_type=force_type)

        def forces(pos_flat, k):
            return jax.grad(energy_fn, argnums=0)(pos_flat, k)

        vel0 = jnp.zeros_like(positions0)
        pos0 = positions0.at[source_nodes_dof].set(imposed_positions)
        grads0 = forces(pos0, stiffnesses) * mask
        gnorm0 = jnp.linalg.norm(jnp.where(mask, grads0, 0.0))
        state0 = (pos0, vel0, dt_init, alpha_start, 0, gnorm0, 1.0)

        def body_fn(i, state):
            pos, vel, dt, alpha, npos, gnorm, active = state

            def do_update(inputs):
                pos, vel, dt, alpha, npos, gnorm = inputs
                g = forces(pos, stiffnesses) * mask
                f = -g
                P = jnp.vdot(vel, f)
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
                    return pos, vel, jnp.minimum(dt * finc, dt_max), alpha * falpha, npos

                def neg_power(args):
                    pos, vel, dt, alpha, npos = args
                    return pos, jnp.zeros_like(vel), jnp.maximum(dt * fdec, dt_min), alpha_start, 0

                pos, vel, dt, alpha, npos = jax.lax.cond(
                    P >= 0, pos_power, neg_power, (pos, vel, dt, alpha, npos)
                )
                pos = pos.at[source_nodes_dof].set(imposed_positions)
                vel = vel * mask
                gnorm = jnp.linalg.norm(jnp.where(mask, g, 0.0))
                return (pos, vel, dt, alpha, npos, gnorm, jnp.where(gnorm / n_dof < tol, 0.0, 1.0))

            def skip_update(inputs):
                pos, vel, dt, alpha, npos, gnorm = inputs
                return (pos, vel, dt, alpha, npos, gnorm, active)

            return jax.lax.cond(active > 0.5, do_update, skip_update,
                                (pos, vel, dt, alpha, npos, gnorm))

        final_state = jax.lax.fori_loop(0, max_steps, body_fn, state0)
        pos_final, _, _, _, _, final_gnorm, _ = final_state
        return pos_final, final_gnorm

    def _warn_check(n_dof):
        def _warn(gnorm):
            if float(gnorm) / n_dof >= tol:
                warnings.warn(
                    f"FIRE did not converge after {max_steps} steps: "
                    f"gnorm/n_dof={float(gnorm)/n_dof:.2e}, tol={tol:.2e}.",
                    RuntimeWarning, stacklevel=6,
                )
        return _warn

    @jax.custom_vjp
    def compute_response_fire(stiffnesses, edges, rest_lengths, positions0,
                               source_nodes_dof, imposed_positions):
        pos_final, gnorm = _fire_forward(stiffnesses, edges, rest_lengths,
                                         positions0, source_nodes_dof, imposed_positions)
        jax.debug.callback(_warn_check(positions0.flatten().shape[0]), gnorm)
        return pos_final

    def crf_fwd(stiffnesses, edges, rest_lengths, positions0, source_nodes_dof, imposed_positions):
        pos_final, gnorm = _fire_forward(stiffnesses, edges, rest_lengths,
                                         positions0, source_nodes_dof, imposed_positions)
        jax.debug.callback(_warn_check(positions0.flatten().shape[0]), gnorm)
        saved = (pos_final, stiffnesses, edges, rest_lengths,
                 positions0, source_nodes_dof, imposed_positions)
        return pos_final, saved

    def crf_bwd(saved, cot_pos_flat):
        pos_final, stiffnesses, edges, rest_lengths, positions0, source_nodes_dof, imposed_positions = saved
        n_dof = pos_final.shape[0]
        mask = jnp.ones((n_dof,), dtype=bool).at[jnp.asarray(source_nodes_dof)].set(False)

        def energy_fn(p_flat, k):
            return elastic_energy(p_flat, edges, rest_lengths, k, d=d, force_type=force_type)

        def R(p_flat, k):
            return jnp.where(mask, jax.grad(energy_fn, argnums=0)(p_flat, k), 0.0)

        J = jax.jacobian(R, argnums=0)(pos_final, stiffnesses)
        dR_dk = jax.jacobian(R, argnums=1)(pos_final, stiffnesses)
        J_reg = J + 1e-8 * jnp.eye(J.shape[0], dtype=J.dtype)
        w = jsp_linalg.solve(J_reg.T, cot_pos_flat)
        return (-jnp.dot(dR_dk.T, w), None, None, None, None, None)

    compute_response_fire.defvjp(crf_fwd, crf_bwd)
    return compute_response_fire


# Default solver instance (quadratic potential, project force tolerance)
crf = make_compute_response_fire(d=2, max_steps=1_000_000, tol=FORCE_TOL, force_type="quadratic")


# ============================================================================
# JAX-DIFFERENTIABLE TRAJECTORY AND OBSERVABLES
# ============================================================================

def compute_quasistatic_trajectory_auxetic_jax(crf_fn, stiffnesses, edges, rest_lengths,
                                               positions_flat, top_nodes, bottom_nodes,
                                               compression_strain, n_steps, d=2):
    """
    JAX-differentiable quasistatic compression trajectory.

    Returns:
        final_pos_flat: (N*d,) equilibrium positions at target strain
    """
    positions_flat = jnp.asarray(positions_flat).flatten()
    edges = jnp.asarray(edges)
    rest_lengths = jnp.asarray(rest_lengths)
    top_nodes = jnp.asarray(np.array(top_nodes), dtype=jnp.int32)
    bottom_nodes = jnp.asarray(np.array(bottom_nodes), dtype=jnp.int32)

    pos_2d = jnp.reshape(positions_flat, (-1, d))
    y_top_init = pos_2d[top_nodes, 1]
    y_bottom_init = pos_2d[bottom_nodes, 1]
    y_bottom_mean = jnp.mean(y_bottom_init)
    initial_height = jnp.mean(y_top_init) - y_bottom_mean

    x_top_init = pos_2d[top_nodes, 0]
    x_bottom_init = pos_2d[bottom_nodes, 0]
    top_offsets = y_top_init - jnp.mean(y_top_init)

    all_boundary = jnp.concatenate([top_nodes, bottom_nodes])
    source_nodes_dof = jnp.concatenate([all_boundary * d, all_boundary * d + 1])

    current_pos = positions_flat
    for step in range(n_steps):
        frac = step / (n_steps - 1)
        target_height = initial_height * (1 + compression_strain * frac)
        y_top_new = y_bottom_mean + target_height
        imposed_positions = jnp.concatenate([
            x_top_init, x_bottom_init,
            y_top_new + top_offsets, y_bottom_init,
        ])
        current_pos = crf_fn(stiffnesses, edges, rest_lengths,
                             current_pos, source_nodes_dof, imposed_positions)

    return current_pos


def compute_poisson_ratio_single_jax(crf_fn, stiffnesses, edges, rest_lengths, positions_flat,
                                     top_nodes, bottom_nodes, left_nodes, right_nodes,
                                     compression_strain, n_steps, d=2):
    """
    JAX-differentiable Poisson ratio for a single compression strain.

    Returns:
        poisson_ratio: scalar = -(lateral_strain / compression_strain)
    """
    positions_flat = jnp.asarray(positions_flat).flatten()
    left_nodes = jnp.asarray(left_nodes, dtype=jnp.int32)
    right_nodes = jnp.asarray(right_nodes, dtype=jnp.int32)

    final_pos_flat = compute_quasistatic_trajectory_auxetic_jax(
        crf_fn, stiffnesses, edges, rest_lengths, positions_flat,
        top_nodes, bottom_nodes, compression_strain, n_steps, d=d,
    )

    initial_pos_2d = jnp.reshape(positions_flat, (-1, d))
    final_pos_2d = jnp.reshape(final_pos_flat, (-1, d))

    width_initial = (jnp.mean(initial_pos_2d[right_nodes, 0])
                     - jnp.mean(initial_pos_2d[left_nodes, 0]))
    width_final = (jnp.mean(final_pos_2d[right_nodes, 0])
                   - jnp.mean(final_pos_2d[left_nodes, 0]))

    lateral_strain = (width_final - width_initial) / width_initial
    return -(lateral_strain / compression_strain)
