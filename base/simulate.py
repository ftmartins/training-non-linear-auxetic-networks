"""
Physics simulation functions shared by training and analysis.

Includes: JAX elastic energy, Cython FIRE wrapper, quasistatic trajectory
computation (both Cython-FIRE and JAX-differentiable variants), and the
JAX-differentiable Poisson-ratio observable.
"""

import warnings
import functools
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from .fire_minimize_memview_cy import fire_minimize_dof
from .config import FORCE_TOL


# ============================================================================
# JAX ELASTIC ENERGY
# ============================================================================

def elastic_energy(flat_positions, edges, rest_lengths, stiffnesses, *, d=2, force_type="quadratic"):
    """JAX-differentiable elastic energy. flat_positions: (N*d,)."""
    pos = jnp.reshape(flat_positions, (-1, d))
    edges = jnp.asarray(edges, dtype=jnp.int32)
    k = jnp.asarray(stiffnesses, dtype=jnp.float64)
    L0 = jnp.asarray(rest_lengths, dtype=jnp.float64)

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
# NEWTON QUASISTATIC HELPERS (private)
# ============================================================================

def _spring_forces(pos, edges, rest_lengths, stiffnesses):
    """O(n_edges) numpy force computation for harmonic springs. Returns F = -dE/dx."""
    i_n = edges[:, 0]; j_n = edges[:, 1]
    r_ij = pos[j_n] - pos[i_n]
    norms = np.linalg.norm(r_ij, axis=1)
    fmag = stiffnesses * (norms - rest_lengths) / norms
    F_edge = fmag[:, None] * r_ij
    F = np.zeros_like(pos)
    np.add.at(F, j_n, -F_edge)
    np.add.at(F, i_n,  F_edge)
    return F


def _precompute_stiffness_coo_indices(edges, n_nodes):
    """Precompute COO row/col indices for stiffness matrix assembly (constant for fixed graph)."""
    i_n = edges[:, 0]; j_n = edges[:, 1]
    ri = 2 * i_n; rj = 2 * j_n
    rows = np.concatenate([ri,   ri,   ri+1, ri+1,
                           rj,   rj,   rj+1, rj+1,
                           ri,   ri,   ri+1, ri+1,
                           rj,   rj,   rj+1, rj+1])
    cols = np.concatenate([ri,   ri+1, ri,   ri+1,
                           rj,   rj+1, rj,   rj+1,
                           rj,   rj+1, rj,   rj+1,
                           ri,   ri+1, ri,   ri+1])
    return rows, cols


def _stiffness_matrix_sparse(pos, edges, rest_lengths, stiffnesses, free_idx, n_dof,
                              coo_rows, coo_cols):
    """Assemble sparse stiffness matrix (free DOFs only) via COO → CSC for spsolve."""
    i_n = edges[:, 0]; j_n = edges[:, 1]
    r_ij = pos[j_n] - pos[i_n]
    norms = np.linalg.norm(r_ij, axis=1)
    k = stiffnesses; L0 = rest_lengths
    t1 = k * (1 - L0 / norms)
    t2 = k * L0 / norms ** 3
    dx, dy = r_ij[:, 0], r_ij[:, 1]
    K00 = t1 + t2 * dx * dx
    K01 = t2 * dx * dy
    K11 = t1 + t2 * dy * dy
    vals = np.concatenate([ K00,  K01,  K01,  K11,
                             K00,  K01,  K01,  K11,
                            -K00, -K01, -K01, -K11,
                            -K00, -K01, -K01, -K11])
    H = coo_matrix((vals, (coo_rows, coo_cols)), shape=(n_dof, n_dof)).tocsc()
    return H[free_idx, :][:, free_idx]


def _grad_k_from_adjoint(pos_final, edges, rest_lengths, w_full):
    """
    O(n_edges) IFT gradient: -dR/dk^T @ w.

    For spring e connecting (i,j):
      -dR/dk^T @ w = stretch_e * r_hat_e · (w_i - w_j)
    """
    i_n = edges[:, 0]; j_n = edges[:, 1]
    r_ij = pos_final[j_n] - pos_final[i_n]
    norms = np.linalg.norm(r_ij, axis=1)
    stretch = norms - rest_lengths
    dx = r_ij[:, 0] / norms; dy = r_ij[:, 1] / norms
    wi_x = w_full[2*i_n]; wi_y = w_full[2*i_n + 1]
    wj_x = w_full[2*j_n]; wj_y = w_full[2*j_n + 1]
    return stretch * (dx * (wi_x - wj_x) + dy * (wi_y - wj_y))


@functools.lru_cache(maxsize=8)
def _make_jax_newton_solver(n_dof, n_free, free_idx_tuple, max_newton):
    """
    Return a JIT-compiled Newton equilibrium solver for a fixed DOF layout.

    Cached by (n_dof, n_free, free_idx, max_newton) so compilation happens once
    per network topology and is reused across all training steps and strain steps.

    The returned function signature:
        solve(pos_flat, edges, rest_lengths, k, tol) -> pos_flat_equilibrated
    All arguments are JAX arrays; constrained DOFs must already be set in pos_flat.
    """
    free_idx_j = jnp.array(list(free_idx_tuple), dtype=jnp.int32)

    @jax.jit
    def _solve(pos_flat, edges, rest_lengths, k, tol):
        def _forces_free(pf):
            pos = pf.reshape(-1, 2)
            i_n, j_n = edges[:, 0], edges[:, 1]
            r_ij = pos[j_n] - pos[i_n]
            norms = jnp.linalg.norm(r_ij, axis=1)
            fmag = k * (norms - rest_lengths) / norms
            F_edge = fmag[:, None] * r_ij
            F = jnp.zeros_like(pos)
            F = F.at[j_n].add(-F_edge)
            F = F.at[i_n].add(F_edge)
            return F.flatten()[free_idx_j]

        def _stiffness_free(pf):
            pos = pf.reshape(-1, 2)
            i_n, j_n = edges[:, 0], edges[:, 1]
            r_ij = pos[j_n] - pos[i_n]
            norms = jnp.linalg.norm(r_ij, axis=1)
            t1 = k * (1.0 - rest_lengths / norms)
            t2 = k * rest_lengths / norms ** 3
            dxe, dye = r_ij[:, 0], r_ij[:, 1]
            K00 = t1 + t2 * dxe ** 2
            K01 = t2 * dxe * dye
            K11 = t1 + t2 * dye ** 2
            ri = 2 * i_n; rj = 2 * j_n
            H = jnp.zeros((n_dof, n_dof))
            H = H.at[ri,   ri  ].add(K00); H = H.at[ri,   ri+1].add(K01)
            H = H.at[ri+1, ri  ].add(K01); H = H.at[ri+1, ri+1].add(K11)
            H = H.at[rj,   rj  ].add(K00); H = H.at[rj,   rj+1].add(K01)
            H = H.at[rj+1, rj  ].add(K01); H = H.at[rj+1, rj+1].add(K11)
            H = H.at[ri,   rj  ].add(-K00); H = H.at[ri,   rj+1].add(-K01)
            H = H.at[ri+1, rj  ].add(-K01); H = H.at[ri+1, rj+1].add(-K11)
            H = H.at[rj,   ri  ].add(-K00); H = H.at[rj,   ri+1].add(-K01)
            H = H.at[rj+1, ri  ].add(-K01); H = H.at[rj+1, ri+1].add(-K11)
            return H[jnp.ix_(free_idx_j, free_idx_j)]

        def _cond(carry):
            _, F_free, count = carry
            return (jnp.linalg.norm(F_free) / n_free >= tol) & (count < max_newton)

        def _body(carry):
            pf, F_free, count = carry
            K_ff = _stiffness_free(pf)
            dx = jnp.linalg.solve(K_ff, F_free)
            pf_new = pf.at[free_idx_j].add(dx)
            return pf_new, _forces_free(pf_new), count + 1

        F0 = _forces_free(pos_flat)
        pf_out, _, _ = jax.lax.while_loop(
            _cond, _body, (pos_flat, F0, jnp.int32(0))
        )
        return pf_out

    return _solve


# ============================================================================
# QUASISTATIC TRAJECTORY
# ============================================================================

def compute_quasistatic_trajectory_auxetic(network, compression_strain, top_nodes, bottom_nodes,
                                           n_steps=100, verbose=False, force_type='quadratic',
                                           tol=1e-6, d=2, method='newton', max_newton=100):
    """
    Quasistatic compression trajectory. Ramps strain from 0 to compression_strain over n_steps.

    Args:
        method: 'newton' (default) or 'fire'.
            'newton' uses sparse Newton-Raphson; convergence: ‖F_free‖/n_free < tol.
            'fire'   uses Cython FIRE;            convergence: ‖F_free‖/n_dof < tol.
        max_newton: max Newton iterations per quasistatic step (default 1000; ignored for 'fire').

    Returns:
        traj: list of (N, d) position arrays, length n_steps
    """
    if method == 'newton':
        return _quasistatic_newton(network, compression_strain, top_nodes, bottom_nodes,
                                   n_steps=n_steps, tol=tol, max_newton=max_newton)

    # ── FIRE branch (original implementation) ────────────────────────────────
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


def _quasistatic_newton(network, compression_strain, top_nodes, bottom_nodes,
                         n_steps=100, tol=1e-6, max_newton=100):
    """JAX-accelerated Newton quasistatic trajectory. Called by compute_quasistatic_trajectory_auxetic."""
    pos = np.copy(network.positions)
    edges = np.array(network.edges, dtype=np.int32)
    rest_lengths = np.array(network.rest_lengths, dtype=np.float64)
    k = np.array(network.stiffnesses, dtype=np.float64)
    n_nodes = len(pos)
    n_dof = n_nodes * 2

    y_top = pos[top_nodes, 1]
    y_bot = pos[bottom_nodes, 1]
    init_h = y_top.mean() - y_bot.mean()
    top_offsets = y_top - y_top.mean()
    x_top_init = pos[top_nodes, 0].copy()
    x_bot_init = pos[bottom_nodes, 0].copy()

    all_bdry = np.concatenate([np.asarray(top_nodes), np.asarray(bottom_nodes)])
    bdry_dofs = np.concatenate([all_bdry * 2, all_bdry * 2 + 1])
    free_mask = np.ones(n_dof, dtype=bool)
    free_mask[bdry_dofs] = False
    free_idx = np.where(free_mask)[0]
    n_free = len(free_idx)

    jax_solver = _make_jax_newton_solver(n_dof, n_free, tuple(free_idx), max_newton)
    edges_j = jnp.array(edges)
    rest_j  = jnp.array(rest_lengths)
    k_j     = jnp.array(k)
    tol_j   = jnp.float64(tol)

    traj = [np.copy(pos)]

    for step in range(1, n_steps):
        frac = step / (n_steps - 1)
        target_h = init_h * (1 + compression_strain * frac)
        y_top_new = y_bot.mean() + target_h

        pos[top_nodes, 1] = y_top_new + top_offsets
        pos[bottom_nodes, 1] = y_bot
        pos[top_nodes, 0] = x_top_init
        pos[bottom_nodes, 0] = x_bot_init

        pf = jax_solver(jnp.array(pos.flatten()), edges_j, rest_j, k_j, tol_j)
        pos = np.array(pf).reshape(-1, 2)
        traj.append(pos.copy())

    return traj


# ============================================================================
# IFT GRADIENT (Newton forward + numpy adjoint backward)
# ============================================================================

def compute_ift_gradient(network, compression_strains, target_poissons,
                         top_nodes, bottom_nodes, left_nodes, right_nodes,
                         tol, n_strain_steps):
    """
    Compute MSE loss and gradient w.r.t. stiffnesses using the implicit function theorem.

    Forward pass: Newton-Raphson quasistatic trajectory (sparse spsolve, up to 1000 iters).
    Backward pass: adjoint solve with K_ff at the final equilibrium + O(n_edges) grad formula.
    No JAX autodiff — fully numpy/scipy.

    Args:
        network: ElasticNetwork with .positions, .edges, .rest_lengths, .stiffnesses
        compression_strains: list of compression strain values (e.g. [-0.2])
        target_poissons:     list of target Poisson ratios (e.g. [-0.8])
        top_nodes, bottom_nodes: boundary node indices (constrained during trajectory)
        left_nodes, right_nodes: measurement node indices (for Poisson ratio)
        tol: Newton convergence tolerance (‖F_free‖/n_free < tol)
        n_strain_steps: number of quasistatic steps

    Returns:
        (loss, grad): scalar MSE loss and gradient array of shape (n_edges,)
    """
    pos = np.copy(network.positions)
    edges = np.array(network.edges, dtype=np.int32)
    rest_lengths = np.array(network.rest_lengths, dtype=np.float64)
    k = np.array(network.stiffnesses, dtype=np.float64)
    n_nodes = len(pos)
    n_dof = n_nodes * 2

    all_bdry = np.concatenate([np.asarray(top_nodes), np.asarray(bottom_nodes)])
    bdry_dofs = np.concatenate([all_bdry * 2, all_bdry * 2 + 1])
    free_mask = np.ones(n_dof, dtype=bool)
    free_mask[bdry_dofs] = False
    free_idx = np.where(free_mask)[0]
    n_free = len(free_idx)
    coo_rows, coo_cols = _precompute_stiffness_coo_indices(edges, n_nodes)

    jax_solver = _make_jax_newton_solver(n_dof, n_free, tuple(free_idx), 100)
    edges_j = jnp.array(edges)
    rest_j  = jnp.array(rest_lengths)
    k_j     = jnp.array(k)
    tol_j   = jnp.float64(tol)

    total_loss = 0.0
    total_grad = np.zeros(len(k))
    n_pairs = len(compression_strains)

    for cs, nu_tgt in zip(compression_strains, target_poissons):
        pos_cur = np.copy(pos)
        y_top = pos_cur[top_nodes, 1]
        y_bot = pos_cur[bottom_nodes, 1]
        init_h = y_top.mean() - y_bot.mean()
        top_offsets = y_top - y_top.mean()
        x_top_init = pos_cur[top_nodes, 0].copy()
        x_bot_init = pos_cur[bottom_nodes, 0].copy()

        K_ff_last = None

        for step in range(1, n_strain_steps):
            frac = step / (n_strain_steps - 1)
            target_h = init_h * (1 + cs * frac)
            y_top_new = y_bot.mean() + target_h
            pos_cur[top_nodes, 1] = y_top_new + top_offsets
            pos_cur[bottom_nodes, 1] = y_bot
            pos_cur[top_nodes, 0] = x_top_init
            pos_cur[bottom_nodes, 0] = x_bot_init

            pf = jax_solver(jnp.array(pos_cur.flatten()), edges_j, rest_j, k_j, tol_j)
            pos_cur = np.array(pf).reshape(-1, 2)

            if step == n_strain_steps - 1:
                # K_ff at the fully-converged final position for the IFT adjoint solve
                K_ff_last = _stiffness_matrix_sparse(pos_cur, edges, rest_lengths, k,
                                                      free_idx, n_dof, coo_rows, coo_cols)

        pos_init = pos          # initial (unstrained)
        pos_final = pos_cur     # final (at full compression)

        # Loss: MSE Poisson ratio
        w_init  = pos_init[right_nodes, 0].mean() - pos_init[left_nodes, 0].mean()
        w_final = pos_final[right_nodes, 0].mean() - pos_final[left_nodes, 0].mean()
        nu_obs = -((w_final - w_init) / w_init) / cs
        residual = nu_obs - nu_tgt
        total_loss += residual ** 2

        # dL/d(pos_final): nonzero only on left/right x-DOFs
        dL_dnu = 2.0 * residual / n_pairs
        n_right, n_left = len(right_nodes), len(left_nodes)
        dL_dpos = np.zeros(n_dof)
        for j in right_nodes:
            dL_dpos[j * 2] += dL_dnu * -(1.0 / cs) * (1.0 / w_init) / n_right
        for j in left_nodes:
            dL_dpos[j * 2] += dL_dnu * -(1.0 / cs) * (1.0 / w_init) * (-1.0 / n_left)

        # Adjoint solve: K_ff @ w_free = dL/dpos[free_idx]
        w_free = spsolve(K_ff_last, dL_dpos[free_idx])
        w_full = np.zeros(n_dof)
        w_full[free_idx] = w_free

        # O(n_edges) gradient
        total_grad += _grad_k_from_adjoint(pos_final, edges, rest_lengths, w_full)

    total_loss /= n_pairs
    return total_loss, total_grad


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
        positions0 = jnp.asarray(positions0, dtype=jnp.float64).flatten()
        imposed_positions = jnp.asarray(imposed_positions, dtype=jnp.float64).flatten()
        n_dof = positions0.shape[0]
        mask = jnp.ones(n_dof, dtype=bool).at[jnp.asarray(source_nodes_dof, dtype=jnp.int32)].set(False)

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
        mask = jnp.ones((n_dof,), dtype=bool).at[jnp.asarray(source_nodes_dof, dtype=jnp.int32)].set(False)

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
    positions_flat = jnp.asarray(positions_flat, dtype=jnp.float64).flatten()
    edges = jnp.asarray(edges, dtype=jnp.int32)
    rest_lengths = jnp.asarray(rest_lengths, dtype=jnp.float64)
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
    positions_flat = jnp.asarray(positions_flat, dtype=jnp.float64).flatten()
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
