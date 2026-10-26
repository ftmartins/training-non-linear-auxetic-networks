"""
Analytical Hessian of the elastic energy of finite elastic networks (JAX-differentiable).

Supports both quadratic (Hookean) and quartic spring potentials.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def compute_hessian_jax(positions, edges, stiffnesses, rest_lengths,
                        force_type='quadratic'):
    """
    Analytical Hessian of elastic energy w.r.t. positions, in pure JAX.

    For each edge e = (i, j):
      n = (x_i - x_j) / |x_i - x_j|   (unit bond vector)
      r = |x_i - x_j|,  delta = r - l0

      Quadratic:  H_block = -k * [n (x) n  +  (l0/r)(I - n (x) n)]
      Quartic:    H_block = -[V''(r)(n (x) n) + (V'(r)/r)(I - n (x) n)]

    Uses the incidence-matrix assembly from generalized_susceptibility.py,
    ported to pure JAX for end-to-end differentiability w.r.t. stiffnesses.

    Parameters
    ----------
    positions : array (N, 2)
        Current node positions.
    edges : array (E, 2), int
        Edge connectivity.
    stiffnesses : array (E,)
        Spring stiffnesses.
    rest_lengths : array (E,)
        Rest lengths.
    force_type : str
        'quadratic' or 'quartic'.

    Returns
    -------
    H : jax array (2N, 2N)
        Hessian matrix (differentiable w.r.t. stiffnesses and positions).
    """
    positions = jnp.asarray(positions, dtype=jnp.float64)
    edges = jnp.asarray(edges)
    stiffnesses = jnp.asarray(stiffnesses, dtype=jnp.float64)
    rest_lengths = jnp.asarray(rest_lengths, dtype=jnp.float64)

    N = positions.shape[0]
    E = edges.shape[0]
    d = 2

    # Incidence matrix (E, N): +1 at node i, -1 at node j
    Delta = jnp.zeros((E, N))
    Delta = Delta.at[jnp.arange(E), edges[:, 0]].set(1.0)
    Delta = Delta.at[jnp.arange(E), edges[:, 1]].set(-1.0)

    # Edge geometry
    i_idx = edges[:, 0]
    j_idx = edges[:, 1]
    dist_vec = positions[i_idx] - positions[j_idx]  # (E, 2)
    r = jnp.linalg.norm(dist_vec, axis=1)           # (E,)
    n = dist_vec / r[:, None]                        # (E, 2) unit vectors

    # Projectors
    outer_nn = jnp.einsum('ea,eb->eab', n, n)       # (E, 2, 2)
    I2 = jnp.eye(d)
    P_perp = I2[None, :, :] - outer_nn              # (E, 2, 2)

    if force_type == 'quadratic':
        # H_block = -k * [n(x)n + (l0/r)(I - n(x)n)]
        # parallel coefficient: k
        # perpendicular coefficient: k * (1 - l0/r)
        parallel_coef = stiffnesses                      # (E,)
        perp_coef = stiffnesses * (1 - rest_lengths / r) # (E,)
    elif force_type == 'quartic':
        avg_rest = jnp.mean(rest_lengths)
        scale = 36.0 / (avg_rest**2 + 1e-12)
        delta = r - rest_lengths
        # V'(r) = k * (scale * delta^3 - 0.5 * delta)
        # V''(r) = k * (3 * scale * delta^2 - 0.5)
        Vprime = stiffnesses * (scale * delta**3 - 0.5 * delta)
        Vsecond = stiffnesses * (3.0 * scale * delta**2 - 0.5)
        parallel_coef = Vsecond        # (E,)
        perp_coef = Vprime / r         # (E,)
    else:
        raise ValueError(f"Unknown force_type: {force_type}")

    # Assemble off-diagonal blocks via incidence matrix:
    #   H_offdiag[a,d,b,f] = sum_e coef_e * Delta[e,a] * block[e,d,f] * Delta[e,b]
    parallel_term = jnp.einsum('e, ea, edf, eb -> adbf',
                               parallel_coef, Delta, outer_nn, Delta)
    perp_term = jnp.einsum('e, ea, edf, eb -> adbf',
                           perp_coef, Delta, P_perp, Delta)

    hessian = parallel_term + perp_term  # (N, 2, N, 2)

    # Fix diagonal: H[a,d,a,f] = -sum_{b!=a} H[a,d,b,f]
    # The incidence approach double-counts the diagonal; correct it:
    diag_I = jnp.diag(jnp.ones(N))  # (N, N)
    row_sum = jnp.einsum('ambn -> mbn', hessian)  # (N, 2, 2)
    hessian = hessian - jnp.einsum('ab, mbn -> ambn', diag_I, row_sum)

    return hessian.reshape(2 * N, 2 * N)
