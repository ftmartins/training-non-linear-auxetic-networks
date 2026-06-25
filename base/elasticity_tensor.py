"""
2D elasticity tensor computation for finite elastic networks.

Computes the elasticity tensor via the Schur complement approach:
  1. Analytical Hessian of elastic energy w.r.t. positions (JAX-differentiable)
  2. Split into boundary/interior DOFs
  3. Schur complement gives effective boundary stiffness
  4. Affine strain mapping yields the Voigt elasticity tensor

Supports both quadratic (Hookean) and quartic spring potentials.

2D Voigt convention:
  Index 1 -> (xx), Index 2 -> (yy), Index 3 -> (xy)
  Strain vector: [eps_xx, eps_yy, gamma_xy] where gamma_xy = 2*eps_xy
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
    positions = jnp.asarray(positions)
    edges = jnp.asarray(edges)
    stiffnesses = jnp.asarray(stiffnesses)
    rest_lengths = jnp.asarray(rest_lengths)

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


def precompute_dof_indices(boundary_node_indices, n_nodes, d=2):
    """
    Precompute boundary and interior DOF index arrays using numpy.

    Must be called ONCE before JAX tracing (before jax.jit / jax.grad) so
    that the index arrays are concrete. Pass the results to
    compute_elasticity_tensor_2d.

    Parameters
    ----------
    boundary_node_indices : array of int
        Indices of ALL boundary nodes.
    n_nodes : int
        Total number of nodes.
    d : int
        Spatial dimension (2).

    Returns
    -------
    boundary_dofs : np.ndarray of int
        Sorted DOF indices for boundary nodes.
    interior_dofs : np.ndarray of int
        Sorted DOF indices for interior nodes.
    """
    boundary_node_indices = np.asarray(boundary_node_indices)
    boundary_dofs = np.sort(np.concatenate([
        boundary_node_indices * d,
        boundary_node_indices * d + 1,
    ]))
    all_dofs = np.arange(n_nodes * d)
    interior_mask = np.ones(n_nodes * d, dtype=bool)
    interior_mask[boundary_dofs] = False
    interior_dofs = all_dofs[interior_mask]
    return boundary_dofs, interior_dofs


def compute_elasticity_tensor_2d(positions, edges, stiffnesses, rest_lengths,
                                 boundary_dofs, interior_dofs,
                                 force_type='quadratic'):
    """
    Compute the 3x3 Voigt elasticity tensor for a 2D finite elastic network.

    Method (Schur complement):
      1. Compute analytical Hessian H (2N x 2N)
      2. Split DOFs into boundary and interior (precomputed indices)
      3. Schur complement: H_eff = H_bb - H_bi @ H_ii^{-1} @ H_ib
      4. Build affine mapping A: strain (3,) -> u_boundary (2*n_boundary,)
      5. C_voigt = A^T @ H_eff @ A / Area

    Parameters
    ----------
    positions : array (N, 2)
        Node positions (at the configuration where tensor is evaluated).
    edges : array (E, 2), int
        Edge connectivity.
    stiffnesses : array (E,)
        Spring stiffnesses.
    rest_lengths : array (E,)
        Rest lengths.
    boundary_dofs : array of int
        Boundary DOF indices (from precompute_dof_indices).
    interior_dofs : array of int
        Interior DOF indices (from precompute_dof_indices).
    force_type : str
        'quadratic' or 'quartic'.

    Returns
    -------
    C_voigt : jax array (3, 3)
        Voigt elasticity tensor.
        Rows/cols: [xx, yy, xy] (1-indexed: C_11, C_12, C_13, ...)
    """
    positions = jnp.asarray(positions)
    boundary_dofs = jnp.asarray(boundary_dofs)
    interior_dofs = jnp.asarray(interior_dofs)

    d = 2

    # 1. Compute Hessian
    H = compute_hessian_jax(positions, edges, stiffnesses, rest_lengths,
                            force_type=force_type)

    # 2. Extract sub-blocks using precomputed integer indices
    H_bb = H[jnp.ix_(boundary_dofs, boundary_dofs)]
    H_bi = H[jnp.ix_(boundary_dofs, interior_dofs)]
    H_ib = H[jnp.ix_(interior_dofs, boundary_dofs)]
    H_ii = H[jnp.ix_(interior_dofs, interior_dofs)]

    # 3. Schur complement: H_eff = H_bb - H_bi @ H_ii^{-1} @ H_ib
    H_ii_inv_H_ib = jnp.linalg.solve(H_ii, H_ib)
    H_eff = H_bb - H_bi @ H_ii_inv_H_ib

    # 4. Affine strain mapping
    # Voigt strain: [eps_xx, eps_yy, gamma_xy] where gamma_xy = 2*eps_xy
    # For boundary node m at position (x, y):
    #   u_x = eps_xx * x + eps_xy * y = eps_xx * x + (gamma_xy/2) * y
    #   u_y = eps_xy * x + eps_yy * y = (gamma_xy/2) * x + eps_yy * y
    n_boundary_dofs = boundary_dofs.shape[0]
    dof_node = boundary_dofs // d   # which node each DOF belongs to
    dof_axis = boundary_dofs % d    # 0=x, 1=y

    bnd_x = positions[dof_node, 0]  # x-coordinate of each DOF's node
    bnd_y = positions[dof_node, 1]  # y-coordinate of each DOF's node

    # For x-DOFs (axis=0): u_x = eps_xx * x + (gamma_xy/2) * y
    #   col 0 (eps_xx): x,   col 1 (eps_yy): 0,   col 2 (gamma_xy): y/2
    # For y-DOFs (axis=1): u_y = (gamma_xy/2) * x + eps_yy * y
    #   col 0 (eps_xx): 0,   col 1 (eps_yy): y,   col 2 (gamma_xy): x/2
    is_x = (dof_axis == 0).astype(jnp.float64)
    is_y = (dof_axis == 1).astype(jnp.float64)

    A = jnp.zeros((n_boundary_dofs, 3))
    A = A.at[:, 0].set(is_x * bnd_x)            # eps_xx -> x for x-DOFs
    A = A.at[:, 1].set(is_y * bnd_y)            # eps_yy -> y for y-DOFs
    A = A.at[:, 2].set(is_x * bnd_y       # gamma_xy -> y/2 for x-DOFs
                       + is_y * bnd_x )    # gamma_xy -> x/2 for y-DOFs

    # 5. Elasticity tensor
    # Area = bounding box area
    x_min, x_max = jnp.min(positions[:, 0]), jnp.max(positions[:, 0])
    y_min, y_max = jnp.min(positions[:, 1]), jnp.max(positions[:, 1])
    area = (x_max - x_min) * (y_max - y_min)

    C_voigt = A.T @ H_eff @ A / area  # (3, 3)

    return C_voigt


def extract_moduli_2d(C_voigt):
    """
    Extract isotropic elastic moduli from the 3x3 Voigt elasticity tensor.

    Parameters
    ----------
    C_voigt : array (3, 3)
        Voigt elasticity tensor (from compute_elasticity_tensor_2d).

    Returns
    -------
    moduli : dict with keys:
        'B'  : 2D bulk modulus  = (C11 + C22 + 2*C12) / 4
        'G'  : 2D shear modulus = (C11 + C22 - 2*C12) / 4
        'nu' : 2D Poisson ratio = (B - G) / (B + G)
        'C_voigt' : the input tensor (for convenience)
    """
    C11 = C_voigt[0, 0]
    C22 = C_voigt[1, 1]
    C12 = C_voigt[0, 1]

    B = (C11 + C22 + 2.0 * C12) / 4.0
    G = (C11 + C22 - 2.0 * C12) / 4.0
    nu = (B - G) / (B + G)

    return {'B': B, 'G': G, 'nu': nu, 'C_voigt': C_voigt}
