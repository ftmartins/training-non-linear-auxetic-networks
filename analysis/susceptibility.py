"""
Susceptibility decomposition for elastic spring networks.

Absorbs and extends src/generalized_susceptibility.py.

Public API
----------
compute_susceptibilities(positions, edges, stiffnesses, rest_lengths,
                         constrained_nodes=np.array([]), mask=None)
    → (s_par, s_perp, s_eq, s_tot)   each (E,)

compute_s_shift(positions, edges, stiffnesses, rest_lengths,
                constrained_nodes=np.array([]))
    → s_shift  (E,)   Frobenius norm of per-edge Jacobian block

Lower-level functions from generalized_susceptibility are also exported
for notebooks that call them directly:
    precompute_geometry, compute_physical_hessian_strained,
    compute_constrained_hessian_inverse, compute_full_jacobian_matrixwise,
    susceptibilities_from_jacobian
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _build_incidence_matrix(edges, n_nodes):
    E = len(edges)
    B = np.zeros((E, n_nodes), dtype=float)
    for e, (i, j) in enumerate(edges):
        B[e, i] = +1.0
        B[e, j] = -1.0
    return B


def _geometry(positions, B, rest_lengths):
    disps = B @ positions
    ells  = np.linalg.norm(disps, axis=1)
    nhats = disps / ells[:, None]
    fs    = 1.0 - rest_lengths / ells
    return disps, ells, nhats, fs


# ---------------------------------------------------------------------------
# Public lower-level functions (re-exported from generalized_susceptibility)
# ---------------------------------------------------------------------------

def precompute_geometry(positions, edges, rest_lengths):
    """
    Precompute edge geometry for repeated susceptibility calls.

    Returns dict with keys: 'B', 'disps', 'ells', 'nhats', 'fs'.
    """
    positions = np.asarray(positions)
    edges     = np.asarray(edges, dtype=int)
    B = _build_incidence_matrix(edges, positions.shape[0])
    disps, ells, nhats, fs = _geometry(positions, B, rest_lengths)
    return {'B': B, 'disps': disps, 'ells': ells, 'nhats': nhats, 'fs': fs}


def compute_physical_hessian_strained(stiffnesses, rest_lengths, edges,
                                      final_positions, constrained_idx_dof=None,
                                      tol=1e-15, force_type='quadratic'):
    """Build the physical (2N, 2N) Hessian at the given configuration."""
    positions    = np.asarray(final_positions, dtype=float)
    stiffnesses  = np.asarray(stiffnesses,     dtype=float)
    rest_lengths = np.asarray(rest_lengths,    dtype=float)
    edges        = np.asarray(edges,           dtype=int)
    n_nodes = positions.shape[0]
    B = _build_incidence_matrix(edges, n_nodes)
    _, _, nhats, fs = _geometry(positions, B, rest_lengths)

    nhats_j = jnp.array(nhats)
    fs_j    = jnp.array(fs)
    k_j     = jnp.array(stiffnesses)
    B_j     = jnp.array(B)

    odiags1 = jnp.einsum('iimn->imn', jnp.einsum('im,jn->ijmn', nhats_j, nhats_j))
    kd      = jnp.eye(2)
    odiags2 = jnp.einsum('i,imn->imn', fs_j,
                          jnp.einsum('i,mn->imn', jnp.ones(len(fs_j)), kd) - odiags1)
    odiags  = odiags1 + odiags2
    H4 = jnp.einsum('i,ia,ib,imn->ambn', k_j, B_j, B_j, odiags)
    return np.array(H4).reshape(2 * n_nodes, 2 * n_nodes)


def compute_constrained_hessian_inverse(positions, edges, stiffnesses, rest_lengths,
                                        constrained_nodes=None):
    """
    Constrained Hessian inverse via Lagrange multipliers.

    Parameters
    ----------
    constrained_nodes : list/array of node indices, or None / empty array
        If empty or None, returns the full (unconstrained) Hessian inverse.

    Returns
    -------
    Hinv : (N, 2, N, 2) ndarray
    """
    positions         = np.asarray(positions,    dtype=float)
    stiffnesses       = np.asarray(stiffnesses,  dtype=float)
    rest_lengths      = np.asarray(rest_lengths, dtype=float)
    edges             = np.asarray(edges,        dtype=int)

    if constrained_nodes is None:
        constrained_nodes = []
    constrained_nodes = list(np.asarray(constrained_nodes, dtype=int).ravel())

    n_nodes = positions.shape[0]
    n_dof   = 2 * n_nodes
    H_flat  = compute_physical_hessian_strained(stiffnesses, rest_lengths, edges, positions)

    if len(constrained_nodes) == 0:
        # Full unconstrained inverse
        H_inv_flat = np.linalg.inv(H_flat)
        return H_inv_flat.reshape(n_nodes, 2, n_nodes, 2)

    n_c = len(constrained_nodes)
    Pi  = np.zeros((2 * n_c, n_dof))
    for idx, node in enumerate(constrained_nodes):
        Pi[2 * idx,     2 * node    ] = 1.0
        Pi[2 * idx + 1, 2 * node + 1] = 1.0

    H_ext = np.block([
        [H_flat,             Pi.T                   ],
        [Pi,                 np.zeros((2*n_c, 2*n_c))],
    ])
    H_ext_inv  = np.linalg.inv(H_ext)
    H_inv_flat = H_ext_inv[:n_dof, :n_dof]
    return H_inv_flat.reshape(n_nodes, 2, n_nodes, 2)


def _construct_long(B_j, nhats_j, Hinv_j):
    return -1.0 * jnp.einsum(
        'amcr,ic,ir,ig,id,dgbn->iambn',
        Hinv_j, B_j, nhats_j, nhats_j, B_j, Hinv_j
    )


def _construct_trans(B_j, nhats_j, stiffnesses_j, fs_j, Hinv_j):
    f  = stiffnesses_j * fs_j
    kd = jnp.eye(2)
    p1 = -1.0 * jnp.einsum('i,amcr,ic,rg,id,dgbn->iambn', f, Hinv_j, B_j, kd, B_j, Hinv_j)
    p2 = jnp.einsum('i,amcr,ic,ir,ig,id,dgbn->iambn', f, Hinv_j, B_j, nhats_j, nhats_j, B_j, Hinv_j)
    return p1 + p2


def _construct_eq(B_j, nhats_j, stiffnesses_j, ells_j, rest_lengths_j, Hinv_j):
    fs_j  = 1.0 - rest_lengths_j / ells_j
    f_j   = stiffnesses_j * fs_j
    kd    = jnp.eye(2)
    ones  = jnp.ones(stiffnesses_j.shape[0])

    sijmn = jnp.einsum('ia,ambn,jb->ijmn', B_j, Hinv_j, B_j)

    Gt1 = jnp.einsum('j,i,j,jn,jimg,ig->ijmn',
                     ones - f_j, ells_j, 1.0/ells_j, nhats_j, sijmn, nhats_j)
    Gt2 = jnp.einsum('j,i,j,jm,jing,ig->ijmn',
                     ones - f_j, ells_j, 1.0/ells_j, nhats_j, sijmn, nhats_j)
    Gt3 = jnp.einsum('j,i,j,jb,jibg,ig,mn->ijmn',
                     ones - f_j, ells_j, 1.0/ells_j, nhats_j, sijmn, nhats_j, kd)
    Gt4 = -3.0 * jnp.einsum('j,i,j,jb,jibg,ig,jm,jn->ijmn',
                             ones - f_j, ells_j, 1.0/ells_j, nhats_j, sijmn, nhats_j, nhats_j, nhats_j)
    G = Gt1 + Gt2 + Gt3 + Gt4

    internal = -1.0 * jnp.einsum('i,j,ja,ijmn,jb->iambn', f_j, stiffnesses_j, B_j, G, B_j)
    return -1.0 * jnp.einsum('amcr,icrdg,dgbn->iambn', Hinv_j, internal, Hinv_j)


def compute_full_jacobian_matrixwise(positions, edges, stiffnesses, rest_lengths,
                                     H_ff_inv, mask=None, d=2, H_full_inv=None):
    """
    Decompose dH^{-1}/dk_i into parallel, transverse, and equilibrium parts.

    Returns (Hjac, Hjac_parts, geom_tuple).
    Hjac_parts keys: 'H1' (parallel), 'H2' (transverse), 'H3' (equilibrium), 'H4' (zeros).
    Each has shape (E, N, 2, N, 2).
    """
    positions    = np.asarray(positions,    dtype=float)
    stiffnesses  = np.asarray(stiffnesses,  dtype=float)
    rest_lengths = np.asarray(rest_lengths, dtype=float)
    edges        = np.asarray(edges,        dtype=int)
    n_nodes      = positions.shape[0]
    B            = _build_incidence_matrix(edges, n_nodes)
    disps, ells, nhats, fs = _geometry(positions, B, rest_lengths)

    if H_full_inv is not None:
        Hinv = np.asarray(H_full_inv, dtype=float).reshape(n_nodes, 2, n_nodes, 2)
    else:
        raise ValueError("H_full_inv must be provided.")

    B_j           = jnp.array(B)
    nhats_j       = jnp.array(nhats)
    stiffnesses_j = jnp.array(stiffnesses)
    rest_lengths_j = jnp.array(rest_lengths)
    ells_j        = jnp.array(ells)
    fs_j          = jnp.array(fs)
    Hinv_j        = jnp.array(Hinv)

    H1  = _construct_long(B_j, nhats_j, Hinv_j)
    H2  = _construct_trans(B_j, nhats_j, stiffnesses_j, fs_j, Hinv_j)
    H3  = _construct_eq(B_j, nhats_j, stiffnesses_j, ells_j, rest_lengths_j, Hinv_j)
    H4  = jnp.zeros_like(H1)
    Hjac = H1 + H2 + H3 + H4
    Hjac_parts = {'H1': H1, 'H2': H2, 'H3': H3, 'H4': H4}

    fs_k       = stiffnesses * fs
    Pp         = np.eye(2)[None, :, :] - nhats[:, :, None] * nhats[:, None, :]
    geom_tuple = (disps, fs_k, ells, nhats, Pp)
    return Hjac, Hjac_parts, geom_tuple


def susceptibilities_from_jacobian(Hjac_parts):
    """
    Extract per-edge scalar susceptibilities from Hjac_parts dict.

    Returns (s_par, s_perp, s_eq, s_tot) each (E,).
    """
    H1, H2, H3, H4 = Hjac_parts['H1'], Hjac_parts['H2'], Hjac_parts['H3'], Hjac_parts['H4']
    s_par  = -np.array(jnp.einsum('iamam->i', H1))
    s_perp = -np.array(jnp.einsum('iamam->i', H2))
    s_eq   = -np.array(jnp.einsum('iamam->i', H3 + H4))
    s_tot  = s_par + s_perp + s_eq
    return s_par, s_perp, s_eq, s_tot


# ---------------------------------------------------------------------------
# High-level public API
# ---------------------------------------------------------------------------

def compute_susceptibilities(positions, edges, stiffnesses, rest_lengths,
                              constrained_nodes=None, mask=None):
    """
    Compute per-edge susceptibility decomposition at the given configuration.

    Parameters
    ----------
    constrained_nodes : array-like or None
        Node indices whose DOFs are fixed. Empty / None → unconstrained inverse.

    Returns
    -------
    s_par, s_perp, s_eq, s_tot : each (E,) float array
    """
    if constrained_nodes is None:
        constrained_nodes = np.array([], dtype=int)
    Hinv = compute_constrained_hessian_inverse(
        positions, edges, stiffnesses, rest_lengths,
        constrained_nodes=constrained_nodes,
    )
    _, Hjac_parts, _ = compute_full_jacobian_matrixwise(
        positions, edges, stiffnesses, rest_lengths,
        H_ff_inv=None, H_full_inv=Hinv,
    )
    return susceptibilities_from_jacobian(Hjac_parts)


def compute_s_shift(positions, edges, stiffnesses, rest_lengths,
                    constrained_nodes=None):
    """
    Per-edge response norm: Frobenius norm of the full Jacobian block.

    s_shift[e] = ||Hjac[e]||_F = ||dH^{-1}/dk_e||_F

    Parameters
    ----------
    constrained_nodes : array-like or None
        Empty / None → unconstrained inverse.

    Returns
    -------
    s_shift : (E,) float array
    """
    if constrained_nodes is None:
        constrained_nodes = np.array([], dtype=int)
    Hinv = compute_constrained_hessian_inverse(
        positions, edges, stiffnesses, rest_lengths,
        constrained_nodes=constrained_nodes,
    )
    Hjac, _, _ = compute_full_jacobian_matrixwise(
        positions, edges, stiffnesses, rest_lengths,
        H_ff_inv=None, H_full_inv=Hinv,
    )
    E      = len(edges)
    Hjac_np = np.array(Hjac).reshape(E, -1)
    return np.linalg.norm(Hjac_np, axis=1)
