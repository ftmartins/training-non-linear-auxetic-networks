"""
generalized_susceptibility.py

Reimplementation of susceptibility analysis from ConstructingSusceptibilities_Stephen.py.

The susceptibility decomposes dH^{-1}/dk_i into three physical contributions:
    s_parallel (H1): longitudinal — edge stretching response
    s_perp     (H2): transverse  — edge bending / shear response
    s_eq       (H3+H4): equilibrium coupling — response mediated by position equilibration

Public API
----------
precompute_geometry(positions, edges, rest_lengths) -> dict
compute_physical_hessian_strained(stiffnesses, rest_lengths, edges, final_positions,
                                   constrained_idx_dof=None, tol=1e-15,
                                   force_type='quadratic') -> ndarray (2N, 2N)
compute_constrained_hessian_inverse(positions, edges, stiffnesses, rest_lengths,
                                     constrained_nodes) -> ndarray (N, 2, N, 2)
compute_full_jacobian_matrixwise(positions, edges, stiffnesses, rest_lengths,
                                  H_ff_inv, mask=None, d=2,
                                  H_full_inv=None) -> (Hjac, Hjac_parts, geom_tuple)

Usage in ensemble_correlation_analysis.ipynb
--------------------------------------------
    from generalized_susceptibility import (
        compute_physical_hessian_strained,
        compute_full_jacobian_matrixwise,
        precompute_geometry,
    )
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_incidence_matrix(edges, n_nodes):
    """
    Build (E, N) incidence matrix B from (E, 2) edge array.

    Convention: B[e, edges[e,0]] = +1, B[e, edges[e,1]] = -1.
    """
    E = len(edges)
    B = np.zeros((E, n_nodes), dtype=float)
    for e, (i, j) in enumerate(edges):
        B[e, i] = +1.0
        B[e, j] = -1.0
    return B


def _geometry(positions, B, rest_lengths):
    """
    Compute edge geometry from node positions and incidence matrix.

    Returns
    -------
    disps : (E, 2)  edge displacement vectors  (B @ positions)
    ells  : (E,)    current edge lengths
    nhats : (E, 2)  unit vectors along edges
    fs    : (E,)    scalar stretch  1 - L0 / ell  (dimensionless, NOT multiplied by k)
    """
    disps = B @ positions                                       # (E, 2)
    ells  = np.linalg.norm(disps, axis=1)                      # (E,)
    nhats = disps / ells[:, None]                               # (E, 2)
    fs    = 1.0 - rest_lengths / ells                           # (E,)
    return disps, ells, nhats, fs


# ---------------------------------------------------------------------------
# Public: precompute_geometry
# ---------------------------------------------------------------------------

def precompute_geometry(positions, edges, rest_lengths):
    """
    Precompute and cache edge geometry for repeated susceptibility calls.

    Parameters
    ----------
    positions    : (N, 2) node positions
    edges        : (E, 2) edge connectivity
    rest_lengths : (E,)   rest lengths

    Returns
    -------
    dict with keys:
        'B'      : (E, N) incidence matrix
        'disps'  : (E, 2) edge displacement vectors
        'ells'   : (E,)   current edge lengths
        'nhats'  : (E, 2) unit vectors
        'fs'     : (E,)   scalar stretch 1 - L0/ell
    """
    positions = np.asarray(positions)
    edges     = np.asarray(edges, dtype=int)
    n_nodes   = positions.shape[0]

    B = _build_incidence_matrix(edges, n_nodes)
    disps, ells, nhats, fs = _geometry(positions, B, rest_lengths)

    return {'B': B, 'disps': disps, 'ells': ells, 'nhats': nhats, 'fs': fs}


# ---------------------------------------------------------------------------
# Public: compute_physical_hessian_strained
# ---------------------------------------------------------------------------

def compute_physical_hessian_strained(
    stiffnesses,
    rest_lengths,
    edges,
    final_positions,
    constrained_idx_dof=None,
    tol=1e-15,
    force_type='quadratic',
):
    """
    Build the physical (unconstrained) Hessian of the elastic energy at the
    given configuration and reshape to (2N, 2N).

    Implements Stephen's buildH, reshaped for direct numpy usage.

    Parameters
    ----------
    stiffnesses        : (E,)
    rest_lengths       : (E,)
    edges              : (E, 2)
    final_positions    : (N, 2)
    constrained_idx_dof: list of DOF indices that are constrained (used externally
                         to build free-DOF blocks; not applied here)
    tol                : numerical floor (unused, kept for API compatibility)
    force_type         : 'quadratic' (only supported type; harmonic springs)

    Returns
    -------
    H_flat : (2N, 2N) ndarray
        Physical Hessian reshaped from (N, 2, N, 2).
    """
    positions    = np.asarray(final_positions, dtype=float)
    stiffnesses  = np.asarray(stiffnesses,     dtype=float)
    rest_lengths = np.asarray(rest_lengths,    dtype=float)
    edges        = np.asarray(edges,           dtype=int)

    n_nodes = positions.shape[0]
    B       = _build_incidence_matrix(edges, n_nodes)
    _, _, nhats, fs = _geometry(positions, B, rest_lengths)

    # odiags[e, m, n] = nhats[e,m]*nhats[e,n] + fs[e]*(delta_mn - nhats[e,m]*nhats[e,n])
    # H[a,m,b,n] = sum_e k_e * B[e,a]*B[e,b] * odiags[e,m,n]
    nhats_j  = jnp.array(nhats)
    fs_j     = jnp.array(fs)
    k_j      = jnp.array(stiffnesses)
    B_j      = jnp.array(B)

    # outer product of nhats with itself: (E, E, 2, 2) → diagonal (E, 2, 2)
    odiags1 = jnp.einsum('iimn->imn', jnp.einsum('im,jn->ijmn', nhats_j, nhats_j))
    # transverse projector part
    kd      = jnp.eye(2)
    odiags2 = jnp.einsum('i,imn->imn',
                          fs_j,
                          jnp.einsum('i,mn->imn', jnp.ones(len(fs_j)), kd) - odiags1)
    odiags  = odiags1 + odiags2                                # (E, 2, 2)

    # Hessian (N, 2, N, 2)
    H4 = jnp.einsum('i,ia,ib,imn->ambn', k_j, B_j, B_j, odiags)
    H_flat = np.array(H4).reshape(2 * n_nodes, 2 * n_nodes)

    return H_flat


# ---------------------------------------------------------------------------
# Public: compute_constrained_hessian_inverse
# ---------------------------------------------------------------------------

def compute_constrained_hessian_inverse(
    positions,
    edges,
    stiffnesses,
    rest_lengths,
    constrained_nodes,
):
    """
    Build the constrained Hessian inverse via Lagrange multipliers.

    Generalises Stephen's buildHinvC to an arbitrary list of constrained
    nodes (all 2 DOFs of each node are fixed).

    Parameters
    ----------
    positions         : (N, 2)
    edges             : (E, 2)
    stiffnesses       : (E,)
    rest_lengths      : (E,)
    constrained_nodes : list/array of node indices whose DOFs are fixed

    Returns
    -------
    Hinv : (N, 2, N, 2)  constrained Hessian inverse
    """
    positions         = np.asarray(positions,         dtype=float)
    stiffnesses       = np.asarray(stiffnesses,       dtype=float)
    rest_lengths      = np.asarray(rest_lengths,      dtype=float)
    edges             = np.asarray(edges,             dtype=int)
    constrained_nodes = list(constrained_nodes)

    n_nodes = positions.shape[0]
    n_dof   = 2 * n_nodes
    n_c     = len(constrained_nodes)

    H_flat = compute_physical_hessian_strained(
        stiffnesses, rest_lengths, edges, positions
    )

    # Build Pi projectors: one 2-DOF block per constrained node
    Pi = np.zeros((2 * n_c, n_dof))
    for idx, node in enumerate(constrained_nodes):
        Pi[2 * idx,     2 * node    ] = 1.0
        Pi[2 * idx + 1, 2 * node + 1] = 1.0

    zero_block = np.zeros((2 * n_c, 2 * n_c))
    H_ext = np.block([
        [H_flat,      Pi.T        ],
        [Pi,          zero_block  ],
    ])

    H_ext_inv = np.linalg.inv(H_ext)
    H_inv_flat = H_ext_inv[:n_dof, :n_dof]

    return H_inv_flat.reshape(n_nodes, 2, n_nodes, 2)


# ---------------------------------------------------------------------------
# Internal susceptibility part builders (follow Stephen's code faithfully)
# ---------------------------------------------------------------------------

def _construct_long(B_j, nhats_j, Hinv_j):
    """
    Parallel (longitudinal) contribution to dH^{-1}/dk_i.

    Implements constructLong from ConstructingSusceptibilities_Stephen.py.

    dHinvpar[i,a,m,b,n] = -Hinv[a,m,c,r] * B[i,c] * n[i,r] * n[i,g] * B[i,d] * Hinv[d,g,b,n]

    Returns
    -------
    dHinvpar : (E, N, 2, N, 2)
    """
    return -1.0 * jnp.einsum(
        'amcr,ic,ir,ig,id,dgbn->iambn',
        Hinv_j, B_j, nhats_j, nhats_j, B_j, Hinv_j
    )


def _construct_trans(B_j, nhats_j, stiffnesses_j, fs_j, Hinv_j):
    """
    Transverse (perpendicular) contribution to dH^{-1}/dk_i.

    Implements constructTrans from ConstructingSusceptibilities_Stephen.py.
    Uses f_i = k_i * (1 - L0_i/ell_i)  (note: includes stiffness factor).

    Returns
    -------
    dHinvperp : (E, N, 2, N, 2)
    """
    f = stiffnesses_j * fs_j                                   # (E,), includes k factor
    kd = jnp.eye(2)

    part1 = -1.0 * jnp.einsum(
        'i,amcr,ic,rg,id,dgbn->iambn',
        f, Hinv_j, B_j, kd, B_j, Hinv_j
    )
    part2 = jnp.einsum(
        'i,amcr,ic,ir,ig,id,dgbn->iambn',
        f, Hinv_j, B_j, nhats_j, nhats_j, B_j, Hinv_j
    )
    return part1 + part2


def _construct_eq(B_j, nhats_j, stiffnesses_j, ells_j, rest_lengths_j, Hinv_j):
    """
    Equilibrium-coupling contribution to dH^{-1}/dk_i.

    Implements constructEq from ConstructingSusceptibilities_Stephen.py.
    Accounts for the change in equilibrium positions when k_i is varied.

    Returns
    -------
    dHinveq : (E, N, 2, N, 2)
    """
    fs_j  = 1.0 - rest_lengths_j / ells_j                     # (E,), scalar stretch
    f_j   = stiffnesses_j * fs_j                               # (E,), prestress per unit length * k
    kd    = jnp.eye(2)
    ones  = jnp.ones(stiffnesses_j.shape[0])

    # sijmn[i,j,m,n] = B[i,a] * Hinv[a,m,b,n] * B[j,b]
    sijmn = jnp.einsum('ia,ambn,jb->ijmn', B_j, Hinv_j, B_j)   # (E, E, 2, 2)

    # G tensor: geometric coupling at equilibrium
    # Four terms Gt1..Gt4 follow Stephen's code exactly
    Gt1 = jnp.einsum(
        'j,i,j,jn,jimg,ig->ijmn',
        ones - fs_j, ells_j, 1.0 / ells_j, nhats_j, sijmn, nhats_j
    )
    Gt2 = jnp.einsum(
        'j,i,j,jm,jing,ig->ijmn',
        ones - fs_j, ells_j, 1.0 / ells_j, nhats_j, sijmn, nhats_j
    )
    Gt3 = jnp.einsum(
        'j,i,j,jb,jibg,ig,mn->ijmn',
        ones - fs_j, ells_j, 1.0 / ells_j, nhats_j, sijmn, nhats_j, kd
    )
    Gt4 = -3.0 * jnp.einsum(
        'j,i,j,jb,jibg,ig,jm,jn->ijmn',
        ones - fs_j, ells_j, 1.0 / ells_j, nhats_j, sijmn, nhats_j, nhats_j, nhats_j
    )
    G = Gt1 + Gt2 + Gt3 + Gt4                                  # (E, E, 2, 2)

    # Internal force change: dH/dk_i contribution from equilibrium shift
    internal = -1.0 * jnp.einsum(
        'i,j,ja,ijmn,jb->iambn',
        f_j, stiffnesses_j, B_j, G, B_j
    )                                                           # (E, N, 2, N, 2)

    # Propagate through H^{-1}: -Hinv * internal * Hinv
    dHinveq = -1.0 * jnp.einsum(
        'amcr,icrdg,dgbn->iambn',
        Hinv_j, internal, Hinv_j
    )
    return dHinveq


# ---------------------------------------------------------------------------
# Public: compute_full_jacobian_matrixwise
# ---------------------------------------------------------------------------

def compute_full_jacobian_matrixwise(
    positions,
    edges,
    stiffnesses,
    rest_lengths,
    H_ff_inv,
    mask=None,
    d=2,
    H_full_inv=None,
):
    """
    Decompose dH^{-1}/dk_i into parallel, transverse and equilibrium parts.

    This is the primary susceptibility function called by the analysis notebooks.

    Parameters
    ----------
    positions    : (N, 2) node positions at the strained equilibrium
    edges        : (E, 2) edge connectivity
    stiffnesses  : (E,)   spring stiffnesses
    rest_lengths : (E,)   spring rest lengths
    H_ff_inv     : (N_free, N_free) inverse of the free-DOF Hessian block
                   (used as fallback if H_full_inv is not provided)
    mask         : (2N,) boolean, True = free DOF (unused in core computation,
                   kept for API compatibility)
    d            : spatial dimension (must be 2)
    H_full_inv   : (N, 2, N, 2) full constrained Hessian inverse.
                   *This is the tensor used for all susceptibility computations.*
                   If None, a zero-padded version is built from H_ff_inv + mask.

    Returns
    -------
    Hjac         : (E, N, 2, N, 2)  total dH^{-1}/dk_i
    Hjac_parts   : dict with keys 'H1', 'H2', 'H3', 'H4'
                   H1 = parallel, H2 = transverse, H3+H4 = equilibrium
                   Each has shape (E, N, 2, N, 2); reshape(E, 2N, 2N) for trace.
    geom_tuple   : (disps, fs_k, ells, nhats, Pp)
                   disps : (E, 2) edge vectors
                   fs_k  : (E,)  prestress = k*(1-L0/ell)
                   ells  : (E,)  edge lengths
                   nhats : (E, 2) unit vectors
                   Pp    : (E, 2, 2) perpendicular projectors I - n⊗n
    """
    positions    = np.asarray(positions,    dtype=float)
    stiffnesses  = np.asarray(stiffnesses,  dtype=float)
    rest_lengths = np.asarray(rest_lengths, dtype=float)
    edges        = np.asarray(edges,        dtype=int)

    n_nodes = positions.shape[0]
    B       = _build_incidence_matrix(edges, n_nodes)
    disps, ells, nhats, fs = _geometry(positions, B, rest_lengths)

    # ----------------------------------------------------------------
    # Build full constrained Hessian inverse in (N, 2, N, 2) form
    # ----------------------------------------------------------------
    if H_full_inv is not None:
        Hinv = np.asarray(H_full_inv, dtype=float).reshape(n_nodes, 2, n_nodes, 2)
    else:
        # Fall back: embed the free-DOF inverse into the full (N,2,N,2) tensor
        # (constrained DOFs contribute zeros)
        if mask is None:
            raise ValueError(
                "Either H_full_inv or (H_ff_inv, mask) must be provided."
            )
        raise NotImplementedError(
            "Fallback to H_ff_inv + mask is not implemented. Please provide H_full_inv directly."
        )
        # mask = np.asarray(mask, dtype=bool)
        # n_dof = 2 * n_nodes
        # H_inv_full_flat = np.zeros((n_dof, n_dof), dtype=float)
        # free_idx = np.where(mask)[0]
        # H_inv_full_flat[np.ix_(free_idx, free_idx)] = np.asarray(H_ff_inv)
        # Hinv = H_inv_full_flat.reshape(n_nodes, 2, n_nodes, 2)

    # ----------------------------------------------------------------
    # JAX arrays
    # ----------------------------------------------------------------
    B_j           = jnp.array(B)
    nhats_j       = jnp.array(nhats)
    stiffnesses_j = jnp.array(stiffnesses)
    rest_lengths_j = jnp.array(rest_lengths)
    ells_j        = jnp.array(ells)
    fs_j          = jnp.array(fs)
    Hinv_j        = jnp.array(Hinv)

    # ----------------------------------------------------------------
    # Three susceptibility parts
    # ----------------------------------------------------------------
    H1 = _construct_long(B_j, nhats_j, Hinv_j)                 # (E, N, 2, N, 2)
    H2 = _construct_trans(B_j, nhats_j, stiffnesses_j, fs_j, Hinv_j)
    Heq = _construct_eq(B_j, nhats_j, stiffnesses_j, ells_j, rest_lengths_j, Hinv_j)
    # Split equilibrium into H3 + H4 (H3 = Heq, H4 = zeros kept for API compat)
    H3 = Heq
    H4 = jnp.zeros_like(H1)

    Hjac_parts = {'H1': H1, 'H2': H2, 'H3': H3, 'H4': H4}
    Hjac       = H1 + H2 + H3 + H4

    # ----------------------------------------------------------------
    # Geometry tuple (Δ, V, l, n, Pp)
    # ----------------------------------------------------------------
    fs_k = stiffnesses * fs                                     # prestress k*(1-L0/ell)
    Pp   = np.eye(2)[None, :, :] - nhats[:, :, None] * nhats[:, None, :]  # (E,2,2)
    geom_tuple = (disps, fs_k, ells, nhats, Pp)

    return Hjac, Hjac_parts, geom_tuple


# ---------------------------------------------------------------------------
# Convenience: extract scalar susceptibilities from Hjac_parts
# ---------------------------------------------------------------------------

def susceptibilities_from_jacobian(Hjac_parts):
    """
    Compute per-edge scalar susceptibilities from Hjac_parts.

    s_x[i] = -trace(Hjac_parts['Hx'][i].reshape(2N, 2N))
            = -einsum('iamam->i', Hjac_parts['Hx'])

    Returns
    -------
    s_par  : (E,) parallel susceptibility
    s_perp : (E,) transverse susceptibility
    s_eq   : (E,) equilibrium susceptibility
    s_tot  : (E,) total susceptibility
    """
    H1 = Hjac_parts['H1']
    H2 = Hjac_parts['H2']
    H3 = Hjac_parts['H3']
    H4 = Hjac_parts['H4']

    s_par  = -np.array(jnp.einsum('iamam->i', H1))
    s_perp = -np.array(jnp.einsum('iamam->i', H2))
    s_eq   = -np.array(jnp.einsum('iamam->i', H3 + H4))
    s_tot  = s_par + s_perp + s_eq

    return s_par, s_perp, s_eq, s_tot
