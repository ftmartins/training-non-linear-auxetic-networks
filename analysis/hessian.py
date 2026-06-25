"""Elastic Hessian computation and spectral analysis."""

import numpy as np
import jax.numpy as jnp

from base.elasticity_tensor import compute_hessian_jax
from .susceptibility import compute_physical_hessian_strained


def compute_hessian(positions, edges, stiffnesses, rest_lengths, force_type='quadratic'):
    """
    Full (2N, 2N) elastic Hessian at the given configuration.

    JAX-differentiable w.r.t. stiffnesses and positions.
    """
    return np.array(compute_hessian_jax(
        jnp.asarray(positions, dtype=float),
        jnp.asarray(edges),
        jnp.asarray(stiffnesses, dtype=float),
        jnp.asarray(rest_lengths, dtype=float),
        force_type=force_type,
    ))


def compute_hessian_spectrum(positions, edges, stiffnesses, rest_lengths,
                             constrained_nodes=None, n_modes=None):
    """
    Eigenspectrum of the constrained elastic Hessian (free-DOF block).

    Parameters
    ----------
    constrained_nodes : array-like or None
        Node indices whose DOFs are fixed. Empty / None → full Hessian.
    n_modes : int or None
        Number of modes to return (ascending by eigenvalue). None → all.

    Returns
    -------
    eigenvalues  : (M,) float array, sorted ascending
    eigenvectors : (2N, M) float array (free-DOF space, zero-padded for constrained DOFs)
    """
    positions         = np.asarray(positions,    dtype=float)
    stiffnesses       = np.asarray(stiffnesses,  dtype=float)
    rest_lengths      = np.asarray(rest_lengths, dtype=float)
    edges             = np.asarray(edges,        dtype=int)

    if constrained_nodes is None:
        constrained_nodes = np.array([], dtype=int)
    constrained_nodes = np.asarray(constrained_nodes, dtype=int).ravel()

    n_nodes = positions.shape[0]
    n_dof   = 2 * n_nodes

    H_flat = compute_physical_hessian_strained(stiffnesses, rest_lengths, edges, positions)

    if len(constrained_nodes) == 0:
        free_dofs = np.arange(n_dof)
    else:
        constrained_dofs = np.concatenate([
            constrained_nodes * 2,
            constrained_nodes * 2 + 1,
        ])
        constrained_dofs = np.unique(constrained_dofs)
        free_dofs = np.setdiff1d(np.arange(n_dof), constrained_dofs)

    H_ff = H_flat[np.ix_(free_dofs, free_dofs)]
    vals, vecs = np.linalg.eigh(H_ff)

    if n_modes is not None:
        vals = vals[:n_modes]
        vecs = vecs[:, :n_modes]

    return vals, vecs, H_ff
