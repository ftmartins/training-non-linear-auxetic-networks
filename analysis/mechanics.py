"""Bond-level mechanics: incidence matrix, strains, and stresses."""

import numpy as np


def build_incidence_matrix(edges, n_nodes):
    """
    Build (E, N) incidence matrix B.

    Convention: B[e, edges[e,0]] = +1, B[e, edges[e,1]] = -1.
    """
    edges = np.asarray(edges, dtype=int)
    E = len(edges)
    B = np.zeros((E, n_nodes), dtype=float)
    for e, (i, j) in enumerate(edges):
        B[e, i] = +1.0
        B[e, j] = -1.0
    return B


def bond_strains(positions, edges, rest_lengths):
    """
    Per-edge engineering strain: (ell - L0) / L0.

    Returns (E,) array.
    """
    positions = np.asarray(positions)
    edges = np.asarray(edges, dtype=int)
    rest_lengths = np.asarray(rest_lengths)
    i, j = edges[:, 0], edges[:, 1]
    ells = np.linalg.norm(positions[j] - positions[i], axis=1)
    return (ells - rest_lengths) / rest_lengths


def bond_stresses(positions, edges, stiffnesses, rest_lengths):
    """
    Per-edge stress: k * strain.

    Returns (E,) array.
    """
    strains = bond_strains(positions, edges, rest_lengths)
    return np.asarray(stiffnesses) * strains
