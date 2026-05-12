"""
Network creation and manipulation utilities for ensemble training.

This module contains functions for creating elastic networks from packing
objects, cleaning network topology, and identifying boundary nodes.
"""

import numpy as np
from pathlib import Path

from elastic_network import ElasticNetwork
from packing_utils import Packing
from config import PACKING_PARAMS, PACKING_DURATION, PACKING_FRAMES, BOUNDARY_MARGIN


def create_network_from_packing(packing_object, dim=2):
    """
    Create an ElasticNetwork from a Packing object.

    Args:
        packing_object: Packing object with generated graph
        dim: Spatial dimension (2 or 3)

    Returns:
        network: ElasticNetwork object
    """
    positions = []
    for i in range(len(packing_object.graph.nodes())):
        pos = packing_object.graph.nodes[i]['pos']
        if dim == 3:
            positions.append([pos[0], pos[1], pos[2] if len(pos) > 2 else 0.0])
        else:
            positions.append([pos[0], pos[1]])

    edges = list(packing_object.graph.edges())
    stiffnesses = []

    for edge in packing_object.graph.edges(data=True):
        stiffnesses.append(edge[2].get('stiffness', 1.0))

    net = ElasticNetwork(
        positions=positions,
        edges=edges,
        stiffnesses=stiffnesses,
    )
    net.save_original_parameters()
    return net


def remove_degree_one_nodes(network):
    """
    Iteratively remove degree-1 (dangling) nodes from the network.

    Args:
        network: ElasticNetwork object

    Returns:
        new_network: ElasticNetwork with degree-1 nodes removed
    """
    positions = np.array(network.positions)
    edges = np.array(network.edges)
    stiffnesses = np.array(network.stiffnesses)
    rest_lengths = np.array(network.rest_lengths) if hasattr(network, 'rest_lengths') else None
    n_nodes = positions.shape[0]

    while True:
        # Compute degree for each node
        degree = np.zeros(n_nodes, dtype=int)
        for i, j in edges:
            degree[i] += 1
            degree[j] += 1

        # Find degree-1 nodes
        degree_one_nodes = np.where(degree == 1)[0]
        if len(degree_one_nodes) == 0:
            break  # No more degree-1 nodes

        # Remove degree-1 nodes and associated edges
        mask_nodes = np.ones(n_nodes, dtype=bool)
        mask_nodes[degree_one_nodes] = False

        # Remap node indices
        new_indices = np.cumsum(mask_nodes) - 1

        # Filter positions
        positions = positions[mask_nodes]
        n_nodes = positions.shape[0]

        # Filter edges (keep only edges where both nodes survive)
        mask_edges = np.array([(mask_nodes[i] and mask_nodes[j]) for i, j in edges])
        edges = edges[mask_edges]

        # Remap edge indices to new node numbering
        edges = np.array([(new_indices[i], new_indices[j]) for i, j in edges])

        # Filter stiffnesses and rest_lengths
        stiffnesses = stiffnesses[mask_edges]
        if rest_lengths is not None:
            rest_lengths = rest_lengths[mask_edges]

    # Create new network object
    new_network = ElasticNetwork(
        positions=positions,
        edges=edges,
        stiffnesses=stiffnesses,
        rest_lengths=rest_lengths if rest_lengths is not None else None,
    )
    return new_network


def get_square_boundary_nodes(positions, margin):
    """
    Identify boundary nodes for a square domain.

    Identifies top, bottom, left, and right boundary nodes based on
    proximity to domain boundaries (within margin).

    Args:
        positions: Node positions array (N, 2)
        margin: Tolerance for boundary detection

    Returns:
        top_nodes: Array of top boundary node indices
        bottom_nodes: Array of bottom boundary node indices
        left_nodes: Array of left boundary node indices
        right_nodes: Array of right boundary node indices
    """
    positions = np.array(positions)
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()

    # Find nodes close to each boundary
    top_nodes = np.where(np.isclose(positions[:, 1], y_max, atol=margin))[0]
    bottom_nodes = np.where(np.isclose(positions[:, 1], y_min, atol=margin))[0]
    left_nodes = np.where(np.isclose(positions[:, 0], x_min, atol=margin))[0]
    right_nodes = np.where(np.isclose(positions[:, 0], x_max, atol=margin))[0]

    # Any node that appears in more than one set is removed from all sets it belongs to
    all_nodes = np.concatenate([top_nodes, bottom_nodes, left_nodes, right_nodes])
    unique_vals, counts = np.unique(all_nodes, return_counts=True)
    shared = unique_vals[counts > 1]

    top_nodes = np.setdiff1d(top_nodes, shared)
    bottom_nodes = np.setdiff1d(bottom_nodes, shared)
    left_nodes = np.setdiff1d(left_nodes, shared)
    right_nodes = np.setdiff1d(right_nodes, shared)

    return top_nodes, bottom_nodes, left_nodes, right_nodes


def cut_square_from_packing(network, n_nodes):
    """
    Cut an approximately square subnetwork of n_nodes from a larger packing.

    Selects the n_nodes nodes closest to the centroid in the Chebyshev (L-infinity)
    metric, which defines a square selection region. Edges with both endpoints
    outside the square are removed and node indices are remapped.

    Args:
        network: ElasticNetwork object (typically oversampled)
        n_nodes: Target number of nodes in the square cutout

    Returns:
        new_network: ElasticNetwork with n_nodes nodes in a square region
    """
    positions = np.array(network.positions)
    edges = np.array(network.edges)
    stiffnesses = np.array(network.stiffnesses)

    cx = positions[:, 0].mean()
    cy = positions[:, 1].mean()

    # Chebyshev distance from centroid selects a square neighbourhood
    cheby = np.maximum(np.abs(positions[:, 0] - cx), np.abs(positions[:, 1] - cy))
    n_keep = min(n_nodes, len(positions))
    keep_idx = np.sort(np.argsort(cheby)[:n_keep])

    new_index = np.full(len(positions), -1, dtype=int)
    new_index[keep_idx] = np.arange(n_keep)

    new_positions = positions[keep_idx]

    if len(edges) > 0:
        keep_set = set(keep_idx.tolist())
        edge_mask = np.array([i in keep_set and j in keep_set for i, j in edges])
        new_edges = np.array([(new_index[i], new_index[j]) for i, j in edges[edge_mask]])
        new_stiffnesses = stiffnesses[edge_mask]
    else:
        new_edges = np.empty((0, 2), dtype=int)
        new_stiffnesses = np.array([])

    return ElasticNetwork(
        positions=new_positions,
        edges=new_edges,
        stiffnesses=new_stiffnesses,
    )


def check_disjoint_sets(sets):
    """
    Check that all sets of nodes are disjoint (no overlaps).

    Args:
        sets: List of node index arrays

    Returns:
        is_disjoint: Boolean indicating if all sets are disjoint
    """
    all_elements = np.concatenate(sets)
    unique_elements = np.unique(all_elements)
    return len(all_elements) == len(unique_elements)


def create_auxetic_network(n_nodes, packing_seed, force_type='quadratic',
                           boundary_margin=BOUNDARY_MARGIN, central_force=None):
    """
    High-level function to create a clean auxetic network ready for training.

    Generates an oversampled packing (~n_nodes * pi/2 particles), cuts a square
    region of approximately n_nodes nodes using Chebyshev-distance selection,
    removes degree-1 nodes, and identifies boundary nodes.

    Args:
        n_nodes: Target number of nodes in the final square network
        packing_seed: Random seed for packing generation
        force_type: 'quadratic' or 'quartic' (reserved for future use)
        boundary_margin: Tolerance for boundary node detection
        central_force: Override for the central compression force in the packing
            dynamics. Defaults to PACKING_PARAMS['central'] from config.
            Smaller values yield less hexagonal, more disordered networks.

    Returns:
        network: ElasticNetwork object (cleaned, square domain)
        boundary_dict: Dict with keys 'top', 'bottom', 'left', 'right'
    """
    params = dict(PACKING_PARAMS)
    if central_force is not None:
        params['central'] = central_force

    # Oversample so the inscribed square contains ~n_nodes nodes (pi/2 factor)
    n_big = int(np.ceil(n_nodes * np.pi / 2))
    packing = Packing(n=n_big, dim=2, seed=packing_seed, rfac=0.8, params=params)
    packing.generate(duration=PACKING_DURATION, frames=PACKING_FRAMES)

    # Extract full network, cut square, then clean topology
    network = create_network_from_packing(packing, dim=2)
    network = cut_square_from_packing(network, n_nodes)
    network = remove_degree_one_nodes(network)

    # Identify boundary nodes
    top, bottom, left, right = get_square_boundary_nodes(network.positions, boundary_margin)

    if not check_disjoint_sets([top, bottom, left, right]):
        print("Warning: Boundary node sets are not disjoint!")

    boundary_dict = {
        'top': top,
        'bottom': bottom,
        'left': left,
        'right': right
    }

    return network, boundary_dict


if __name__ == '__main__':
    print("Testing network creation...")

    test_seed = 0
    test_n_nodes = 100

    print(f"\nCreating network with {test_n_nodes} nodes, seed={test_seed}")
    network, boundaries = create_auxetic_network(test_n_nodes, test_seed)

    print(f"\nNetwork properties:")
    print(f"  Nodes: {len(network.positions)}")
    print(f"  Edges: {len(network.edges)}")
    print(f"  Boundary nodes:")
    print(f"    Top: {len(boundaries['top'])}")
    print(f"    Bottom: {len(boundaries['bottom'])}")
    print(f"    Left: {len(boundaries['left'])}")
    print(f"    Right: {len(boundaries['right'])}")
    print(f"  Total boundary: {sum(len(boundaries[k]) for k in boundaries)}")
    print(f"  Interior: {len(network.positions) - sum(len(boundaries[k]) for k in boundaries)}")

    all_boundary = [boundaries['top'], boundaries['bottom'], boundaries['left'], boundaries['right']]
    print(f"\nBoundary sets disjoint: {check_disjoint_sets(all_boundary)}")
    print("\nNetwork creation test successful!")
