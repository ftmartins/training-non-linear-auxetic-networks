"""
Network creation and manipulation utilities for ensemble training.

This module contains functions for creating elastic networks from packing
objects, cleaning network topology, and identifying boundary nodes.
"""

import numpy as np
from pathlib import Path

from elastic_network import ElasticNetwork
from packing_utils import Packing


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

    # Remove intersections (corners belong to top/bottom only)
    left_nodes = np.setdiff1d(left_nodes, np.concatenate([top_nodes, bottom_nodes]))
    right_nodes = np.setdiff1d(right_nodes, np.concatenate([top_nodes, bottom_nodes]))
    top_nodes = np.setdiff1d(top_nodes, np.concatenate([left_nodes, right_nodes]))
    bottom_nodes = np.setdiff1d(bottom_nodes, np.concatenate([left_nodes, right_nodes]))

    return top_nodes, bottom_nodes, left_nodes, right_nodes


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


def create_auxetic_network(n_nodes, packing_seed, force_type='quadratic', boundary_margin=0.05):
    """
    High-level function to create a clean auxetic network ready for training.

    This function:
    1. Creates a packing with specified seed
    2. Generates the packing (runs dynamics)
    3. Extracts ElasticNetwork from packing
    4. Removes degree-1 nodes
    5. Identifies boundary nodes

    Args:
        n_nodes: Number of nodes for packing
        packing_seed: Random seed for packing generation
        force_type: 'quadratic' or 'quartic'
        boundary_margin: Margin for boundary node detection

    Returns:
        network: ElasticNetwork object (cleaned)
        boundary_dict: Dict with keys 'top', 'bottom', 'left', 'right'
    """
    # Create packing
    packing = Packing(n=n_nodes, dim=2, seed=packing_seed, rfac=0.8)
    packing.generate()

    # Extract network
    network = create_network_from_packing(packing, dim=2)

    # Clean topology (remove dangling nodes)
    network = remove_degree_one_nodes(network)

    # Identify boundary nodes
    top, bottom, left, right = get_square_boundary_nodes(network.positions, boundary_margin)

    # Verify boundaries are disjoint
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
    # Test network creation
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

    # Check disjoint
    all_boundary = [boundaries['top'], boundaries['bottom'], boundaries['left'], boundaries['right']]
    is_disjoint = check_disjoint_sets(all_boundary)
    print(f"\nBoundary sets disjoint: {is_disjoint}")

    print("\nNetwork creation test successful!")
