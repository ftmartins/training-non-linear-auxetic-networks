"""
Network creation and manipulation utilities for ensemble training.

This module contains functions for creating elastic networks from packing
objects, cleaning network topology, and identifying boundary nodes.
"""

import numpy as np
from pathlib import Path
from scipy.spatial import ConvexHull, cKDTree

from .elastic_network import ElasticNetwork
from .packing_utils import Packing
from .config import PACKING_PARAMS, PACKING_DURATION, PACKING_FRAMES, BOUNDARY_MARGIN


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


def get_hull_boundary_nodes(positions, margin=None):
    """
    Identify boundary nodes near the convex hull of the point cloud, then
    split them into top/bottom/left/right sides for a quasi-square domain.

    The hull's *vertices* are only the outermost corner points — for a
    jittered lattice, most of a nominally flat side sits slightly behind the
    straight line connecting two adjacent hull vertices (by up to the jitter
    amplitude), so restricting to `hull.vertices` misses most of that side.
    Instead, every node within `margin` of the hull's supporting hyperplanes
    (the facets connecting consecutive hull vertices) is treated as boundary:
    for each facet, `ConvexHull.equations` gives a unit normal `n` and offset
    `c` such that `n . x + c <= 0` for points inside the hull, with equality
    on that facet. `max_facet(n . x + c)` is the signed distance from `x` to
    the nearest supporting hyperplane (~0 for points on the surface, more
    negative for interior points), so thresholding it selects a full boundary
    layer rather than just the hull's corner points.

    Args:
        positions: Node positions array (N, 2)
        margin: Distance tolerance from the hull surface. Defaults to
            0.75x the median nearest-neighbor spacing, which is wide enough to
            catch a full row/column of lattice nodes behind the hull surface
            but narrow enough to exclude interior nodes.

    Returns:
        top_nodes, bottom_nodes, left_nodes, right_nodes: arrays of node indices
    """
    positions = np.array(positions)
    hull = ConvexHull(positions)

    if margin is None:
        tree = cKDTree(positions)
        nn_dist, _ = tree.query(positions, k=2)
        margin = 0.75 * np.median(nn_dist[:, 1])

    dist_to_hull = (positions @ hull.equations[:, :-1].T + hull.equations[:, -1]).max(axis=1)
    boundary_idx = np.where(dist_to_hull > -margin)[0]

    centroid = positions.mean(axis=0)
    rel = positions[boundary_idx] - centroid

    top, bottom, left, right = [], [], [], []
    for node_idx, (dx, dy) in zip(boundary_idx, rel):
        if abs(dy) >= abs(dx):
            (top if dy > 0 else bottom).append(node_idx)
        else:
            (right if dx > 0 else left).append(node_idx)

    return (np.array(sorted(top)), np.array(sorted(bottom)),
            np.array(sorted(left)), np.array(sorted(right)))


def create_lattice_square_network(L, p, R, seed=None, dilution=0.05):
    """
    Build an ElasticNetwork from a jittered triangular lattice cut to a
    quasi-square domain, with boundary nodes assigned via the convex hull.

    Same lattice + noise construction as
    `training/runners/allosteric_trainer.py::create_network` (triangular
    lattice with basis vectors a1=(1,0), a2=(0.5, sqrt(3)/2), each node
    perturbed by uniform noise of amplitude `p`), but nodes are windowed to
    a square (max(|x|, |y|) < L/2) instead of a circle, giving four flat
    sides suitable for top/bottom/left/right boundary training.

    A random `dilution` fraction of the lattice (distance-cutoff) bonds is
    removed before any non-lattice bonds are added, so the diluted lattice is
    the base topology that later disorder/rewiring steps build on top of.

    Args:
        L: Lattice size parameter (domain side length is ~L)
        p: Disorder amplitude (uniform jitter on each node position)
        R: Distance cutoff for bond formation
        seed: Optional RNG seed for reproducibility
        dilution: Fraction of lattice bonds to randomly remove (default 0.05)

    Returns:
        network: ElasticNetwork object
        boundary_dict: Dict with keys 'top', 'bottom', 'left', 'right'
            (convex-hull-derived boundary node indices)
    """
    rng = np.random.RandomState(seed)
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3) / 2.0])

    n_candidates = 2 * L ** 2
    moves = (rng.rand(n_candidates, 2) - 0.5) * 2 * p

    nodes = []
    for xidx in range(L):
        for yidx in range(int((2 / np.sqrt(3)) * L)):
            node = ((xidx - int((L / 2) * (1 - 1 / np.sqrt(3))) - np.floor(yidx / 2)) * a1
                    + (yidx - int((1 / np.sqrt(3)) * L)) * a2
                    + moves[len(nodes)])
            if np.max(np.abs(node)) < L / 2:
                nodes.append(node)
    nodes = np.array(nodes)

    edges = []
    for i in range(len(nodes)):
        for j in range(i):
            if np.linalg.norm(nodes[i] - nodes[j]) < R:
                edges.append((j, i))
    edges = np.array(edges)

    if dilution > 0 and len(edges) > 0:
        n_remove = int(round(dilution * len(edges)))
        remove_idx = rng.choice(len(edges), size=n_remove, replace=False)
        edges = np.delete(edges, remove_idx, axis=0)

    network = ElasticNetwork(positions=nodes, edges=edges, stiffnesses=np.ones(len(edges)))
    network.save_original_parameters()

    top, bottom, left, right = get_hull_boundary_nodes(nodes)
    if not check_disjoint_sets([top, bottom, left, right]):
        print("Warning: Boundary node sets are not disjoint!")

    boundary_dict = {'top': top, 'bottom': bottom, 'left': left, 'right': right}
    return network, boundary_dict


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
                           boundary_margin=BOUNDARY_MARGIN, central_force=None,
                           network_type='jammed', lattice_jitter=0.15,
                           lattice_cutoff=1.6, lattice_dilution=0.05):
    """
    High-level function to create a clean auxetic network ready for training.

    Supports two network generation methods, selected via `network_type`:
      - 'jammed' (default): generates an oversampled jammed packing
        (~n_nodes * pi/2 particles) and cuts a square region of n_nodes
        nodes from it.
      - 'lattice': generates an oversampled perturbed (jittered) triangular
        lattice square and cuts it down to n_nodes nodes.

    In both cases the cut network has degree-1 nodes removed and boundary
    nodes identified via the convex hull (`get_hull_boundary_nodes`).

    Args:
        n_nodes: Target number of nodes in the final square network
        packing_seed: Random seed for network generation (packing or lattice)
        force_type: 'quadratic' or 'quartic' (reserved for future use)
        boundary_margin: Distance tolerance for hull-based boundary node
            detection (passed to `get_hull_boundary_nodes`). If None, an
            automatic margin based on nearest-neighbor spacing is used.
        central_force: Override for the central compression force in the
            packing dynamics. Only used when network_type='jammed'. Defaults
            to PACKING_PARAMS['central'] from config. Smaller values yield
            less hexagonal, more disordered networks.
        network_type: 'jammed' or 'lattice' — selects the network generation
            method (see above).
        lattice_jitter: Jitter amplitude for lattice node positions. Only
            used when network_type='lattice'.
        lattice_cutoff: Distance cutoff for lattice bond formation. Only
            used when network_type='lattice'.
        lattice_dilution: Fraction of lattice bonds randomly removed before
            cutting to n_nodes. Only used when network_type='lattice'.

    Returns:
        network: ElasticNetwork object (cleaned, square domain)
        boundary_dict: Dict with keys 'top', 'bottom', 'left', 'right'
    """
    if network_type == 'jammed':
        params = dict(PACKING_PARAMS)
        if central_force is not None:
            params['central'] = central_force

        # Oversample so the inscribed square contains ~n_nodes nodes (pi/2 factor)
        n_big = int(np.ceil(n_nodes * np.pi / 2))
        packing = Packing(n=n_big, dim=2, seed=packing_seed, rfac=0.8, params=params)
        packing.generate(duration=PACKING_DURATION, frames=PACKING_FRAMES)
        network = create_network_from_packing(packing, dim=2)
    elif network_type == 'lattice':
        # Oversample the lattice side length so the square cut below has
        # n_nodes to choose from. Unit-spacing triangular lattice density
        # is 2/sqrt(3) nodes per unit area.
        density = 2.0 / np.sqrt(3.0)
        l_target = np.sqrt(n_nodes / density)
        l_big = int(np.ceil(l_target * 1.25)) + 2
        network, _ = create_lattice_square_network(
            l_big, lattice_jitter, lattice_cutoff,
            seed=packing_seed, dilution=lattice_dilution
        )
    else:
        raise ValueError(f"Unknown network_type: {network_type!r} (expected 'jammed' or 'lattice')")

    # Cut square region, then clean topology
    network = cut_square_from_packing(network, n_nodes)
    network = remove_degree_one_nodes(network)

    # Identify boundary nodes via the convex hull
    top, bottom, left, right = get_hull_boundary_nodes(network.positions, boundary_margin)

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
