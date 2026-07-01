"""Quasistatic actuation trajectories and trajectory I/O."""

import numpy as np
from pathlib import Path

from base.simulate import compute_quasistatic_trajectory_auxetic, fire_minimize_dof


def compute_auxetic_trajectory(network, compression_strain, boundary, n_steps=100,
                               force_type='quadratic', tol=1e-6, method='fire'):
    """
    Quasistatic compression trajectory for an auxetic (Poisson-ratio) task.

    Parameters
    ----------
    network : ElasticNetwork
    compression_strain : float   (e.g. -0.08 for 8% compression)
    boundary : dict with keys 'top', 'bottom', 'left', 'right' (node index arrays)
    n_steps : int
    force_type : str
    tol : float   convergence tolerance
    method : str  'fire' (Cython FIRE) or 'newton' (Newton-Raphson, faster)

    Returns
    -------
    list of (N, 2) position arrays, length n_steps
    """
    top    = boundary['top']
    bottom = boundary['bottom']
    return compute_quasistatic_trajectory_auxetic(
        network, compression_strain, top, bottom,
        n_steps=n_steps, force_type=force_type, tol=tol, method=method,
    )


def compute_allosteric_trajectory(network, input_node, output_node, displacement,
                                  n_steps=100, force_type='quadratic', tol=1e-6):
    """
    Quasistatic allosteric trajectory: displace input_node, measure output_node.

    Parameters
    ----------
    network : ElasticNetwork
    input_node : int     index of the driven node
    output_node : int    index of the sensor node
    displacement : float target displacement amplitude of the input node
    n_steps : int
    force_type : str
    tol : float

    Returns
    -------
    list of (N, 2) position arrays, length n_steps
    """

    positions = np.copy(network.positions)
    x0 = positions[input_node, 0]
    traj = [np.copy(positions)]

    force_type_int = 1 if force_type == 'quartic' else 0
    constrained = [input_node * 2, input_node * 2 + 1]  # both x and y of input node

    for step in range(1, n_steps):
        frac = step / (n_steps - 1)
        positions_step = np.copy(positions)
        positions_step[input_node, 0] = x0 + frac * displacement

        min_pos, force_norm, _ = fire_minimize_dof(
            positions_step,
            np.array(network.edges,       dtype=np.int32),
            np.array(network.rest_lengths, dtype=np.float64),
            np.array(network.stiffnesses,  dtype=np.float64),
            1e-2, 1_000_000, tol, constrained, force_type_int,
        )
        positions = min_pos
        traj.append(np.copy(min_pos))

    return traj


def save_trajectory(positions_list, filepath):
    """
    Save a trajectory (list of position arrays) to a .npz file.

    Parameters
    ----------
    positions_list : list of (N, 2) arrays
    filepath : str or Path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    arrays = np.stack([np.asarray(p) for p in positions_list], axis=0)  # (T, N, 2)
    np.savez_compressed(filepath, trajectory=arrays)


def load_trajectory(filepath):
    """
    Load a trajectory from a .npz file.

    Returns
    -------
    list of (N, 2) position arrays
    """
    data = np.load(filepath)
    arrays = data['trajectory']   # (T, N, 2)
    return [arrays[t] for t in range(arrays.shape[0])]
