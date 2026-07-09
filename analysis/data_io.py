"""Network and trajectory I/O for analysis notebooks."""

import json
import pickle
import numpy as np
from pathlib import Path

from base.elastic_network import ElasticNetwork
from base.network_utils import get_square_boundary_nodes
from base.config import (
    TARGETED_DATA_DIR, DATA_DIR, ALLOSTERIC_DATA_DIR, BOUNDARY_MARGIN,
)
from training.src.checkpoint_manager import _nt_filename


# ---------------------------------------------------------------------------
# Network loaders
# ---------------------------------------------------------------------------

def load_auxetic_network(task_seed, real_seed, data_dir=None, network_type=None):
    """
    Load a trained auxetic network from disk.

    Parameters
    ----------
    task_seed : int
    real_seed : int
    data_dir : str or Path or None  defaults to base.config.TARGETED_DATA_DIR
    network_type : str or None  'jammed'/'lattice'/etc. (see create_auxetic_network).
        When given, looks for the network-type-suffixed filename (e.g.
        'final_network_jammed.pkl') as written by checkpoint_manager.save_training_results.
        When None (default), looks for the plain 'final_network.pkl'.

    Returns
    -------
    network : ElasticNetwork  (with final trained stiffnesses)
    boundary : dict with keys 'top', 'bottom', 'left', 'right'
    """
    if data_dir is None:
        data_dir = TARGETED_DATA_DIR
    filename = ('final_network.pkl' if network_type is None
                else _nt_filename('final_network.pkl', network_type))
    path = (Path(data_dir)
            / f'task_{task_seed:02d}'
            / f'realization_{real_seed:02d}'
            / filename)

    with open(path, 'rb') as f:
        net_dict = pickle.load(f)

    network = ElasticNetwork(
        positions=net_dict['positions'],
        edges=net_dict['edges'],
        stiffnesses=net_dict['stiffnesses'],
        rest_lengths=net_dict['rest_lengths'],
    )

    # Prefer the boundary fixed at network-creation time and saved alongside
    # training results — final_network.pkl's positions are the network's
    # *final trained/deformed* state, so re-deriving boundary nodes from them
    # geometrically (get_square_boundary_nodes) can silently pick a different
    # node set than the one training actually used (deformation moves nodes
    # off/onto the margin-based threshold). Only fall back to recomputing for
    # older result directories saved before boundary_nodes.json existed.
    boundary_filename = ('boundary_nodes.json' if network_type is None
                          else _nt_filename('boundary_nodes.json', network_type))
    boundary_path = path.parent / boundary_filename
    if boundary_path.exists():
        with open(boundary_path) as f:
            boundary = {k: np.asarray(v, dtype=int) for k, v in json.load(f).items()}
    else:
        top, bottom, left, right = get_square_boundary_nodes(
            np.array(net_dict['positions']), BOUNDARY_MARGIN
        )
        boundary = {'top': top, 'bottom': bottom, 'left': left, 'right': right}
    return network, boundary


def load_allosteric_network(geometry_id, task_id, real_id, data_dir=None):
    """
    Load an allosteric network ensemble result.

    Parameters
    ----------
    geometry_id, task_id, real_id : int
    data_dir : str or Path or None  defaults to base.config.ALLOSTERIC_DATA_DIR

    Returns
    -------
    dict with keys:
        'nodes', 'stiffnesses', 'eq_lengths', 'incidence_matrix', 'tasks',
        'mse1', 'mse2', and optionally 'cost_hessian_eigs'
    """
    if data_dir is None:
        data_dir = ALLOSTERIC_DATA_DIR
    path = (Path(data_dir)
            / f'geometry_{geometry_id}'
            / f'task_{task_id}'
            / f'realization_{real_id}')

    result = {}
    for key, fname in [
        ('nodes',            'nodes.npy'),
        ('stiffnesses',      'stiffnesses.npy'),
        ('eq_lengths',       'eq_lengths.npy'),
        ('incidence_matrix', 'incidence_matrix.npy'),
        ('mse1',             'mse1.npy'),
        ('mse2',             'mse2.npy'),
    ]:
        f = path / fname
        if f.exists():
            result[key] = np.load(f)

    tasks_file = path / 'tasks.txt'
    if tasks_file.exists():
        result['tasks'] = tasks_file.read_text()

    eigs_file = path / 'cost_hessian_eigs.npz'
    if eigs_file.exists():
        result['cost_hessian_eigs'] = np.load(eigs_file)

    return result


# ---------------------------------------------------------------------------
# Trajectory I/O (thin wrappers around analysis.trajectory)
# ---------------------------------------------------------------------------

def save_trajectory(positions_list, filepath):
    """Save a list of (N, 2) position arrays as a .npz file."""
    from .trajectory import save_trajectory as _save
    _save(positions_list, filepath)


def load_trajectory(filepath):
    """Load a .npz trajectory file. Returns list of (N, 2) arrays."""
    from .trajectory import load_trajectory as _load
    return _load(filepath)
