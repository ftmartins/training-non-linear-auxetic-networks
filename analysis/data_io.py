"""Network and trajectory I/O for analysis notebooks."""

import errno
import os
import numpy as np
from pathlib import Path

from base.elastic_network import ElasticNetwork
from base.realization_record import RealizationRecord
from base.network_utils import get_square_boundary_nodes
from base.config import (
    TARGETED_DATA_DIR, DATA_DIR, BOUNDARY_MARGIN,
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
    record = RealizationRecord.locate(data_dir, 'auxetic', task_seed, real_seed,
                                      must_exist=False, network_type=network_type)

    def exact(name):
        return record.run_dir / (name if network_type is None else _nt_filename(name, network_type))

    # Only the exact file name for network_type counts here (no fallback to plain names).
    path = exact('final_network.pkl')
    if not path.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))
    net = record.network
    network = ElasticNetwork(
        positions=net.positions,
        edges=net.edges,
        stiffnesses=record.current_stiffnesses,
        rest_lengths=net.rest_lengths,
    )

    # Prefer the boundary fixed at network-creation time and saved alongside
    # training results — final_network.pkl's positions are the network's
    # *final trained/deformed* state, so re-deriving boundary nodes from them
    # geometrically (get_square_boundary_nodes) can silently pick a different
    # node set than the one training actually used (deformation moves nodes
    # off/onto the margin-based threshold). Only fall back to recomputing for
    # older result directories saved before boundary_nodes.json existed.
    if exact('boundary_nodes.json').exists():
        boundary = record.boundary
    else:
        top, bottom, left, right = get_square_boundary_nodes(
            np.array(net.positions), BOUNDARY_MARGIN
        )
        boundary = {'top': top, 'bottom': bottom, 'left': left, 'right': right}
    return network, boundary


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
