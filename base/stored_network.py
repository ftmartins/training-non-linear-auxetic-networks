"""
Stored auxetic networks: export, load, and fingerprint.

The jammed packing behind create_auxetic_network() is integrated with an
adaptive ODE solver (LSODA) whose result depends on the hardware, so the same
packing seed has produced different networks on different machines (see
docs/adr/0002-auxetic-ensemble-on-stored-network-a.md and GitHub issue #2).
Runs that must share a network therefore load it from a file written by
save_stored_network() instead of regenerating it.

A network's identity is its fingerprint: a hash of the exact node positions
and bond list. Stiffnesses are excluded (they are what training changes), and
rest lengths follow from the positions.
"""
import hashlib
from pathlib import Path

import numpy as np

from base.elastic_network import ElasticNetwork

STORED_NETWORKS_DIR = Path(__file__).parent / 'networks'

# Canonical auxetic network (Network A in GLOSSARY.md): 194 bonds, the network
# 280 of the marmalade targeted runs trained on.
NETWORK_A_FILE = STORED_NETWORKS_DIR / 'auxetic_network_A.npz'
NETWORK_A_FINGERPRINT = '86a25180f56a10ea'
# Other networks packing seed 42 produced (kept for labelling old runs only).
NETWORK_B_FINGERPRINT = 'f14702529023d395'  # marmalade, 194 bonds
NETWORK_C_FINGERPRINT = '2c0b6e48b5bf0840'  # betty, 195 bonds

_BOUNDARY_SIDES = ('top', 'bottom', 'left', 'right')


def network_fingerprint(positions, edges):
    """Hex digest identifying a network by its exact positions and bonds."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(positions, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(edges, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def save_stored_network(path, positions, edges, rest_lengths, boundary_dict):
    """Write a network (geometry + boundary node sets, no stiffnesses) to `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fp = network_fingerprint(positions, edges)
    np.savez(
        path,
        positions=np.asarray(positions, dtype=np.float64),
        edges=np.asarray(edges, dtype=np.int64),
        rest_lengths=np.asarray(rest_lengths, dtype=np.float64),
        fingerprint=np.array(fp),
        **{f'boundary_{s}': np.asarray(boundary_dict[s], dtype=np.int64) for s in _BOUNDARY_SIDES},
    )
    return fp


def load_stored_network(path):
    """
    Load a network written by save_stored_network().

    Returns (network, boundary_dict, fingerprint), the same first two values
    create_auxetic_network() returns. Raises ValueError if the stored geometry
    does not hash to the fingerprint recorded in the file.
    """
    with np.load(path) as z:
        positions = z['positions']
        edges = z['edges']
        rest_lengths = z['rest_lengths']
        recorded = str(z['fingerprint'])
        boundary_dict = {s: [int(i) for i in z[f'boundary_{s}']] for s in _BOUNDARY_SIDES}
    fp = network_fingerprint(positions, edges)
    if fp != recorded:
        raise ValueError(f"{path}: geometry hashes to {fp}, file records {recorded}")
    network = ElasticNetwork(positions, edges, rest_lengths=rest_lengths)
    return network, boundary_dict, fp
