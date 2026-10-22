"""
Tests for stored auxetic networks and distinct-seed ensemble selection
(docs/adr/0001-ensemble-members-have-distinct-seeds.md, 0002-auxetic-ensemble-on-stored-network-a.md).

Run: python -m unittest tests.test_ensemble_distinct_seeds
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'analysis' / 'notebooks' / 'figures'))

from base.stored_network import (
    NETWORK_A_FILE, NETWORK_A_FINGERPRINT, load_stored_network, network_fingerprint,
    save_stored_network,
)


def _toy_network():
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    edges = np.array([[0, 1], [0, 2], [1, 3], [2, 3], [0, 3]])
    rest = np.linalg.norm(positions[edges[:, 1]] - positions[edges[:, 0]], axis=1)
    boundary = {'top': [2, 3], 'bottom': [0, 1], 'left': [0, 2], 'right': [1, 3]}
    return positions, edges, rest, boundary


class TestStoredNetwork(unittest.TestCase):
    def test_round_trip_preserves_geometry_and_fingerprint(self):
        positions, edges, rest, boundary = _toy_network()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'net.npz'
            fp = save_stored_network(path, positions, edges, rest, boundary)
            net, bd, fp_loaded = load_stored_network(path)
        self.assertEqual(fp, fp_loaded)
        np.testing.assert_array_equal(net.positions, positions)
        np.testing.assert_array_equal(net.edges, edges)
        np.testing.assert_array_equal(net.rest_lengths, rest)
        self.assertEqual(bd, boundary)

    def test_fingerprint_sees_tiny_position_change(self):
        positions, edges, _, _ = _toy_network()
        moved = positions.copy()
        moved[3, 0] = np.nextafter(moved[3, 0], 2.0)
        self.assertNotEqual(network_fingerprint(positions, edges),
                            network_fingerprint(moved, edges))

    def test_load_rejects_file_whose_geometry_was_altered(self):
        positions, edges, rest, boundary = _toy_network()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'net.npz'
            save_stored_network(path, positions, edges, rest, boundary)
            with np.load(path) as z:
                data = {k: z[k] for k in z.files}
            data['positions'] = data['positions'] + 1e-12
            np.savez(path, **data)
            with self.assertRaises(ValueError):
                load_stored_network(path)

    def test_shipped_network_a_matches_its_fingerprint(self):
        net, boundary, fp = load_stored_network(NETWORK_A_FILE)
        self.assertEqual(fp, NETWORK_A_FINGERPRINT)
        self.assertEqual(len(net.edges), 194)
        self.assertEqual(set(boundary), {'top', 'bottom', 'left', 'right'})


class TestBestDistinctSeeds(unittest.TestCase):
    def setUp(self):
        from aug26_common import _best_distinct_seeds
        self.pick = _best_distinct_seeds

    def test_duplicate_seed_keeps_lower_ratio_and_fills_from_spares(self):
        # (ratio, real_id, seed): rids 4 and 13 share seed 7.
        cands = [(1e-6, 4, 7), (5e-7, 13, 7), (2e-6, 1, 3), (3e-6, 2, 5), (9e-6, 8, 9)]
        self.assertEqual(self.pick(cands, 3), [1, 2, 13])

    def test_bit_identical_pair_keeps_lower_real_id(self):
        cands = [(1e-6, 13, 7), (1e-6, 4, 7), (2e-6, 1, 3)]
        self.assertEqual(self.pick(cands, 2), [1, 4])

    def test_short_cell_returns_fewer_than_n_best(self):
        cands = [(1e-6, 0, 7), (2e-6, 1, 7)]
        self.assertEqual(self.pick(cands, 5), [0])


if __name__ == '__main__':
    unittest.main()
