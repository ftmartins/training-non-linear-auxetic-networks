#!/usr/bin/env python3
"""
Export the network an auxetic run trained on to a stored-network file.

Reads final_network_<nt>.pkl (positions/edges/rest_lengths; training only
changes stiffnesses) and boundary_nodes_<nt>.json from a realization
directory and writes them with base.stored_network.save_stored_network.

Usage:
    python training/runners/export_stored_network.py \
        --run-dir <TARGETED_RESULTS_DIR>/jax/task_00/realization_00 \
        --out base/networks/auxetic_network_A.npz
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from base.stored_network import save_stored_network


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--network-type', default='jammed')
    args = parser.parse_args()

    nt = args.network_type
    with open(args.run_dir / f'final_network_{nt}.pkl', 'rb') as f:
        net = pickle.load(f)
    with open(args.run_dir / f'boundary_nodes_{nt}.json') as f:
        boundary = json.load(f)

    fp = save_stored_network(args.out, net['positions'], net['edges'],
                             net['rest_lengths'], boundary)
    print(f"Wrote {args.out}: {len(net['positions'])} nodes, {len(net['edges'])} edges, "
          f"fingerprint {fp}")


if __name__ == '__main__':
    main()
