#!/usr/bin/env python
"""
Loss reconstruction verification — script version of
analysis/notebooks/loss_reconstruction_verification.ipynb

Verifies that losses saved during training (history['loss']) can be exactly
reconstructed by recomputing from the saved stiffnesses/positions, using both
sources of stiffnesses (history.pkl and stiffness_trajectory.npy).

Cross-system workflow (cluster vs local pkl/reconstruction drift):
    1. On the cluster:
        python loss_reconstruction_verification.py --mode train --results-dir $SCRATCH/loss_recon_test
    2. Download $SCRATCH/loss_recon_test to a local directory.
    3. Locally:
        python loss_reconstruction_verification.py --mode verify --results-dir ./downloaded_results
    Each run writes reconstruction_summary.json (with platform/library
    versions and max reconstruction error). Diff the cluster-side summary
    (written during --mode train... see note below) against the local-side
    summary (written during --mode verify) to see whether recomputing the
    same physics on a different machine reproduces the stored losses.

    Note: --mode train does NOT itself verify (it only trains+saves); run
    --mode verify on the SAME machine right after --mode train if you also
    want a same-machine baseline summary to diff against the downloaded one.
    Easiest: just use --mode all on the cluster once, then --mode verify
    locally on the downloaded directory, and diff the two
    reconstruction_summary.json files.

Usage:
    # Full local run (train + verify), like the notebook
    python loss_reconstruction_verification.py --mode all

    # Cluster: train + save only
    python loss_reconstruction_verification.py --mode train --results-dir $SCRATCH/loss_recon_test

    # Local: verify only, against downloaded results
    python loss_reconstruction_verification.py --mode verify --results-dir ./downloaded_results
"""

import argparse
import copy
import json
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent.parent  # project root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from base.config import FORCE_TOL, FORCE_TYPE, VMIN, VMAX, NETWORK_TYPE
from base.elastic_network import ElasticNetwork
from base.network_utils import create_auxetic_network
from base.simulate import compute_ift_gradient
from training.src.training_functions import (
    finish_training_GD_auxetic_batch,
    poisson_loss_batch_parallel,
)
from training.src.checkpoint_manager import save_training_results, get_training_result_path, _nt_filename
from tqdm import tqdm

def env_info():
    """Environment fingerprint — the first thing to diff when cluster and
    local reconstructions disagree (BLAS/JAX backend differences are the
    usual culprit for non-bit-identical floating point results)."""
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'numpy_version': np.__version__,
        'hostname': platform.node(),
    }
    try:
        import scipy
        info['scipy_version'] = scipy.__version__
    except ImportError:
        pass
    try:
        import jax
        info['jax_version'] = jax.__version__
        info['jax_backend'] = jax.default_backend()
    except ImportError:
        pass
    return info


def train_and_save(results_dir, packing_seed, n_nodes, compression_strains, target_poisson,
                    n_strain_steps, n_train_steps_newton, n_train_steps_fire,
                    boundary_margin=0.02, central_force=0.00005, methods=('newton', 'fire'),
                    network_type=NETWORK_TYPE):
    target_extensions = [-(nu * cs) for nu, cs in zip(target_poisson, compression_strains)]

    print(f'Creating network: packing_seed={packing_seed}, n_nodes={n_nodes}, network_type={network_type}')
    network_base, boundary_dict = create_auxetic_network(
        n_nodes=n_nodes, packing_seed=packing_seed, force_type=FORCE_TYPE,
        boundary_margin=boundary_margin, central_force=central_force,
        network_type=network_type,
    )
    top, bottom = boundary_dict['top'], boundary_dict['bottom']
    left, right = boundary_dict['left'], boundary_dict['right']
    print(f'  Nodes: {len(network_base.positions)}  Edges: {len(network_base.edges)}')
    print(f'  Boundary — top:{len(top)} bottom:{len(bottom)} left:{len(left)} right:{len(right)}')

    task_config_base = {
        'compression_strains': compression_strains,
        'target_poisson_ratios': target_poisson,
        'n_nodes': n_nodes, 'packing_seed': packing_seed,
        'force_tol': FORCE_TOL, 'n_strain_steps': n_strain_steps,
    }

    if 'newton' in methods:
        print('\n--- Training: Newton ---')
        network_newton = copy.deepcopy(network_base)
        t0 = time.time()
        history_newton, trained_newton = finish_training_GD_auxetic_batch(
            network=network_newton, history={}, learning_rate=1e-4,
            n_steps=n_train_steps_newton,
            top_nodes=top, bottom_nodes=bottom, left_nodes=left, right_nodes=right,
            force_type=FORCE_TYPE, n_strain_steps=n_strain_steps,
            source_compression_strain_list=compression_strains,
            desired_target_extension_list=target_extensions,
            force_tol=FORCE_TOL, vmin=VMIN, vmax=VMAX,
            method='newton', verbose=False,
        )
        print(f'  Newton training done in {time.time() - t0:.1f}s, '
              f'{len(history_newton["loss"])} steps')
        save_training_results(
            task_seed=0, realization_seed=0,
            history=history_newton, network=trained_newton,
            task_config={**task_config_base, 'method': 'newton'},
            results_dir=Path(results_dir) / 'newton',
            network_type=network_type,
        )

    if 'fire' in methods:
        print('\n--- Training: FIRE ---')
        network_fire = copy.deepcopy(network_base)
        t0 = time.time()
        history_fire, trained_fire = finish_training_GD_auxetic_batch(
            network=network_fire, history={}, learning_rate=1e-2,
            n_steps=n_train_steps_fire,
            top_nodes=top, bottom_nodes=bottom, left_nodes=left, right_nodes=right,
            force_type=FORCE_TYPE, n_strain_steps=n_strain_steps,
            source_compression_strain_list=compression_strains,
            desired_target_extension_list=target_extensions,
            force_tol=FORCE_TOL, vmin=VMIN, vmax=VMAX,
            method='fire', verbose=False,
        )
        print(f'  FIRE training done in {time.time() - t0:.1f}s, '
              f'{len(history_fire["loss"])} steps')
        save_training_results(
            task_seed=0, realization_seed=0,
            history=history_fire, network=trained_fire,
            task_config={**task_config_base, 'method': 'fire'},
            results_dir=Path(results_dir) / 'fire',
            network_type=network_type,
        )

    # Boundary indices + env fingerprint needed by `verify` (possibly on another machine)
    meta = {
        'boundary': {k: np.asarray(v).tolist() for k, v in boundary_dict.items()},
        'methods': list(methods),
        'env_info_train': env_info(),
        'network_type': network_type,
    }
    meta_path = Path(results_dir) / 'verification_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'\nSaved verification metadata -> {meta_path}')


def load_result(result_path, network_type='jammed'):
    with open(result_path / _nt_filename('history.pkl', network_type), 'rb') as f:
        history = pickle.load(f)
    with open(result_path / _nt_filename('final_network.pkl', network_type), 'rb') as f:
        net_dict = pickle.load(f)
    with open(result_path / _nt_filename('task_config.json', network_type), 'r') as f:
        cfg = json.load(f)
    stiff_traj = np.load(result_path / _nt_filename('stiffness_trajectory.npy', network_type))
    loss_traj = np.load(result_path / _nt_filename('loss_trajectory.npy', network_type))
    return history, net_dict, cfg, stiff_traj, loss_traj


def reconstruct_newton(stiffnesses_per_step, positions_per_step, net_dict, cfg, boundary):
    top, bottom = boundary['top'], boundary['bottom']
    left, right = boundary['left'], boundary['right']
    losses = []
    for i in tqdm(range(len(stiffnesses_per_step)), desc="Reconstructing Newton trajectory"):
        net_i = ElasticNetwork(
            positions=positions_per_step[i], edges=net_dict['edges'],
            rest_lengths=net_dict['rest_lengths'], stiffnesses=stiffnesses_per_step[i],
        )
        loss_i, _ = compute_ift_gradient(
            net_i,
            compression_strains=cfg['compression_strains'],
            target_poissons=cfg['target_poisson_ratios'],
            top_nodes=top, bottom_nodes=bottom, left_nodes=left, right_nodes=right,
            tol=cfg['force_tol'], n_strain_steps=cfg['n_strain_steps'],
        )
        losses.append(loss_i)
    return np.array(losses)


def reconstruct_fire(stiffnesses_per_step, positions_per_step, net_dict, cfg, boundary):
    top, bottom = boundary['top'], boundary['bottom']
    left, right = boundary['left'], boundary['right']
    losses = []
    for i in tqdm(range(len(stiffnesses_per_step)), desc="Reconstructing FIRE trajectory"):
        net_i = ElasticNetwork(
            positions=positions_per_step[i], edges=net_dict['edges'],
            rest_lengths=net_dict['rest_lengths'], stiffnesses=stiffnesses_per_step[i],
        )
        loss_i, _ = poisson_loss_batch_parallel(
            net_i,
            target_poisson_list=cfg['target_poisson_ratios'],
            top_nodes=top, bottom_nodes=bottom, left_nodes=left, right_nodes=right,
            compression_strain_list=cfg['compression_strains'],
            n_strain_steps=cfg['n_strain_steps'],
            force_type=FORCE_TYPE, tol=cfg['force_tol'],
        )
        losses.append(loss_i)
    return np.array(losses)


RECONSTRUCTORS = {'newton': reconstruct_newton, 'fire': reconstruct_fire}


def verify(results_dir, make_plot=True):
    results_dir = Path(results_dir)
    with open(results_dir / 'verification_meta.json') as f:
        meta = json.load(f)
    boundary = {k: np.array(v) for k, v in meta['boundary'].items()}
    methods = meta['methods']
    network_type = meta.get('network_type', 'jammed')

    summary = {
        'env_info_verify': env_info(),
        'env_info_train': meta.get('env_info_train'),
        'results': {},
    }
    plot_data = {}

    for method in methods:
        print(f'\n--- Verifying: {method} ---')
        history, net_dict, cfg, stiff_npy, loss_traj = load_result(
            get_training_result_path(0, 0, results_dir=results_dir / method),
            network_type=network_type,
        )
        recon_fn = RECONSTRUCTORS[method]

        # recon_pkl = recon_fn(history['stiffnesses'], history['positions'], net_dict, cfg, boundary)
        # recon_npy = recon_fn(stiff_npy, history['positions'], net_dict, cfg, boundary)

        stored = np.asarray(history['loss'])
        print(stored.min(), stored.max())
        # err_pkl = np.abs(stored - recon_pkl)
        # err_npy = np.abs(stored - recon_npy)
        stiff_hist = np.asarray(history['stiffnesses'])
        stiff_equal = bool(np.array_equal(stiff_hist, stiff_npy))
        stiff_maxdiff = float(np.max(np.abs(stiff_hist - stiff_npy)))

        print(f'  steps: {len(stored)}')
        print(f'  stiffness pkl==npy: {stiff_equal}  (max|diff|={stiff_maxdiff:.2e})')
        # print(f'  max|stored-recon(pkl)| = {err_pkl.max():.4e}')
        # print(f'  max|stored-recon(npy)| = {err_npy.max():.4e}')  
        
        assert 1==2

        summary['results'][method] = {
            'n_steps': int(len(stored)),
            'stiffness_pkl_eq_npy': stiff_equal,
            'stiffness_max_abs_diff': stiff_maxdiff,
            # 'max_abs_err_pkl': float(err_pkl.max()),
            # 'max_abs_err_npy': float(err_npy.max()),
            # 'mean_abs_err_pkl': float(err_pkl.mean()),
            # 'mean_abs_err_npy': float(err_npy.mean()),
        }
        # plot_data[method] = dict(stored=stored, recon_pkl=recon_pkl, recon_npy=recon_npy)

    summary_path = results_dir / 'reconstruction_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved verification summary -> {summary_path}')

    if make_plot and plot_data:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n_methods = len(plot_data)
        fig, axes = plt.subplots(2, n_methods, figsize=(6.5 * n_methods, 8), squeeze=False)
        for col, (method, d) in enumerate(plot_data.items()):
            steps = np.arange(len(d['stored']))
            ax = axes[0, col]
            ax.plot(steps, d['stored'], 'o-', color='steelblue', label='stored')
            ax.plot(steps, d['recon_pkl'], 's--', color='tomato', label='recon (pkl)', alpha=0.8)
            ax.plot(steps, d['recon_npy'], 'x:', color='green', label='recon (npy)', alpha=0.8)
            ax.set_title(f'{method} — loss trajectory')
            ax.set_xlabel('step'); ax.set_ylabel('loss'); ax.set_yscale('log')
            ax.legend(fontsize=8)

            ax2 = axes[1, col]
            ax2.plot(steps, np.abs(d['stored'] - d['recon_pkl']), 'o-',
                      color='tomato', label='|stored-recon(pkl)|')
            ax2.plot(steps, np.abs(d['stored'] - d['recon_npy']), 'x--',
                      color='green', label='|stored-recon(npy)|')
            ax2.set_title(f'{method} — absolute reconstruction error')
            ax2.set_xlabel('step'); ax2.set_ylabel('|error|'); ax2.set_yscale('log')
            ax2.legend(fontsize=8)

        plt.tight_layout()
        plot_path = results_dir / 'reconstruction_summary.pdf'
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved plot -> {plot_path}')

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Verify training losses can be exactly reconstructed from saved history/stiffnesses.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--mode', choices=['train', 'verify', 'all'], default='all')
    parser.add_argument('--results-dir', type=str, default='/tmp/loss_reconstruction_test')
    parser.add_argument('--methods', nargs='+', choices=['newton', 'fire'], default=['newton', 'fire'])
    parser.add_argument('--packing-seed', type=int, default=7)
    parser.add_argument('--n-nodes', type=int, default=100)
    parser.add_argument('--network-type', choices=['jammed', 'lattice'], default=NETWORK_TYPE)
    parser.add_argument('--n-strain-steps', type=int, default=100)
    parser.add_argument('--n-train-steps-newton', type=int, default=1000)
    parser.add_argument('--n-train-steps-fire', type=int, default=10)
    parser.add_argument('--compression-strains', type=float, nargs='+', default=[-0.10])
    parser.add_argument('--target-poisson', type=float, nargs='+', default=[-0.5])
    parser.add_argument('--no-plot', action='store_true', help='Skip plot generation (verify mode)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f'Mode: {args.mode}')
    print(f'Results dir: {results_dir}')
    print(f'Environment: {env_info()}\n')

    if args.mode in ('train', 'all'):
        train_and_save(
            results_dir=results_dir,
            packing_seed=args.packing_seed, n_nodes=args.n_nodes,
            compression_strains=args.compression_strains, target_poisson=args.target_poisson,
            n_strain_steps=args.n_strain_steps,
            n_train_steps_newton=args.n_train_steps_newton,
            n_train_steps_fire=args.n_train_steps_fire,
            methods=args.methods,
            network_type=args.network_type,
        )

    if args.mode in ('verify', 'all'):
        verify(results_dir, make_plot=not args.no_plot)


if __name__ == '__main__':
    main()
