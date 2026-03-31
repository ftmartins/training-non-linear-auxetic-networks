"""
Compute cost Hessian eigenvectors for a single (task, realization) pair.

Usage:
    python compute_cost_hessian.py --task TASK_SEED --realization REAL_SEED [options]

Saves to:
    <data_dir>/task_XX/realization_XX/cost_hessian_eigs.npz
with keys:
    eigenvalues  : (k_eigs,)       ascending by value
    eigenvectors : (n_edges, k_eigs)
    subtask_idx  : (k_subtasks,)   which subtask each slice belongs to
    compression_strains : (k_subtasks,)
    target_poissons     : (k_subtasks,)
"""

import sys
import argparse
import copy
import pickle
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse.linalg import LinearOperator, eigsh

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from task_generator import generate_task_config
from config import N_NODES, BOUNDARY_MARGIN, FORCE_TYPE, N_STRAIN_STEPS
from network_utils import create_auxetic_network, get_square_boundary_nodes
from elastic_network import ElasticNetwork
from training_functions_with_toggle import (
    compute_poisson_ratio_single,
    compute_poisson_ratio_single_jax,
    crf as _crf_default,
)

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# ── Data helpers ──────────────────────────────────────────────────────────────

def load_trajectories(task_seed, realization_seed, data_dir):
    path = Path(data_dir) / f'task_{task_seed:02d}' / f'realization_{realization_seed:02d}'
    loss = np.load(path / 'loss_trajectory.npy')
    stiffness = np.load(path / 'stiffness_trajectory.npy')
    return loss, stiffness


def load_network_topology(task_seed, realization_seed, data_dir):
    path = Path(data_dir) / f'task_{task_seed:02d}' / f'realization_{realization_seed:02d}'
    pkl_file = path / 'final_network.pkl'

    if pkl_file.exists():
        with open(pkl_file, 'rb') as f:
            net_dict = pickle.load(f)
        network = ElasticNetwork(
            positions=net_dict['positions'],
            edges=net_dict['edges'],
            stiffnesses=net_dict['stiffnesses'],
            rest_lengths=net_dict['rest_lengths'],
        )
        top, bottom, left, right = get_square_boundary_nodes(
            np.array(net_dict['positions']), BOUNDARY_MARGIN
        )
        boundary_dict = {'top': top, 'bottom': bottom, 'left': left, 'right': right}
    else:
        print(f"  WARNING: final_network.pkl missing — regenerating topology.")
        network, boundary_dict = create_auxetic_network(
            n_nodes=N_NODES,
            packing_seed=task_seed,
            boundary_margin=BOUNDARY_MARGIN,
        )
    return network, boundary_dict


# ── Hessian computation ───────────────────────────────────────────────────────

def compute_cost_hessian_single_subtask(
    network, target_poisson, compression_strain,
    top_nodes, bottom_nodes, left_nodes, right_nodes,
    n_strain_steps=N_STRAIN_STEPS, force_type='quadratic',
    k_eigs=20, hvp_epsilon=1e-4,
    # kept for backward compatibility but unused:
    epsilon=None, n_jobs=None,
):
    """
    Top-k eigenvalues/vectors of the loss Hessian via Lanczos + matrix-free HVPs.

    Loss = (attained_poisson - target_poisson)^2

    Gradient ∇L is computed via JAX autodiff (jax.grad) — one forward+backward
    pass instead of 2×n_edges finite-difference simulations.

    HVP:  H·v ≈ (∇L(k + ε·v) - ∇L(k)) / ε  using JAX gradients.

    Returns:
        eigenvalues  : (k_eigs,)        sorted ascending by value
        eigenvectors : (n_edges, k_eigs)
    """
    n_edges = len(network.stiffnesses)
    base_k = np.array(network.stiffnesses, dtype=float)

    # ── JAX setup: convert static arrays once ─────────────────────────────────
    edges_jax        = jnp.asarray(np.array(network.edges,        dtype=np.int32))
    rest_lengths_jax = jnp.asarray(np.array(network.rest_lengths, dtype=np.float64))
    positions_jax    = jnp.asarray(np.array(network.positions,    dtype=np.float64).flatten())

    def loss_jax(k_jax):
        pr = compute_poisson_ratio_single_jax(
            _crf_default, k_jax, edges_jax, rest_lengths_jax, positions_jax,
            top_nodes, bottom_nodes, left_nodes, right_nodes,
            compression_strain, n_strain_steps,
        )
        return (pr - target_poisson) ** 2

    grad_loss = jax.jit(jax.grad(loss_jax))

    def jax_gradient(k):
        return np.array(grad_loss(jnp.asarray(k)))

    # Precompute base gradient once; reuse for every HVP.
    t0 = time.time()
    print(f"      Computing base gradient ({n_edges} edges) ...", flush=True)
    g0 = jax_gradient(base_k)
    print(f"      Base gradient done in {time.time() - t0:.1f}s", flush=True)

    hvp_count = [0]
    hvp_t0 = [time.time()]

    def hvp(v):
        v = np.asarray(v, dtype=float)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-14:
            return np.zeros_like(v)
        hvp_count[0] += 1
        t_call = time.time()
        g_fwd = jax_gradient(base_k + hvp_epsilon * v / norm_v)
        elapsed = time.time() - t_call
        total_elapsed = time.time() - hvp_t0[0]
        print(
            f"      HVP #{hvp_count[0]:3d}  call={elapsed:.1f}s  total={total_elapsed:.1f}s",
            flush=True,
        )
        return norm_v * (g_fwd - g0) / hvp_epsilon

    H_op = LinearOperator((n_edges, n_edges), matvec=hvp, dtype=float)
    k = min(k_eigs, n_edges - 1)
    print(f"      Starting eigsh (k={k}) ...", flush=True)
    t_eig = time.time()
    eigenvalues, eigenvectors = eigsh(H_op, k=k, which='LM')
    print(f"      eigsh done in {time.time() - t_eig:.1f}s ({hvp_count[0]} HVPs total)", flush=True)

    order = np.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Compute cost Hessian eigenvectors.')
    parser.add_argument('--task',         type=int, required=True)
    parser.add_argument('--realization',  type=int, required=True)
    parser.add_argument('--data_dir',     type=str, default='/data2/shared/felipetm/auxetic_networks/ensemble_training_new/')
    parser.add_argument('--epsilon',      type=float, default=1e-5)
    parser.add_argument('--hvp_epsilon',  type=float, default=1e-4)
    parser.add_argument('--k_eigs',       type=int,   default=20)
    parser.add_argument('--n_jobs',       type=int,   default=4)
    parser.add_argument('--n_strain_steps', type=int, default=N_STRAIN_STEPS)
    parser.add_argument('--force_type',   type=str,   default=FORCE_TYPE)
    parser.add_argument('--overwrite',    action='store_true',
                        help='Recompute even if output file already exists.')
    args = parser.parse_args()

    out_path = (
        Path(args.data_dir)
        / f'task_{args.task:02d}'
        / f'realization_{args.realization:02d}'
        / 'cost_hessian_eigs.npz'
    )

    if out_path.exists() and not args.overwrite:
        print(f"Output already exists: {out_path}  (use --overwrite to recompute)")
        return

    print(f"Task {args.task}, Realization {args.realization}")

    # ── Load network and best stiffness ───────────────────────────────────────
    loss, stiffness = load_trajectories(args.task, args.realization, args.data_dir)
    best_step = np.argmin(loss)
    best_stiffness = stiffness[best_step]
    print(f"  Best step: {best_step}, min loss: {loss[best_step]:.6f}")

    network, boundary_dict = load_network_topology(args.task, args.realization, args.data_dir)
    network.stiffnesses = np.array(best_stiffness)

    top_nodes    = boundary_dict['top']
    bottom_nodes = boundary_dict['bottom']
    left_nodes   = boundary_dict['left']
    right_nodes  = boundary_dict['right']

    # ── Task subtasks ──────────────────────────────────────────────────────────
    task_config = generate_task_config(args.task)
    compression_strains = task_config['compression_strains']
    target_poissons     = task_config['target_poisson_ratios']

    # ── Compute per subtask ────────────────────────────────────────────────────
    all_eigenvalues  = []
    all_eigenvectors = []

    t_total = time.time()
    for subtask_idx, (cs, tp) in enumerate(zip(compression_strains, target_poissons)):
        t_sub = time.time()
        print(f"  Subtask {subtask_idx}: compression={cs:.3f}, target_poisson={tp:.3f}", flush=True)
        evals, evecs = compute_cost_hessian_single_subtask(
            network, tp, cs,
            top_nodes, bottom_nodes, left_nodes, right_nodes,
            epsilon=args.epsilon,
            hvp_epsilon=args.hvp_epsilon,
            k_eigs=args.k_eigs,
            n_jobs=args.n_jobs,
            n_strain_steps=args.n_strain_steps,
            force_type=args.force_type,
        )
        all_eigenvalues.append(evals)
        all_eigenvectors.append(evecs)
        print(f"  Subtask {subtask_idx} done in {time.time() - t_sub:.1f}s", flush=True)

    # ── Save ───────────────────────────────────────────────────────────────────
    np.savez(
        out_path,
        eigenvalues          = np.array(all_eigenvalues),    # (n_subtasks, k_eigs)
        eigenvectors         = np.array(all_eigenvectors),   # (n_subtasks, n_edges, k_eigs)
        subtask_indices      = np.arange(len(compression_strains)),
        compression_strains  = np.array(compression_strains),
        target_poissons      = np.array(target_poissons),
        task_seed            = np.array(args.task),
        realization_seed     = np.array(args.realization),
    )
    print(f"  Saved → {out_path}")
    print(f"  Total time: {time.time() - t_total:.1f}s", flush=True)


if __name__ == '__main__':
    main()
