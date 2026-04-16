"""
compute_actuation_modes.py
==========================
Command-line script for computing actuation trajectories and elastic-mode
overlap data for a single task.  Designed to be submitted as a SLURM array
job (one element per task seed).

Usage
-----
    python compute_actuation_modes.py --task 11 [options]

Output
------
    <output_dir>/task_<NN>/all_results.pkl   – list of per-realization result dicts
    <output_dir>/task_<NN>/all_mode_data.pkl – dict keyed by (task, real)

Each result dict contains:
    task_seed, realization_seed, min_loss, network, boundary_dict, subtasks

Each subtask dict contains:
    compression_strain, target_poisson, trajectory,
    mode_eigenvalues, mode_eigenvectors, mode_overlaps, displacement_norms

all_mode_data values are lists (one per subtask) of dicts returned by
compute_unified_mode_data, which include overlap heatmaps, cosine similarity,
Pearson/Spearman correlations, and cached Analysis-9 feature vectors.
"""

import argparse
import contextlib
import copy
import io
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

import jax
jax.config.update("jax_enable_x64", True)

from config import BOUNDARY_MARGIN, FORCE_TYPE, get_n_strain_steps, get_n_nodes
from network_utils import create_auxetic_network, get_square_boundary_nodes
from elastic_network import ElasticNetwork
from task_generator import generate_task_config
from training_functions_with_toggle import compute_quasistatic_trajectory_auxetic
from generalized_susceptibility import (
    compute_physical_hessian_strained,
    compute_constrained_hessian_inverse,
    compute_full_jacobian_matrixwise,
    susceptibilities_from_jacobian,
)

try:
    from fire_minimize_memview_cy import fire_minimize_dof
    print("Cython FIRE minimiser loaded.")
except ImportError:
    fire_minimize_dof = None
    print("Warning: Cython FIRE not available — JAX fallback will be used.")

# ---------------------------------------------------------------------------
# Data-loading helpers (mirrored from notebook)
# ---------------------------------------------------------------------------

def discover_realizations(task_seed, data_dir):
    data_dir = Path(data_dir)
    task_dir = data_dir / f"task_{task_seed:02d}"
    realizations = []
    if not task_dir.exists():
        return realizations
    for d in sorted(task_dir.iterdir()):
        if d.is_dir() and d.name.startswith("realization_"):
            if (d / "loss_trajectory.npy").exists():
                realizations.append(int(d.name.split("_")[1]))
    return realizations


def load_trajectories(task_seed, realization_seed, data_dir):
    path = (
        Path(data_dir)
        / f"task_{task_seed:02d}"
        / f"realization_{realization_seed:02d}"
    )
    loss = np.load(path / "loss_trajectory.npy")
    stiffness = np.load(path / "stiffness_trajectory.npy")
    return loss, stiffness


def load_network_topology(task_seed, realization_seed, data_dir):
    path = (
        Path(data_dir)
        / f"task_{task_seed:02d}"
        / f"realization_{realization_seed:02d}"
    )
    pkl_file = path / "final_network.pkl"
    if pkl_file.exists():
        with open(pkl_file, "rb") as f:
            net_dict = pickle.load(f)
        network = ElasticNetwork(
            positions=net_dict["positions"],
            edges=net_dict["edges"],
            stiffnesses=net_dict["stiffnesses"],
            rest_lengths=net_dict["rest_lengths"],
        )
        top, bottom, left, right = get_square_boundary_nodes(
            np.array(net_dict["positions"]), BOUNDARY_MARGIN
        )
        boundary_dict = {
            "top": top, "bottom": bottom, "left": left, "right": right
        }
        return network, boundary_dict
    else:
        print(
            f"  WARNING: final_network.pkl missing for task {task_seed}, "
            f"real {realization_seed}. Regenerating from packing seed."
        )
        network, boundary_dict = create_auxetic_network(
            n_nodes=get_n_nodes(task_seed),
            packing_seed=task_seed,
            boundary_margin=BOUNDARY_MARGIN,
        )
        return network, boundary_dict


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _suppress_stdout():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def compute_unconstrained_hessian(network, positions):
    with _suppress_stdout():
        H = compute_physical_hessian_strained(
            stiffnesses=np.array(network.stiffnesses, dtype=float),
            rest_lengths=np.array(network.rest_lengths, dtype=float),
            edges=np.array(network.edges, dtype=int),
            final_positions=np.array(positions, dtype=float),
        )
    return H


def compute_actuation_and_modes(
    task_seed,
    realization_seed,
    data_dir,
    n_strain_steps=None,
    force_type=FORCE_TYPE,
):
    if n_strain_steps is None:
        n_strain_steps = get_n_strain_steps(task_seed)

    loss, stiffness_traj = load_trajectories(task_seed, realization_seed, data_dir)
    best_step = int(np.argmin(loss))
    best_stiffness = stiffness_traj[best_step]

    task_config = generate_task_config(task_seed)
    compression_strains = task_config["compression_strains"]
    target_poissons = task_config["target_poisson_ratios"]

    network, boundary_dict = load_network_topology(task_seed, realization_seed, data_dir)
    network.stiffnesses = np.array(best_stiffness, dtype=float)

    top_nodes = boundary_dict["top"]
    bot_nodes = boundary_dict["bottom"]

    subtask_results = []
    for cs, tp in zip(compression_strains, target_poissons):
        print(f"  [subtask] strain={cs:.3f}  target_ν={tp:.3f}")

        net_copy = copy.deepcopy(network)
        traj = compute_quasistatic_trajectory_auxetic(
            net_copy, cs, top_nodes, bot_nodes,
            n_steps=n_strain_steps, verbose=False,
            force_type=force_type, tol=1e-9,
        )
        traj = [np.asarray(pos, dtype=float) for pos in traj]
        T = len(traj)
        N = traj[0].shape[0]
        print(f"    Trajectory: {T} frames, {N} nodes, {2*N} DOFs")

        evals_list, evecs_list, overlaps_list, dr_norms = [], [], [], []

        for t in tqdm(range(T - 1), desc="    Hessian modes", leave=False):
            H = compute_unconstrained_hessian(network, traj[t])
            evals, evecs = np.linalg.eigh(H)

            dr = (traj[t + 1] - traj[t]).ravel()
            norm_dr = float(np.linalg.norm(dr))
            assert norm_dr > 0, f"Zero displacement at step {t}"

            c = evecs[:, 3:].T @ dr
            overlap = c ** 2 / norm_dr

            evals_list.append(evals[3:])
            evecs_list.append(evecs[:, 3:])
            overlaps_list.append(overlap)
            dr_norms.append(norm_dr)

        subtask_results.append({
            "compression_strain": cs,
            "target_poisson": tp,
            "trajectory": traj,
            "mode_eigenvalues": np.array(evals_list),
            "mode_eigenvectors": np.array(evecs_list),
            "mode_overlaps": np.array(overlaps_list),
            "displacement_norms": np.array(dr_norms),
        })

    return {
        "task_seed": task_seed,
        "realization_seed": realization_seed,
        "min_loss": float(loss[best_step]),
        "network": network,
        "boundary_dict": boundary_dict,
        "subtasks": subtask_results,
    }


# ---------------------------------------------------------------------------
# Unified mode-data computation (mirrored from notebook)
# ---------------------------------------------------------------------------

def _edge_geometry_from_positions(positions, edges, eps=1e-12):
    edge_vec = positions[edges[:, 1]] - positions[edges[:, 0]]
    edge_len = np.linalg.norm(edge_vec, axis=1)
    edge_hat = edge_vec / np.maximum(edge_len[:, None], eps)
    return edge_hat, edge_len


def _mode_edge_strain(mode_vec_flat, positions, edges, eps=1e-12):
    n_nodes = positions.shape[0]
    u = np.asarray(mode_vec_flat, dtype=float).reshape(n_nodes, 2)
    du = u[edges[:, 1]] - u[edges[:, 0]]
    edge_hat, _ = _edge_geometry_from_positions(positions, edges, eps=eps)
    return np.abs(np.einsum("ed,ed->e", du, edge_hat))


def _susceptibility_components_per_edge(positions, network, boundary_dict):
    edges = np.asarray(network.edges, dtype=int)
    stiff = np.asarray(network.stiffnesses, dtype=float)
    restl = np.asarray(network.rest_lengths, dtype=float)
    constrained_nodes = np.unique(
        np.concatenate([boundary_dict["top"], boundary_dict["bottom"]])
    ).astype(int)

    H_full_inv = compute_constrained_hessian_inverse(
        positions=positions,
        edges=edges,
        stiffnesses=stiff,
        rest_lengths=restl,
        constrained_nodes=constrained_nodes,
    )
    _, Hjac_parts, _ = compute_full_jacobian_matrixwise(
        positions=positions,
        edges=edges,
        stiffnesses=stiff,
        rest_lengths=restl,
        H_ff_inv=None,
        H_full_inv=H_full_inv,
    )
    s_par, s_perp, s_eq, s_tot = susceptibilities_from_jacobian(Hjac_parts)
    return (
        np.abs(np.asarray(s_par, dtype=float)),
        np.abs(np.asarray(s_perp, dtype=float)),
        np.abs(np.asarray(s_eq, dtype=float)),
        np.abs(np.asarray(s_tot, dtype=float)),
    )


def _pearson_batch(x, Y, eps=1e-12):
    xc = x - x.mean()
    Yc = Y - Y.mean(axis=0)
    num = xc @ Yc
    denom = np.linalg.norm(xc) * np.linalg.norm(Yc, axis=0) + eps
    return num / denom


def _spearman_batch(x, Y, eps=1e-12):
    xr = rankdata(x).astype(float)
    Yr = np.apply_along_axis(rankdata, 0, Y).astype(float)
    return _pearson_batch(xr, Yr, eps=eps)


def compute_unified_mode_data(
    result,
    subtask_idx,
    n_modes=25,
    thresholds=None,
    subsample=4,
    n_corr_modes=10,
    eps=1e-12,
):
    if thresholds is None:
        thresholds = np.linspace(0, 95, 40)

    subtask = result["subtasks"][subtask_idx]
    traj = subtask["trajectory"]
    evals_all = np.asarray(subtask["mode_eigenvalues"], dtype=float)
    evecs_all = np.asarray(subtask["mode_eigenvectors"], dtype=float)
    network = result["network"]
    edges = np.asarray(network.edges, dtype=int)
    restl = np.asarray(network.rest_lengths, dtype=float)
    stiff = np.abs(np.asarray(network.stiffnesses, dtype=float))

    T_m1_full, M = evals_all.shape
    k = min(n_modes, M)
    k_corr = min(n_corr_modes, k)
    n_nodes = np.asarray(traj[0], dtype=float).shape[0]
    thresholds = np.asarray(thresholds, dtype=float)
    P = len(thresholds)

    t_indices = np.arange(0, T_m1_full, subsample)
    T_sub = len(t_indices)

    P_raw = np.zeros((T_sub, k), dtype=float)
    cosine = np.zeros((T_sub, k), dtype=float)
    pearson_vals = np.full((T_sub, P, k_corr), np.nan)
    spearman_vals = np.full((T_sub, P, k_corr), np.nan)

    feat_acc = None
    n_feat_frames = 0

    for ti, t in enumerate(t_indices):
        pos_t = np.asarray(traj[t], dtype=float)
        s_par_abs, s_perp_abs, s_eq_abs, s_tot_abs = _susceptibility_components_per_edge(
            pos_t, result["network"], result["boundary_dict"]
        )
        norm_s = float(np.linalg.norm(s_tot_abs))

        edge_hat, edge_len = _edge_geometry_from_positions(pos_t, edges, eps=eps)

        U = evecs_all[t, :, :k].reshape(n_nodes, 2, k)
        dU = U[edges[:, 1]] - U[edges[:, 0]]
        sigma = np.abs(np.einsum("ed,edk->ek", edge_hat, dU))

        P_raw[ti] = s_tot_abs @ sigma
        norm_sigma = np.linalg.norm(sigma, axis=0)
        cosine[ti] = P_raw[ti] / (norm_s * norm_sigma + eps)

        sigma_c = sigma[:, :k_corr]
        for pi, thr in enumerate(thresholds):
            cutoff = np.percentile(s_tot_abs, thr)
            mask = s_tot_abs >= cutoff
            if mask.sum() < 3:
                continue
            s_m = s_tot_abs[mask]
            sig_m = sigma_c[mask]
            valid = np.std(sig_m, axis=0) > eps
            if np.std(s_m) < eps or not valid.any():
                continue
            p_r = _pearson_batch(s_m, sig_m, eps=eps)
            s_r = _spearman_batch(s_m, sig_m, eps=eps)
            pearson_vals[ti, pi, valid] = p_r[valid]
            spearman_vals[ti, pi, valid] = s_r[valid]

        strain_abs = np.abs((edge_len - restl) / (restl + eps))
        stress_abs = np.abs(stiff * strain_abs)
        hess_edge_mode = sigma[:, 0] if k > 0 else np.zeros_like(s_tot_abs)

        if feat_acc is None:
            feat_acc = {k_: np.zeros_like(v) for k_, v in [
                ("s_par", s_par_abs), ("s_perp", s_perp_abs),
                ("s_eq", s_eq_abs), ("s_tot", s_tot_abs),
                ("stiffness", stiff), ("strain", strain_abs),
                ("stress", stress_abs), ("hessian_evec", hess_edge_mode),
            ]}

        feat_acc["s_par"] += s_par_abs
        feat_acc["s_perp"] += s_perp_abs
        feat_acc["s_eq"] += s_eq_abs
        feat_acc["s_tot"] += s_tot_abs
        feat_acc["stiffness"] += stiff
        feat_acc["strain"] += strain_abs
        feat_acc["stress"] += stress_abs
        feat_acc["hessian_evec"] += hess_edge_mode
        n_feat_frames += 1

    row_sum = P_raw.sum(axis=1, keepdims=True)
    P_norm = np.divide(P_raw, row_sum, out=np.zeros_like(P_raw), where=row_sum > eps)

    if n_feat_frames > 0 and feat_acc is not None:
        feat_vec = {kname: v / float(n_feat_frames) for kname, v in feat_acc.items()}
    else:
        n_edges = len(edges)
        feat_vec = {
            "s_par": np.zeros(n_edges), "s_perp": np.zeros(n_edges),
            "s_eq": np.zeros(n_edges), "s_tot": np.zeros(n_edges),
            "stiffness": np.abs(stiff), "strain": np.zeros(n_edges),
            "stress": np.zeros(n_edges), "hessian_evec": np.zeros(n_edges),
        }

    compression_full = np.linspace(0.0, float(subtask["compression_strain"]), T_m1_full)
    compression_sub = compression_full[t_indices]

    return {
        "compression": compression_sub,
        "P_raw": P_raw,
        "P_norm": P_norm,
        "cosine": cosine,
        "eigenvalues": evals_all[t_indices, :k],
        "n_modes": k,
        "compression_strain": float(subtask["compression_strain"]),
        "weighted_raw": P_raw,
        "weighted_norm": P_norm,
        "cosine_raw": cosine,
        "selected_modes_nontrivial": np.arange(k, dtype=int),
        "thresholds": thresholds,
        "pearson": pearson_vals,
        "spearman": spearman_vals,
        "pearson_tmean": np.nanmean(pearson_vals, axis=0),
        "spearman_tmean": np.nanmean(spearman_vals, axis=0),
        "n_corr_modes": k_corr,
        "analysis9_feature_vectors": feat_vec,
        "analysis9_feature_labels": [
            "s_par", "s_perp", "s_eq", "s_tot",
            "stiffness", "strain", "stress", "hessian_evec",
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute actuation modes for one task seed and save to file."
    )
    parser.add_argument(
        "--task", type=int, required=True,
        help="Task seed (SLURM array index).",
    )
    parser.add_argument(
        "--data-dir", default="./data/results_new/",
        help="Directory containing task_XX/realization_YY subdirectories.",
    )
    parser.add_argument(
        "--output-dir", default="./figure_data/",
        help="Root output directory; results saved under <output_dir>/task_<NN>/.",
    )
    parser.add_argument(
        "--loss-threshold", type=float, default=0.01,
        help="Relative min-loss threshold for filtering realizations (default 0.01).",
    )
    parser.add_argument(
        "--traj-subsample", type=int, default=4,
        help="Use every Nth trajectory frame for mode-data computation (default 4).",
    )
    parser.add_argument(
        "--n-modes", type=int, default=25,
        help="Number of non-trivial modes to retain (default 25).",
    )
    parser.add_argument(
        "--n-corr-modes", type=int, default=10,
        help="Number of modes used for correlation curves (default 10).",
    )
    parser.add_argument(
        "--n-strain-steps", type=int, default=None,
        help="Quasistatic trajectory steps (default: get_n_strain_steps(task)).",
    )
    args = parser.parse_args()
    if args.n_strain_steps is None:
        args.n_strain_steps = get_n_strain_steps(args.task)

    task = args.task
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir) / f"task_{task:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Task {task}  |  data_dir={data_dir}  |  output={out_dir}")
    print(f"loss_threshold={args.loss_threshold}  subsample={args.traj_subsample}  "
          f"n_modes={args.n_modes}")
    print(f"{'='*60}\n")

    # -- Discover and filter realizations ------------------------------------
    realizations = discover_realizations(task, data_dir)
    if not realizations:
        print(f"No realizations found for task {task} in {data_dir}. Exiting.")
        sys.exit(1)
    print(f"Found {len(realizations)} realizations: {realizations}")

    candidates = []
    for real in realizations:
        try:
            loss, _ = load_trajectories(task, real, data_dir)
            rel_min = float(np.nanmin(loss)) / float(loss[0])
            if rel_min > args.loss_threshold:
                print(f"  SKIP  real={real}  rel_min_loss={rel_min:.4f} > {args.loss_threshold}")
            else:
                print(f"  OK    real={real}  rel_min_loss={rel_min:.4f}")
                candidates.append(real)
        except Exception as exc:
            print(f"  WARN  real={real}  failed to load: {exc}")

    if not candidates:
        print("No realizations pass the loss threshold. Exiting.")
        sys.exit(0)

    print(f"\n{len(candidates)} realization(s) selected: {candidates}\n")

    # -- Compute and collect -------------------------------------------------
    all_results = []
    all_mode_data = {}

    for real in candidates:
        print(f"\n--- Task {task}, Realization {real} ---")
        try:
            result = compute_actuation_and_modes(
                task, real, data_dir,
                n_strain_steps=args.n_strain_steps,
            )
        except Exception as exc:
            import traceback
            print(f"  ERROR during actuation computation: {exc}")
            traceback.print_exc()
            continue

        key = (task, real)
        subtask_data_list = []
        for si in range(len(result["subtasks"])):
            print(f"  Computing unified mode data for subtask {si} ...")
            try:
                md = compute_unified_mode_data(
                    result, si,
                    n_modes=args.n_modes,
                    subsample=args.traj_subsample,
                    n_corr_modes=args.n_corr_modes,
                )
                subtask_data_list.append(md)
            except Exception as exc:
                import traceback
                print(f"  ERROR during mode-data computation (subtask {si}): {exc}")
                traceback.print_exc()
                subtask_data_list.append(None)

        all_results.append(result)
        all_mode_data[key] = subtask_data_list

    # -- Save ----------------------------------------------------------------
    results_path = out_dir / "all_results.pkl"
    mode_data_path = out_dir / "all_mode_data.pkl"

    with open(results_path, "wb") as f:
        pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved all_results   → {results_path}")

    with open(mode_data_path, "wb") as f:
        pickle.dump(all_mode_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved all_mode_data → {mode_data_path}")

    print(f"\nDone. Processed {len(all_results)}/{len(candidates)} realizations.")


if __name__ == "__main__":
    main()
