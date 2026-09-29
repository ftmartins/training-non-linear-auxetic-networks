#!/usr/bin/env python3
"""
Cluster-side per-checkpoint computation for explorations_sep21.ipynb section 5b
(mode-overlap evolution through training).

This was the notebook's slowest cell by far: for each of REP_ALLO / REP_AUX (the
single representative networks Figs 2/3/5 use) it serially computes a full
per-subtask cost-Hessian (autodiff + eigsh) plus an actuation/compression pass
at N_EVOL_CKPT checkpoints -- each checkpoint several minutes -- which repeatedly
overran nbconvert's timeout (see project_explorations_sep21 memory).

This script is a byte-for-byte port of evolution_allo()/evolution_aux()'s
per-checkpoint body, parallelized across checkpoints via a small SLURM array
(one task per checkpoint) instead of one process looping over all of them.

Usage:
    # 1. how many checkpoints exist for a family+n_ckpt (aux's count can be <
    #    n_ckpt after np.unique dedup -- print it before sizing the array):
    python compute_evolution_checkpoint.py --count --family allo --n-ckpt 8

    # 2. compute ONE checkpoint (SLURM array index into range(count)):
    python compute_evolution_checkpoint.py --family allo --n-ckpt 8 --ckpt-idx K \
        --data-root /data2/shared/felipetm/aug26_shim

    # 3. once every checkpoint for a family is done, assemble them into the
    #    SAME pickle format+name the notebook's own `cached()` call expects
    #    (figure_data/explore_sep21/_cache/evol_<family..>.pkl) -- rsync that
    #    file back locally and the notebook picks it up with no recompute:
    python compute_evolution_checkpoint.py --assemble --family allo --n-ckpt 8
"""
import argparse
import copy
import pickle
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_FIGDIR = _ROOT / "analysis" / "notebooks" / "figures"
for _p in (_ROOT, _FIGDIR, _ROOT / "training" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import aug26_common as C

K_EIGS = 4
REP_ALLO_DEFAULT = ("targeted", 0, 0)
REP_AUX_DEFAULT = (4, 0)
CKPT_DIR_DEFAULT = Path("/data2/shared/felipetm/explore_sep21/evolution_cache")

# cache-key suffix each family uses in the notebook -- allo carries a "_v3" (the
# true-initial-stiffness fix); aux never needed one (its stiff_traj[0] IS already
# the true untrained draw, see the notebook's evolution_aux docstring).
CACHE_KEY = {"allo": "evol_allo_v3_{n}", "aux": "evol_aux_{n}"}


def configure_data_root(root):
    """Point aug26_common's data-loading globals at `root` (see
    compute_combined_cost_hessian.py's identical helper / docstring for why:
    REP_ALLO/REP_AUX live in the PRODUCTION allosteric_nets_aug /
    auxetic_nets_aug trees, whose raw cluster layout differs from the
    repo-relative `data/` mirror aug26_common defaults to)."""
    import aug26_common as C
    root = Path(root)
    C.REPO_ROOT = root
    C.ALLO_DATA_DIR = root / "data" / "allosteric_nets_aug"
    C.AUX_DATA_DIR = root / "data" / "auxetic_nets_aug" / C.AUX_FAMILY


def _pick(lst, n):
    if n is None or len(lst) <= n:
        return list(lst)
    return [lst[i] for i in np.unique(np.linspace(0, len(lst) - 1, n).round().astype(int))]


def _safe_top(L, V):
    try:
        return C.top_positive_eigvec(L, V)
    except ValueError:
        return np.full(V.shape[0], np.nan)


# ---- checkpoint selection: identical to evolution_allo/evolution_aux's setup ----

def allo_checkpoints(d, n_ckpt):
    """[(step, k), ...] -- true untrained state first, then n_ckpt-1 log-by-loss checkpoints."""
    ti, _lck = C.select_steps(d["mse1"], d["mse2"], d["stiff_traj_steps"])
    steps = [int(d["stiff_traj_steps"][t]) for t in _pick(list(ti), n_ckpt - 1)]
    ks = [d["stiff_traj"][list(d["stiff_traj_steps"]).index(s_)] for s_ in steps]
    return [(0, C.allo_initial_stiffnesses(d))] + list(zip(steps, ks))


def aux_checkpoints(a, n_ckpt):
    """[(step, k), ...] -- aux's stiff_traj[0] is already the true untrained draw."""
    L = np.asarray(a["loss_traj"], float)
    L = L.mean(axis=1) if L.ndim > 1 else L
    b = int(np.nanargmin(L))
    idx = np.unique(np.r_[0, np.geomspace(1, max(b, 2), n_ckpt - 1).astype(int)])
    stiff_traj = np.asarray(a["stiff_traj"])
    return [(int(t), stiff_traj[t]) for t in idx]


def n_checkpoints(family, n_ckpt, allo_ids, aux_ids):
    if family == "allo":
        d = C.load_allosteric(*allo_ids)
        return len(allo_checkpoints(d, n_ckpt))
    a = C.load_auxetic(*aux_ids)
    return len(aux_checkpoints(a, n_ckpt))


# ---- per-checkpoint compute: identical to the notebook's loop bodies ----

def compute_allo_ckpt(d, step, k, allo_solver):
    ch = C.allo_cost_hessian_at(d, k, k_eigs=K_EIGS)
    comb = (d["mse1"] + d["mse2"]) / 2
    phis, ss = [], []
    for sub, key in ((0, "mse1"), (1, "mse2")):
        _, fr = C.allo_actuation_frames(d["nodes"], d["incidence_matrix"], k, d["task_config"], sub, solver=allo_solver)
        ss.append(C.edge_susceptibilities(fr[-1], d["edges"], k, d["eq_lengths"])["s_shift"])
        phis.append(_safe_top(*ch[key]))
    loss = float(comb[step]) if step < len(comb) else float(comb[-1])
    return dict(step=step, loss=loss, phi=phis, sshift=ss)


def compute_aux_ckpt(a, step, k):
    from analysis.timestep_sweep import global_constrained_nodes
    net, bd = a["network"], a["boundary"]
    edges = np.asarray(net.edges)
    pos = np.asarray(net.positions, float)
    rl = np.linalg.norm(pos[edges[:, 1]] - pos[edges[:, 0]], axis=1)
    cn = global_constrained_nodes(bd)
    n2 = copy.deepcopy(net)
    n2.stiffnesses = np.asarray(k, float)
    subs = C.aux_subtasks(a["task_config"])[:2]
    L = np.asarray(a["loss_traj"], float)
    L = L.mean(axis=1) if L.ndim > 1 else L
    phis, ss = [], []
    for cs, tp in subs:
        fr = C.aux_compression_frames(n2, bd, cs, n_steps=a["task_config"].get("n_strain_steps", 100))
        try:
            s_shift = C.edge_susceptibilities(fr[-1], edges, k, rl, constrained_nodes=None, source_nodes=cn)["s_shift"]
        except np.linalg.LinAlgError as e:
            # Singular constrained elastic Hessian (a near-floppy configuration, seen at some
            # early/untrained checkpoints) -- degrade to NaN for this one checkpoint/subtask
            # rather than crashing the whole array task; C.spearman already drops NaN pairs,
            # so downstream evolution_stats() correlations at OTHER checkpoints are unaffected.
            print(f"  WARNING: singular Hessian at step={step}, cs={cs} ({e}) -- s_shift set to NaN")
            s_shift = np.full(len(edges), np.nan)
        ss.append(s_shift)
        phis.append(_safe_top(*C.aux_cost_hessian(n2, bd, cs, tp, k_eigs=K_EIGS,
                                                  n_strain_steps=a["task_config"].get("n_strain_steps", 100))))
    return dict(step=step, loss=float(L[step]), phi=phis, sshift=ss)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["allo", "aux"], required=True)
    p.add_argument("--n-ckpt", type=int, default=8)
    p.add_argument("--ckpt-idx", type=int)
    p.add_argument("--count", action="store_true", help="print how many checkpoints this family/n-ckpt has, then exit")
    p.add_argument("--assemble", action="store_true", help="combine per-checkpoint pkls into the notebook cache pkl")
    p.add_argument("--allo-ids", nargs=3, default=list(REP_ALLO_DEFAULT))
    p.add_argument("--aux-ids", nargs=2, type=int, default=list(REP_AUX_DEFAULT))
    p.add_argument("--data-root", default="/data2/shared/felipetm/aug26_shim")
    p.add_argument("--ckpt-dir", default=str(CKPT_DIR_DEFAULT))
    p.add_argument("--notebook-cache-dir", default=None,
                   help="where to write the assembled notebook-format pkl (default: "
                        "figure_data/explore_sep21/_cache under this checkout)")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    configure_data_root(a.data_root)
    allo_ids = (a.allo_ids[0], int(a.allo_ids[1]), int(a.allo_ids[2]))
    aux_ids = tuple(a.aux_ids)
    ckpt_dir = Path(a.ckpt_dir) / a.family
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if a.count:
        print(n_checkpoints(a.family, a.n_ckpt, allo_ids, aux_ids))
        return

    if a.assemble:
        n = n_checkpoints(a.family, a.n_ckpt, allo_ids, aux_ids)
        pieces = []
        for i in range(n):
            fp = ckpt_dir / f"ckpt_{a.n_ckpt}_{i}.pkl"
            if not fp.exists():
                print(f"MISSING {fp} -- not all checkpoints are done yet, aborting assembly")
                sys.exit(1)
            pieces.append(pickle.load(open(fp, "rb")))
        out = dict(
            step=np.array([pc["step"] for pc in pieces]),
            loss=np.array([pc["loss"] for pc in pieces]),
            phi=np.array([pc["phi"] for pc in pieces]),
            sshift=np.array([pc["sshift"] for pc in pieces]),
        )
        cache_dir = Path(a.notebook_cache_dir) if a.notebook_cache_dir else (_ROOT / "figure_data" / "explore_sep21" / "_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / f"{CACHE_KEY[a.family].format(n=a.n_ckpt)}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(out, f)
        print(f"assembled {n} checkpoints -> {out_path}")
        return

    if a.ckpt_idx is None:
        p.error("--ckpt-idx is required unless --count or --assemble")

    out_path = ckpt_dir / f"ckpt_{a.n_ckpt}_{a.ckpt_idx}.pkl"
    if out_path.exists() and not a.force:
        print(f"SKIP {a.family}/{a.ckpt_idx}: already computed at {out_path}")
        return

    allo_solver = "lammps" if C.allo_solver_available("lammps") else "jax_fire"
    if a.family == "allo":
        d = C.load_allosteric(*allo_ids)
        ckpts = allo_checkpoints(d, a.n_ckpt)
        step, k = ckpts[a.ckpt_idx]
        print(f"allo ckpt {a.ckpt_idx}/{len(ckpts)}: step={step} solver={allo_solver}")
        result = compute_allo_ckpt(d, step, k, allo_solver)
    else:
        aa = C.load_auxetic(*aux_ids)
        ckpts = aux_checkpoints(aa, a.n_ckpt)
        step, k = ckpts[a.ckpt_idx]
        print(f"aux ckpt {a.ckpt_idx}/{len(ckpts)}: step={step}")
        result = compute_aux_ckpt(aa, step, k)

    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"OK {a.family}/{a.ckpt_idx} (step {step}) -> {out_path}")


if __name__ == "__main__":
    main()
