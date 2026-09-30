#!/usr/bin/env python3
"""
Cluster-side per-network computation for explorations_sep21.ipynb section 2
(z-sweep edge susceptibilities + cost-Hessian top eigenvector).

This is a byte-for-byte port of that notebook's process_record / _sus_allo /
_sus_aux / _rho_summary / z_records / select_records cells -- the slow part of
that notebook (minutes of autodiff cost-Hessian + eigsh per network). Running
it here means the notebook itself only ever needs to READ pre-computed
results (once rsynced back into figure_data/explore_sep21/_cache/); nothing
in the notebook has to change beyond what it already does (it already skips
any case_id whose z_<family>_<case_id>.pkl exists).

Output: one pickle per case, in the SAME format+filename the notebook's own
`cached()` writes (z_<family>_<case_id>.pkl), written by default to
/data2/shared/felipetm/explore_sep21/z_sweep_cache/ (kept separate from the
training trees; nothing there is touched). Rsync that whole directory into
the local figure_data/explore_sep21/_cache/ and the notebook picks it up.

Usage:
    # 1. build the lookup (one line per (family, case_id) selected exactly as
    #    select_records() would, against the CURRENT converged set on disk):
    python compute_z_sweep_cache.py --build-lookup --lookup-file z_sweep_lookup.txt

    # 2. compute one case (SLURM array index into that lookup file):
    python compute_z_sweep_cache.py --case-index N --lookup-file z_sweep_lookup.txt
"""
import argparse
import copy
import json
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_FIGDIR = _ROOT / "analysis" / "notebooks" / "figures"
for _p in (_ROOT, _FIGDIR, _ROOT / "training" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import aug26_common as C
from training.src import explore_sep21_cases as EC

EXPLORE_ROOT_DEFAULT = Path("/data2/shared/felipetm/explore_sep21")
OUT_DIR_DEFAULT = EXPLORE_ROOT_DEFAULT / "z_sweep_cache"
K_EIGS = 4
MAX_RATIO = 1e-4


@contextmanager
def data_root(allo=None, aux=None):
    """Temporarily point aug26_common's loaders at another tree (same helper as
    the notebook's own -- kept identical rather than imported to avoid a
    notebook-execution dependency for a pure batch script)."""
    a0, x0 = C.ALLO_DATA_DIR, C.AUX_DATA_DIR
    if allo is not None:
        C.ALLO_DATA_DIR = Path(allo)
    if aux is not None:
        C.AUX_DATA_DIR = Path(aux)
    try:
        yield
    finally:
        C.ALLO_DATA_DIR, C.AUX_DATA_DIR = a0, x0


def _loss_ratio_allo(d):
    m1, m2 = np.load(d / "mse1.npy"), np.load(d / "mse2.npy")
    c = (m1 + m2) / 2.0
    return float(np.nanmin(c) / c[0]) if len(c) else np.nan


def _loss_ratio_aux(d):
    p = d / "loss_trajectory_jammed.npy"
    if not p.exists():
        return np.nan
    L = np.load(p)
    L = L.mean(axis=1) if L.ndim > 1 else L
    return float(np.nanmin(L) / L[0]) if len(L) else np.nan


# Extra IC seeds submitted 2026-09-26 for allo (task,z) cells short of 5 converged reps
# (see project_explorations_sep21 memory / submit_allo_extra.sh); must be folded into
# selection here or their networks never get picked up for the susceptibility cache.
ALLO_Z_EXTRA_CELLS = [(2, 0), (0, 1), (2, 1)]
ALLO_Z_EXTRA_IC_RANGE = (5, 20)


def _allo_z_all_cases():
    return EC.allo_z_cases() + EC.allo_z_extra_cases(ALLO_Z_EXTRA_CELLS, *ALLO_Z_EXTRA_IC_RANGE)


def z_records(family, exp_root):
    """Identical to the notebook's z_records(), against `exp_root` (the raw
    cluster tree, not a local rsync) so selection reflects the freshest
    on-disk convergence state."""
    recs = []
    for c in (EC.aux_z_cases() if family == "aux_z" else _allo_z_all_cases()):
        if family == "aux_z":
            # NOTE: the raw cluster tree nests an extra "jax" (gradient-method) level that the
            # local rsync mirror strips (`rsync .../aux_z/jax/ data/explore_sep21/aux_z/`) --
            # this script reads the raw tree directly, so it must include it explicitly.
            d = exp_root / "aux_z" / "jax" / f"task_{c['case_id']:02d}" / "realization_00"
            ratio = _loss_ratio_aux(d) if d.is_dir() else np.nan
            meta_p = d / "explore_case.json"
            task = c["orig_task"]
        else:
            d = EC.allo_z_dir(c, exp_root) / "geometry_targeted" / f"task_{c['task']}" / f"realization_{c['ic']}"
            ratio = _loss_ratio_allo(d) if (d / "mse1.npy").exists() else np.nan
            meta_p = d / "explore_case.json"
            task = c["task"]
        z = json.loads(meta_p.read_text())["z"] if meta_p.exists() else np.nan
        recs.append(dict(c, family=family, dir=d, ratio=ratio, z=z, task=task))
    return recs


def select_records(R, max_per_cell=2):
    """Identical to the notebook's select_records()."""
    seen, out = {}, []
    for r in R:
        if not (np.isfinite(r["ratio"]) and r["ratio"] <= MAX_RATIO):
            continue
        key = (r["task"], r["z_idx"], r["topo"])
        if max_per_cell is not None and seen.get(key, 0) >= max_per_cell:
            continue
        seen[key] = seen.get(key, 0) + 1
        out.append(r)
    return out


# ---- _sus_allo / _sus_aux / _rho_summary: identical to the notebook's ----

def _safe_susceptibilities(*args, **kwargs):
    """C.edge_susceptibilities, degrading to NaN on a singular constrained elastic Hessian
    (a near-floppy configuration, seen at some best-loss z-sweep states -- e.g. aux_z case 226)
    instead of crashing the whole array task. Downstream Spearman/plotting already drop NaNs."""
    try:
        return C.edge_susceptibilities(*args, **kwargs)
    except np.linalg.LinAlgError as e:
        n_edges = len(args[1])   # edges is always the 2nd positional arg
        print(f"  WARNING: singular Hessian ({e}) -- susceptibilities set to NaN")
        nan = np.full(n_edges, np.nan)
        return dict(s_par=nan, s_perp=nan, s_eq=nan, s_tot=nan, s_shift=nan)


def _sus_allo(d, k, allo_solver):
    ch = C.allo_cost_hessian_at(d, k, k_eigs=K_EIGS)
    rows = {}
    for sub, key in ((0, "mse1"), (1, "mse2")):
        _, fr = C.allo_actuation_frames(d["nodes"], d["incidence_matrix"], k, d["task_config"], sub, solver=allo_solver)
        sus = _safe_susceptibilities(fr[-1], d["edges"], k, d["eq_lengths"])
        ev, evec = ch[key]   # ev descending (allo_cost_hessian_at contract)
        rows[sub] = dict(sus, v_top=C.top_positive_eigvec(ev, evec), k=np.asarray(k, float),
                         cost_evals=np.sort(np.asarray(ev, float))[::-1])   # descending, top eigenvalue first
    return rows


def _sus_aux(a, k, use_sweep=True):
    from analysis.timestep_sweep import global_constrained_nodes
    net, bd = a["network"], a["boundary"]
    edges = np.asarray(net.edges)
    pos_ref = np.asarray(net.positions, float)
    rl = np.linalg.norm(pos_ref[edges[:, 1]] - pos_ref[edges[:, 0]], axis=1)
    cn = global_constrained_nodes(bd)
    n2 = copy.deepcopy(net)
    n2.stiffnesses = np.asarray(k, float)
    subs = C.aux_subtasks(a["task_config"])[:2]
    sw = a.get("sweep") if use_sweep else None
    raw_order = np.argsort(-np.abs(np.asarray(a["task_config"]["compression_strains"], float)))
    rows = {}
    for si, (cs, tp) in enumerate(subs):
        fr = C.aux_compression_frames(n2, bd, cs, n_steps=a["task_config"].get("n_strain_steps", 100))
        sus = _safe_susceptibilities(fr[-1], edges, k, rl, constrained_nodes=cn)
        if sw is not None and "cost_hessian_after_eigvecs" in sw:
            j = int(raw_order[si])
            L = np.asarray(sw["cost_hessian_after_eigvals"])[j]
            V = np.asarray(sw["cost_hessian_after_eigvecs"])[j]
        else:
            L, V = C.aux_cost_hessian(n2, bd, cs, tp, k_eigs=K_EIGS,
                                      n_strain_steps=a["task_config"].get("n_strain_steps", 100))
        rows[si] = dict(sus, v_top=C.top_positive_eigvec(L, V), k=np.asarray(k, float),
                        cost_evals=np.sort(np.asarray(L, float))[::-1])   # descending, top eigenvalue first
    return rows


def _rho_summary(rows):
    out = {}
    for s, r in rows.items():
        out[f"rho_shift_{s}"] = C.spearman(r["v_top"], r["s_shift"])
    if 0 in rows and 1 in rows:
        out["rho_v12"] = C.spearman(rows[0]["v_top"], rows[1]["v_top"])
    return out


def compute_one(r, exp_root, allo_solver):
    """Identical to the notebook's process_record()'s inner _run()."""
    if r["family"] == "aux_z":
        with data_root(aux=exp_root / "aux_z" / "jax"):   # raw-tree "jax" level; see z_records() note
            a = C.load_auxetic(r["case_id"], 0)
        if a is None or not a.get("task_config"):
            return None
        _, k, _ = C.aux_best_loss_state(a, load_positions=False)
        rows = _sus_aux(a, k)
        edges = np.asarray(a["network"].edges)
        pos = np.asarray(a["network"].positions)
    else:
        with data_root(allo=EC.allo_z_dir(r, exp_root)):
            d = C.load_allosteric("targeted", r["task"], r["ic"])
        if d is None:
            return None
        k = d["best_stiffnesses"] if d["best_stiffnesses"] is not None else d["stiff_traj"][-1]
        rows = _sus_allo(d, k, allo_solver)
        edges, pos = d["edges"], d["nodes"]
    return dict(rows=rows, edges=edges, pos=pos, **_rho_summary(rows))


def build_lookup(exp_root, max_per_cell=2):
    lines = []
    for name in ("allo_z", "aux_z"):
        R = z_records(name, exp_root)
        sel = select_records(R, max_per_cell)
        for r in sel:
            lines.append(f"{name} {r['case_id']}")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build-lookup", action="store_true", help="write the lookup file and exit")
    p.add_argument("--lookup-file", default="z_sweep_lookup.txt")
    p.add_argument("--case-index", type=int)
    p.add_argument("--exp-root", default=str(EXPLORE_ROOT_DEFAULT))
    p.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    p.add_argument("--max-per-cell", type=int, default=2)
    p.add_argument("--force", action="store_true", help="recompute even if the pkl already exists")
    a = p.parse_args()
    exp_root = Path(a.exp_root)

    if a.build_lookup:
        lines = build_lookup(exp_root, a.max_per_cell)
        Path(a.lookup_file).write_text("\n".join(lines) + "\n")
        print(f"wrote {len(lines)} cases to {a.lookup_file}")
        return

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [l for l in Path(a.lookup_file).read_text().splitlines() if l.strip()]
    family, case_id_s = lines[a.case_index].split()
    case_id = int(case_id_s)
    out_path = out_dir / f"z_{family}_{case_id}.pkl"
    if out_path.exists() and not a.force:
        print(f"SKIP {family}/{case_id}: already cached at {out_path}")
        return

    allo_solver = "lammps" if C.allo_solver_available("lammps") else "jax_fire"
    print(f"case {a.case_index}: family={family} case_id={case_id} allo_solver={allo_solver}")
    R = z_records(family, exp_root)
    r = next(x for x in R if x["case_id"] == case_id)
    result = compute_one(r, exp_root, allo_solver)
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"OK {family}/{case_id} -> {out_path}" + (" (result is None)" if result is None else ""))


if __name__ == "__main__":
    main()
