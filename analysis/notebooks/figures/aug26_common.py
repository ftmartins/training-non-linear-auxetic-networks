"""
Shared infrastructure for the Aug-26 figure set (``NewFiguresAug26_calc`` /
``NewFiguresAug26_plot``).

Everything that both the *calculation* notebook and the *plotting* notebook need
lives here so the notebooks stay thin:

    calc notebook  :  loads raw training output -> heavy computation -> figure_data/aug26_figs/*.npz
    plot notebook  :  figure_data/aug26_figs/*.npz -> matplotlib figures

Data source (fixed by design decision 2026-08-26): the post-2026-08-20
``data/allosteric_nets_aug`` tree and the ``jax`` auxetic tree under
``data/auxetic_nets/targeted_results_sqr``.  See the project memory
``project_allosteric_aug_training_status`` for the layout.

NOTE ``data/allosteric_nets_aug`` was originally pulled *without*
``timestep_sweep.npz``.  Sweep-derived quantities are recomputed here from
``stiffnesses_traj.npy`` when the sweep file is absent; when it *is* present
(rsync'd down after regeneration), ``load_allosteric`` / ``load_auxetic`` expose
it as ``d['sweep']`` and the cost-Hessian helpers below
(``allo_cost_hessian_at`` / ``aux_cost_hessian_at``) read the per-subtask +
combined eigen-pairs straight from it for the before/after checkpoints instead of
recomputing.  Any local recompute still mirrors training's own evaluation path
(``evaluate_actuation`` with the solver recorded in ``training_meta.json``) and
the calc notebook asserts the recomputed combined loss matches the stored
``mse1+mse2`` before using anything downstream.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Repo path plumbing (same three entries the June notebooks add)              #
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (REPO_ROOT, REPO_ROOT / "training" / "src", REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #
ALLO_DATA_DIR = REPO_ROOT / "data" / "allosteric_nets_aug"
AUX_GRADIENT_METHOD = "jax"
# New "_aug" auxetic ensemble (2026-08 pipeline): merged marmalade+betty tree,
# split into general/ and targeted/, with network-type-suffixed filenames
# (loss_trajectory_jammed.npy, final_network_jammed.pkl, ...). The Aug-26
# single-network figure panels use the *targeted* family (as the old
# targeted_results_sqr tree did).
AUX_FAMILY = os.environ.get("AUG26_AUX_FAMILY", "targeted")   # "targeted" | "general"
AUX_NETWORK_TYPE = "jammed"
AUX_DATA_DIR = REPO_ROOT / "data" / "auxetic_nets_aug" / AUX_FAMILY

FIG_DATA_DIR = REPO_ROOT / "figure_data" / "aug26_figs"
FIG_DATA_ALLO = FIG_DATA_DIR / "allosteric"
FIG_DATA_AUX = FIG_DATA_DIR / "auxetic"
for _d in (FIG_DATA_ALLO, FIG_DATA_AUX):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
# allosteric_trainer constants (single source of truth: training/runners/allosteric_trainer.py)
ALLO_STRAIN_INPUT = 1.0       # subtask 1 input strain (the -0.8 output target)
ALLO_STRAIN_INPUT2 = 0.5      # subtask 2 input strain (the -0.6 output target)
ALLO_NSTEPS_TASK1 = 40
ALLO_NSTEPS_TASK2 = 20
ALLO_CONSTRAINED_NODES = np.array([0, 1, 2, 3])   # analysis.timestep_sweep.LOCAL_CONSTRAINED_NODES


@dataclass
class SweepConfig:
    """Knobs for the loss-threshold step selection + Hessian sampling."""
    n_thresh_steps: int = 40          # log-spaced-by-loss checkpoints to keep
    eps_min: float = 1e-8             # smallest (L-Lmin)/Lmin threshold
    k_cost_eigs: int = 8             # top cost-Hessian eigen-pairs per subtask
    n_hessian_traj_steps: int = 20   # points along the actuation trajectory for the elastic Hessian
    n_modes_track: int = 10          # lowest branches followed for diabatic avoided-crossing tracking


CFG = SweepConfig()

# Loss convergence bar (combined (mse1+mse2)/2, running-min / initial).
MIN_SUCCESS_RATIO = 1e-4

# Tolerance for "recomputed loss is faithful to the saved one" (relative).
LOSS_FAITHFUL_RTOL = 5e-2


# --------------------------------------------------------------------------- #
# figure_data I/O                                                             #
# --------------------------------------------------------------------------- #
def _fig_dir(kind: str) -> Path:
    return {"allo": FIG_DATA_ALLO, "aux": FIG_DATA_AUX}[kind]


def save_fig_data(kind: str, name: str, **arrays) -> Path:
    """``save_fig_data('allo', 'fig2c_targeted', lam0=...)`` -> .../allosteric/fig2c_targeted.npz"""
    path = _fig_dir(kind) / f"{name}.npz"
    np.savez_compressed(path, **arrays)
    return path


def load_fig_data(kind: str, name: str) -> dict:
    path = _fig_dir(kind) / f"{name}.npz"
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def fig_data_exists(kind: str, name: str) -> bool:
    return (_fig_dir(kind) / f"{name}.npz").exists()


# Per-item cache (keeps expensive per-realization results so ensemble loops and
# killed runs are resumable). Lives under figure_data/aug26_figs/<kind>/_cache/.
def _cache_dir(kind: str) -> Path:
    d = _fig_dir(kind) / "_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_get(kind: str, key: str) -> dict | None:
    p = _cache_dir(kind) / f"{key}.npz"
    if not p.exists():
        return None
    with np.load(p, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def cache_put(kind: str, key: str, **arrays) -> None:
    np.savez_compressed(_cache_dir(kind) / f"{key}.npz", **arrays)


# --------------------------------------------------------------------------- #
# Loss / step selection helpers                                               #
# --------------------------------------------------------------------------- #
def combined_loss(mse1, mse2):
    """The quantity the allosteric trainer's stop bar is expressed in."""
    return (np.asarray(mse1) + np.asarray(mse2)) / 2.0


def convergence_ratio(mse1, mse2) -> float:
    c = combined_loss(mse1, mse2)
    return float(c.min() / c[0]) if len(c) and c[0] > 0 else np.inf


def select_steps(mse1, mse2, steps, cfg: SweepConfig = CFG):
    """Checkpoint-aligned, log-spaced-by-loss step selection.

    Returns ``(t_indices, loss_ckpt)`` where ``t_indices`` index into the
    checkpoint axis of ``stiffnesses_traj`` and ``loss_ckpt`` is the combined
    (mse1+mse2) loss at every checkpoint (NOT halved -- matches
    ``analysis.timestep_sweep._select_local_threshold_indices``).
    """
    from analysis.timestep_sweep import _select_local_threshold_indices
    return _select_local_threshold_indices(steps, mse1, mse2, cfg.n_thresh_steps, cfg.eps_min)


# --------------------------------------------------------------------------- #
# Allosteric loading (data/allosteric_nets_aug)                               #
# --------------------------------------------------------------------------- #
def _incmat_to_edges(incidence_matrix):
    """(E, 2) endpoint-index array from a signed incidence matrix -- pure numpy.

    Identical result to ``training.lammps_utils.incidence_to_edges`` but without
    importing that module (which pulls in the ``lammps`` extension at import
    time; the plotting env need not have it).
    """
    incidence_matrix = np.asarray(incidence_matrix)
    edges = np.zeros((incidence_matrix.shape[0], 2), dtype=int)
    for e, row in enumerate(incidence_matrix):
        cols = np.where(np.abs(row) > 0.5)[0]
        edges[e] = cols[0], cols[1]
    return edges


def allo_solver_available(solver: str) -> bool:
    """True if the requested allosteric physics backend can be imported here."""
    if solver == "jax_fire":
        return True
    if solver == "lammps":
        try:
            import lammps  # noqa: F401
            return True
        except Exception:
            return False
    return False


ALLO_GEOMETRIES = ["targeted", 0, 1, 2, 3, 4]


def _allo_geom_dir(geometry_id) -> Path:
    name = "geometry_targeted" if geometry_id == "targeted" else f"geometry_{geometry_id}"
    return ALLO_DATA_DIR / name


def allo_task_config(nodes, tasks_txt_vals) -> dict:
    """Rebuild the ``task_config`` dict exactly as ``post_training_sweep._run_allosteric`` does.

    ``tasks.txt`` holds ``(gseed, strain_output2, strain_output)`` -- note the
    order: subtask 1 (``tod``) uses ``strain_output`` (line 3), subtask 2
    (``tod2``) uses ``strain_output2`` (line 2).
    """
    _gseed, strain_output2, strain_output = [float(v) for v in tasks_txt_vals]
    d23 = float(np.linalg.norm(nodes[3] - nodes[2]))
    d01 = float(np.linalg.norm(nodes[0] - nodes[1]))
    return dict(
        tod=(1 + strain_output) * d23,
        tod2=(1 + strain_output2) * d23,
        dinputdistance=ALLO_STRAIN_INPUT * d01,
        dinputdistance2=ALLO_STRAIN_INPUT2 * d01,
        nsteps=ALLO_NSTEPS_TASK1,
        nsteps2=ALLO_NSTEPS_TASK2,
        strain_output=strain_output,
        strain_output2=strain_output2,
        strain_input=ALLO_STRAIN_INPUT,
        strain_input2=ALLO_STRAIN_INPUT2,
    )


def load_allosteric(geometry_id, task_id, real_id) -> dict | None:
    """Load one allosteric realization from the ``_aug`` tree.

    Returns ``None`` if the realization directory or its essential files are
    missing. Keys: ``nodes, incidence_matrix, edges, eq_lengths, stiffnesses,
    best_stiffnesses, stiff_traj, stiff_traj_steps, mse1, mse2, task_config,
    solver, geometry_id, task_id, real_id, dir``.
    """
    d = _allo_geom_dir(geometry_id) / f"task_{task_id}" / f"realization_{real_id}"
    need = ["nodes.npy", "incidence_matrix.npy", "eq_lengths.npy",
            "stiffnesses_traj.npy", "stiffnesses_traj_steps.npy",
            "mse1.npy", "mse2.npy", "tasks.txt"]
    if not d.is_dir() or not all((d / f).exists() for f in need):
        return None

    nodes = np.load(d / "nodes.npy").astype(float)
    incidence_matrix = np.load(d / "incidence_matrix.npy")
    tasks_vals = np.loadtxt(d / "tasks.txt")

    solver = "lammps"
    if (d / "training_meta.json").exists():
        solver = json.loads((d / "training_meta.json").read_text()).get("solver", "lammps")

    best_k = None
    if (d / "best_stiffnesses.npy").exists():
        best_k = np.abs(np.load(d / "best_stiffnesses.npy").astype(float))

    # Precomputed post-training sweep, if it has been rsync'd down. When present
    # (and new enough to carry the per-subtask + combined cost-Hessian keys,
    # 2026-08-27+), the figure code reads its cost / elastic Hessian eigen-pairs
    # straight from here instead of recomputing locally -- see
    # allo_cost_hessian_at / _cost_hessian_from_sweep.
    sweep = None
    if (d / "timestep_sweep.npz").exists():
        with np.load(d / "timestep_sweep.npz", allow_pickle=True) as z:
            sweep = {k: z[k] for k in z.files}

    return dict(
        nodes=nodes,
        incidence_matrix=incidence_matrix,
        edges=_incmat_to_edges(incidence_matrix),
        eq_lengths=np.load(d / "eq_lengths.npy").astype(float),
        stiffnesses=np.abs(np.load(d / "stiffnesses.npy").astype(float))
        if (d / "stiffnesses.npy").exists() else None,
        best_stiffnesses=best_k,
        stiff_traj=np.abs(np.load(d / "stiffnesses_traj.npy").astype(float)),
        stiff_traj_steps=np.load(d / "stiffnesses_traj_steps.npy").astype(int),
        mse1=np.load(d / "mse1.npy").astype(float),
        mse2=np.load(d / "mse2.npy").astype(float),
        task_config=allo_task_config(nodes, tasks_vals),
        solver=solver,
        sweep=sweep,
        geometry_id=geometry_id, task_id=task_id, real_id=real_id,
        dir=str(d),
    )


def discover_allosteric(pool: str = "targeted", max_per_task: int | None = None,
                        require_converged: bool = True) -> list[tuple]:
    """List ``(geometry_id, task_id, real_id)`` triples present on disk.

    pool : 'targeted' | 'general' | 'all'
        'targeted' -> geometry_targeted only; 'general' -> geometry_0..4;
        'all' -> both.
    """
    if pool == "targeted":
        geoms = ["targeted"]
    elif pool == "general":
        geoms = [0, 1, 2, 3, 4]
    elif pool == "all":
        geoms = ALLO_GEOMETRIES
    else:
        raise ValueError(pool)

    found = []
    for gid in geoms:
        gdir = _allo_geom_dir(gid)
        if not gdir.is_dir():
            continue
        for task_dir in sorted(gdir.glob("task_*"), key=lambda p: int(p.name.split("_")[1])):
            tid = int(task_dir.name.split("_")[1])
            reals = sorted(task_dir.glob("realization_*"), key=lambda p: int(p.name.split("_")[1]))
            if max_per_task is not None:
                reals = reals[:max_per_task]
            for rdir in reals:
                rid = int(rdir.name.split("_")[1])
                if not (rdir / "mse1.npy").exists():
                    continue
                if require_converged:
                    mse1 = np.load(rdir / "mse1.npy")
                    mse2 = np.load(rdir / "mse2.npy")
                    if convergence_ratio(mse1, mse2) > MIN_SUCCESS_RATIO:
                        continue
                found.append((gid, tid, rid))
    return found


# --------------------------------------------------------------------------- #
# Auxetic loading (data/auxetic_nets_aug/<family>)                            #
# --------------------------------------------------------------------------- #
def _auxfile(d: Path, stem: str, ext: str) -> Path:
    """Resolve <stem>.<ext> in realization dir `d`, preferring the network-type-
    suffixed name the "_aug" pipeline writes (``<stem>_jammed.<ext>``) and
    falling back to the plain name used by the older targeted_results_sqr tree."""
    j = d / f"{stem}_jammed.{ext}"
    return j if j.exists() else d / f"{stem}.{ext}"


def load_auxetic(task_id, real_id) -> dict | None:
    """Load one auxetic realization.

    Wraps ``analysis.data_io.load_auxetic_network`` (network + boundary) and
    adds the trajectories + task_config the calc notebook needs. Returns
    ``None`` when the realization is missing.
    """
    d = AUX_DATA_DIR / f"task_{task_id:02d}" / f"realization_{real_id:02d}"
    if not d.is_dir() or not _auxfile(d, "final_network", "pkl").exists():
        return None
    from analysis.data_io import load_auxetic_network

    network, boundary = load_auxetic_network(
        task_id, real_id, data_dir=AUX_DATA_DIR, network_type=AUX_NETWORK_TYPE)

    task_config = {}
    tc_path = _auxfile(d, "task_config", "json")
    if tc_path.exists():
        task_config = json.loads(tc_path.read_text())

    lt_path = _auxfile(d, "loss_trajectory", "npy")
    st_path = _auxfile(d, "stiffness_trajectory", "npy")
    loss_traj = np.load(lt_path) if lt_path.exists() else None
    stiff_traj = np.load(st_path) if st_path.exists() else None

    # Precomputed post-training sweep (network-type-suffixed filename possible).
    # Used by aux_cost_hessian_at when its operating point can be matched to the
    # sweep's before/after checkpoint (needs stiffness_trajectory.npy too).
    sweep = None
    for cand in sorted(d.glob("timestep_sweep*.npz")):
        with np.load(cand, allow_pickle=True) as z:
            sweep = {k: z[k] for k in z.files}
        break

    return dict(
        network=network, boundary=boundary, task_config=task_config,
        loss_traj=loss_traj, stiff_traj=stiff_traj, sweep=sweep,
        task_id=task_id, real_id=real_id, dir=str(d),
    )


def discover_auxetic(max_per_task: int | None = None, require_converged: bool = True,
                     min_success_ratio: float = 1e-4) -> list[tuple]:
    """List ``(task_id, real_id)`` pairs under the auxetic ``<family>`` tree.

    Convergence bar defaults to 1e-4 (min loss / initial loss), matching the
    allosteric bar; pass ``min_success_ratio=1e-2`` for the older 2-decade bar.
    """
    found = []
    if not AUX_DATA_DIR.is_dir():
        return found
    for task_dir in sorted(AUX_DATA_DIR.glob("task_*"), key=lambda p: int(p.name.split("_")[1])):
        tid = int(task_dir.name.split("_")[1])
        reals = sorted(task_dir.glob("realization_*"), key=lambda p: int(p.name.split("_")[1]))
        if max_per_task is not None:
            reals = reals[:max_per_task]
        for rdir in reals:
            rid = int(rdir.name.split("_")[1])
            lt = _auxfile(rdir, "loss_trajectory", "npy")
            if not lt.exists():
                continue
            if require_converged:
                loss = np.asarray(np.load(lt), dtype=float)
                s = loss.mean(axis=1) if loss.ndim > 1 else loss   # per-step scalar
                if len(s) == 0 or s[0] <= 0 or float(np.nanmin(s)) / s[0] > min_success_ratio:
                    continue
            found.append((tid, rid))
    return found


# --------------------------------------------------------------------------- #
# Physics: allosteric actuation + elastic / cost Hessians                     #
# --------------------------------------------------------------------------- #
def _subtask_params(task_config, subtask: int):
    if subtask == 0:
        return task_config["tod"], task_config["dinputdistance"], task_config["nsteps"]
    return task_config["tod2"], task_config["dinputdistance2"], task_config["nsteps2"]


def allo_actuation_frames(nodes, incidence_matrix, stiffnesses, task_config, subtask: int,
                          solver: str = "jax_fire"):
    """Full free quasistatic actuation trajectory for one subtask (0 or 1).

    Returns ``(mse, frames)``; ``frames`` is a list of ``(N, 2)`` arrays, one
    per pull step. ``mse = (||f[-1][2] - f[-1][3]|| - tod)**2`` -- the exact
    quantity ``training.runners.allosteric_trainer.evaluate_actuation`` reports.

    solver
        ``'jax_fire'`` (default) -- calls ``training.jax_actuation.strain_network_jax``
        (``FREE_CRF``) directly, no LAMMPS import needed. This is the same FIRE
        physics ``evaluate_actuation(solver='jax_fire')`` uses.
        ``'lammps'`` -- routes through ``evaluate_actuation`` for a
        solver-matched recompute; requires the ``lammps`` module. Falls back to
        ``'jax_fire'`` with a printed warning when it can't be imported.
    """
    edges = _incmat_to_edges(incidence_matrix)
    rest_lengths = np.linalg.norm(nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1)
    tod, dd, ns = _subtask_params(task_config, subtask)
    dx = dd / ns

    if solver == "lammps" and allo_solver_available("lammps"):
        from training.runners.allosteric_trainer import evaluate_actuation
        mse, _nf, _nc, frames = evaluate_actuation(
            nodes, incidence_matrix, np.asarray(stiffnesses, float), tod, dx, ns,
            return_trajectory=True, compute_clamped=False, solver="lammps")
        return float(mse), [np.asarray(f) for f in frames]

    if solver == "lammps":
        print("  [allo_actuation_frames] lammps unavailable -> falling back to jax_fire")

    import training.jax_actuation as jx_act
    frames = jx_act.strain_network_jax(
        jx_act.FREE_CRF, np.asarray(nodes, float), edges, rest_lengths,
        np.asarray(stiffnesses, float), 0, 1, dx=dx, nsteps=ns)
    frames = [np.asarray(f) for f in frames]
    mse = float((np.linalg.norm(frames[-1][2] - frames[-1][3]) - tod) ** 2)
    return mse, frames


def allo_elastic_spectrum(frame_positions, edges, stiffnesses, eq_lengths,
                          constrained_nodes=ALLO_CONSTRAINED_NODES, n_modes=None):
    """Constrained elastic Hessian eigen-pairs at one configuration.

    Thin wrapper around ``analysis.hessian.compute_hessian_spectrum`` --
    ``constrained_nodes=None`` gives the unconstrained (full, minus rigid
    modes) spectrum for the Fig 2c toggle.
    """
    from analysis.hessian import compute_hessian_spectrum
    vals, vecs, _ = compute_hessian_spectrum(
        frame_positions, edges, stiffnesses, eq_lengths,
        constrained_nodes=constrained_nodes, n_modes=n_modes,
    )
    return vals, vecs


def lowest_nontrivial(vals, constrained: bool) -> float:
    """Lowest *non-trivial* eigenvalue from an ascending elastic spectrum.

    The constrained (pinned-node) Hessian has no rigid-body nullspace -> take
    ``vals[0]``. The unconstrained Hessian keeps the 3 rigid modes of a 2-D
    network (2 translations + 1 rotation) at ~0 -> take ``vals[3]``.
    """
    vals = np.asarray(vals, float)
    return float(vals[0] if constrained else vals[min(3, len(vals) - 1)])


def allo_per_task_cost_hessian(nodes, incidence_matrix, task_config, stiffnesses,
                               k_eigs: int = CFG.k_cost_eigs, verbose: bool = False):
    """Top-``k_eigs`` eigen-pairs of the MSE1 and MSE2 cost Hessians *separately*.

    The saved ``timestep_sweep.npz`` only stores the *combined* (0.5*(mse1+mse2))
    cost Hessian. This rebuilds the two per-subtask scalar losses through the
    differentiable JAX-FIRE ramp (``training.jax_actuation``) -- same machinery
    as ``analysis.timestep_sweep._compute_cost_hessian_jax`` -- and Lanczos-solves
    each one. Returns ``dict(mse1=(evals, evecs), mse2=(evals, evecs))`` with
    ``evecs`` shaped ``(n_edges, k)``.
    """
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from scipy.sparse.linalg import LinearOperator, eigsh
    import training.jax_actuation as jx_act

    tod, tod2 = task_config["tod"], task_config["tod2"]
    ns, ns2 = task_config["nsteps"], task_config["nsteps2"]
    dx = task_config["dinputdistance"] / ns
    dx2 = task_config["dinputdistance2"] / ns2

    base_k = np.asarray(stiffnesses, float)
    n_edges = len(base_k)
    edges = _incmat_to_edges(incidence_matrix)
    rest_lengths = np.linalg.norm(nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1)

    def _loss_factory(dx_, ns_, tod_):
        def scalar_loss(k_jax):
            pos = jx_act.strain_network_jax_final_traced(
                jx_act.FREE_CRF, nodes, edges, rest_lengths, k_jax, 0, 1, dx=dx_, nsteps=ns_
            ).reshape(-1, 2)
            return (jnp.linalg.norm(pos[2] - pos[3]) - tod_) ** 2
        return scalar_loss

    out = {}
    for name, (dx_, ns_, tod_) in (("mse1", (dx, ns, tod)), ("mse2", (dx2, ns2, tod2))):
        scalar_loss = _loss_factory(dx_, ns_, tod_)

        @jax.jit
        def hvp_jax(k_jax, v_jax, _sl=scalar_loss):
            _, Hv = jax.jvp(jax.grad(_sl), (k_jax,), (v_jax,))
            return Hv

        k0 = jnp.asarray(base_k, dtype=jnp.float64)

        def hvp(v, _h=hvp_jax, _k0=k0):
            return np.asarray(_h(_k0, jnp.asarray(v, dtype=jnp.float64)))

        k = min(k_eigs, n_edges - 1)
        if verbose:
            print(f"    [{name}] eigsh k={k} ...", flush=True)
        evals, evecs = eigsh(LinearOperator((n_edges, n_edges), matvec=hvp, dtype=float),
                             k=k, which="LA")
        order = np.argsort(evals)[::-1]     # largest algebraic first
        out[name] = (evals[order], evecs[:, order])
    return out


# --------------------------------------------------------------------------- #
# Cost Hessian: prefer the precomputed sweep, fall back to local recompute    #
# --------------------------------------------------------------------------- #
_SWEEP_WHEN = {"before": "before", "init": "before", "after": "after", "best": "after"}


def _cost_hessian_from_sweep(sweep, when, k_eigs):
    """Per-subtask + combined cost-Hessian eigen-pairs pulled from a
    ``timestep_sweep.npz`` dict.

    Returns ``dict(per_subtask=[(evals, evecs), ...], combined=(evals, evecs))``
    with eigen-order normalised to DESCENDING (col 0 = largest algebraic) and
    sliced to the top ``k_eigs`` -- matching ``allo_per_task_cost_hessian`` /
    ``aux_cost_hessian``. ``evecs`` shaped ``(n_edges, k_eigs)``.

    Returns ``None`` when ``sweep`` is missing, is an *old* file without the
    2026-08-27+ per-subtask keys, or stores fewer than ``k_eigs`` eigen-pairs
    (caller then recomputes locally). ``combined`` is ``None`` for 2026-08-28+
    sweeps, which no longer store the combined Hessian (dropped for cluster
    wall-time reasons) -- the caller recomputes it locally if it needs it.
    """
    if not sweep:
        return None
    tag = _SWEEP_WHEN.get(when, when)
    kv, ke = f"cost_hessian_{tag}_eigvals", f"cost_hessian_{tag}_eigvecs"
    kvc, kec = f"{kv}_combined", f"{ke}_combined"
    if kv not in sweep:
        return None
    sub_vals = np.asarray(sweep[kv])        # (n_sub, k) ascending
    sub_vecs = np.asarray(sweep[ke])        # (n_sub, n_edges, k)
    if sub_vals.ndim != 2 or sub_vals.shape[1] < k_eigs:
        return None                         # old allo layout was 1-D combined, or too few eigs

    def _desc_top(vals, vecs):
        return np.ascontiguousarray(vals[..., ::-1][..., :k_eigs]), \
               np.ascontiguousarray(vecs[..., ::-1][..., :k_eigs])

    per_sub = [_desc_top(sub_vals[s], sub_vecs[s]) for s in range(sub_vals.shape[0])]
    combined = (_desc_top(np.asarray(sweep[kvc]), np.asarray(sweep[kec]))
                if kvc in sweep and kec in sweep else None)
    return dict(per_subtask=per_sub, combined=combined)


def _sweep_checkpoint_stiffness(d, end):
    """Stiffness vector at the sweep's ``end`` ('before'|'after') checkpoint, or
    None if ``d`` has no usable sweep. ``end`` -> index into ``d['stiff_traj']``."""
    sweep = d.get("sweep")
    if sweep is None or "t_indices" not in sweep:
        return None
    ti = np.asarray(sweep["t_indices"], dtype=int)
    idx = ti[0] if end == "before" else ti[-1]
    st = d["stiff_traj"]
    return st[idx] if 0 <= idx < len(st) else None


def allo_operating_point(d, at="best"):
    """``(k_vec, tag)`` for the allosteric cost-Hessian operating point.

    When ``d`` carries a precomputed sweep, snap to that sweep's
    before/after *checkpoint* stiffnesses (``tag`` 'before'/'after') so the cost
    Hessian is sweep-served and any susceptibility computed at ``k_vec`` is
    evaluated at the same configuration. Otherwise: ``at='best'`` ->
    ``best_stiffnesses`` (fallback ``stiff_traj[ti[-1]]``), ``at='before'`` ->
    ``stiff_traj[ti[0]]``.
    """
    end = "after" if at in ("best", "after") else "before"
    k_sweep = _sweep_checkpoint_stiffness(d, end)
    if k_sweep is not None:
        return k_sweep, end
    from analysis.timestep_sweep import _select_local_threshold_indices
    ti, _ = _select_local_threshold_indices(
        d["stiff_traj_steps"], d["mse1"], d["mse2"], CFG.n_thresh_steps, CFG.eps_min)
    if at in ("best", "after") and d.get("best_stiffnesses") is not None:
        return d["best_stiffnesses"], "best"
    return d["stiff_traj"][ti[-1] if end == "after" else ti[0]], at


def allo_cost_hessian_at(d, k_vec, k_eigs=CFG.k_cost_eigs, verbose=False):
    """Per-subtask allosteric cost Hessian at stiffness ``k_vec``.

    Prefers ``d['sweep']`` when ``k_vec`` *is* the sweep's before/after
    checkpoint stiffness (exact array match); otherwise recomputes locally with
    ``allo_per_task_cost_hessian``. Return contract matches the latter:
    ``dict(mse1=(evals, evecs), mse2=(evals, evecs))`` (plus ``combined`` when
    served from the sweep), eigenvalues DESCENDING, ``evecs`` ``(n_edges, k_eigs)``.
    """
    k_vec = np.asarray(k_vec, dtype=float)
    for end in ("before", "after"):
        k_ck = _sweep_checkpoint_stiffness(d, end)
        if k_ck is not None and k_ck.shape == k_vec.shape and np.array_equal(k_ck, k_vec):
            got = _cost_hessian_from_sweep(d["sweep"], end, k_eigs)
            if got is not None:
                ps = got["per_subtask"]
                out = {"mse1": ps[0],
                       "mse2": ps[1] if len(ps) > 1 else ps[0]}
                if got["combined"] is not None:
                    out["combined"] = got["combined"]
                return out
    return allo_per_task_cost_hessian(
        d["nodes"], d["incidence_matrix"], d["task_config"], k_vec,
        k_eigs=k_eigs, verbose=verbose)


def aux_cost_hessian_at(aa, subtask_idx, compression_strain, target_poisson,
                        k_eigs=CFG.k_cost_eigs, verbose=False):
    """Cost Hessian for one auxetic subtask (``subtask_idx`` in
    ``aux_subtasks`` order), preferring ``aa['sweep']``.

    The sweep stacks its per-subtask Hessians in *raw* ``task_config`` order,
    while ``aux_subtasks`` sorts by descending |compression|, so the index is
    remapped. The sweep is only trusted when its after-checkpoint stiffnesses
    (needs ``stiffness_trajectory.npy``) exactly match ``aa['network']``'s --
    otherwise falls back to ``aux_cost_hessian``. Returns ``(evals, evecs)``,
    largest-algebraic first.
    """
    sweep = aa.get("sweep")
    if sweep is not None and "t_indices" in sweep and aa.get("stiff_traj") is not None:
        ti = np.asarray(sweep["t_indices"], dtype=int)
        st = np.asarray(aa["stiff_traj"])
        k_net = np.asarray(aa["network"].stiffnesses, dtype=float)
        idx = int(ti[-1])
        if 0 <= idx < len(st) and st[idx].shape == k_net.shape and np.array_equal(
                np.asarray(st[idx], dtype=float), k_net):
            cs_raw = np.asarray(aa["task_config"]["compression_strains"], float)
            raw_order = np.argsort(-np.abs(cs_raw))          # aux_subtasks[i] -> raw index raw_order[i]
            if subtask_idx < len(raw_order):
                got = _cost_hessian_from_sweep(sweep, "after", k_eigs)
                if got is not None and raw_order[subtask_idx] < len(got["per_subtask"]):
                    return got["per_subtask"][int(raw_order[subtask_idx])]
    return aux_cost_hessian(aa["network"], aa["boundary"],
                            compression_strain, target_poisson,
                            k_eigs=k_eigs, verbose=verbose)


# --------------------------------------------------------------------------- #
# Susceptibilities & bond quantities                                          #
# --------------------------------------------------------------------------- #
def edge_susceptibilities(positions, edges, stiffnesses, rest_lengths,
                          constrained_nodes=ALLO_CONSTRAINED_NODES) -> dict:
    """Per-edge susceptibility decomposition + shift susceptibility at one config.

    Returns ``dict(s_par, s_perp, s_eq, s_tot, s_shift)`` -- each ``(E,)``.
    """
    from analysis.susceptibility import compute_susceptibilities, compute_s_shift
    cn = None if constrained_nodes is None else np.asarray(constrained_nodes, int)
    s_par, s_perp, s_eq, s_tot = compute_susceptibilities(
        positions, edges, stiffnesses, rest_lengths, constrained_nodes=cn)
    s_shift = compute_s_shift(positions, edges, stiffnesses, rest_lengths, constrained_nodes=cn)
    return dict(s_par=s_par, s_perp=s_perp, s_eq=s_eq, s_tot=s_tot, s_shift=s_shift)


def bond_quantities(positions, edges, stiffnesses, rest_lengths) -> dict:
    """Per-edge stiffness / stress / strain at one configuration."""
    from analysis.mechanics import bond_strains, bond_stresses
    return dict(
        stiffness=np.asarray(stiffnesses, float),
        stress=bond_stresses(positions, edges, stiffnesses, rest_lengths),
        strain=bond_strains(positions, edges, rest_lengths),
    )


# --------------------------------------------------------------------------- #
# Avoided crossings via diabatic tracking                                     #
# --------------------------------------------------------------------------- #
def spectrum_and_lowmodes(frames, stiffnesses, eq_lengths, edges,
                          constrained_nodes=ALLO_CONSTRAINED_NODES,
                          n_modes_track: int = CFG.n_modes_track):
    """Full eigenvalue spectrum + lowest ``n_modes_track+1`` eigenvectors at every frame."""
    from analysis.hessian import compute_hessian_spectrum
    T = len(frames)
    specs, vecs_low = [], []
    for pos in frames:
        vals, vecs, _ = compute_hessian_spectrum(
            pos, edges, stiffnesses, eq_lengths,
            constrained_nodes=constrained_nodes, n_modes=None)
        specs.append(vals)
        K = min(n_modes_track + 1, vecs.shape[1])
        vecs_low.append(vecs[:, :K].T)     # (K, M)
    return np.stack(specs), np.stack(vecs_low)


def aux_spectrum_and_lowmodes(frames, network, boundary, constrained: bool = True,
                              n_modes_track: int = CFG.n_modes_track):
    """Auxetic analogue of ``spectrum_and_lowmodes``: full eigenvalue spectrum +
    lowest ``n_modes_track+1`` eigenvectors of the (constrained) elastic Hessian
    at every compression frame, for diabatic avoided-crossing tracking along an
    auxetic compression trajectory.

    ``constrained`` -> pin the union of the top/bottom boundary nodes
    (``analysis.timestep_sweep.global_constrained_nodes``); rest lengths are the
    network's undeformed edge lengths and the stiffnesses are ``network.stiffnesses``
    (set these on the passed network to track a before/after-training spectrum).
    """
    from analysis.hessian import compute_hessian_spectrum
    from analysis.timestep_sweep import global_constrained_nodes
    cn = global_constrained_nodes(boundary) if constrained else None
    pos0 = np.asarray(network.positions)
    edges = np.asarray(network.edges)
    rest_lengths = np.linalg.norm(pos0[edges[:, 1]] - pos0[edges[:, 0]], axis=1)
    stiff = np.asarray(network.stiffnesses)
    specs, vecs_low = [], []
    for pos in frames:
        vals, vecs, _ = compute_hessian_spectrum(
            pos, edges, stiff, rest_lengths, constrained_nodes=cn, n_modes=None)
        specs.append(vals)
        K = min(n_modes_track + 1, vecs.shape[1])
        vecs_low.append(vecs[:, :K].T)     # (K, M)
    return np.stack(specs), np.stack(vecs_low)


def detect_avoided_crossings_diabatic(vecs_low, n_modes_track: int = CFG.n_modes_track):
    """(t, m) events: following mode *character* across ``[t, t+1]`` requires
    swapping the energy-sorted labels of adjacent branches ``m`` and ``m+1`` --
    the standard signature of an avoided crossing. Sign is a free gauge, so
    overlaps are compared in absolute value.
    """
    T, K, _ = vecs_low.shape
    n_modes_track = min(n_modes_track, K - 1)
    events = []
    for t in range(T - 1):
        Vt, Vt1 = vecs_low[t], vecs_low[t + 1]
        for m in range(n_modes_track):
            o_mm = abs(float(np.dot(Vt[m], Vt1[m])))
            o_mp = abs(float(np.dot(Vt[m], Vt1[m + 1])))
            o_pm = abs(float(np.dot(Vt[m + 1], Vt1[m])))
            o_pp = abs(float(np.dot(Vt[m + 1], Vt1[m + 1])))
            if o_mp > o_mm and o_pm > o_pp:
                events.append((t, m))
    return events


# --------------------------------------------------------------------------- #
# Correlation helpers                                                         #
# --------------------------------------------------------------------------- #
def spearman(x, y) -> float:
    """Spearman rank correlation with NaN pairs dropped; NaN if < 3 valid pairs."""
    from scipy.stats import spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    rho, _ = spearmanr(x[m], y[m])
    return float(rho)


# --------------------------------------------------------------------------- #
# Plotting helpers                                                            #
# --------------------------------------------------------------------------- #
def draw_network(ax, positions, edges, values, cmap="magma", log_color=True,
                 vmin=None, vmax=None, lw=1.6, alpha=0.95):
    """Draw a 2-D spring network, edges colored by ``values`` (e.g. stiffness).

    Returns the ``LineCollection`` so the caller can attach a colorbar.
    """
    import matplotlib.pyplot as plt
    from matplotlib import collections as mc
    from matplotlib.colors import Normalize

    positions = np.asarray(positions, float)
    v = np.log10(np.abs(values) + 1e-14) if log_color else np.asarray(values, float)
    if vmin is None:
        vmin = np.nanpercentile(v, 2)
    if vmax is None:
        vmax = np.nanpercentile(v, 98)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap(cmap)
    segs = [positions[e] for e in edges]
    lc = mc.LineCollection(segs, colors=[cm(norm(val)) for val in v], linewidths=lw, alpha=alpha)
    ax.add_collection(lc)
    ax.set_xlim(positions[:, 0].min() - 0.05, positions[:, 0].max() + 0.05)
    ax.set_ylim(positions[:, 1].min() - 0.05, positions[:, 1].max() + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")
    lc.set_clim(vmin, vmax)
    lc.set_cmap(cmap)
    return lc


SUBTASK_COLORS = ("#1f77b4", "#d62728")   # subtask 1, subtask 2
BEFORE_AFTER_COLORS = ("#7f7f7f", "#d62728")   # before training, best-loss step


# --------------------------------------------------------------------------- #
# Physics: auxetic ("global") compression + Poisson + Hessians                #
# --------------------------------------------------------------------------- #
AUX_FORCE_TYPE = "quadratic"
AUX_TOL = 1e-9


def aux_subtasks(task_config) -> list[tuple]:
    """``[(compression_strain, target_poisson), ...]`` for an auxetic task,
    ordered by descending ``|compression|`` (subtask 1 = deepest compression),
    matching ``analysis.timestep_sweep``'s convention."""
    cs = np.asarray(task_config["compression_strains"], float)
    tp = np.asarray(task_config["target_poisson_ratios"], float)
    order = np.argsort(-np.abs(cs))
    return list(zip(cs[order].tolist(), tp[order].tolist()))


def aux_compression_frames(network, boundary, compression_strain, n_steps=100):
    """Quasistatic compression trajectory (list of ``(N, 2)`` arrays).

    ``method='fire'`` (NOT the function's ``'newton'`` default): for the
    near-mechanism jammed auxetic networks the sparse Newton solver "converges"
    (``|F_free| < tol``) onto a spurious buckled branch that gives a wildly
    wrong lateral strain -- e.g. targeted task 0 relaxes to nu ~ -2.2 under
    Newton vs the trained nu ~ -0.8. Cython FIRE lands on the same equilibrium
    as training's own solver (``base.simulate.crf`` /
    ``compute_poisson_ratio_single_jax``), so the Fig 1/2/4 response curves then
    match the recomputed subtask losses.
    """
    import copy
    from base.simulate import compute_quasistatic_trajectory_auxetic
    return compute_quasistatic_trajectory_auxetic(
        copy.deepcopy(network), compression_strain,
        boundary["top"], boundary["bottom"],
        n_steps=n_steps, verbose=False, force_type=AUX_FORCE_TYPE, tol=AUX_TOL,
        method="fire",
    )


def poisson_along_traj(frames, boundary):
    """``(eps_yy, nu)`` arrays along a compression trajectory."""
    top, bot = boundary["top"], boundary["bottom"]
    left, right = boundary["left"], boundary["right"]
    p0 = np.asarray(frames[0])
    h0 = p0[top, 1].mean() - p0[bot, 1].mean()
    w0 = p0[right, 0].mean() - p0[left, 0].mean()
    eps_yy, nu = [], []
    for pos in frames:
        pos = np.asarray(pos)
        h = pos[top, 1].mean() - pos[bot, 1].mean()
        w = pos[right, 0].mean() - pos[left, 0].mean()
        ey = (h - h0) / h0
        ex = (w - w0) / w0
        eps_yy.append(ey)
        nu.append(-ex / ey if abs(ey) > 1e-10 else 0.0)
    return np.asarray(eps_yy), np.asarray(nu)


def aux_elastic_spectrum(frame_positions, network, boundary, constrained=True, n_modes=None):
    """Constrained (union of top+bottom fixed) or unconstrained elastic Hessian
    eigen-pairs at one compressed configuration."""
    from analysis.hessian import compute_hessian_spectrum
    from analysis.timestep_sweep import global_constrained_nodes
    cn = global_constrained_nodes(boundary) if constrained else None
    rest_lengths = np.linalg.norm(
        np.asarray(network.positions)[np.asarray(network.edges)[:, 1]]
        - np.asarray(network.positions)[np.asarray(network.edges)[:, 0]], axis=1)
    vals, vecs, _ = compute_hessian_spectrum(
        frame_positions, np.asarray(network.edges), np.asarray(network.stiffnesses),
        rest_lengths, constrained_nodes=cn, n_modes=n_modes)
    return vals, vecs


def aux_recompute_loss(network, boundary, compression_strain, target_poisson, n_strain_steps):
    """Faithful auxetic subtask loss ``(nu(K) - target)^2`` via the *training* JAX
    solver (``base.simulate.compute_poisson_ratio_single_jax``) -- the same call
    ``analysis.cost_utils.compute_cost_hessian`` differentiates. Use this (not the
    ``poisson_along_traj`` endpoint estimate) for the loss-faithfulness check."""
    import jax.numpy as jnp
    from base.simulate import compute_poisson_ratio_single_jax, crf as _crf
    edges = jnp.asarray(np.asarray(network.edges, dtype=np.int32))
    rest = jnp.asarray(np.asarray(network.rest_lengths, dtype=np.float64))
    pos = jnp.asarray(np.asarray(network.positions, dtype=np.float64).flatten())
    k = jnp.asarray(np.asarray(network.stiffnesses, dtype=np.float64))
    nu = compute_poisson_ratio_single_jax(
        _crf, k, edges, rest, pos,
        boundary["top"], boundary["bottom"], boundary["left"], boundary["right"],
        compression_strain, n_strain_steps)
    return float((float(nu) - target_poisson) ** 2)


def aux_cost_hessian(network, boundary, compression_strain, target_poisson,
                     k_eigs: int = CFG.k_cost_eigs, verbose: bool = False):
    """Top-``k_eigs`` cost-Hessian eigen-pairs for ONE auxetic subtask
    (``(nu(K) - target)^2`` w.r.t. stiffnesses). Thin wrapper around
    ``analysis.cost_utils.compute_cost_hessian``; returns ``(evals, evecs)``
    with ``evecs`` shaped ``(n_edges, k)``, largest-algebraic first."""
    from analysis.cost_utils import compute_cost_hessian
    evals, evecs = compute_cost_hessian(
        network, compression_strain, target_poisson, boundary,
        k_eigs=k_eigs, verbose=verbose)
    order = np.argsort(evals)[::-1]
    return evals[order], evecs[:, order]
