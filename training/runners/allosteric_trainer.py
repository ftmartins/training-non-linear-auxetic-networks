#!/usr/bin/env python3
"""
Allosteric network trainer — single-network script for SLURM array submission.

Dimensions:
  geometry    — controls network topology (random node perturbations)
  task        — controls training targets (soi1 / soi2 → strain pairs)
  realization — controls initial stiffnesses (IID uniform draw)

Output layout:
  <output_dir>/geometry_<gid>/task_<tid>/realization_<rid>/
    tasks.txt          — geometry_seed, strain_output2, strain_output
    stiffnesses.npy
    mse1.npy
    mse2.npy

Training runs for `training_steps` (see --training-steps); to train longer,
re-invoke with a larger --training-steps (resumes from the existing
checkpoint). Learning rate starts at --learning-rate and is scaled down by
training.src.lr_schedule every 1000 steps if loss has plateaued or
overshot — see that module for details.
"""

import argparse
import os
import sys
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import fsolve
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import training.lammps_utils as f
import training.jax_actuation as jx
from training.src.run_provenance import (
    save_run_provenance,
    save_training_meta,
    save_code_snapshot,
    has_critical_mismatch,
    HyperparameterMismatchError,
    DEFAULT_CRITICAL_KEYS,
)
from training.src import lr_schedule
from training.src.good_realizations import get_realization_seed

# Code archived alongside each realization's results (see save_code_snapshot)
# so the exact code that produced it is recoverable without relying on git
# history/dirty state alone.
_CODE_SNAPSHOT_FILES = [
    Path(__file__),
    Path(f.__file__),
    Path(jx.__file__),
    Path(__file__).resolve().parent.parent / 'src' / 'run_provenance.py',
]

# ── Constants ─────────────────────────────────────────────────────────────────
K_MIN    = 1e-4
K_MAX    = 1e1
ETA      = 1.0
K_OUTPUT = 1e3
LEARNING_RATE = 1.0     # starting lr, decayed by lr_schedule; override with --learning-rate

# Physics backend for evaluate_actuation: 'jax_fire' (default, same
# differentiable FIRE solver as auxetic training) or 'lammps' (legacy).
DEFAULT_SOLVER = 'lammps'

# Fixed input-actuation schedule, shared by training and by post-training
# analysis (analysis/timestep_sweep.py) so both use identical physics calls.
STRAIN_INPUT   = 1.0
STRAIN_INPUT2  = 0.5
NSTEPS_TASK1   = 40
NSTEPS_TASK2   = 20

N_GEOMETRIES   = 5
N_TASKS        = 5
N_REALIZATIONS = 5
N_TRAINING_STEPS = 1_000

# Non-overlapping seed namespaces
_GEOMETRY_BASE    = 1_000_000
_TASK_BASE        = 2_000_000
_REALIZATION_BASE = 3_000_000

# ── Targeted ensemble ─────────────────────────────────────────────────────────
# Fixed geometry shared across all 5 tasks; IC seed independent of task.
# Each entry: strain_output2 = target at input_strain 0.5,
#             strain_output  = target at input_strain 1.0.
_TARGETED_GEOMETRY_SEED = _GEOMETRY_BASE + 100   # = 1_000_100

TARGETED_ENSEMBLE = [
    {'strain_output2': -0.6, 'strain_output': -0.8},
    {'strain_output2': -0.8, 'strain_output': -0.8},
    {'strain_output2': -1.0, 'strain_output': -0.8},
    {'strain_output2': -0.4, 'strain_output': -0.8},
    {'strain_output2': -0.2, 'strain_output': -0.8},
]


# ── Seed helpers ──────────────────────────────────────────────────────────────

def geometry_seed(gid: int) -> int:
    return _GEOMETRY_BASE + gid

def task_rng(tid: int) -> np.random.RandomState:
    return np.random.RandomState(_TASK_BASE + tid)

def realization_rng(rid: int) -> np.random.RandomState:
    return np.random.RandomState(_REALIZATION_BASE + rid)


# ── Network creation (verbatim from notebook) ─────────────────────────────────

def create_network(L, p, R):
    """Geometry seeded externally via random.seed() before this call."""
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3) / 2.0])
    moves = np.array([[(random.random() - 0.5) * 2 * p for _ in range(2)]
                      for _ in range(2 * L ** 2)])
    nodes = np.array([])
    for xidx in range(L):
        for yidx in range(int((2 / np.sqrt(3)) * L)):
            node = ((xidx - int((L / 2) * (1 - 1 / np.sqrt(3))) - np.floor(yidx / 2)) * a1
                    + (yidx - int((1 / np.sqrt(3)) * L)) * a2
                    + moves[len(nodes) + 1])
            if np.linalg.norm(node) < L / 2 and len(nodes) == 0:
                nodes = node
            elif np.linalg.norm(node) < L / 2:
                nodes = np.vstack((nodes, node))

    incidence_matrix = np.array([])
    for i in range(len(nodes)):
        for j in range(i):
            if np.linalg.norm(nodes[i] - nodes[j]) < R:
                row = np.zeros(len(nodes))
                row[j] = 1; row[i] = -1
                if len(incidence_matrix) == 0:
                    incidence_matrix = row
                else:
                    incidence_matrix = np.vstack((incidence_matrix, row))

    in_node_1 = np.where(
        np.abs(nodes[:, 0]) + nodes[:, 1]
        == np.min(np.abs(nodes[:, 0]) + nodes[:, 1])
    )[0][0]
    bonds = np.where(np.abs(incidence_matrix[:, in_node_1]) == 1)[0]
    nbrs = [
        np.delete(
            np.where(np.abs(incidence_matrix[bonds[i]]) == 1)[0],
            np.where(np.where(np.abs(incidence_matrix[bonds[i]]) == 1)[0] == in_node_1)[0][0],
        )[0]
        for i in range(len(bonds))
    ]
    absxpy = [np.abs(nodes[nbrs[i]][0]) + nodes[nbrs[i]][1] for i in range(len(nbrs))]
    in_node_2 = nbrs[np.where(absxpy == min(absxpy))[0][0]]
    if nodes[in_node_1][0] > nodes[in_node_2][0]:
        in_node_1, in_node_2 = in_node_2, in_node_1

    out_node_1 = np.where(
        np.abs(nodes[:, 0]) - nodes[:, 1]
        == np.min(np.abs(nodes[:, 0]) - nodes[:, 1])
    )[0][0]
    bonds = np.where(np.abs(incidence_matrix[:, out_node_1]) == 1)[0]
    nbrs = [
        np.delete(
            np.where(np.abs(incidence_matrix[bonds[i]]) == 1)[0],
            np.where(np.where(np.abs(incidence_matrix[bonds[i]]) == 1)[0] == out_node_1)[0][0],
        )[0]
        for i in range(len(bonds))
    ]
    absxpy = [np.abs(nodes[nbrs[i]][0]) - nodes[nbrs[i]][1] for i in range(len(nbrs))]
    out_node_2 = nbrs[np.where(absxpy == min(absxpy))[0][0]]

    def Rot(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta),  np.cos(theta)]])

    def _f(theta):
        return (np.sin(theta) * (nodes[in_node_2] - nodes[in_node_1])[0]
                + np.cos(theta) * (nodes[in_node_2] - nodes[in_node_1])[1])

    theta_sol, _, ier, msg = fsolve(_f, 0, full_output=True)
    if ier != 1:
        raise RuntimeError(f"fsolve failed to find rotation angle: {msg}")
    theta = theta_sol[0]
    for i in range(len(nodes)):
        nodes[i] = Rot(theta) @ (nodes[i] - nodes[in_node_1]) + nodes[in_node_1]

    special = [in_node_1, in_node_2, out_node_1, out_node_2]
    nodesnew = nodes[in_node_1]
    nodesnew = np.vstack((nodesnew, nodes[in_node_2]))
    nodesnew = np.vstack((nodesnew, nodes[out_node_1]))
    nodesnew = np.vstack((nodesnew, nodes[out_node_2]))
    for i in [i for i in range(len(nodes)) if i not in special]:
        nodesnew = np.vstack((nodesnew, nodes[i]))
    nodes = nodesnew

    incidence_matrix = np.array([])
    for i in range(len(nodes)):
        for j in range(i):
            if np.linalg.norm(nodes[i] - nodes[j]) < R:
                row = np.zeros(len(nodes))
                row[j] = 1; row[i] = -1
                if len(incidence_matrix) == 0:
                    incidence_matrix = row
                else:
                    incidence_matrix = np.vstack((incidence_matrix, row))

    incidence_matrix = np.delete(incidence_matrix, 0, axis=0)
    eq_lengths  = np.linalg.norm(incidence_matrix @ nodes, axis=1)
    stiffnesses = np.ones(len(incidence_matrix))
    return nodes, incidence_matrix, eq_lengths, stiffnesses


# ── Geometry: load from disk or recreate from seed ────────────────────────────

def load_or_create_geometry(output_path, gseed):
    """
    Load nodes/incidence_matrix/eq_lengths from output_path if present; otherwise
    recreate from gseed and save them.  Raises RuntimeError if existing stiffnesses
    imply a different edge count than the geometry.
    """
    nodes_path = os.path.join(output_path, 'nodes.npy')
    inc_path   = os.path.join(output_path, 'incidence_matrix.npy')
    eq_path    = os.path.join(output_path, 'eq_lengths.npy')

    if (os.path.exists(nodes_path) and os.path.exists(inc_path)
            and os.path.exists(eq_path)):
        nodes            = np.load(nodes_path)
        incidence_matrix = np.load(inc_path)
        eq_lengths       = np.load(eq_path)
        print("  Geometry: loaded from disk.")
    else:
        print("  Geometry: files missing — recreating from seed.")
        random.seed(gseed)
        nodes, incidence_matrix, eq_lengths, _ = create_network(10, 0.15, 1.6)
        np.save(nodes_path, nodes)
        np.save(inc_path,   incidence_matrix)
        np.save(eq_path,    eq_lengths)

    stiff_path = os.path.join(output_path, 'stiffnesses.npy')
    if os.path.exists(stiff_path):
        n_edges_stiff = len(np.load(stiff_path))
        n_edges_geom  = len(incidence_matrix)
        if n_edges_stiff != n_edges_geom:
            raise RuntimeError(
                f"Edge-count mismatch: stiffnesses.npy has {n_edges_stiff} edges "
                f"but the geometry produces {n_edges_geom} edges. "
                f"The network-creation code may have been modified."
            )

    return nodes, incidence_matrix, eq_lengths


# ── Resume: find the latest NaN-free stiffness state ──────────────────────────

def load_resume_state(output_path):
    """
    Return (stiffnesses, mse1, mse2, start_step) if a prior run exists, else None.

    stiffnesses.npy is the primary source; stiffnesses_ckpt.npy + ckpt_step.txt
    (written at each 500-step checkpoint) serve as a fallback if the primary file
    contains NaNs (run diverged after the last clean save).
    """
    stiff_path = os.path.join(output_path, 'stiffnesses.npy')
    if not os.path.exists(stiff_path):
        return None

    mse1_path = os.path.join(output_path, 'mse1.npy')
    mse2_path = os.path.join(output_path, 'mse2.npy')
    mse1 = np.load(mse1_path) if os.path.exists(mse1_path) else np.array([])
    mse2 = np.load(mse2_path) if os.path.exists(mse2_path) else np.array([])

    stiff = np.load(stiff_path)
    if not np.any(np.isnan(stiff)):
        start_step = len(mse1)
        print(f"  Resume: found clean stiffnesses at step {start_step}.")
        return stiff, mse1, mse2, start_step

    # stiffnesses.npy has NaNs — fall back to last NaN-free checkpoint
    ckpt_stiff_path = os.path.join(output_path, 'stiffnesses_ckpt.npy')
    ckpt_step_path  = os.path.join(output_path, 'ckpt_step.txt')
    if os.path.exists(ckpt_stiff_path) and os.path.exists(ckpt_step_path):
        ckpt_stiff = np.load(ckpt_stiff_path)
        ckpt_step  = int(np.loadtxt(ckpt_step_path))
        print(f"  Resume: stiffnesses.npy contains NaNs; "
              f"rolling back to checkpoint at step {ckpt_step}.")
        return ckpt_stiff, mse1[:ckpt_step], mse2[:ckpt_step], ckpt_step

    print("  Resume: stiffnesses.npy contains NaNs and no clean checkpoint found; "
          "starting fresh.")
    return None


# ── Actuation evaluation (free + clamped actuation pair at fixed stiffnesses) ─

def evaluate_actuation(nodes, incidence_matrix, stiffnesses, tod, dx, nsteps, eta=ETA,
                       return_trajectory=False, solver=DEFAULT_SOLVER, compute_clamped=True):
    """
    Run one free(+clamped) actuation pair at fixed stiffnesses.

    This is the read-only half of the training step (no stiffness update) —
    used both by the training loop and by post-training analysis
    (analysis/timestep_sweep.py) so recomputed loss uses the exact same
    physics calls as training.

    Parameters
    ----------
    return_trajectory : bool
        If True, also return the full quasistatic trajectory (one frame per
        pull step, length `nsteps`) instead of just its final frame — used by
        analysis.timestep_sweep.sweep_allosteric to sample the elastic
        Hessian along the actuation trajectory. The clamped run's trajectory
        is returned unless compute_clamped=False, in which case the free
        run's trajectory is returned instead.
    solver : 'jax_fire' (default) or 'lammps'
        Physics backend. 'jax_fire' uses the same differentiable FIRE solver
        (base.simulate.make_compute_response_fire) as auxetic training;
        'lammps' is the original LAMMPS-based implementation, kept as an
        option.
    compute_clamped : bool
        `mse` only ever depends on the free run (see below) — the clamped
        (output-spring) run exists solely to build the Hebbian learning
        signal (learning_update). Callers that only need `mse` (e.g. a
        gradient-descent loss, or the cost-Hessian finite-difference recipe
        in analysis/timestep_sweep.py) can pass False to skip that extra,
        smaller-timestep FIRE ramp entirely. Ignored by the 'lammps' solver,
        which always computes both runs.

    Returns
    -------
    mse : float                    (||nodes_free[2]-nodes_free[3]|| - tod)^2
    nodes_free : (N, 2) array      equilibrium positions, free (uncalibrated) run
    nodes_clamped : (N, 2) array or None
        Equilibrium positions, clamped (output-spring) run; None when
        compute_clamped=False.
    frames : list of (N, 2) arrays, length nsteps
        Only returned when return_trajectory=True. The clamped run's frames,
        unless compute_clamped=False, in which case the free run's frames.
    """
    if solver == 'jax_fire':
        return _evaluate_actuation_jax(nodes, incidence_matrix, stiffnesses, tod, dx, nsteps,
                                       eta=eta, return_trajectory=return_trajectory,
                                       compute_clamped=compute_clamped)
    elif solver == 'lammps':
        return _evaluate_actuation_lammps(nodes, incidence_matrix, stiffnesses, tod, dx, nsteps,
                                          eta=eta, return_trajectory=return_trajectory)
    raise ValueError(f"Unknown solver: {solver!r} (expected 'jax_fire' or 'lammps')")


def _evaluate_actuation_jax(nodes, incidence_matrix, stiffnesses, tod, dx, nsteps, eta=ETA,
                            return_trajectory=False, compute_clamped=True):
    edges = f.incidence_to_edges(incidence_matrix)
    rest_lengths = np.linalg.norm(nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1)

    # Only return_trajectory=True needs every frame; otherwise the jitted
    # final-frame-only ramp is dramatically cheaper for repeated calls (see
    # strain_network_jax_final's docstring) and gives identical values.
    if return_trajectory:
        frames_free = jx.strain_network_jax(jx.FREE_CRF, nodes, edges, rest_lengths, stiffnesses,
                                            0, 1, dx=dx, nsteps=nsteps)
        nodes_free = frames_free[nsteps - 1]
    else:
        frames_free = None
        nodes_free = jx.strain_network_jax_final(jx.FREE_CRF, nodes, edges, rest_lengths,
                                                  stiffnesses, 0, 1, dx=dx, nsteps=nsteps)
    mse = (np.linalg.norm(nodes_free[2] - nodes_free[3]) - tod) ** 2

    if not compute_clamped:
        if return_trajectory:
            return mse, nodes_free, None, frames_free
        return mse, nodes_free, None

    cod = np.linalg.norm(nodes_free[3] - nodes_free[2])

    edges_clamped        = np.vstack([edges, (2, 3)])
    rest_lengths_clamped = np.append(rest_lengths, eta * tod + (1 - eta) * cod)
    stiffnesses_clamped  = np.append(stiffnesses, K_OUTPUT)

    # CLAMPED_CRF, not FREE_CRF: the output spring's stiffness (K_OUTPUT)
    # is far above the base network's range and destabilizes FREE_CRF's
    # larger timestep (see training/jax_actuation.py).
    if return_trajectory:
        frames_clamped = jx.strain_network_jax(jx.CLAMPED_CRF, nodes, edges_clamped,
                                               rest_lengths_clamped, stiffnesses_clamped,
                                               0, 1, dx=dx, nsteps=nsteps)
        nodes_clamped = frames_clamped[nsteps - 1]
        return mse, nodes_free, nodes_clamped, frames_clamped

    nodes_clamped = jx.strain_network_jax_final(jx.CLAMPED_CRF, nodes, edges_clamped,
                                                rest_lengths_clamped, stiffnesses_clamped,
                                                0, 1, dx=dx, nsteps=nsteps)
    return mse, nodes_free, nodes_clamped


def _evaluate_actuation_lammps(nodes, incidence_matrix, stiffnesses, tod, dx, nsteps, eta=ETA,
                               return_trajectory=False):
    # Each call gets its own scratch dir: evaluate_actuation is invoked both
    # from the training loop (already cwd-isolated per SLURM array task) and
    # from post_training_sweep.py (runs in the shared $SLURM_SUBMIT_DIR), so
    # relying on the caller's cwd let concurrent array tasks clobber each
    # other's data_free.network / bond_coeffs_free.in.
    work_dir = tempfile.mkdtemp(prefix="allosteric_actuation_")
    try:
        f.write_lammps_data("data_free.network", nodes, incidence_matrix, stiffnesses,
                            work_dir=work_dir)
        nodes_free = f.strain_network("data_free.network", 0, 1, clamped=False,
                                      dx=dx, nsteps=nsteps, work_dir=work_dir)[nsteps - 1]
        cod = np.linalg.norm(nodes_free[3] - nodes_free[2])
        f.write_lammps_data("data_clamped.network", nodes, incidence_matrix, stiffnesses,
                            id_outA=2, id_outB=3,
                            target_output_distance=eta * tod + (1 - eta) * cod,
                            k_output=K_OUTPUT, work_dir=work_dir)
        frames_clamped = f.strain_network("data_clamped.network", 0, 1, clamped=True,
                                          dx=dx, nsteps=nsteps, work_dir=work_dir)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    nodes_clamped = frames_clamped[nsteps - 1]
    mse = (np.linalg.norm(nodes_free[2] - nodes_free[3]) - tod) ** 2
    if return_trajectory:
        return mse, nodes_free, nodes_clamped, frames_clamped
    return mse, nodes_free, nodes_clamped


# ── Learning rule ─────────────────────────────────────────────────────────────

def learning_update(nodesfree, nodesclamped, tod, eq_lengths,
                    stiffnesses, incidence_matrix, eta, learning_rate):
    dVfree    = np.linalg.norm(incidence_matrix @ nodesfree,    axis=1) - eq_lengths
    dVclamped = np.linalg.norm(incidence_matrix @ nodesclamped, axis=1) - eq_lengths
    factors   = (dVfree - dVclamped) * dVfree
    delta_K   = (1 / eta) * stiffnesses * factors
    stiffnesses = np.clip(stiffnesses + delta_K, K_MIN, K_MAX)
    mse = (np.linalg.norm(nodesfree[2] - nodesfree[3]) - tod) ** 2
    return stiffnesses, mse, delta_K


# ── Core training loop ────────────────────────────────────────────────────────
def _run_training_loop(nodes, incidence_matrix, eq_lengths, stiffnesses,
                       learning_rate, tod, tod2, dinputdistance, dinputdistance2,
                       nsteps, nsteps2, n_steps, output_path,
                       msearray=None, msearray2=None, step_offset=0,
                       best_stiffnesses=None, best_combined_mse=np.inf,
                       solver=DEFAULT_SOLVER):
    if msearray  is None: msearray  = np.array([])
    if msearray2 is None: msearray2 = np.array([])
    if best_stiffnesses is None:
        best_stiffnesses = stiffnesses.copy()

    dx  = dinputdistance  / nsteps
    dx2 = dinputdistance2 / nsteps2
    update_mag = np.nan

    best_updated = False

    pbar = tqdm(range(n_steps),
                desc=f'(mse1={np.nan:.4e}, mse2={np.nan:.4e}, best_combined={best_combined_mse:.4e}), updated_mag={update_mag:.4e}')
    for j in pbar:
        # Task 1 and task 2 both evaluate against the SAME stiffnesses (the
        # value that will be checkpointed/saved for this iteration) instead
        # of task 2 seeing task 1's already-applied update — that alternating
        # scheme left mse2[j] computed at an intermediate stiffness state
        # that was never itself saved anywhere, so total_loss/mse1/mse2 could
        # never be exactly reconstructed from a saved stiffness snapshot (see
        # analysis.timestep_sweep._select_local_threshold_indices). Both
        # updates are computed here but only applied together afterward, so
        # mse1[j] and mse2[j] are both exactly loss(stiffnesses-at-start-of-
        # iteration-j).
        _, nodes_free, nodes_clamped = evaluate_actuation(
            nodes, incidence_matrix, stiffnesses, tod, dx, nsteps, solver=solver)
        _, mse, delta_K1 = learning_update(
            nodes_free, nodes_clamped, tod, eq_lengths,
            stiffnesses, incidence_matrix, ETA, learning_rate)

        _, nodes_free, nodes_clamped = evaluate_actuation(
            nodes, incidence_matrix, stiffnesses, tod2, dx2, nsteps2, solver=solver)
        _, mse2, delta_K2 = learning_update(
            nodes_free, nodes_clamped, tod2, eq_lengths,
            stiffnesses, incidence_matrix, ETA, learning_rate)

        # Snapshot the stiffnesses that mse/mse2 above were actually measured
        # at, BEFORE folding in this iteration's update — this (not the
        # post-update `stiffnesses` a few lines down) is what best_stiffnesses
        # and stiffnesses_traj save below, so every persisted (stiffness,
        # loss) pair for analysis is exactly self-consistent, including at
        # the very last checkpoint of a run (previously that pairing only
        # became correct retroactively, once training had produced a *later*
        # mse entry to catch up to the checkpointed stiffness — which never
        # happens for the final checkpoint, and was being silently papered
        # over by an index-clamp in
        # analysis.timestep_sweep._select_local_threshold_indices instead of
        # actually matching). `stiffnesses.npy`/`stiffnesses_ckpt.npy` below
        # deliberately keep saving the POST-update value (unchanged) since
        # those are the resume checkpoint — resuming needs the state the
        # next iteration should start from, not the state whose loss was
        # last measured; do not "align" them to measured_stiffnesses too, or
        # resume will silently replay this iteration's update.
        measured_stiffnesses = stiffnesses
        # lr_scale is a pure function of the (mse1+mse2)/2 trajectory so far
        # (steps completed before this one) — see training.src.lr_schedule.
        # It's recomputed from msearray/msearray2, which are themselves
        # reloaded on resume, so lr_scale needs no separate persisted state.
        lr_scale, _ = lr_schedule.lr_scale_for_step((msearray + msearray2) / 2.0)
        current_lr = learning_rate * lr_scale
        stiffnesses = np.clip(stiffnesses + current_lr * (delta_K1 + delta_K2), K_MIN, K_MAX)

        update_mag = np.mean(np.log10(np.abs(current_lr * (delta_K1 + delta_K2))))

        msearray  = np.append(msearray,  mse)
        msearray2 = np.append(msearray2, mse2)

        combined = (mse + mse2) / 2.0
        if combined < best_combined_mse and not np.any(np.isnan(stiffnesses)):
            best_combined_mse = combined
            best_stiffnesses  = measured_stiffnesses.copy()
            best_updated      = True

        pbar.set_description(
                f'(loss={(mse+mse2):.4e}, mse1={mse:.4e}, mse2={mse2:.4e}, best_combined={best_combined_mse:.4e}), '
                f'update_mag={update_mag:.4e}, lr_scale={lr_scale:.3g}')

        global_step = step_offset + j + 1
        if global_step % 50 == 0:
            print(f"  step {global_step}: MSE1={mse:.4e}  MSE2={mse2:.4e}"
                  f"  best_combined={best_combined_mse:.4e}")
            np.save(os.path.join(output_path, 'stiffnesses.npy'), stiffnesses)
            np.save(os.path.join(output_path, 'mse1.npy'),        msearray)
            np.save(os.path.join(output_path, 'mse2.npy'),        msearray2)
            if not np.any(np.isnan(stiffnesses)):
                np.save(os.path.join(output_path, 'stiffnesses_ckpt.npy'), stiffnesses)
                np.savetxt(os.path.join(output_path, 'ckpt_step.txt'),
                           [global_step], fmt='%d')
            if best_updated:
                np.save(os.path.join(output_path, 'best_stiffnesses.npy'), best_stiffnesses)
                np.savetxt(os.path.join(output_path, 'best_combined_mse.txt'),
                           [best_combined_mse])
                best_updated = False
            # Append measured_stiffnesses (see note above the snapshot,
            # earlier in this loop) — the stiffness mse1/mse2's last entry
            # was actually computed at — paired with its own exact array
            # index (len(msearray) - 1), not global_step. That index is
            # always in range for mse1[steps[c]]/mse2[steps[c]] (see
            # analysis.timestep_sweep._select_local_threshold_indices) since
            # it is, by construction, the index msearray/msearray2 were just
            # appended to — including on this, possibly the run's last,
            # checkpoint.
            traj_path  = os.path.join(output_path, 'stiffnesses_traj.npy')
            steps_path = os.path.join(output_path, 'stiffnesses_traj_steps.npy')
            if os.path.exists(traj_path) and os.path.exists(steps_path):
                traj  = np.vstack([np.load(traj_path),  measured_stiffnesses])
                steps = np.append(np.load(steps_path), len(msearray) - 1)
            else:
                traj  = measured_stiffnesses[np.newaxis, :]
                steps = np.array([len(msearray) - 1])
            np.save(traj_path,  traj)
            np.save(steps_path, steps)

        if mse/msearray[0] < 8e-5 and mse2/msearray2[0] < 8e-5:
            np.save(os.path.join(output_path, 'stiffnesses.npy'), stiffnesses)
            np.save(os.path.join(output_path, 'mse1.npy'),        msearray)
            np.save(os.path.join(output_path, 'mse2.npy'),        msearray2)
            if not np.any(np.isnan(stiffnesses)):
                np.save(os.path.join(output_path, 'stiffnesses_ckpt.npy'), stiffnesses)
                np.savetxt(os.path.join(output_path, 'ckpt_step.txt'),
                           [global_step], fmt='%d')
            if best_updated:
                np.save(os.path.join(output_path, 'best_stiffnesses.npy'), best_stiffnesses)
                np.savetxt(os.path.join(output_path, 'best_combined_mse.txt'),
                           [best_combined_mse])
                best_updated = False
            # Append measured_stiffnesses (see note above the snapshot,
            # earlier in this loop) — the stiffness mse1/mse2's last entry
            # was actually computed at — paired with its own exact array
            # index (len(msearray) - 1), not global_step. That index is
            # always in range for mse1[steps[c]]/mse2[steps[c]] (see
            # analysis.timestep_sweep._select_local_threshold_indices) since
            # it is, by construction, the index msearray/msearray2 were just
            # appended to — including on this, possibly the run's last,
            # checkpoint.
            traj_path  = os.path.join(output_path, 'stiffnesses_traj.npy')
            steps_path = os.path.join(output_path, 'stiffnesses_traj_steps.npy')
            if os.path.exists(traj_path) and os.path.exists(steps_path):
                traj  = np.vstack([np.load(traj_path),  measured_stiffnesses])
                steps = np.append(np.load(steps_path), len(msearray) - 1)
            else:
                traj  = measured_stiffnesses[np.newaxis, :]
                steps = np.array([len(msearray) - 1])
            np.save(traj_path,  traj)
            np.save(steps_path, steps)


            print(f"  Early stop at step {global_step}: both tasks converged.")
            break

    return msearray, msearray2, stiffnesses, best_stiffnesses, best_combined_mse


# ── Success check ─────────────────────────────────────────────────────────────

def check_success(msearray1, msearray2):
    if len(msearray1) == 0 or len(msearray2) == 0:
        return False
    combined = (msearray1 + msearray2) / 2.0
    ratio = np.min(combined) / combined[0]
    print(f"  Success check: min_loss/loss[0] = {ratio:.4e} "
          f"({'PASS' if ratio < 1e-6 else 'FAIL'})")
    return ratio < 0.01



# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train one allosteric network (geometry × task × realization).')
    parser.add_argument('--geometry-id',    type=int, default=0,
                        help=f'Geometry index (0 to {N_GEOMETRIES-1}); ignored with --targeted-ensemble')
    parser.add_argument('--task-id',        type=int, required=True,
                        help=f'Task index (0 to {N_TASKS-1})')
    parser.add_argument('--realization-id', type=int, required=True,
                        help=f'Realization index (0 to {N_REALIZATIONS-1})')
    parser.add_argument('--training-steps', type=int, default=N_TRAINING_STEPS)
    parser.add_argument('--output-dir',     type=str,
                        # '_aug' keeps the raw-gradient/lr-schedule runs (this
                        # branch) from writing into the pre-existing
                        # normalized-gradient results — resume/checkpoint
                        # logic is unchanged, it just now resumes from
                        # whatever's under this new path instead.
                        default='/data2/shared/felipetm/allosteric_nets_aug')
    parser.add_argument('--targeted-ensemble', action='store_true',
                        help='Use TARGETED_ENSEMBLE fixed tasks with a shared fixed geometry')
    parser.add_argument('--solver', choices=['jax_fire', 'lammps'], default=DEFAULT_SOLVER,
                        help="Physics backend for evaluate_actuation (default: %(default)s)")
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE,
                        help="Starting learning rate; scaled down by training.src.lr_schedule "
                             "every 1000 steps on plateau/overshoot (default: %(default)s)")
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow this run to replace previously recorded solver/learning_rate/'
                             'k_min/k_max/eta in training_meta.json instead of failing on mismatch')
    args = parser.parse_args()

    gid = args.geometry_id
    tid = args.task_id
    rid = args.realization_id
    training_steps = args.training_steps
    output_dir     = args.output_dir
    targeted       = args.targeted_ensemble
    solver         = args.solver
    learning_rate  = args.learning_rate
    overwrite      = args.overwrite

    log_dir = '/home1/felipetm/auxetic_networks/ensemble_training/Logs'

    mode_tag = 'targeted' if targeted else f'geometry={gid}'
    print(f"=== Allosteric trainer: {mode_tag}, task={tid}, realization={rid}, solver={solver} ===")

    # rid (0..N_REALIZATIONS-1) is a serial index into the screened-good seeds
    # for this task[, geometry], not a literal RNG seed — see
    # training/src/good_realizations.py / docs/realization_screening.md.
    # Computed here (before the critical-hparams check below) so a mismatch
    # between this run's screened seed and whatever seed actually produced a
    # resumed checkpoint is caught the same way as any other critical key,
    # instead of silently resuming a trajectory screening never selected.
    _seed_kind = 'allosteric_targeted' if targeted else 'allosteric_general'
    screened_seed = get_realization_seed(_seed_kind, tid, rid, geometry_id=None if targeted else gid)

    # Per-job temp dir so parallel LAMMPS jobs don't share files
    work_dir = f"/tmp/allosteric_{'tgt' if targeted else f'g{gid}'}_t{tid}_r{rid}_{os.getpid()}"
    os.makedirs(work_dir, exist_ok=True)
    original_dir = os.getcwd()
    os.chdir(work_dir)

    # Output subfolder
    geom_dir = 'geometry_targeted' if targeted else f'geometry_{gid}'
    output_path = os.path.join(output_dir, geom_dir, f'task_{tid}', f'realization_{rid}')

    print(geom_dir, output_path, 'test outputpath')
    os.makedirs(output_path, exist_ok=True)

    # Persist run-level hyperparameters and provenance for this realization.
    # training_meta.json is the source of truth for which solver, learning
    # rate, and stiffness bounds actually produced this realization's saved
    # trajectory (these are otherwise hardcoded/CLI values with no other
    # per-run record); run_provenance.json additionally records the git
    # commit that produced it, appended on every invocation.
    #
    # Checked against THIS invocation's values BEFORE any resume state is
    # read — a mismatch on solver/learning_rate/k_min/k_max/eta fails outright
    # unless --overwrite is passed. On --overwrite, the realization's saved
    # state (stiffnesses/mse/best-state/geometry/meta) is wiped and restarted
    # from scratch under the new hyperparameters: resuming a trajectory
    # trained under the OLD settings and merely relabeling training_meta.json
    # with the NEW ones would make it describe a run that never actually
    # happened, defeating the point of tracking it. n_training_steps is
    # exempt: resuming to train for longer is expected and fine.
    #
    # eta is the coupled-learning nudge strength (rest_lengths_clamped =
    # eta*tod + (1-eta)*cod; delta_K is also divided by eta) — silently
    # changing it mid-run would mix trajectories computed under different
    # nudge strengths, same risk as changing learning_rate/k_min/k_max.
    _ALLOSTERIC_CRITICAL_KEYS = DEFAULT_CRITICAL_KEYS | {'eta'}
    current_hparams = {
        'solver': solver,
        'learning_rate': learning_rate,
        'k_min': K_MIN,
        'k_max': K_MAX,
        'eta': ETA,
        'n_training_steps': training_steps,
        'realization_seed': screened_seed,
    }
    if has_critical_mismatch(output_path, current_hparams, critical_keys=_ALLOSTERIC_CRITICAL_KEYS):
        if not overwrite:
            os.chdir(original_dir)
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HyperparameterMismatchError(
                f"{output_path}/training_meta.json already recorded different solver/"
                f"learning_rate/k_min/k_max/eta/realization_seed than this run's. Resuming would "
                f"mix incompatible settings into one training trajectory. Pass --overwrite to wipe "
                f"it and restart from scratch under the new hyperparameters."
            )
        print(f"  --overwrite set: recorded hyperparameters differ from this run's — wiping "
              f"{output_path} and restarting from scratch under the new hyperparameters.")
        shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)

    try:
        save_training_meta(output_path, current_hparams, critical_keys=_ALLOSTERIC_CRITICAL_KEYS)
        save_run_provenance(output_path, extra=current_hparams)
        save_code_snapshot(output_path, _CODE_SNAPSHOT_FILES)
    except Exception:
        os.chdir(original_dir)
        shutil.rmtree(work_dir, ignore_errors=True)
        raise

    try:
        # ── Build geometry (load if present, recreate from seed if missing) ───
        gseed = _TARGETED_GEOMETRY_SEED if targeted else geometry_seed(gid)
        nodes, incidence_matrix, eq_lengths = load_or_create_geometry(output_path, gseed)

        # ── Resolve task strains ──────────────────────────────────────────────
        strain_input  = STRAIN_INPUT
        strain_input2 = STRAIN_INPUT2
        if targeted:
            if tid >= len(TARGETED_ENSEMBLE):
                raise ValueError(f'--task-id {tid} out of range for TARGETED_ENSEMBLE '
                                 f'(max {len(TARGETED_ENSEMBLE)-1})')
            strain_output  = TARGETED_ENSEMBLE[tid]['strain_output']
            strain_output2 = TARGETED_ENSEMBLE[tid]['strain_output2']
            print(f"  Geometry seed : {gseed} (targeted, fixed)")
            print(f"  Task          : TARGETED_ENSEMBLE[{tid}]  "
                  f"→  strain_out={strain_output:.1f}, strain_out2={strain_output2:.1f}")
        else:
            trng = task_rng(tid)
            soi1 = int(trng.randint(1, 11))   # inclusive [1, 10], excludes strain=0
            soi2 = int(trng.randint(1, 11))
            strain_output  = -0.1 * soi2
            strain_output2 = -0.1 * soi1
            print(f"  Geometry seed : {gseed}")
            print(f"  Task          : soi1={soi1}, soi2={soi2}  "
                  f"→  strain_out={strain_output:.1f}, strain_out2={strain_output2:.1f}")

        np.savetxt(os.path.join(output_path, 'tasks.txt'),
                   [gseed, strain_output2, strain_output])

        tod  = (1 + strain_output)  * np.linalg.norm(nodes[3] - nodes[2])
        tod2 = (1 + strain_output2) * np.linalg.norm(nodes[3] - nodes[2])

        dinputdistance  = strain_input  * np.linalg.norm(nodes[0] - nodes[1])
        dinputdistance2 = strain_input2 * np.linalg.norm(nodes[0] - nodes[1])

        nsteps  = NSTEPS_TASK1
        nsteps2 = NSTEPS_TASK2

        # ── Resume or fresh start ─────────────────────────────────────────────
        resume = load_resume_state(output_path)
        if resume is not None:
            stiffnesses, msearray, msearray2, start_step = resume
            print(f"  Stiffnesses   : [{stiffnesses.min():.2f}, {stiffnesses.max():.2f}]"
                  f"  (resumed from step {start_step})")
        else:
            # screened_seed computed earlier, up with current_hparams.
            rrng = realization_rng(screened_seed)
            stiffnesses = rrng.uniform(K_MIN, K_MAX, size=len(incidence_matrix))
            msearray  = np.array([])
            msearray2 = np.array([])
            start_step = 0
            print(f"  Stiffnesses   : [{stiffnesses.min():.2f}, {stiffnesses.max():.2f}]")

        # Load best state (survives resume across SLURM restarts)
        best_path = os.path.join(output_path, 'best_stiffnesses.npy')
        bmse_path = os.path.join(output_path, 'best_combined_mse.txt')
        if os.path.exists(best_path) and os.path.exists(bmse_path):
            best_stiffnesses  = np.load(best_path)
            best_combined_mse = float(np.loadtxt(bmse_path))
            print(f"  Best state    : combined_mse={best_combined_mse:.4e} (loaded from disk)")
        else:
            best_stiffnesses  = None   # _run_training_loop initialises from stiffnesses
            best_combined_mse = np.inf

        print(f"  Training steps: {training_steps:,}")

        # ── Training (or its remaining portion, if resumed) ────────────────────
        remaining_steps = max(0, training_steps - start_step)
        if remaining_steps > 0:
            tag = ("Training" if start_step == 0
                   else f"Training resumed — {remaining_steps} steps remaining")
            print(f"\n--- {tag} ---")
            print('Learning rate: ', learning_rate)
            msearray, msearray2, stiffnesses, best_stiffnesses, best_combined_mse = \
                _run_training_loop(
                    nodes, incidence_matrix, eq_lengths, stiffnesses,
                    learning_rate, tod, tod2, dinputdistance, dinputdistance2,
                    nsteps, nsteps2, remaining_steps, output_path,
                    msearray=msearray, msearray2=msearray2,
                    step_offset=start_step,
                    best_stiffnesses=best_stiffnesses,
                    best_combined_mse=best_combined_mse,
                    solver=solver)

        if check_success(msearray, msearray2):
            print("\nTraining succeeded.")
        else:
            print("\nTraining did not reach the success threshold within "
                  f"{training_steps:,} steps. Re-invoke with a larger "
                  "--training-steps to continue from this checkpoint.")

        # ── Final save ────────────────────────────────────────────────────────
        np.save(os.path.join(output_path, 'stiffnesses.npy'),      stiffnesses)
        np.save(os.path.join(output_path, 'best_stiffnesses.npy'), best_stiffnesses)
        np.save(os.path.join(output_path, 'mse1.npy'),             msearray)
        np.save(os.path.join(output_path, 'mse2.npy'),             msearray2)
        np.savetxt(os.path.join(output_path, 'best_combined_mse.txt'), [best_combined_mse])
        print(f"\nResults saved to {output_path}"
              f"  (best combined MSE: {best_combined_mse:.4e})")

    finally:
        os.chdir(original_dir)
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
