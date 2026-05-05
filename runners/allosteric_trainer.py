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

Retry logic: attempt 1 runs N_TRAINING_STEPS; on failure attempt 2 runs
2× steps with a recalibrated LR. If both fail the job resubmits with a
fresh realization index (rid + N_REALIZATIONS).
"""

import argparse
import os
import sys
import random
import shutil
import subprocess
import textwrap

import numpy as np
from scipy.optimize import fsolve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data', 'ToFelipe0422'))
import functions as f  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
K_MIN    = 1e-3
K_MAX    = 1e1
ETA      = 1.0
K_OUTPUT = 1e3
LR_TARGET_LOG = -3.5   # target mean(log10|delta_K|)

N_GEOMETRIES   = 5
N_TASKS        = 5
N_REALIZATIONS = 5
N_TRAINING_STEPS = 100_000

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

    theta = fsolve(_f, 0)[0]
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


# ── Learning rule ─────────────────────────────────────────────────────────────

def learning_update(nodesfree, nodesclamped, tod, eq_lengths,
                    stiffnesses, incidence_matrix, eta, learning_rate):
    dVfree    = np.linalg.norm(incidence_matrix @ nodesfree,    axis=1) - eq_lengths
    dVclamped = np.linalg.norm(incidence_matrix @ nodesclamped, axis=1) - eq_lengths
    factors   = (dVfree - dVclamped) * dVfree
    delta_K   = (learning_rate / eta) * stiffnesses * factors
    stiffnesses = np.clip(stiffnesses + delta_K, K_MIN, K_MAX)
    mse = (np.linalg.norm(nodesfree[2] - nodesfree[3]) - tod) ** 2
    return stiffnesses, mse, delta_K


# ── Learning-rate calibration ─────────────────────────────────────────────────

def calibrate_learning_rate(nodes, incidence_matrix, eq_lengths, stiffnesses,
                             eta, tod, dx, nsteps):
    """One LAMMPS step at lr=1; scale so mean(log10|delta_K|) = LR_TARGET_LOG."""
    f.write_lammps_data("data_free.network", nodes, incidence_matrix, stiffnesses)
    nodes_free = f.strain_network("data_free.network", 0, 1, clamped=False,
                                  dx=dx, nsteps=nsteps)[nsteps - 1]
    cod = np.linalg.norm(nodes_free[3] - nodes_free[2])

    f.write_lammps_data("data_clamped.network", nodes, incidence_matrix, stiffnesses,
                        id_outA=2, id_outB=3,
                        target_output_distance=eta * tod + (1 - eta) * cod,
                        k_output=K_OUTPUT)
    nodes_clamped = f.strain_network("data_clamped.network", 0, 1, clamped=True,
                                     dx=dx, nsteps=nsteps)[nsteps - 1]

    dVfree    = np.linalg.norm(incidence_matrix @ nodes_free,    axis=1) - eq_lengths
    dVclamped = np.linalg.norm(incidence_matrix @ nodes_clamped, axis=1) - eq_lengths
    factors   = (dVfree - dVclamped) * dVfree

    base = (1.0 / eta) * stiffnesses * factors
    mask = base != 0
    if not np.any(mask):
        print("  [calibrate] All delta_K zero; defaulting lr=1e-3")
        return 1e-3

    base_log_mean = np.mean(np.log10(np.abs(base[mask])))
    lr = float(10 ** (LR_TARGET_LOG - base_log_mean))
    print(f"  [calibrate] base_log_mean={base_log_mean:.3f} → lr={lr:.3e}")
    return lr


# ── Core training loop ────────────────────────────────────────────────────────

def _run_training_loop(nodes, incidence_matrix, eq_lengths, stiffnesses,
                       learning_rate, tod, tod2, dinputdistance, dinputdistance2,
                       nsteps, nsteps2, n_steps, output_path,
                       msearray=None, msearray2=None, step_offset=0):
    if msearray  is None: msearray  = np.array([])
    if msearray2 is None: msearray2 = np.array([])

    dx  = dinputdistance  / nsteps
    dx2 = dinputdistance2 / nsteps2

    for j in range(n_steps):
        # Task 1
        f.write_lammps_data("data_free.network", nodes, incidence_matrix, stiffnesses)
        nodes_free = f.strain_network("data_free.network", 0, 1, clamped=False,
                                      dx=dx, nsteps=nsteps)[nsteps - 1]
        cod = np.linalg.norm(nodes_free[3] - nodes_free[2])
        f.write_lammps_data("data_clamped.network", nodes, incidence_matrix, stiffnesses,
                            id_outA=2, id_outB=3,
                            target_output_distance=ETA * tod + (1 - ETA) * cod,
                            k_output=K_OUTPUT)
        nodes_clamped = f.strain_network("data_clamped.network", 0, 1, clamped=True,
                                         dx=dx, nsteps=nsteps)[nsteps - 1]
        stiffnesses, mse, _ = learning_update(
            nodes_free, nodes_clamped, tod, eq_lengths,
            stiffnesses, incidence_matrix, ETA, learning_rate)

        # Task 2
        f.write_lammps_data("data_free.network", nodes, incidence_matrix, stiffnesses)
        nodes_free = f.strain_network("data_free.network", 0, 1, clamped=False,
                                      dx=dx2, nsteps=nsteps2)[nsteps2 - 1]
        cod = np.linalg.norm(nodes_free[3] - nodes_free[2])
        f.write_lammps_data("data_clamped.network", nodes, incidence_matrix, stiffnesses,
                            id_outA=2, id_outB=3,
                            target_output_distance=ETA * tod2 + (1 - ETA) * cod,
                            k_output=K_OUTPUT)
        nodes_clamped = f.strain_network("data_clamped.network", 0, 1, clamped=True,
                                         dx=dx2, nsteps=nsteps2)[nsteps2 - 1]
        stiffnesses, mse2, _ = learning_update(
            nodes_free, nodes_clamped, tod2, eq_lengths,
            stiffnesses, incidence_matrix, ETA, learning_rate)

        msearray  = np.append(msearray,  mse)
        msearray2 = np.append(msearray2, mse2)

        global_step = step_offset + j + 1
        if global_step % 500 == 0:
            print(f"  step {global_step}: MSE1={mse:.4e}  MSE2={mse2:.4e}")
            np.save(os.path.join(output_path, 'stiffnesses.npy'), stiffnesses)
            np.save(os.path.join(output_path, 'mse1.npy'),        msearray)
            np.save(os.path.join(output_path, 'mse2.npy'),        msearray2)
            if not np.any(np.isnan(stiffnesses)):
                np.save(os.path.join(output_path, 'stiffnesses_ckpt.npy'), stiffnesses)
                np.savetxt(os.path.join(output_path, 'ckpt_step.txt'),
                           [global_step], fmt='%d')

        if mse < 5e-8 and mse2 < 5e-8:
            print(f"  Early stop at step {global_step}: both tasks converged.")
            break

    return msearray, msearray2, stiffnesses


# ── Success check ─────────────────────────────────────────────────────────────

def check_success(msearray1, msearray2):
    if len(msearray1) == 0 or len(msearray2) == 0:
        return False
    combined = (msearray1 + msearray2) / 2.0
    ratio = np.min(combined) / combined[0]
    print(f"  Success check: min_loss/loss[0] = {ratio:.4e} "
          f"({'PASS' if ratio < 1e-6 else 'FAIL'})")
    return ratio < 0.01


# ── Resubmission with a fresh realization ─────────────────────────────────────

def resubmit_new_realization(gid, tid, rid, training_steps, output_dir, log_dir,
                              conda_env='auxetic_nets', targeted=False):
    new_rid = rid + N_REALIZATIONS   # step outside the normal index range
    script_path = os.path.abspath(__file__)
    targeted_flag = '\n            --targeted-ensemble' if targeted else ''

    script = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH -t 5-00:00:00
        #SBATCH --qos=low
        #SBATCH --partition=low
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=2
        #SBATCH --mem=4gb
        #SBATCH --job-name=allosteric_g{gid}t{tid}r{new_rid}
        #SBATCH --output={log_dir}/allosteric_g{gid}t{tid}r{new_rid}_%j.out
        #SBATCH --error={log_dir}/allosteric_g{gid}t{tid}r{new_rid}_%j.err

        eval "$(conda shell.bash hook)"
        conda activate {conda_env}

        python {script_path} \\
            --geometry-id    {gid} \\
            --task-id        {tid} \\
            --realization-id {new_rid} \\
            --training-steps {training_steps} \\
            --output-dir     {output_dir}{targeted_flag}
    """)

    tmp = f"/tmp/allosteric_resubmit_g{gid}t{tid}r{new_rid}.sh"
    with open(tmp, 'w') as fh:
        fh.write(script)

    result = subprocess.run(['sbatch', tmp], capture_output=True, text=True)
    os.remove(tmp)

    if result.returncode == 0:
        print(f"  Resubmitted g{gid}/t{tid} with new realization {new_rid}: "
              f"{result.stdout.strip()}")
    else:
        print(f"  sbatch failed: {result.stderr.strip()}")


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
                        default='/data2/shared/felipetm/allosteric_nets')
    parser.add_argument('--targeted-ensemble', action='store_true',
                        help='Use TARGETED_ENSEMBLE fixed tasks with a shared fixed geometry')
    args = parser.parse_args()

    gid = args.geometry_id
    tid = args.task_id
    rid = args.realization_id
    training_steps = args.training_steps
    output_dir     = args.output_dir
    targeted       = args.targeted_ensemble

    log_dir = '/home1/felipetm/auxetic_networks/ensemble_training/Logs'

    mode_tag = 'targeted' if targeted else f'geometry={gid}'
    print(f"=== Allosteric trainer: {mode_tag}, task={tid}, realization={rid} ===")

    # Per-job temp dir so parallel LAMMPS jobs don't share files
    work_dir = f"/tmp/allosteric_{'tgt' if targeted else f'g{gid}'}_t{tid}_r{rid}_{os.getpid()}"
    os.makedirs(work_dir, exist_ok=True)
    original_dir = os.getcwd()
    os.chdir(work_dir)

    # Output subfolder
    geom_dir = 'geometry_targeted' if targeted else f'geometry_{gid}'
    output_path = os.path.join(output_dir, geom_dir, f'task_{tid}', f'realization_{rid}')
    os.makedirs(output_path, exist_ok=True)

    try:
        # ── Build geometry (load if present, recreate from seed if missing) ───
        gseed = _TARGETED_GEOMETRY_SEED if targeted else geometry_seed(gid)
        nodes, incidence_matrix, eq_lengths = load_or_create_geometry(output_path, gseed)

        # ── Resolve task strains ──────────────────────────────────────────────
        strain_input  = 1.0
        strain_input2 = 0.5
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
            soi1 = int(trng.randint(0, 11))   # inclusive [0, 10]
            soi2 = int(trng.randint(0, 11))
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

        nsteps  = 100
        nsteps2 = 50
        dx  = dinputdistance  / nsteps
        dx2 = dinputdistance2 / nsteps2

        # ── Resume or fresh start ─────────────────────────────────────────────
        resume = load_resume_state(output_path)
        if resume is not None:
            stiffnesses, msearray, msearray2, start_step = resume
            print(f"  Stiffnesses   : [{stiffnesses.min():.2f}, {stiffnesses.max():.2f}]"
                  f"  (resumed from step {start_step})")
        else:
            rrng = realization_rng(rid)
            stiffnesses = rrng.uniform(K_MIN, K_MAX, size=len(incidence_matrix))
            msearray  = np.array([])
            msearray2 = np.array([])
            start_step = 0
            print(f"  Stiffnesses   : [{stiffnesses.min():.2f}, {stiffnesses.max():.2f}]")

        print(f"  Training steps: {training_steps:,}")

        # ── Attempt 1 (or its remaining portion) ──────────────────────────────
        attempt1_remaining = max(0, training_steps - start_step)
        if attempt1_remaining > 0:
            tag = ("Attempt 1" if start_step == 0
                   else f"Attempt 1 resumed — {attempt1_remaining} steps remaining")
            print(f"\n--- {tag} ---")
            lr = calibrate_learning_rate(nodes, incidence_matrix, eq_lengths,
                                         stiffnesses.copy(), ETA, tod, dx, nsteps)
            msearray, msearray2, stiffnesses = _run_training_loop(
                nodes, incidence_matrix, eq_lengths, stiffnesses,
                lr, tod, tod2, dinputdistance, dinputdistance2,
                nsteps, nsteps2, attempt1_remaining, output_path,
                msearray=msearray, msearray2=msearray2,
                step_offset=start_step)

        if check_success(msearray, msearray2):
            print("\nAttempt 1 succeeded.")
        else:
            # ── Attempt 2: recalibrated LR, up to 2× more steps ──────────────
            # Total budget = training_steps (attempt 1) + 2 * training_steps (attempt 2).
            # If resuming mid-attempt-2, only the remaining portion is run.
            step_after_a1    = len(msearray)
            attempt2_remaining = max(0, 3 * training_steps - step_after_a1)
            if attempt2_remaining > 0:
                tag = ("Attempt 2 (2× steps, recalibrated LR)" if step_after_a1 <= training_steps
                       else f"Attempt 2 resumed — {attempt2_remaining} steps remaining")
                print(f"\n--- {tag} ---")
                lr2 = calibrate_learning_rate(nodes, incidence_matrix, eq_lengths,
                                              stiffnesses.copy(), ETA, tod, dx, nsteps)
                msearray, msearray2, stiffnesses = _run_training_loop(
                    nodes, incidence_matrix, eq_lengths, stiffnesses,
                    lr2, tod, tod2, dinputdistance, dinputdistance2,
                    nsteps, nsteps2, attempt2_remaining, output_path,
                    msearray=msearray, msearray2=msearray2,
                    step_offset=step_after_a1)

            if check_success(msearray, msearray2):
                print("\nAttempt 2 succeeded.")
            else:
                print("\nBoth attempts failed. Deleting output and resubmitting.")
                os.chdir(original_dir)
                shutil.rmtree(output_path, ignore_errors=True)
                resubmit_new_realization(gid, tid, rid, training_steps,
                                         output_dir, log_dir, targeted=targeted)
                return

        # ── Final save ────────────────────────────────────────────────────────
        np.save(os.path.join(output_path, 'stiffnesses.npy'), stiffnesses)
        np.save(os.path.join(output_path, 'mse1.npy'),        msearray)
        np.save(os.path.join(output_path, 'mse2.npy'),        msearray2)
        print(f"\nResults saved to {output_path}")

    finally:
        os.chdir(original_dir)
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
