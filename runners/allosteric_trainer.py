#!/usr/bin/env python3
"""
Allosteric network trainer — single-network script for SLURM array submission.

Adapts EnsembleGenerator030726.ipynb with:
  - Uniformly distributed initial stiffnesses in [1e-3, 1e1]
  - No stiffness rescaling; clip to [1e-3, 1e1] instead
  - Auto-calibrated learning rate (mean log10 |delta_K| in [-4, -3])
  - Post-training quality check with one automatic retry at 2x steps
  - Self-resubmission via sbatch if training fails after retry
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

# functions.py lives next to the source notebook
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data', 'ToFelipe0422'))
import functions as f  # noqa: E402  (LAMMPS helper; writes bond_coeffs_*.in to cwd)

K_MIN = 1e-3
K_MAX = 1e1
ETA = 1.0
K_OUTPUT = 1e3
LR_TARGET_LOG = -3.5   # midpoint of [-4, -3]


# ---------------------------------------------------------------------------
# Network creation  (verbatim from notebook cell 1)
# ---------------------------------------------------------------------------

def create_network(L, p, R):
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
                row[j] = 1
                row[i] = -1
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
    rest = [i for i in range(len(nodes)) if i not in special]
    for i in rest:
        nodesnew = np.vstack((nodesnew, nodes[i]))
    nodes = nodesnew

    incidence_matrix = np.array([])
    for i in range(len(nodes)):
        for j in range(i):
            if np.linalg.norm(nodes[i] - nodes[j]) < R:
                row = np.zeros(len(nodes))
                row[j] = 1
                row[i] = -1
                if len(incidence_matrix) == 0:
                    incidence_matrix = row
                else:
                    incidence_matrix = np.vstack((incidence_matrix, row))

    # remove edge between input nodes
    incidence_matrix = np.delete(incidence_matrix, 0, axis=0)

    eq_lengths = np.linalg.norm(incidence_matrix @ nodes, axis=1)
    stiffnesses = np.ones(len(incidence_matrix))
    return nodes, incidence_matrix, eq_lengths, stiffnesses


# ---------------------------------------------------------------------------
# Learning rule  (modified: clip instead of rescale, returns delta_K)
# ---------------------------------------------------------------------------

def learning_update(nodesfree, nodesclamped, tod, eq_lengths,
                    stiffnesses, incidence_matrix, eta, learning_rate):
    dVfree    = np.linalg.norm(incidence_matrix @ nodesfree,    axis=1) - eq_lengths
    dVclamped = np.linalg.norm(incidence_matrix @ nodesclamped, axis=1) - eq_lengths
    factors   = (dVfree - dVclamped) * dVfree
    delta_K   = (learning_rate / eta) * stiffnesses * factors
    stiffnesses = np.clip(stiffnesses + delta_K, K_MIN, K_MAX)
    mse = (np.linalg.norm(nodesfree[2] - nodesfree[3]) - tod) ** 2
    return stiffnesses, mse, delta_K


# ---------------------------------------------------------------------------
# Learning rate calibration (called once per training attempt)
# ---------------------------------------------------------------------------

def calibrate_learning_rate(nodes, incidence_matrix, eq_lengths, stiffnesses,
                             eta, tod, dx, nsteps):
    """Run one LAMMPS step with lr=1 and scale so mean(log10|delta_K|) = LR_TARGET_LOG."""
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
        print("  [calibrate] All delta_K are zero; defaulting lr=1e-3")
        return 1e-3

    base_log_mean = np.mean(np.log10(np.abs(base[mask])))
    lr = float(10 ** (LR_TARGET_LOG - base_log_mean))
    print(f"  [calibrate] base_log_mean={base_log_mean:.3f} → lr={lr:.3e} "
          f"(target log10|delta_K|={LR_TARGET_LOG})")
    return lr


# ---------------------------------------------------------------------------
# Core training loop for one attempt
# ---------------------------------------------------------------------------

def _run_training_loop(nodes, incidence_matrix, eq_lengths, stiffnesses,
                       learning_rate, tod, tod2, dinputdistance, dinputdistance2,
                       nsteps, nsteps2, n_steps, output_path, network_id, seed,
                       msearray=None, msearray2=None, step_offset=0):
    """
    Run training loop for n_steps steps.
    Continues from existing mse arrays if provided (for retry attempts).
    Returns (msearray1, msearray2, stiffnesses).
    """
    if msearray is None:
        msearray = np.array([])
    if msearray2 is None:
        msearray2 = np.array([])

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
        if global_step % 100 == 0:
            print(f"  step {global_step}: MSE1={mse:.4e}  MSE2={mse2:.4e}")
            np.save(os.path.join(output_path, 'stiffnesses.npy'), stiffnesses)
            np.save(os.path.join(output_path, 'mse1.npy'),        msearray)
            np.save(os.path.join(output_path, 'mse2.npy'),        msearray2)

        if mse < 5e-4 and mse2 < 5e-4:
            print(f"  Early stop at step {global_step}: both tasks converged.")
            break

    return msearray, msearray2, stiffnesses


# ---------------------------------------------------------------------------
# Success check
# ---------------------------------------------------------------------------

def check_success(msearray1, msearray2):
    """True if combined loss dropped by at least 2 orders of magnitude."""
    if len(msearray1) == 0 or len(msearray2) == 0:
        return False
    combined = (msearray1 + msearray2) / 2.0
    ratio = np.min(combined) / combined[0]
    print(f"  Success check: min_loss/loss[0] = {ratio:.4e} "
          f"({'PASS' if ratio < 0.01 else 'FAIL'})")
    return ratio < 0.01


# ---------------------------------------------------------------------------
# Resubmission via sbatch with a new seed
# ---------------------------------------------------------------------------

def resubmit_new_seed(network_id, seed, training_steps, output_dir, log_dir,
                      conda_env='auxetic_nets'):
    rng = np.random.default_rng(seed)
    new_seed = int(rng.integers(2 ** 31))
    script_path = os.path.abspath(__file__)

    script = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH -t 5-00:00:00
        #SBATCH --qos=low
        #SBATCH --partition=low
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=2
        #SBATCH --mem=4gb
        #SBATCH --job-name=allosteric_{network_id}
        #SBATCH --output={log_dir}/allosteric_{network_id}_%j.out
        #SBATCH --error={log_dir}/allosteric_{network_id}_%j.err

        eval "$(conda shell.bash hook)"
        conda activate {conda_env}

        python {script_path} \\
            --network-id {network_id} \\
            --seed {new_seed} \\
            --training-steps {training_steps} \\
            --output-dir {output_dir}
    """)

    tmp = f"/tmp/allosteric_resubmit_{network_id}_{new_seed}.sh"
    with open(tmp, 'w') as fh:
        fh.write(script)

    result = subprocess.run(['sbatch', tmp], capture_output=True, text=True)
    os.remove(tmp)

    if result.returncode == 0:
        print(f"  Resubmitted net {network_id} with new seed {new_seed}: "
              f"{result.stdout.strip()}")
    else:
        print(f"  sbatch failed: {result.stderr.strip()}")
    return new_seed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train one allosteric network and save results.')
    parser.add_argument('--network-id',    type=int, required=True)
    parser.add_argument('--seed',          type=int, required=True)
    parser.add_argument('--training-steps', type=int, default=10000)
    parser.add_argument('--output-dir',    type=str,
                        default='/data2/shared/felipetm/allosteric_nets')
    args = parser.parse_args()

    network_id    = args.network_id
    seed          = args.seed
    training_steps = args.training_steps
    output_dir    = args.output_dir

    log_dir = '/home1/felipetm/auxetic_networks/ensemble_training/Logs'

    print(f"=== Allosteric trainer: net {network_id}, seed {seed} ===")

    # Per-job working directory so parallel LAMMPS jobs don't share temp files
    work_dir = f"/tmp/allosteric_{network_id}_{seed}_{os.getpid()}"
    os.makedirs(work_dir, exist_ok=True)
    original_dir = os.getcwd()
    os.chdir(work_dir)

    # Output directory for this run
    output_path = os.path.join(output_dir, f'net_{network_id}_seed_{seed}')
    os.makedirs(output_path, exist_ok=True)

    try:
        # Seed both RNGs before network creation
        random.seed(seed)
        np.random.seed(seed)

        nodes, incidence_matrix, eq_lengths, _ = create_network(10, 0.15, 1.6)

        # Uniform stiffness initialisation (NOT all-ones, NOT log-uniform)
        stiffnesses = np.random.uniform(K_MIN, K_MAX, size=len(incidence_matrix))

        soi1 = random.randint(0, 10)
        soi2 = random.randint(0, 10)

        strain_input   = 1.0
        strain_input2  = 0.5
        strain_output  = -0.1 * soi2
        strain_output2 = -0.1 * soi1

        np.savetxt(os.path.join(output_path, 'tasks.txt'),
                   [seed, strain_output2, strain_output])

        tod  = (1 + strain_output)  * np.linalg.norm(nodes[3] - nodes[2])
        tod2 = (1 + strain_output2) * np.linalg.norm(nodes[3] - nodes[2])

        dinputdistance  = strain_input  * np.linalg.norm(nodes[0] - nodes[1])
        dinputdistance2 = strain_input2 * np.linalg.norm(nodes[0] - nodes[1])

        nsteps  = 100
        nsteps2 = 50
        dx  = dinputdistance  / nsteps
        dx2 = dinputdistance2 / nsteps2

        # ------------------------------------------------------------------
        # Attempt 1
        # ------------------------------------------------------------------
        print("\n--- Attempt 1 ---")
        print("  Calibrating learning rate...")
        lr = calibrate_learning_rate(nodes, incidence_matrix, eq_lengths,
                                     stiffnesses.copy(), ETA, tod, dx, nsteps)

        msearray, msearray2, stiffnesses = _run_training_loop(
            nodes, incidence_matrix, eq_lengths, stiffnesses,
            lr, tod, tod2, dinputdistance, dinputdistance2,
            nsteps, nsteps2, training_steps, output_path, network_id, seed)

        if check_success(msearray, msearray2):
            print("\nAttempt 1 succeeded.")
        else:
            # ------------------------------------------------------------------
            # Attempt 2: continue from current stiffnesses, 2× steps
            # ------------------------------------------------------------------
            print("\n--- Attempt 2 (2× steps, recalibrated LR) ---")
            print("  Calibrating learning rate with evolved stiffnesses...")
            lr2 = calibrate_learning_rate(nodes, incidence_matrix, eq_lengths,
                                          stiffnesses.copy(), ETA, tod, dx, nsteps)

            msearray, msearray2, stiffnesses = _run_training_loop(
                nodes, incidence_matrix, eq_lengths, stiffnesses,
                lr2, tod, tod2, dinputdistance, dinputdistance2,
                nsteps, nsteps2, 2 * training_steps, output_path, network_id, seed,
                msearray=msearray, msearray2=msearray2,
                step_offset=len(msearray))

            if check_success(msearray, msearray2):
                print("\nAttempt 2 succeeded.")
            else:
                print("\nBoth attempts failed. Deleting output and resubmitting.")
                os.chdir(original_dir)
                shutil.rmtree(output_path, ignore_errors=True)
                resubmit_new_seed(network_id, seed, training_steps,
                                  output_dir, log_dir)
                return

        # Final save
        np.save(os.path.join(output_path, 'stiffnesses.npy'), stiffnesses)
        np.save(os.path.join(output_path, 'mse1.npy'),        msearray)
        np.save(os.path.join(output_path, 'mse2.npy'),        msearray2)
        print(f"\nResults saved to {output_path}")

    finally:
        os.chdir(original_dir)
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
