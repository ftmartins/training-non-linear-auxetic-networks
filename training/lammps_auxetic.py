"""
LAMMPS actuation solver for the auxetic (Poisson-ratio) task.

Provides a quasi-static boundary-displacement compression routine built on
training.lammps_utils.strain_network_auxetic, as a LAMMPS-backed
alternative to the JAX-FIRE / Cython-FIRE / Newton solvers in base.simulate
(compute_quasistatic_trajectory_auxetic, compute_poisson_ratio_single_jax).
Mirrors how training.jax_actuation provides a JAX-FIRE alternative to
training.lammps_utils for the allosteric task, just in the opposite
direction: here LAMMPS is the alternative backend, and the JAX/Newton path
in base.simulate is production.

The task itself (boundary node sets from base.network_utils.
create_auxetic_network, compression strain, Poisson ratio observable) is
solver-agnostic — see training/src/task_generator.py — so this module only
needs to reproduce the equilibrium mechanics, not redefine the task.
"""
import shutil
import tempfile

import numpy as np

from training.lammps_utils import (
    write_lammps_data, strain_network_auxetic, strain_network_auxetic_clamped,
)


def edges_to_incidence(edges, n_nodes):
    """Build an (n_edges, n_nodes) incidence matrix from an (n_edges, 2) edge list.

    Inverse of training.lammps_utils.incidence_to_edges — ElasticNetwork
    stores edges directly, but write_lammps_data expects an incidence matrix.
    """
    edges = np.asarray(edges, dtype=int)
    incidence = np.zeros((len(edges), n_nodes))
    incidence[np.arange(len(edges)), edges[:, 0]] = 1.0
    incidence[np.arange(len(edges)), edges[:, 1]] = -1.0
    return incidence


def compute_quasistatic_trajectory_auxetic_lammps(positions, edges, stiffnesses,
                                                   top_nodes, bottom_nodes,
                                                   compression_strain, n_steps=100,
                                                   tol=1e-8, work_dir=None):
    """
    LAMMPS analogue of base.simulate.compute_quasistatic_trajectory_auxetic:
    same network (positions, edges, stiffnesses) and boundary-displacement
    protocol (top/bottom clamped, y ramped 0 -> compression_strain over
    n_steps), equilibrated via LAMMPS FIRE instead of Cython FIRE / Newton.

    tol : passed through to strain_network_auxetic (see its docstring for
        the JAX-FIRE-equivalent tolerance convention used here).

    Returns:
        traj: list of (N, 2) position arrays, length n_steps
    """
    n_nodes = positions.shape[0]
    incidence = edges_to_incidence(edges, n_nodes)

    own_work_dir = work_dir is None
    if own_work_dir:
        work_dir = tempfile.mkdtemp(prefix="lammps_auxetic_")
    try:
        write_lammps_data("data_auxetic.network", positions, incidence, stiffnesses,
                          work_dir=work_dir)
        traj = strain_network_auxetic("data_auxetic.network", top_nodes, bottom_nodes,
                                      compression_strain, n_steps=n_steps, tol=tol,
                                      work_dir=work_dir)
    finally:
        if own_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
    return traj


def compute_poisson_ratio_lammps(positions, edges, stiffnesses, top_nodes, bottom_nodes,
                                 left_nodes, right_nodes, compression_strain,
                                 n_steps=100, tol=1e-8, work_dir=None):
    """
    LAMMPS analogue of base.simulate.compute_poisson_ratio_single_jax: same
    task definition (boundary node sets, compression strain, lateral
    measurement via left/right nodes), equilibrated via LAMMPS FIRE.

    Returns:
        poisson_ratio: scalar = -(lateral_strain / compression_strain)
    """
    traj = compute_quasistatic_trajectory_auxetic_lammps(
        positions, edges, stiffnesses, top_nodes, bottom_nodes,
        compression_strain, n_steps=n_steps, tol=tol, work_dir=work_dir,
    )
    left_nodes = np.asarray(left_nodes, dtype=int)
    right_nodes = np.asarray(right_nodes, dtype=int)

    width_initial = traj[0][right_nodes, 0].mean() - traj[0][left_nodes, 0].mean()
    width_final = traj[-1][right_nodes, 0].mean() - traj[-1][left_nodes, 0].mean()
    lateral_strain = (width_final - width_initial) / width_initial
    return -(lateral_strain / compression_strain)


def compute_free_and_clamped_auxetic_lammps(positions, edges, stiffnesses, top_nodes,
                                             bottom_nodes, left_nodes, right_nodes,
                                             compression_strain, target_poisson_ratio,
                                             eta, n_steps=100, tol=1e-8, work_dir=None):
    """
    Coupled-learning free+clamped pair for one (compression_strain,
    target_poisson_ratio) pair of an auxetic task — the LAMMPS analogue of
    allosteric_trainer.py's _evaluate_actuation_jax free-then-clamped
    sequencing (allosteric_trainer.py:357-399), adapted from a single
    pulled-node/output-spring pair to auxetic's two boundary-group
    compression protocol.

    1. Free run: compute_quasistatic_trajectory_auxetic_lammps (unchanged,
       top/bottom clamped only, left/right free) -> nodes_free.
    2. Nudge target: blend the free run's own observed width with the width
       that would give EXACTLY target_poisson_ratio at this compression,
       weighted by eta (same role as allosteric's `eta*tod + (1-eta)*cod`).
    3. Clamped run: same top/bottom ramp, plus left/right pinned
       symmetrically (strain_network_auxetic_clamped) to the nudged width
       -> nodes_clamped.

    Both runs start from the same `positions`/`stiffnesses` (the network's
    current training-step state) via a single shared LAMMPS data file, same
    dependency order as allosteric: the clamped run's boundary condition
    needs the free run's own result first.

    Returns:
        nodes_free, nodes_clamped, observed_nu_free
    """
    n_nodes = positions.shape[0]
    incidence = edges_to_incidence(edges, n_nodes)

    own_work_dir = work_dir is None
    if own_work_dir:
        work_dir = tempfile.mkdtemp(prefix="lammps_auxetic_cl_")
    try:
        write_lammps_data("data_auxetic.network", positions, incidence, stiffnesses,
                          work_dir=work_dir)

        traj_free = strain_network_auxetic(
            "data_auxetic.network", top_nodes, bottom_nodes, compression_strain,
            n_steps=n_steps, tol=tol, work_dir=work_dir)
        nodes_free = traj_free[-1]

        left_idx = np.asarray(left_nodes, dtype=int)
        right_idx = np.asarray(right_nodes, dtype=int)
        width_initial = positions[right_idx, 0].mean() - positions[left_idx, 0].mean()
        width_free = nodes_free[right_idx, 0].mean() - nodes_free[left_idx, 0].mean()
        # Width consistent with EXACTLY target_poisson_ratio at this
        # compression — inverts lateral_strain = -nu*cs,
        # lateral_strain = (w_final - w_init) / w_init (compute_poisson_ratio_lammps
        # above computes the forward direction of this same relation).
        width_target = width_initial * (1 - target_poisson_ratio * compression_strain)
        width_nudged = (1 - eta) * width_free + eta * width_target

        traj_clamped = strain_network_auxetic_clamped(
            "data_auxetic.network", top_nodes, bottom_nodes, left_idx, right_idx,
            compression_strain, width_nudged, n_steps=n_steps, tol=tol, work_dir=work_dir)
        nodes_clamped = traj_clamped[-1]
    finally:
        if own_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

    lateral_strain_free = (width_free - width_initial) / width_initial
    observed_nu_free = -(lateral_strain_free / compression_strain)
    return nodes_free, nodes_clamped, observed_nu_free
