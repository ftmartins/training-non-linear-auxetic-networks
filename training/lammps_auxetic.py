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
    write_lammps_data, strain_network_auxetic,
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
