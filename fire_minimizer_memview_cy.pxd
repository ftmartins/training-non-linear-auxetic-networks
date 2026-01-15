import numpy as np
cimport numpy as np

cpdef fire_minimize(
    double[:, :] positions,
    int[:, :] edges,
    double[:] rest_lengths,
    double[:] stiffnesses,
    double deltaT,
    int max_steps,
    double force_tol,
    object constrained_idx,
    int force_type
)

cpdef fire_minimize_dof(
    double[:, :] positions,
    int[:, :] edges,
    double[:] rest_lengths,
    double[:] stiffnesses,
    double deltaT,
    int max_steps,
    double force_tol,
    object constrained_dof_idx,
    int force_type
)