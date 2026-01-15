# cython: boundscheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void compute_forces_inplace(
    double[:, :] pos,
    int[:, :] edges,
    double[:] rest_lengths,
    double[:] stiffnesses,
    int force_type,
    double[:, :] forces
):
    cdef int m = edges.shape[0]
    cdef int d = pos.shape[1]
    cdef int i, j, e
    cdef double dx, dy, dz, norm, delta, force_mag
    cdef double avg_rest_length = 0.0
    cdef double scale_coeff = 1.0
    cdef int npos = pos.shape[0]

    # zero out forces
    for i in range(npos):
        for j in range(d):
            forces[i, j] = 0.0

    # compute avg rest length only if quartic forces requested
    if force_type == 1:
        for e in range(m):
            avg_rest_length += rest_lengths[e]
        avg_rest_length /= m
        scale_coeff = 36.0 / (avg_rest_length * avg_rest_length + 1e-12)

    # accumulate forces
    for e in range(m):
        i = edges[e, 0]
        j = edges[e, 1]

        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        if d == 3:
            dz = pos[j, 2] - pos[i, 2]
        else:
            dz = 0.0

        norm = dx * dx + dy * dy + dz * dz
        norm = sqrt(norm)
        delta = norm - rest_lengths[e]

        if norm > 1e-12:
            if force_type == 0:
                # harmonic
                force_mag = stiffnesses[e] * delta / norm
            else:
                # quartic: stiffness * (scale_coeff * delta^3 - delta) / norm
                # replaced delta**3 with delta * delta * delta
                force_mag = stiffnesses[e] * (scale_coeff * (delta * delta * delta) - delta) / norm

            forces[i, 0] += force_mag * dx
            forces[j, 0] -= force_mag * dx
            forces[i, 1] += force_mag * dy
            forces[j, 1] -= force_mag * dy
            if d == 3:
                forces[i, 2] += force_mag * dz
                forces[j, 2] -= force_mag * dz


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple fire_minimize(
    double[:, :] positions,
    int[:, :] edges,
    double[:] rest_lengths,
    double[:] stiffnesses,
    double deltaT,
    int max_steps,
    double force_tol,
    object constrained_idx,
    int force_type
):
    cdef int step, n, d, i, j, N_min, n_neg, n_pos
    cdef double force_norm, v_dot_f, v_norm, alpha, dt, finc, fdec, astart, fa, scaling, deltaTMax, deltaTMin
    cdef double[:, :] v
    cdef double[:, :] f
    cdef double[:, :] x

    deltaTMax = 1e-1
    deltaTMin = 1e-4
    n = positions.shape[0]
    d = positions.shape[1]
    N_min = 5
    n_neg = 0
    n_pos = 0
    finc = 1.1
    fdec = 0.5
    astart = 0.1
    fa = 0.99
    alpha = 0.1
    dt = deltaT

    # preallocate arrays once
    v_np = np.zeros((n, d), dtype=np.float64)
    f_np = np.zeros((n, d), dtype=np.float64)
    x_np = np.copy(positions)

    v = v_np
    f = f_np
    x = x_np

    # --- NEW: convert constrained_idx to C-level mask for O(1) checks ---
    cdef bint use_constraints = constrained_idx is not None
    cdef np.ndarray[np.int8_t, ndim=1] is_constrained_arr = np.zeros(n, dtype=np.int8)
    cdef np.int8_t[:] is_constrained = is_constrained_arr
    if use_constraints:
        for i in constrained_idx:
            is_constrained[i] = 1

    # --- NEW: validate force_type once ---
    if force_type not in (0, 1):
        raise ValueError("Unknown force_type: %s" % force_type)

    for step in range(max_steps):

        # Compute forces (writes into preallocated f_np)
        compute_forces_inplace(x, edges, rest_lengths, stiffnesses, force_type, f)

        # Update positions (x) and half-step velocities
        for i in range(n):
            if use_constraints and is_constrained[i]:
                continue
            for j in range(d):
                x[i, j] += v[i, j] * dt + 0.5 * f[i, j] * dt * dt
                v[i, j] += 0.5 * f[i, j] * dt

        # Recompute forces (overwrite f)
        compute_forces_inplace(x, edges, rest_lengths, stiffnesses, force_type, f)

        # Update velocities (second half)
        for i in range(n):
            if use_constraints and is_constrained[i]:
                continue
            for j in range(d):
                v[i, j] += 0.5 * f[i, j] * dt

        # Compute norms
        force_norm = 0.0
        v_dot_f = 0.0
        v_norm = 0.0
        for i in range(n):
            if use_constraints and is_constrained[i]:
                continue
            for j in range(d):
                force_norm += f[i, j] * f[i, j]
                v_dot_f += v[i, j] * f[i, j]
                v_norm += v[i, j] * v[i, j]
        force_norm = sqrt(force_norm)
        v_norm = sqrt(v_norm)

        if force_norm < force_tol:
            break

        if force_norm > 0.:
            # replaced pow(..., 0.5) with sqrt(...)
            scaling = sqrt(v_norm / (force_norm + 1e-12))
        else:
            scaling = 0.0

        # Velocity modification
        for i in range(n):
            if use_constraints and is_constrained[i]:
                continue
            for j in range(d):
                v[i, j] = (1 - alpha) * v[i, j] + alpha * f[i, j] * scaling

        if v_dot_f > 0:
            n_pos += 1
            n_neg = 0
            if n_pos > N_min:
                dt = min(dt * finc, deltaTMax)
                alpha *= fa
        else:
            n_neg += 1
            n_pos = 0
            dt = max(dt * fdec, deltaTMin)
            alpha = astart
            for i in range(n):
                for j in range(d):
                    v[i, j] = 0.0

    # return x (array), force_norm (double), and f (the last computed forces array)
    return np.asarray(x), force_norm, f_np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple fire_minimize_dof(
    double[:, :] positions,
    int[:, :] edges,
    double[:] rest_lengths,
    double[:] stiffnesses,
    double deltaT,
    int max_steps,
    double force_tol,
    object constrained_dof_idx,   # NEW: per-dof constraints
    int force_type
):
    cdef int step, n, d, i, j, N_min, n_neg, n_pos, dof
    cdef double force_norm, v_dot_f, v_norm, alpha, dt, finc, fdec, astart, fa, scaling, deltaTMax, deltaTMin

    cdef double[:, :] v
    cdef double[:, :] f
    cdef double[:, :] x

    deltaTMax = 1e-1
    deltaTMin = 1e-4
    n = positions.shape[0]
    d = positions.shape[1]
    N_min = 5
    n_neg = 0
    n_pos = 0
    finc = 1.1
    fdec = 0.5
    astart = 0.1
    fa = 0.99
    alpha = 0.1
    dt = deltaT

    # preallocate arrays once
    v_np = np.zeros((n, d), dtype=np.float64)
    f_np = np.zeros((n, d), dtype=np.float64)
    x_np = np.copy(positions)

    v = v_np
    f = f_np
    x = x_np

    # --- NEW: convert constrained_dof_idx to mask ---
    cdef bint use_constraints = constrained_dof_idx is not None
    cdef np.ndarray[np.int8_t, ndim=1] is_constrained_arr = np.zeros(n * d, dtype=np.int8)
    cdef np.int8_t[:] is_constrained = is_constrained_arr
    if use_constraints:
        for dof in constrained_dof_idx:
            is_constrained[dof] = 1

    # --- validate force_type once ---
    if force_type not in (0, 1):
        raise ValueError("Unknown force_type: %s" % force_type)

    for step in range(max_steps):

        # Compute forces (writes into preallocated f_np)
        compute_forces_inplace(x, edges, rest_lengths, stiffnesses, force_type, f)

        # Update positions (x) and half-step velocities
        for i in range(n):
            for j in range(d):
                if use_constraints and is_constrained[i * d + j]:
                    continue
                x[i, j] += v[i, j] * dt + 0.5 * f[i, j] * dt * dt
                v[i, j] += 0.5 * f[i, j] * dt

        # Recompute forces (overwrite f)
        compute_forces_inplace(x, edges, rest_lengths, stiffnesses, force_type, f)

        # Update velocities (second half)
        for i in range(n):
            for j in range(d):
                if use_constraints and is_constrained[i * d + j]:
                    continue
                v[i, j] += 0.5 * f[i, j] * dt

        # Compute norms
        force_norm = 0.0
        v_dot_f = 0.0
        v_norm = 0.0
        for i in range(n):
            for j in range(d):
                if use_constraints and is_constrained[i * d + j]:
                    continue
                force_norm += f[i, j] * f[i, j]
                v_dot_f += v[i, j] * f[i, j]
                v_norm += v[i, j] * v[i, j]
        force_norm = sqrt(force_norm)
        v_norm = sqrt(v_norm)

        if force_norm < force_tol:
            break

        if force_norm > 0.:
            scaling = sqrt(v_norm / (force_norm + 1e-12))
        else:
            scaling = 0.0

        # Velocity modification
        for i in range(n):
            for j in range(d):
                if use_constraints and is_constrained[i * d + j]:
                    continue
                v[i, j] = (1 - alpha) * v[i, j] + alpha * f[i, j] * scaling

        if v_dot_f > 0:
            n_pos += 1
            n_neg = 0
            if n_pos > N_min:
                dt = min(dt * finc, deltaTMax)
                alpha *= fa
        else:
            n_neg += 1
            n_pos = 0
            dt = max(dt * fdec, deltaTMin)
            alpha = astart
            for i in range(n):
                for j in range(d):
                    v[i, j] = 0.0

    # return x (array), force_norm (double), and f (the last computed forces array)
    return np.asarray(x), force_norm, f_np
