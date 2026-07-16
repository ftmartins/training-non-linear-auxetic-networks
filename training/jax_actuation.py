"""
JAX-FIRE actuation solver for the allosteric trainer.

Provides a quasi-static pulling routine built on the same JAX-differentiable
FIRE solver (base.simulate.make_compute_response_fire) used by auxetic
training, as a drop-in alternative to the LAMMPS-based one in
training.lammps_utils.
"""
import numpy as np
import jax.numpy as jnp

from base.simulate import make_compute_response_fire

# Force tolerance (per-DOF-normalized: gnorm/n_dof < tol) for the allosteric
# actuation solves. Deliberately a dedicated constant rather than reusing
# base.config.FORCE_TOL (1e-10, tuned for auxetic): verified empirically on
# the production network that relaxing 1e-10 -> 1e-8 roughly halves wall
# time with only ~2e-5 position drift after 15 accumulated pulling steps —
# negligible next to the O(1-10) network scale.
ACTUATION_TOL = 1e-8

# Own instance (not base.simulate's shared `crf` singleton) so this tol
# choice doesn't couple to auxetic training's solver.
FREE_CRF = make_compute_response_fire(
    d=2, dt_init=1e-2, dt_max=1e-1, dt_min=1e-4,
    max_steps=1_000_000, tol=ACTUATION_TOL, force_type="quadratic",
)

# The allosteric "clamped" run adds an output spring with stiffness
# K_OUTPUT=1e3 (training.runners.allosteric_trainer.K_OUTPUT) — ~100x the
# base network's K_MAX=10. FREE_CRF's dt_max=0.1 (tuned for auxetic-scale
# stiffnesses) is well past the leapfrog stability limit at that stiffness
# (dt < ~2/sqrt(k)) and produces NaN positions. Verified empirically on the
# production network: dt_max=1e-3 is stable but slow (~10s/step); dt_max=5e-3
# is still NaN-free with ~5x fewer iterations needed (~2s/step); dt_max=1e-1
# (FREE_CRF's value) blows up. Use a separate, smaller-timestep instance for
# any run that includes the output spring.
CLAMPED_CRF = make_compute_response_fire(
    d=2, dt_init=5e-3, dt_max=5e-3, dt_min=1e-6,
    max_steps=1_000_000, tol=ACTUATION_TOL, force_type="quadratic",
)


def strain_network_jax(crf_fn, positions0, edges, rest_lengths, stiffnesses,
                       id_fixed, id_pull, dx=0.025, nsteps=200):
    """
    Quasi-static pulling of id_pull away from id_fixed via crf_fn (a
    make_compute_response_fire instance). id_fixed never moves; id_pull's
    x-coordinate advances by dx each step while its y stays put. Both nodes
    are held at their imposed positions at every FIRE sub-step.

    JAX-FIRE analogue of training.lammps_utils.strain_network.

    Returns frames: list of (N, 2) node position arrays, one per step.
    """
    edges_j = jnp.asarray(edges, dtype=jnp.int32)
    rest_j  = jnp.asarray(rest_lengths, dtype=jnp.float64)
    k_j     = jnp.asarray(stiffnesses, dtype=jnp.float64)

    x_fixed, y_fixed = positions0[id_fixed]
    x_pull,  y_pull   = positions0[id_pull]

    source_dof = jnp.array(
        [id_fixed * 2, id_pull * 2, id_fixed * 2 + 1, id_pull * 2 + 1],
        dtype=jnp.int32,
    )

    current_pos = jnp.asarray(positions0.flatten(), dtype=jnp.float64)
    frames = []
    for _ in range(nsteps):
        x_pull = x_pull + dx
        imposed = jnp.array([x_fixed, x_pull, y_fixed, y_pull], dtype=jnp.float64)
        current_pos = crf_fn(k_j, edges_j, rest_j, current_pos, source_dof, imposed)
        frames.append(np.asarray(current_pos).reshape(-1, 2).copy())
    return frames
