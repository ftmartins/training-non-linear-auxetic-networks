"""
JAX-FIRE actuation solver for the allosteric trainer.

Provides a quasi-static pulling routine built on the same JAX-differentiable
FIRE solver (base.simulate.make_compute_response_fire) used by auxetic
training, as a drop-in alternative to the LAMMPS-based one in
training.lammps_utils.
"""
import numpy as np
import jax
import jax.numpy as jnp

from base.simulate import make_compute_response_fire

# Force tolerance (per-DOF-normalized: gnorm/n_dof < tol) for the allosteric
# actuation solves. Deliberately a dedicated constant rather than reusing
# base.config.FORCE_TOL (1e-10, tuned for auxetic): verified empirically on
# the production network that relaxing 1e-10 -> 1e-8 roughly halves wall
# time with only ~2e-5 position drift after 15 accumulated pulling steps —
# negligible next to the O(1-10) network scale.
ACTUATION_TOL = 1e-6

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


# Cache of jax.jit-compiled ramps, keyed by (id(crf_fn), id_fixed, id_pull, nsteps).
# Plain dict (not functools.lru_cache) so we don't depend on crf_fn's hashability —
# id() is always defined. jax.jit's own cache (keyed on argument shapes/dtypes)
# sits underneath this, so a given entry still recompiles if edges/rest_lengths
# shape changes (e.g. the clamped run's extra output-spring edge).
_JIT_RAMP_CACHE = {}


def _make_jit_ramp(crf_fn, id_fixed, id_pull, nsteps):
    key = (id(crf_fn), id_fixed, id_pull, nsteps)
    ramp = _JIT_RAMP_CACHE.get(key)
    if ramp is None:
        @jax.jit
        def ramp(stiffnesses, edges, rest_lengths, positions0_flat, dx):
            x_fixed = positions0_flat[id_fixed * 2]
            y_fixed = positions0_flat[id_fixed * 2 + 1]
            x_pull0 = positions0_flat[id_pull * 2]
            y_pull  = positions0_flat[id_pull * 2 + 1]
            source_dof = jnp.array(
                [id_fixed * 2, id_pull * 2, id_fixed * 2 + 1, id_pull * 2 + 1],
                dtype=jnp.int32,
            )

            # jax.lax.scan instead of a Python for-loop, same reasoning as
            # base.simulate.compute_quasistatic_trajectory_auxetic_jax: the
            # unrolled loop traces nsteps separate copies of crf_fn's
            # custom_vjp into the compiled graph, and this ramp gets
            # differentiated (jax.grad, and forward-over-reverse
            # jax.jvp(jax.grad(...)) for analysis/timestep_sweep.py's
            # cost-Hessian) so each unrolled copy's backward/double-backward
            # rule was paying to compile too. Measured on the production
            # geometry_targeted/task_0/realization_0 network (n_edges=253,
            # nsteps=40): compile time drops ~3x for the forward-only ramp,
            # ~4.5x for jax.grad, ~6x for the jax.jvp(jax.grad(...)) HVP
            # (25s -> 4s). Verified bit-identical to the old unrolled version
            # (forward position, jax.grad, and the HVP all match to 0.0 abs
            # diff) using that same network's real trained stiffnesses;
            # mean per-step wall time is unchanged (this only cuts compile
            # cost, as with the auxetic conversion). x_pull is carried
            # through scan's carry (cumulative += dx, not step*dx) so the
            # floating-point arithmetic matches the eager loop exactly
            # rather than merely approximately.
            def scan_body(carry, _):
                current_pos, x_pull = carry
                x_pull = x_pull + dx
                imposed = jnp.array([x_fixed, x_pull, y_fixed, y_pull], dtype=jnp.float64)
                new_pos = crf_fn(stiffnesses, edges, rest_lengths, current_pos,
                                 source_dof, imposed)
                return (new_pos, x_pull), None

            (final_pos, _), _ = jax.lax.scan(
                scan_body, (positions0_flat, x_pull0), None, length=nsteps
            )
            return final_pos

        _JIT_RAMP_CACHE[key] = ramp
    return ramp


def strain_network_jax_final_traced(crf_fn, positions0, edges, rest_lengths, stiffnesses,
                                    id_fixed, id_pull, dx=0.025, nsteps=200):
    """
    Same physics as strain_network_jax_final, but returns the raw (flattened)
    JAX array instead of converting to numpy, so callers can differentiate
    through it with jax.grad / jax.jvp (analysis.timestep_sweep's exact-autodiff
    cost-Hessian is the motivating caller: it needs the whole ramp to stay a
    traced jnp value all the way to the scalar loss, since converting to numpy
    anywhere in between breaks the autodiff graph and forces finite differences).
    strain_network_jax_final itself is now a thin numpy-converting wrapper
    around this.
    """
    ramp = _make_jit_ramp(crf_fn, id_fixed, id_pull, nsteps)
    edges_j = jnp.asarray(edges, dtype=jnp.int32)
    rest_j  = jnp.asarray(rest_lengths, dtype=jnp.float64)
    k_j     = jnp.asarray(stiffnesses, dtype=jnp.float64)
    positions0_flat = jnp.asarray(positions0.flatten(), dtype=jnp.float64)
    return ramp(k_j, edges_j, rest_j, positions0_flat, dx)


def strain_network_jax_final(crf_fn, positions0, edges, rest_lengths, stiffnesses,
                             id_fixed, id_pull, dx=0.025, nsteps=200):
    """
    Same physics as strain_network_jax, but jax.jit-compiled and returning only
    the final (N, 2) position array (no per-step frames).

    strain_network_jax calls crf_fn nsteps times as separate eager (un-jitted)
    dispatches; each one pays make_compute_response_fire's fixed per-call
    overhead from its internal jax.lax.fori_loop always running its full
    max_steps trip count (early convergence only skips the update body via
    jax.lax.cond — it doesn't shorten the loop), which adds up to a very
    consistent, nontrivial cost per call regardless of when the physics
    actually converges (empirically ~0.15s per 100 max_steps=1_000_000 substeps
    on the production allosteric network). Fusing the whole ramp into one
    compiled XLA program (as training's jax.jit(jax.value_and_grad(...)) already
    does) avoids paying that overhead nsteps times per call. Compiled once per
    (crf_fn, id_fixed, id_pull, nsteps) and reused for any stiffnesses/dx/
    positions0/edges of matching shape — a large win for callers that invoke
    this hundreds of times per gradient estimate and don't need the per-step
    trajectory (that still goes through strain_network_jax above).
    """
    final_flat = strain_network_jax_final_traced(
        crf_fn, positions0, edges, rest_lengths, stiffnesses, id_fixed, id_pull, dx, nsteps)
    return np.asarray(final_flat).reshape(-1, 2)
