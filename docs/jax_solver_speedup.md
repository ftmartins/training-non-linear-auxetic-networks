# JAX targeted-auxetic solver speedup

Optimization of the JAX-autodiff training path for targeted auxetic networks
(`gradient_method='jax'` in `targeted_ensemble_runner.py`,
`finish_training_GD_auxetic_batch_jax` in `training/src/training_functions.py`,
`make_compute_response_fire` / `compute_quasistatic_trajectory_auxetic_jax` in
`base/simulate.py`). Explicitly kept as **FIRE minimization + the quasistatic
compression protocol** throughout — no algorithm swap (in particular, no
Newton-Raphson forward solver, which already exists as the separate, faster
`gradient_method='newton'` production default and was intentionally left out
of scope here).

Benchmarked on a real trained network (`task_00`, realization 0: 98 nodes,
194 edges, compression -0.2, target Poisson ratio -0.8 — an intentionally
extreme target, which turned out to matter: this network sits close to a
mechanical soft mode/mechanism, and that fragility shaped which optimizations
were safe).

**Net result: ~6-9x on the forward pass** (the ~97-99% of total per-call
time), a further **~3.5x on the backward/gradient pass** (already a small
fraction of the total, so its effect on end-to-end wall-clock is minor), all
validated to reach the *same* physical equilibrium as the original — same
force-balance tolerance, same edge topology/crossing count.

## Shipped optimizations

### Forward pass (`_fire_forward` inside `make_compute_response_fire`)

| Change | Effect | Why it's safe |
|---|---|---|
| Tuned FIRE step-size hyperparameters (`dt_max` 0.1→1.0, `finc` 1.1→1.3), `tol=1e-10` kept at **every** strain step | ~6-9x (n=100: 14.48s→2.42s; n=400: ~93-158s→16.75s) | Every step still individually converges to the identical `force_balance` tolerance as before — verified to land on the exact same equilibrium (same edge-crossing count) as the untuned defaults, not merely "a" force-balanced point |
| Analytic (closed-form) `dE/dp` instead of `jax.grad(energy_fn)` in the FIRE inner loop | ~1.4% (measured back-to-back: 2072.7ms→2043.3ms at n=100) | Matches `jax.grad`'s output to float64 machine precision (7e-17); free, no risk |

### Backward / gradient pass (`crf_bwd` inside `make_compute_response_fire`)

| Change | Effect | Why it's safe |
|---|---|---|
| `jax.lax.stop_gradient` on stiffnesses for all-but-the-last strain step | Changes backward cost from O(n_strain_steps) adjoint solves to O(1) | `crf_fn`'s custom_vjp already returns no gradient through `positions0`, so every intermediate step's cotangent is analytically zero — this only removes wasted computation, exact zero-diff match |
| Analytic Hessian/adjoint (closed-form per-edge assembly) replacing generic `jax.jacobian` | ~1.37x on the one remaining backward call | Exact — validated end-to-end through the real `custom_vjp` chain, grad relative diff ~1.7e-12 |
| Native sparse solve (`jax.experimental.sparse.linalg.spsolve`, CSR structure precomputed once from network topology) replacing the dense linear solve | Additional ~2.37x (1.11ms→0.47ms) | Matches to ~1e-9 relative — limited by the adjoint system's own ~6.5×10⁹ condition number (see below), not by this method |

Combined backward: **~3.5x** vs. the original generic-Jacobian dense solve.
In absolute terms this is a rounding error against a ~16.5s forward pass, but
it costs nothing and was kept per instruction ("if there is no cost, even if
minimal improvement, keep it").

### A genuine bug found and fixed along the way

The first version of the analytic backward was wrong by ~5 orders of
magnitude. Cause: `R` (the masked force residual used in the IFT adjoint) is
*identically* zero at constrained (boundary) dof, so `jax.jacobian` correctly
gives it zero rows w.r.t. stiffness there too — but the closed-form
replacement didn't re-apply that masking to the solved adjoint vector `w`
before contracting. A constrained dof's column in the regularized adjoint
matrix is ~singular (only the `1e-8` regularizer pins it), so its solved
`w`-value is a huge, physically meaningless number that leaked into the
result unmasked. Fixed by masking `w` the same way `R` is masked, before use.

## A load-bearing finding: this network is numerically fragile

The adjoint linear system (`crf_bwd`'s IFT solve) has condition number
**~6.5×10⁹** (smallest singular value ~3.6×10⁻¹⁰, right at the `1e-8`
regularization floor). This is inherent to the *original* code too — not
something introduced here — and it explains several negative results below:
extreme-Poisson-ratio auxetic networks are deliberately trained close to an
internal mechanism (soft mode), and that is exactly what produces near-zero
Hessian eigenvalues.

Consequence for optimization: **any change to the numerical path** (not just
tolerance) risks landing on a different, though still fully force-balanced,
equilibrium. Verified directly — checked edge-edge crossing counts before vs.
after compression across all 6 real targeted-auxetic tasks (task_00–05); the
*original* solver already produces self-crossing ("folded") geometry at this
compression level (e.g. task_00: 0→39 crossings), so folding at large
compression is inherent to the physics of a network this close to a
mechanism, not an artifact of any change made here — but it also means the
bar for "safe" is not just "still force-balanced," it's "lands on the same
branch as the original," which several attempted optimizations violated.

## Rejected / abandoned attempts

| Attempt | Result | Why rejected |
|---|---|---|
| Swap the FIRE forward solver for Newton-Raphson (reusing the existing `_make_jax_newton_solver`) | Diverged to positions ~1e148 within the custom_vjp forward pass at moderate compression | Out of scope per explicit instruction (kept the JAX branch as a from-scratch FIRE-based check, independent of the analytic Newton/IFT production path); also unstable without damping/line search |
| Loosen `tol` on intermediate (warm-start-only) strain steps, keep `tol=1e-10` only on the final step | ~30-50x forward speedup, but landed on a **different** equilibrium branch (loss 0.15-0.35 vs. the correct ~0.026; different crossing count) | Confirmed via the same-input-different-tol test: even at `tol=1e-8` (100x tighter than the naive first try), gradient still differed substantially — this network's near-mechanism character means under-converged intermediate steps can tip the trajectory into a different basin |
| More aggressive FIRE hyperparameters (`dt_max` up to 2.0) with `tol=1e-10` kept everywhere | `dt_max=1.4` and above reliably diverges to NaN; `dt_max=1.2` is actually *slower* than 1.0 | Hard CFL-type stability limit tied to the stiffest spring's natural frequency — not a tunable margin |
| FIRE 2.0-style `N_min` ramp-gating (require several consecutive "good" steps before growing the timestep), tested up to `N_min=20` at `dt_max` up to 1.8 | Still diverges to NaN | Confirms the `dt_max≥1.4` failure is a hard stability wall, not an "over-confident ramp-up" issue that gentler ramping could fix |
| Linear/secant extrapolation warm-starting between strain steps (classic continuation-method predictor) | 1.0-1.4x speedup, but chaotically sensitive: damping factor 0.1 stays on the correct branch (32 crossings), 0.2 flips to a wrong one (35), 0.3 flips back (32), 0.5/1.0 flip again (37, 40) | This network sits right at a bifurcation somewhere along the trajectory; every "wrong" branch found also had *more* folding than the validated one, i.e. this direction actively works against the "don't want folded networks" requirement |
| Reducing `max_steps` (50,000 → 3,000) | No measurable effect | The post-convergence "skip" tail (via the `active` flag) was already cheap; the cost is genuinely in pre-convergence iterations |
| `pure_callback` to `scipy.sparse.linalg.spsolve` for the backward solve | 19% **slower** than dense | Host-callback/data-marshaling overhead dominates at this small matrix size (196×196, ~5% density) |

## A note on where the win actually lands

Because of the `stop_gradient` fix, backward cost after tuning is dominated
by a single adjoint solve per training step, not by `n_strain_steps` of them.
That means the analytic-Hessian/sparse-solve work barely moves *training*
wall-clock (forward still dominates ~99.97% of a call), but would matter far
more in a workflow that calls the JAX gradient repeatedly at *fixed*
(non-training-drifting) configurations — e.g. a verification/cross-check
notebook comparing the JAX-autodiff gradient against the analytic Newton/IFT
gradient at a saved trained network's exact stiffnesses.

## Validation methodology

- **Force balance**: residual `‖F_free‖ / n_free` computed independently
  (not relying on internal solver warnings) after each candidate change;
  required to match the `1e-10` tolerance.
- **Same-branch check**: edge-edge crossing count on the final compressed
  geometry, compared against the original solver's output, across all 6 real
  targeted-auxetic tasks (task_00–05).
- **Gradient correctness**: compared against `jax.jacobian`-based reference
  gradients through the actual `custom_vjp` chain (not a reimplementation),
  at both loose (relative ~1e-12, well-conditioned cases) and tight (~1e-9,
  the ill-conditioned real network) tolerances.

## Where this lives

All work developed and validated in `jax_speedup_branch/` (a full sandbox
copy of `base/`, `training/`, `analysis/`) before being ported into
production `base/simulate.py` and `training/src/training_functions.py`, so
that all of the above was verified without any risk to the working
production training pipeline.
