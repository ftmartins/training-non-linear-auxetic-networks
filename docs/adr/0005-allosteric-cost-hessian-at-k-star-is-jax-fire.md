# Allosteric cost Hessians at K* are JAX-FIRE, although the runs trained with LAMMPS

Every allosteric `_aug` realization trained with LAMMPS, but every cost Hessian at its best-loss
stiffnesses K* (post-training sweep and local recompute in the figure code) is the JAX-FIRE
autodiff Hessian. The solver-matched alternative, a finite-difference LAMMPS Hessian, never
finished within cluster wall limits (none of 143 sweep jobs completed a single eigensolve). We
decided to accept JAX-FIRE cost Hessians at K* for these runs and to label them as such: sweeps
record `cost_hessian_solver: jax_fire` in `timestep_sweep_meta.json`, and the figure code does
not refuse a local JAX-FIRE recompute for a LAMMPS-trained run.

The mismatch is not small. On 15 canonical runs (2026-10-23) the JAX-FIRE combined loss at K*
differs from the stored LAMMPS loss by a relative factor of median 0.83, minimum 0.16 and
maximum about 1000, while staying 8e-8 to 1e-3 in absolute terms (initial losses 0.1 to 0.5). K*
is therefore in the JAX-FIRE low-loss region but not at its minimum.

## Consequences

- Cost-Hessian eigenvectors at K* describe the JAX-FIRE landscape, not the LAMMPS one the run
  was trained in. Whether the top eigenvector the figures use changes under FD-LAMMPS is open and
  to be investigated.
- The record still refuses a sweep whose after Hessian was not taken at K* (ADR 0004); it does
  not check the solver.
