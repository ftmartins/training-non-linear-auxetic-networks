# Best-loss stiffnesses (K*) have one definition and no fallback

Figures and sweeps are meant to analyse each realization at its best-loss stiffnesses, but
several scripts silently substitute something else when K* is missing (the last trajectory
entry, the checkpoint stiffness, or a cost Hessian recomputed with a different solver). We
decided K* is, for both task kinds, the pre-update stiffnesses at the training step with the
lowest recorded training loss ((mse1 + mse2)/2 for allosteric, the Poisson loss for auxetic),
taken over every step, ties to the first such step, NaN losses excluded. The realization record
is the only place K* is read, and it raises instead of falling back when K* cannot be determined
exactly: missing data, all-NaN losses, loss and stiffness trajectories of different lengths,
stored best stiffnesses that disagree with the stored best loss, or a sweep cost Hessian that was
not evaluated at K*.

## Consequences

- The stiffnesses at a checkpoint or at the final step get their own names and are never called
  K*.
- Sweeps computed before 2026-09-10 (cost Hessian not at K*) raise and are recomputed on the
  cluster rather than read.
- The allosteric trainer saved best stiffnesses post-update before commit 3c24f68 (2026-08-10);
  runs first started before then raise for K* (decided from run provenance), and any canonical
  ensemble member among them is swapped for the next distinct-seed spare, as in ADR 0001.
