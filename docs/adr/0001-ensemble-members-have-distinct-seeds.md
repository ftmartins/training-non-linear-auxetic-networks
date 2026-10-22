# Canonical ensemble members have distinct realization seeds

The cluster seed tables were re-aggregated (45 screening candidates, 20 kept) after training had
started, so some seeds already trained under one realization id were reassigned to another id
and trained again. The canonical ensemble then counted the same initial condition twice: 4 of
150 allosteric and 39 of 300 auxetic members (many pairs bit-identical runs). We decided that
ensemble selection keys on the realization seed recorded in each run's `training_meta*.json`,
never on the realization id or the seed table, and admits at most one realization per seed per
cell (the lower best/initial ratio; ties to the lower realization id). Duplicate run directories
are kept on disk and flagged, not deleted, so the as-trained history stays reproducible.

## Consequences

- Slots freed by duplicates are filled from existing converged spares when the cell has them
  (every allosteric cell does); new training happens only where a cell runs short.
- New realizations get fresh ids (auxetic: 20 upward) and the as-trained seed table is extended,
  never re-aggregated in place.
- A one-off audit hashes the initial stiffnesses of all members to confirm no two different seeds
  give the same initial condition.
