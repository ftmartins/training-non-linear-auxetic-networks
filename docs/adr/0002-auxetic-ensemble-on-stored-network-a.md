# Auxetic ensemble trains on one stored network, loaded from file

All auxetic targeted tasks were meant to share one network (packing seed 42), but the jammed
packing is integrated with an adaptive ODE solver whose result depends on the hardware: stored
runs hold three different networks (A: 280 runs on marmalade, 194 bonds; B: 20 runs on
marmalade; C: 180 runs on betty, 195 bonds), and a Mac produces a fourth. We decided the
canonical auxetic ensemble uses Network A only, and that new runs load Network A's stored
positions, bonds and rest lengths from a file instead of regenerating the packing, since even
marmalade produced both A and B. Ensemble membership is decided by the network fingerprint of
the run's stored network, not by where it ran.

## Considered Options

- Keep the mixed A/B/C ensemble and only document it: rejected, ensemble statistics would mix
  realizations of physically different networks.
- Regenerate the packing on marmalade: rejected, not reproducible (marmalade gave A and B).
- Make the packing deterministic: deferred to a separate issue; loading from file is needed
  regardless to reproduce the existing Network A runs.

## Consequences

- Tasks 8, 9 and 12-19 lose their B/C members and need new Network A runs; every task is brought
  to 20 distinct-seed Network A attempts (165 new runs), first re-training the B/C seeds on A,
  then unused screening candidates in rank order.
- B/C run directories stay on disk, outside the ensemble.
