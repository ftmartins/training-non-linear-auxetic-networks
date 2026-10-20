# Nonlinear mechanical networks

Training elastic spring networks, by tuning bond stiffnesses, to perform nonlinear mechanical
tasks (allosteric response, targeted auxetic Poisson ratios), and analysing the trained ensembles.

## Tasks and networks

**Allosteric task**:
A pair of subtasks in which a strain applied at input nodes must produce a prescribed strain at
output nodes.
_Avoid_: allostery problem

**Auxetic task**:
A set of compression strains with a target Poisson ratio at each strain.
_Avoid_: Poisson task

**Geometry**:
One fixed allosteric network layout (node positions and bonds) shared by every realization of
its tasks. The "targeted" geometry is a geometry like the numbered ones.

**Packing**:
The jammed particle configuration that an auxetic network's nodes and bonds are derived from,
identified by its packing seed.

**Network**:
The concrete node positions, bonds and rest lengths a run trains on. Two runs share a network
only if these are identical, regardless of the packing seed that was requested.
_Avoid_: geometry (for auxetic), topology

**Network A**:
The canonical auxetic network (194 bonds) every auxetic ensemble member must train on. Networks
B and C are other networks that the same packing seed produced on different hardware.

**Network fingerprint**:
The identity of a network as computed from its actual positions and bonds, used to decide
whether a run trained on Network A.
_Avoid_: network group (a label in old manifests)

## Runs and ensembles

**Cell**:
One (geometry, task) for allosteric or one task for auxetic; the unit within which realizations
are compared and the ensemble is drawn.

**Realization**:
One training run inside a cell, starting from one initial condition.
_Avoid_: replicate, trial, seed (when meaning the run)

**Realization id**:
The serial number of a realization within its cell (its directory number). It carries no
information about the initial condition.

**Realization seed**:
The random seed that generated a realization's initial stiffnesses, as recorded with the run.
The recorded value is authoritative; seed tables are not.
_Avoid_: realization id, screened index

**Initial condition**:
The initial bond stiffnesses of a realization, fully determined by its realization seed (given
the network).
_Avoid_: IC seed

**Screening**:
Short trial trainings over a pool of candidate seeds per cell, used to rank candidates before
full training.

**Seed table**:
The per-cell list mapping realization id to realization seed.

**Converged**:
A realization whose best loss is at most 1e-4 of its initial loss.

**Best-loss stiffnesses**:
The stiffnesses at the training step with the lowest loss (K*), as opposed to the final step.
_Avoid_: final network (for this meaning)

**Canonical ensemble**:
The set of realizations every ensemble figure is computed from: per cell, the best-trained
converged realizations with distinct realization seeds (5 per allosteric cell, 15 per auxetic
task 0-19), auxetic members restricted to Network A.
_Avoid_: figure pool, good realizations

**Duplicate realization**:
A realization whose realization seed already belongs to another realization in the same cell.
Duplicates stay on disk but never enter the canonical ensemble.
