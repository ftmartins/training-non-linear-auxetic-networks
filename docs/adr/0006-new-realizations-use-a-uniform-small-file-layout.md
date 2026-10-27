# New realizations use one uniform layout of small plain files

ADR 0003 left the design of a new run-directory format to a separate pass. Existing runs are never
migrated and a run keeps the format it started in; this decision applies to realizations created
after the format ticket lands. Today's two layouts differ by task kind (`mse1.npy`/`tasks.txt`
versus `loss_trajectory_jammed.npy`/`final_network_jammed.pkl`) and carry suffix and padding variants
that only the realization record has to cope with.

We decided that new realizations of both task kinds write the same file set, with no network-type
suffix in names:

```text
realization_NN/
  record.json      # schema_version, kind, realization seed, config, task, status, step
  network.npz      # nodes, edges, rest lengths, boundary nodes
  loss.npy         # (n_steps, n_subtasks); the training loss is the mean over subtasks
  stiffness.npz    # steps, stiffnesses (the trajectory snapshots)
  resume.npz       # step, current stiffnesses, best-so-far; removed when the realization is done
  sweep.npz        # post-training sweep
  provenance.json  # plus the code snapshot
```

The record gets a third private reader for this layout and detects it from `record.json`. Reading
and the resume state are unchanged for callers, and old-layout runs resumed or extended keep writing
their old layout.

## Equivalence criterion

An old-format and a new-format run of the same realization are equivalent when everything read
through the record is bit-identical: config, realization seed, network, loss and stiffness
trajectories, best-loss stiffnesses (K*) and resume state. Files may differ.

## Considered Options

- One HDF5 file per realization: rejected, it saves inodes on the nearly full cluster disk but needs
  h5py, cannot be inspected or diffed with plain numpy and rsync, and a crash mid-write can corrupt
  the whole record.
- Keep today's layouts and only tidy the record: rejected by the user in favour of one uniform
  layout for new runs.

## Consequences

- New layout reaches `main` only after the Network A campaign has ended (ADR 0003); it is written
  by the lifecycle through the record, so runners do not change.
- Details settled when implementing #14: snapshot cadence of `stiffness.npz`, and whether
  `record.json` or `resume.npz` holds best-so-far.
