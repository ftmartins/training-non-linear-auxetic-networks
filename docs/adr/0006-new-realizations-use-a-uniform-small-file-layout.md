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
  record.json            # marks the layout: schema_version, kind
  training_meta.json     # configuration, realization seed (written by run_provenance, as today)
  run_provenance.json    # plus the code snapshot (.gz), as today
  network.npz            # allosteric: nodes, incidence_matrix, eq_lengths;
                         # auxetic: positions, edges, rest_lengths, boundary_<side>
  task.json              # allosteric: geometry_seed, strain_output(2); auxetic: task config
  loss.npy               # (n_steps, n_subtasks) allosteric; (n_steps,) auxetic
  stiffness.npz          # steps, stiffnesses (the trajectory snapshots)
  current.npy            # current (post-update) stiffnesses
  best.npz               # allosteric best-so-far: stiffnesses, training_loss
  resume.npz             # allosteric: last NaN-free checkpoint; auxetic: step, stiffnesses, loss
  trainer_state.pkl      # auxetic only: the trainer's own history, beside resume.npz
  positions.npy          # auxetic per-step relaxed positions
  sweep.npz              # post-training sweep
  training_complete.txt, training_complete_small_loss.txt   # markers, as today
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
- Settled in #14: best-so-far lives in `best.npz`; `stiffness.npz` takes one snapshot per
  checkpoint exactly as the legacy trajectory does; `resume.npz` is dropped only where the legacy
  checkpoint was (auxetic training that converged), because an allosteric extend resumes from it.
  Every file is written to a temporary name and renamed.
- Not carried over: the auxetic `freetraj` history field and `history.pkl`'s duplicate of the
  trajectories (no reader uses them); the trainer's private history stays in `trainer_state.pkl`.
- Fresh run directories get this layout only when `NMN_RECORD_LAYOUT=uniform`; the default stays
  legacy until the user flips it after the Network A campaign. `post_training_sweep.py` writes
  its sweep to the names the record gives (`record.sweep_files()`): `sweep.npz` and
  `sweep_meta.json` here, the legacy names for legacy runs.
