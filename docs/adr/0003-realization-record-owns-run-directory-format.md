# Realization record owns the run-directory format; old layouts are read forever

Every training runner, sweep script, analysis module and figure notebook reads run directories
on its own, so the on-disk format has grown many variants (allosteric vs auxetic checkpoints,
padded vs unpadded task directories, `_jammed` suffixes, `tasks.txt` parsed in six places) and
each reader copes with them differently. We decided that one realization record module owns the
format: it is the only code that knows file names and layouts, it opens a run directory (task
kind detected from its files) or iterates a data root by cell, and it presents one task-
independent interface, including the resume state (step, current stiffnesses, loss history)
whatever files hold it. Existing runs are never migrated: every layout that exists on disk today
stays readable forever, and a run keeps the format it started in, so resuming or extending an old
run writes the old layout.

## Considered Options

- Migrate all stored runs to one new layout: rejected, the as-trained data on marmalade, betty
  and Box is the record behind the published figures and the replication repo; rewriting it risks
  silent damage and breaks byte-level provenance.
- Keep per-script readers and only document the variants: rejected, the variants already cause
  real disagreements (e.g. fallbacks to the final step where K* is meant).

## Consequences

- The refactor first keeps today's format exactly (training outputs byte-identical, fresh and
  resumed); a new format, if any, is designed separately and applies to new runs only.
- Format changes reach `main` and the replication repo only after the active Network A campaign
  ends and a replication impact check (canonical seed tables, Figs 1-6 caches, replay) passes.
- Not-migrated code (legacy `src/`,
  `runners/`, speedup experiments) keeps writing its own layouts; the record must still read them.
