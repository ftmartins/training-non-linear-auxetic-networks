# Realization screening

Motivation: a task's random initial-stiffness draw ("realization") turned out
to matter a lot more than expected. In a live run of the allosteric targeted
ensemble (5 tasks × 5 realizations, LR=1.0), one specific realization index
stalled at <0.2% loss improvement over ~150 steps *across all 5 tasks*, while
another realization index converged 3-7% over the same window *across all 5
tasks* — a per-realization effect, not noise, not LR, not instability (every
sampled trajectory was smooth/monotonic, no jumps). Rather than accept
whichever 5 realizations happen to land at index 0-4, each task is now
screened over a larger candidate pool first, and only the best 5 are used for
the real, full-length run.

Applies uniformly to all four ensembles: auxetic targeted/general, allosteric
targeted/general.

## Procedure

1. **Screen**: for each task (each `(geometry, task)` pair for the general
   ensembles), run `POOL_SIZE=15` candidate realizations for a short window
   (`N_STEPS=150`) using the exact production training path — same optimizer,
   same hyperparameters as the real run would use (screening imports
   `LEARNING_RATE`/`USE_OPT_FIRE`/`OPT_FIRE_FINC` directly from the relevant
   runner module, so it can never silently drift out of sync with production).
   Candidate seeds live in a disjoint range (`good_realizations.SCREEN_SEED_BASE
   = 1_000_000`+) so they can never collide with any other seed value.
2. **Score**: candidates that produced a NaN are discarded outright. The rest
   are ranked by **final loss** (not minimum loss) after the short window —
   using the final value rather than the best-seen value means a candidate
   that dipped low and then jumped back up scores as if it had jumped back
   up, so this one ranking criterion captures both "made progress" and
   "didn't overshoot/oscillate" without a separate jumpiness penalty.
3. **Keep the best 5**, in ranked order. Their (screening) seeds are stored,
   indexed **serially by realization_id** (`realization_id=0` is always "the
   best candidate found for this task", regardless of what its underlying
   seed value happens to be) — this is also what lets the real full-length
   job array keep numbering realizations `0..4` exactly as before.
4. **Real run**: the production trainer looks up `realization_id` through
   this table instead of using it as a literal seed. Everything else about
   the training loop, checkpointing, and output layout is unchanged.

## Files

| File | Role |
|---|---|
| `training/src/good_realizations.py` | Lookup module: `get_realization_seed(kind, task_id, realization_id, geometry_id=None)`, `save_table`, `load_table` |
| `training/src/good_realizations/*.json` | The four lookup tables (one per kind), written by the aggregation step |
| `training/runners/screen_realizations_auxetic.py` | One screening trial (auxetic); `--kind {targeted,general}` |
| `training/runners/screen_realizations_allosteric.py` | One screening trial (allosteric); `--kind {targeted,general}` |
| `training/runners/screen_aggregate.py` | Reads all trial results for a kind, scores, writes the lookup table; `--physics {auxetic,allosteric} --kind {targeted,general}` |

`kind` here means "targeted vs. general ensemble"; the lookup table's own key
is `"<task_id>"` (targeted, and auxetic-general) or `"<geometry_id>_<task_id>"`
(allosteric-general, since geometry varies independently of task there).

## Scope screened (per kind)

| Kind | Tasks | Realizations kept | Screening pool | Real jobs |
|---|---|---|---|---|
| Auxetic targeted | 5 (`task_id` 0-4) | 5 | 15/task → 75 trials | 25 |
| Auxetic general | 30 (`task_id` 0-29, `base.config.N_TASKS`) | 5 | 15/task → 450 trials | 150 |
| Allosteric targeted | 5 (`task_id` 0-4, shared fixed geometry) | 5 | 15/task → 75 trials | 25 |
| Allosteric general | 5 geometries × 5 tasks = 25 pairs | 5 | 15/pair → 375 trials | 125 |

## Running it

Each kind is: submit the screening array (SLURM `--array` index maps to
`(task[, geometry], candidate)` — see `build_grid`/`POOL_SIZE` in the
screening script) → submit the aggregation job with a SLURM
`--dependency=afterok:<screening_array_job_id>` → submit the real full-length
array with `--dependency=afterok:<aggregation_job_id>`. This lets all four
kinds' screening, aggregation, and real runs be submitted at once while
guaranteeing the real jobs never start reading a lookup table that isn't
finished yet.

```bash
# example, auxetic targeted
sbatch --array=0-74 screen_realizations_auxetic.py --kind targeted --results-dir <scratch>
# (wrapped in a submit_*.sh in practice; see the sibling screening submit scripts)
sbatch --dependency=afterok:<screen_job_id> screen_aggregate.py --physics auxetic --kind targeted --results-dir <scratch>
sbatch --dependency=afterok:<aggregate_job_id> submit_targeted.sh
```

## Re-screening

The lookup tables are plain JSON, safe to inspect, hand-edit, or delete and
regenerate. There's no versioning/staleness check tying a table to the code
that produced it — if the network/task definitions change materially, rerun
screening for the affected kind before trusting its table. A production run
that can't find its table (or its task's key) raises `FileNotFoundError` /
`KeyError` rather than silently falling back to an unscreened seed.
