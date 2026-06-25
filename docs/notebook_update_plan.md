# Notebook Update Request and Implementation Plan

## User Requests

1. Add a global boolean `PLOT_VIDEOS` in the notebook to control whether actuation videos are generated and saved in the current run.
2. Add a new bottom section that computes and plots distributions of Pearson and Spearman correlation coefficients.
   - Correlations are computed between `|S_tot^i|` and `|Psi_i|`, where `i` is the subtask index.
   - Use realizations from the best 5 tasks, where best is defined by lowest mean loss.
   - Check whether a loss threshold is currently applied across notebook calculations; if not, prepare for consistent thresholding.
   - Plot requirements:
     - Figure 1: 2 subplots (subtask 0 and subtask 1).
       - Each subplot overlays two distributions (Pearson and Spearman).
       - Use 40 bins over the same range within each subplot.
     - Figure 2: aggregate across subtasks.
       - One subplot with overlaid Pearson and Spearman distributions.
3. Add heatmap analysis for pairwise correlations among:
   - `|S_tot|`, `|S_par|`, `|S_perp|`, `|S_eq|`, `|Psi|`, `|strain|`, `|stress|`
   - Correlations should be computed per `(task, realization, subtask)` sample, then aggregated.
   - Display average and standard deviation heatmaps.

## Confirmed Decisions

- Scope: only `ensemble_correlation_analysis.ipynb`.
- Task selection: rank tasks by mean final loss using all discovered realizations.
- Threshold policy: introduce a global `LOSS_THRESHOLD` and apply it consistently across notebook analyses.
- Heatmap coefficient type: include both Pearson and Spearman.

## Implementation Plan

### Phase 1: Global Controls and Candidate Selection

1. Add top-level notebook config variables:
   - `PLOT_VIDEOS`
   - `LOSS_THRESHOLD`
   - `N_BEST_TASKS = 5`
   - `HIST_BINS = 40`
2. Add helper logic to:
   - scan all discovered tasks/realizations,
   - compute per-task mean final loss,
   - select the best 5 tasks,
   - build analysis candidates from those tasks.
3. Replace current fixed candidate slicing logic with best-task selection + threshold-aware filtering.

### Phase 2: Consistent Thresholding

1. Ensure thresholding is applied once and consistently before expensive analysis.
2. Exclude threshold-failing realizations from:
   - `all_analysis_results`,
   - all downstream DataFrames,
   - all new plots/aggregations.
3. Log skip reasons with task/realization/loss for traceability.

### Phase 3: Correlation Distribution Section

1. Add a new section at the bottom of the notebook.
2. For each included `(task, realization, subtask)`, compute:
   - Pearson(`|S_tot|`, `|Psi|`)
   - Spearman(`|S_tot|`, `|Psi|`)
3. Create Figure 1 with two subplots:
   - subplot for subtask 0,
   - subplot for subtask 1,
   - each overlays Pearson and Spearman histograms,
   - exactly 40 bins, shared range per subplot.
4. Create Figure 2 with one subplot:
   - aggregate both subtasks,
   - overlay Pearson and Spearman histograms,
   - 40 bins.
5. Print summary statistics table (count, mean, std, min, max) by subtask and coefficient type.

### Phase 4: Pairwise Correlation Heatmaps

1. Build feature vectors per sample for:
   - `|S_tot|`, `|S_par|`, `|S_perp|`, `|S_eq|`, `|Psi|`, `|strain|`, `|stress|`
2. Compute per-sample 7x7 matrices for:
   - Pearson correlations,
   - Spearman correlations.
3. Aggregate over all samples to produce:
   - mean matrix,
   - standard deviation matrix,
   for each coefficient type.
4. Plot two figures:
   - Pearson figure: mean and std heatmaps (1x2),
   - Spearman figure: mean and std heatmaps (1x2).

### Phase 5: Video Gating

1. Keep existing video-generation logic.
2. Wrap video execution block with `if PLOT_VIDEOS:`.
3. When `PLOT_VIDEOS` is false, print an explicit skip message.
4. Clarify interaction between global `LOSS_THRESHOLD` and video-specific `ANIM_LOSS_THRESH`.

## Validation Checklist

1. With `PLOT_VIDEOS=False`, verify video generation is skipped and other analyses run.
2. Verify selected tasks are the 5 lowest mean-loss tasks from discovered realizations.
3. Verify excluded realizations never enter downstream analyses.
4. Verify histogram settings:
   - 40 bins,
   - Pearson/Spearman overlaid,
   - shared range in each subplot.
5. Verify heatmap settings:
   - shape 7x7,
   - variable order: `|S_tot|`, `|S_par|`, `|S_perp|`, `|S_eq|`, `|Psi|`, `|strain|`, `|stress|`,
   - mean/std shown for both Pearson and Spearman.
