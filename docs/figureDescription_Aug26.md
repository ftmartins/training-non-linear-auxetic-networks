# Figure description Aug 26th

## Figure 1

a. One well trained auxetic targeted network with node at rest positions, edges colored by stiffness. Magma colormap. 
b. Strain input-output trajectories for 5 different tasks trained on the same geometry.
c-d: same as above but for auxetic networks.

a. Eigenvalues of the **elastic Hessian of the network at rest** (undeformed node
positions, no actuation) as a function of error/init_error for one of the targeted
allosteric tasks, at the log-spaced-by-loss checkpoints (`select_steps` /
`_select_local_threshold_indices`). One faint line per eigen-branch, lowest
non-trivial branch highlighted. (Was: MSE1/MSE2 cost-Hessian eigenvalue curves —
changed 2026-09-01.)
b. Constrained elastic hessian eigenvalues along a single actuation trajectory — the
deepest subtask only, whose ramp already spans the shallower subtask's range — as a
function of the actual imposed input-strain magnitude. Two sets of curves, one for
before training and one for the best loss step; a dotted vertical line marks where the
shallower subtask's ramp ends (`strain_shallow`).
c. Overlapping distributions of the **ratio** `λ0(after) / λ0(before)` — the lowest
non-trivial elastic-Hessian eigenvalue at best-loss K divided by the same quantity at the
initial (first selected checkpoint) K — computed at 3 compressions: no compression,
subtask 1 end state, subtask 2 end state. Across **all converged targeted
tasks/realizations** (`FIG2_ALLO_POOL` — every realization passing the 1e-4 convergence
bar; no per-task cap, no `general` variant). Log x-axis, dashed line at ratio 1. Series
labelled by the imposed strain of each subtask, not by subtask number. Global toggle
`NOCOMP_HESSIAN` picks constrained vs unconstrained for the no-compression series.
(Was: raw λ0 distributions at trained K — changed to the after/before ratio 2026-09-02.)
d. **Histogram of where along the actuation trajectory** diabatic-tracked avoided
crossings occur — normalised position `(t + 0.5)/(T - 1)` of every event, pooled over
both subtasks and **all converged targeted realizations**, before vs after training.
(Was: count of avoided crossings per realization — changed 2026-09-01. Mirrors the
ensemble event-strain histogram in `NewFiguresJune_allosteric.ipynb`.)
e-h: Same as above, but for auxetic tasks (`FIG2_AUX_POOL` = all converged targeted
auxetic realizations; 2h compression resolution set by `AUG26_FIG2H_NSTEPS`).

## Figure 3

a. |Top cost hessian eigenvectors| for each of the two subtasks plotted against shift susceptibility magnitude; each point is for an edge.
    Q: Do we have in the timestep data, the top cost hessian eigenvector for each of the tasks? Instead of considering the hessian of the total MSE, considering MSE_1, MSE_2 separately.
b. Distribution of the spearman rank correlation across subtasks, tasks, realizations of the allosteric targeted taks.
c-d: same as above, but for auxetic tasks.

## Figure 4
All subplots contained in the Fig3Maker*.ipynb for one allosteric and one auxetic targeted task. But try to make it so that all subplots for each of the tasks are plotted in the same matplotlib figure.

## Figure 5
a-c: pick targeted allosteric task.
a. s_i^parallel vs. |strain_i|; both quantities are calculated at each subtask, leading to two sets of points to be distinguished by color.
b. s_i^perp vs. sigma. Same as above. 
c. s_i^eq vs. sigma. Same as above.
d. Heatmap of spearman correlations across subtasks, tasks, realizations in the general ensemble. Each row corresponds to one of |s_i^parallel|, |s_i^perp|, |s_i^eq|, |s_shift|; columns correspond to stiffness, stress, strain.
e-h: same as above, but for auxetic task.