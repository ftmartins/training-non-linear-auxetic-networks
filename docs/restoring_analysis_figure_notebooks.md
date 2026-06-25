# LLM Restoration Skeleton for Missing Analysis Artifacts

## How to Use This File

For each missing or incomplete product:

1. Record what is known from surviving notebooks, scripts, imports, figures, saved outputs, and README notes.
2. Distinguish direct evidence from inference.
3. Describe the expected inputs, intermediate calculations, outputs, and figures.
4. Define validation criteria for deciding whether a restoration attempt is plausible.
5. Keep a short log of attempted prompts, generated artifacts, and failure modes.

## Evidence Classes

- Direct evidence: existing notebook cells, imports, helper functions, saved arrays, file names, plot titles, comments, printed output, and docstrings.
- Indirect evidence: similar notebooks in this repo, analogous analysis pipelines, naming conventions, and module usage in other files.
- Open assumptions: anything not explicitly supported by surviving code or outputs.

## Global Restoration Checklist

- Product name and type identified.
- Current status documented: missing, partially missing, or import-broken.
- Source evidence linked or copied into this file.
- Required data inputs identified.
- Required internal modules identified.
- Intermediate calculations enumerated.
- Expected figures and tables enumerated.
- Expected side effects or saved files enumerated.
- Validation tests or sanity checks listed.
- Restoration attempt log updated.

## Generic Product Template

### Product Summary

- Product name:
- Product type: notebook / python module / helper script / figure pipeline / other
- Current status: missing entirely / partially missing / import reference broken / unknown
- Priority:
- Owner or restorer:
- Last updated:

### Goal

- Scientific or analysis goal:
- Concrete user-facing purpose:
- What question the product should answer:

### Known Evidence

- Surviving references:
- Relevant imports:
- Relevant function names:
- Relevant variable names:
- Existing outputs or printed summaries:
- Related files in repo:
- Notes from comments, markdown cells, or docstrings:

### Required Inputs

- Input datasets:
- Directory or file layout expected:
- Parameters or configuration constants:
- Upstream scripts or notebooks that generate these inputs:
- Input assumptions that still need confirmation:

### Intermediate Calculations

- Reconstruction steps in likely execution order:
- Important helper functions or module calls:
- Derived quantities that must be computed:
- Temporary data structures expected in memory:
- Any numerical procedures, optimization, minimization, or tensor assembly involved:

### Outputs

- Returned objects or in-memory results:
- Saved files:
- Tables, metrics, or printed summaries:
- Notebook cells or sections expected to exist:

### Figures

- Figure titles or themes:
- Plot types:
- Axes, labels, and color encodings:
- Number of panels or subplots:
- Which intermediate quantities each figure depends on:

### Validation

- Minimal executable test:
- Shape, dtype, or dimensional consistency checks:
- Expected qualitative behavior:
- Comparison target from surviving outputs:
- Known failure modes:

### Restoration Strategy

- Best source artifacts to imitate:
- What can be restored directly from evidence:
- What must be inferred:
- What should be left explicitly marked as uncertain:

### Attempt Log

- Attempt date:
- Prompt or method used:
- Files produced:
- What worked:
- What failed:
- Next correction:

## Restoration Targets in This Repository

## Product 1: ensemble_figures.ipynb

### Product Summary

- Product name: ensemble_figures.ipynb
- Product type: analysis notebook
- Current status: missing entirely
- Priority: high

### Goal

- Reconstruct the notebook that likely generated summary figures for the ensemble training results.
- Recover the intended narrative order of analysis, figure generation, and any exported panels used elsewhere.
- Identify whether the notebook was exploratory only or also acted as a reproducible figure pipeline.

### Known Evidence

- Surviving references: none yet located in this repo by filename search.
- Related surviving notebooks: ensemble_convergence_analysis.ipynb, ensemble_correlation_analysis.ipynb.
- Related scripts likely feeding analysis data: data_loader.py, ensemble_runner.py, moduli_training.py, training_functions_with_toggle.py.
- Related input tree: data/results/task_*/realization_*/.
- Likely figure subject matter from repository scope: convergence behavior, task-to-task comparisons, realization-level statistics, auxetic response metrics, susceptibility or elasticity summaries.
- ensemble_figures used to export json file now locatd in figure_data.
- To understand the idea of cost, consider the training scripts like ensemble_runner and its dependencies.
- To understand the idea of mechanical actuation trajectory, consider the training scripts like ensemble_runner and its dependencies.

### Required Inputs

- Realization result directories under data/results or any alternate ensemble data directory.
- Training trajectories such as loss trajectories and stiffness trajectories if present.
- Task configuration metadata for each task and realization.
- Any cached summary tables produced by data_loader.py or related scripts.
- Confirmation of whether figures were built from raw realization files or pre-aggregated pandas DataFrames.

### Intermediate Calculations

- Discover available tasks and realizations.
- Load per-realization outputs and metadata.
- Aggregate metrics across realizations for each task.
- Compute summary statistics such as means, variances, convergence fractions, attained Poisson ratios, or stiffness distributions.
- Prepare plotting DataFrames and panel-specific subsets.

### Outputs

- Notebook sections for loading data, aggregating ensemble metrics, and producing figures.
- Possibly exported figures or panels saved to disk.
- Printed summaries describing number of completed jobs, task coverage, and aggregate behavior.

### Figures
- Be careful to do a filter of poorly trained models (check if this is done in the ensemble_correlation_analysis.ipynb).
- Figures:
    - First figure: In the first figure, we had three subplots. A plot of the network with edges between connected nodes with the edges colored by their stiffnesses, grays colormap. A plot of the poisson ratio along actuation (calculation actuation trajectory, then calculate the poisson ratio with respect to the boundary nodes top-bottom, left-right) for each of the targeted networks. The targeted networks were used in this case. Third subplot a scatter plot of absolute cost eigenvector entry for the highest eigenvalue vector versus the magnitude of the total susceptibility. In this case, you will need to compute the cost eigenvector corresponding to each subtask compression strain, and the susceptibility at each compression strain.
    - Second figure: In the second figure, plot the scatter plots of the different parts of the susceptibility (parallel=longitudinal, perp=transverse, equilibrium=eq) vs physical mechanical proxies, and then a subplot showing the network coloring the edges by the ratio of |s_tot|(1)/|s_tot(2)| where 1 and 2 denote each subtask of a chosen target task. At each subtask strain, do parallel vs strain, |perp| and perp vs stress, eq and |eq| vs stress. 2 versions of a 4 subplot figure (the two versions corresponding to |quantity| or quantity being used in the scatter subplots)- the fourth subplot in each is the ratio |s_tot|(1)/|s_tot(2)|. Each scatter-subplot has two sets of dots (orange, blue) corresponding to each subtask. 
    - Third figure: For each of the tasks of the ensemble_runner, I want individual scatter plots of |eigenvector(i)| vs |s_tot(i)| - orange and blue in the same scatter plot for each task. I then also want a heatmap of density averaging across realizations of each task. Do the same for the rank of |s_tot(i)| and |eigenvector(i)|.
	- Fourth figure: two subplots, one histogram with two distributions of pearson/spearman correlation coefficients, accumulate across tasks and realizations. Second subplot, average spearman/pearson coefficient versus (top-)percentile of the susceptibility kept. Show the envelope around the average plot line.

### Restoration Strategy

- Use ensemble_convergence_analysis.ipynb as the closest structural template if it covers adjacent analysis.
- Reuse existing data loading utilities before inventing new I/O code.
- Mark any figure panel whose existence is inferred but not evidenced.
- Ask clarification questions as needed. Pay attention to the data available.

## Product 2: ensemble_correlation_analysis.ipynb

### Product Summary

- Product name: ensemble_correlation_analysis.ipynb
- Product type: analysis notebook
- Current status: partially missing sections
- Priority: high

### Goal

- Restore the missing sections so the notebook forms a coherent, executable analysis of correlations between susceptibilities and cost-eigenvector structure across the trained ensemble.

### Known Evidence

- Surviving top-level notebook description lists six analyses:
	- correlations between susceptibilities and cost eigenvectors
	- susceptibilities at different strains
	- cost eigenvectors from individual Hessians
	- different kinds of susceptibilities: parallel, perp, eq
	- relative susceptibility ratio across strains or subtasks
	- relative susceptibility vs relative cost eigenvector
- Surviving imported missing module: generalized_susceptibility
- Surviving imported helpers include:
	- compute_quasistatic_trajectory_auxetic
	- compute_poisson_ratio_single
	- fire_minimize_network
	- fire_minimize_dof
- Surviving data-loading utilities already reconstruct tasks, realizations, and networks.
- Surviving analysis code computes:
	- susceptibility components per edge
	- cost Hessian information for a subtask
	- cross-strain or cross-subtask comparisons
	- relative susceptibility and relative eigenvector comparisons
- Surviving plot text reveals expected figures and printed summaries, including:
	- susceptibility vs eigenvector scatter plots
	- cross-strain susceptibility comparison plots
	- susceptibility component magnitude plots
	- relative susceptibility distributions
	- relative susceptibility vs relative eigenvector scatter plots

### Required Inputs

- Ensemble realization outputs from the expected data directory.
- Reconstructed network geometry for each task.
- Stiffness trajectories or final stiffness values per realization.
- Task configuration: compression strains and target Poisson ratios.
- generalized_susceptibility module or a compatible replacement.

### Intermediate Calculations

- Discover available tasks and realizations.
- Reconstruct the network for each task.
- Load trained stiffnesses and assign them to the network.
- Compute strained equilibrium states for each subtask.
- Compute physical Hessians and constrained inverse operators.
- Compute Jacobians and decompose susceptibility into parallel, perpendicular, and equilibrium-coupling contributions.
- Compute cost Hessians or their approximations for each subtask.
- Extract leading cost eigenvectors or related importance measures.
- Compare susceptibility vectors with eigenvector-derived edge importance metrics.
- Aggregate statistics across realizations.

### Outputs

- A notebook with complete markdown narrative and executable code cells.
- Per-realization analysis result objects, likely stored in dictionaries keyed by subtask.
- Aggregate printed summaries across the ensemble.

### Figures

- Figure family 1: susceptibility vs cost-eigenvector correlation scatter plots.
- Figure family 2: susceptibility comparisons across different strains.
- Figure family 3: susceptibility component contribution plots.
- Figure family 4: histograms or distributions of relative susceptibility.
- Figure family 5: relative susceptibility vs relative eigenvector comparisons.
- Todo: Figure family 6: video of actuation of auxetic networks for each realization of each task. First subplot of gif: network colored by stiffnesses, node positions in each frame follow actuation trajectory. next subplots: the different components of the susceptibility color the network edges, one subplot per parallel, perp, eq. 

### Validation

- Notebook should execute from top to bottom once missing imports are restored.
- Restored sections should be consistent with surviving section titles, plot labels, and printed summaries.
- Array lengths for susceptibility and eigenvector quantities should match number of edges.
- Cross-subtask comparisons should only run when both subtasks exist.

### Restoration Strategy

- Preserve surviving notebook order unless clear evidence supports reordering.
- Use existing function names and variable naming patterns from surviving cells.
- Prefer filling missing glue cells and narrative cells before rewriting surviving analysis logic.
- Keep explicit TODO markers where evidence is insufficient.
- Ask questions when needed.

## Product 3: generalized_susceptibility.py

### Product Summary

- Product name: generalized_susceptibility.py
- Product type: python module
- Current status: referenced import missing
- Priority: high
- Owner or restorer:
- Last updated:

### Goal

- Restore the missing module well enough to support notebook execution and to preserve the intended mathematical meaning of susceptibility-related calculations.

### Known Evidence

- Imported symbols in ensemble_correlation_analysis.ipynb:
	- compute_physical_hessian_strained
	- compute_full_jacobian_matrixwise
	- precompute_geometry
- elasticity_tensor.py explicitly states that its incidence-matrix assembly is derived from generalized_susceptibility.py.
- Surviving notebook code expects the module to support constrained Hessian construction, Jacobian assembly, and geometry decomposition for finite elastic networks.
- The notebook indicates the susceptibility decomposition uses parts H1, H2, H3, and H4 returned inside Hjac_parts.
- Consider what's done in ConstructingSusceptibilities_Stephen, and simply reimplement it into the new module file with the intended signature.

### Required Inputs

- positions
- edges
- stiffnesses
- rest_lengths
- constrained degree-of-freedom indices or masks
- force_type and numerical tolerances
- potentially inverse Hessian blocks or full inverse Hessian tensors

### Intermediate Calculations

- Build edge incidence matrix.
- Compute edge vectors, lengths, unit vectors, and projector tensors.
- Assemble physical Hessian at the strained configuration.
- Separate free and constrained DOFs where needed.
- Construct full Jacobian of the Hessian-derived response with respect to edge stiffnesses.
- Return decomposed pieces H1, H2, H3, and H4 in shapes compatible with notebook usage.

### Outputs

- compute_physical_hessian_strained(...): physical Hessian matrix, probably over DOFs.
- compute_full_jacobian_matrixwise(...): total Jacobian, component decomposition, and reusable geometry terms.
- precompute_geometry(...): cached geometry tensors or edgewise factors used repeatedly.

### Validation

- Imported functions must satisfy the notebook call signatures already present.
- Returned shapes must match downstream reshape and trace operations.
- Numerical results should be qualitatively consistent with notebook expectations:
	- finite Hessians
	- invertible free-DOF blocks where expected
	- edgewise susceptibility arrays with length equal to number of edges
- Compare implementation logic against elasticity_tensor.py where overlap exists.
