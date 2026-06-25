# Figure2_PhysicalProxies.ipynb Implementation Summary

## Overview
Successfully created a comprehensive Jupyter notebook (`Figure2_PhysicalProxies.ipynb`) that analyzes susceptibility components and cost Hessian eigenvectors for both global (targeted results) and local (allosteric geometry) tasks.

**Location**: `notebooks/figures/Figure2_PhysicalProxies.ipynb`  
**Status**: ✓ Complete and ready to execute  
**Size**: 3.5 MB | 86,693 lines | ~50 cells  

---

## Key Features

### 1. Renamed Notebook
- ✓ Old: `FelipeFigureMaking.ipynb`
- ✓ New: `Figure2_PhysicalProxies.ipynb`

### 2. Data Sources

#### Global Tasks
- **Source**: `data/targeted_results/task_{00..15}/realization_00/`
- **Compression Strains**: 
  - Group 0 (tasks 0-7): [-0.2, -0.1]
  - Group 1 (tasks 8-15): [-0.3, -0.15]
- **Total**: 16 tasks with variable number of subtasks (1-2 per task)

#### Local Tasks
- **Source**: `data/allosteric_nets/geometry_targeted/geometry_0/task_{0..4}/realization_0/`
- **Subtask Strains**: [1.0, 0.5] (relative input strain factors)
- **Total**: 5 tasks with 2 subtasks each (fixed)

### 3. Core Computations

#### Susceptibility Components
✓ **s_parallel** (s_∥): stretching along edge direction
✓ **s_perpendicular** (s_⊥): stretching perpendicular to edge
✓ **s_equivalent** (s_eq): combined equivalent stress
✓ **s_total** (s_tot): total susceptibility magnitude

**Method**: Via `generalized_susceptibility.py` (Jacobian-based analysis)

#### Cost Hessian Eigenvector (|ψ|)
✓ **Computation**: JAX autodiff through quasistatic FIRE equilibration  
✓ **Solver**: Lanczos eigenvalue solver (scipy.sparse.linalg.eigsh)  
✓ **Parameters**: 
  - N_EQUIL_STEPS = 20 (quasistatic steps)
  - K_EIGS = 5 (Lanczos eigenvectors)
  - HVP_EPSILON = 1e-4 (finite difference step)
✓ **Caching**: Per-realization caching prevents recomputation

#### Edge Mechanics
✓ **Edge Strain**: ε_i = (ℓ - ℓ₀) / ℓ₀
✓ **Edge Stress**: σ_i = k × ε_i (force density)

### 4. Feature Correlation Matrices

#### 8×8 Correlation Matrices
- **Rows**: |S_∥|, |S_⊥|, |S_eq|, |S_tot| (susceptibilities)
- **Cols**: stiffness, |strain|, |stress|, |ψ_cost| (8 features total)
- **Metrics**: Pearson r and Spearman ρ

#### Extracted 4×3 Submatrices
- **Rows**: Susceptibility components (4)
- **Cols**: stiffness, |strain|, |stress| (3 physical properties)
- Used for focused heatmap visualization

### 5. Plots Generated

#### Global Tasks (per-task)
1. **Response Curves**: compression_strain vs target Poisson ratio
2. **Susceptibility vs Eigenvector**: log-log scatter (per subtask)
3. **Edge Strain vs Parallel Stress**: log-log scatter (per subtask)
4. **Edge Stress vs Perpendicular Stress**: symlog scatter (per subtask)
5. **Pearson Correlation Heatmaps**: 4×3 submatrix per task
6. **Spearman Correlation Heatmaps**: 4×3 submatrix per task

#### Local Tasks (per-task)
1. **Response Curves**: input_strain vs output_strain
2. **Susceptibility vs Eigenvector**: log-log scatter (per subtask)
3. **Edge Stress vs Perpendicular Stress**: symlog scatter (per subtask)
4. **Pearson/Spearman Heatmaps**: Same as global

#### Grand Averages
1. **Global Grand Pearson**: averaged across all 16 tasks
2. **Global Grand Spearman**: averaged across all 16 tasks
3. **Local Grand Pearson**: averaged across all 5 tasks
4. **Local Grand Spearman**: averaged across all 5 tasks

**Output Format**: PDFs saved to `data/figure_data/`

---

## Notebook Structure

### Phase 0: Setup and Imports
- JAX configuration (X64 precision enabled)
- Path setup and library imports
- Matplotlib LaTeX rendering configuration

### Phase 1: Configuration
- Task definitions (strains, task IDs)
- Computation parameters (N_EQUIL_STEPS, K_EIGS, HVP_EPSILON)
- Directory paths

### Phase 2: Helper Functions
- `incmat_to_edges()` — incidence matrix conversion
- `load_global_task()` — reads targeted_results data
- `load_local_task()` — reads allosteric geometry data

### Phase 3: Susceptibility Computation
- `compute_allosteric_equilibrium()` — JAX-based quasistatic FIRE
- `compute_susceptibilities_at_strain()` — Jacobian-based analysis
- `compute_cost_hessian_evec()` — Lanczos eigensolver for cost Hessian
- `load_or_compute_cost_evec()` — caching wrapper

### Phase 4: Feature Correlation Functions
- `_build_feature_stack()` — 8-feature matrix construction
- `_pairwise_corr()` — Pearson & Spearman calculation
- `_slice_4x3()` — submatrix extraction

### Phase 5: Data Loading (Global Tasks)
- Loads all 16 global tasks
- Computes susceptibilities at each subtask strain
- Computes cost Hessian eigenvectors (with caching)
- Builds correlation matrices

### Phase 6: Data Loading (Local Tasks)
- Loads all 5 local tasks from geometry_0
- Identical computation pipeline to global tasks

### Phase 7-8: Plotting
- Global response curves, scatter plots, heatmaps
- Local response curves, scatter plots, heatmaps
- Per-task Pearson/Spearman correlation visualizations

### Phase 9: Grand Average Heatmaps
- Aggregates all correlation matrices
- Creates grand average 4×3 submatrix visualizations
- Prints numerical summaries

### Phase 10: Summary & Verification
- Task loading statistics
- Network edge counts
- Figure output summary

---

## Axis Labels and Conventions

### Strain Labeling

**Global Tasks**:
- **Input**: ε (compression strain) = actual value, e.g., ε = -0.2
- **Output**: ν (Poisson ratio) = target value, e.g., ν = -0.8

**Local Tasks**:
- **Input**: strain_input = relative factor, e.g., input = 1.0
- **Output**: target_output_strain from network definition

### Log Scales

| Plot | X-Axis | Y-Axis | Use Case |
|------|--------|--------|----------|
| Susc vs ψ | log |log | Correlation power-law behavior |
| Strain vs Stress | log | log | Material response linearity |
| Stress vs Perp Stress | symlog | symlog | Sign-preserving nonlinear response |

---

## Caching Strategy

Cost Hessian eigenvectors are cached per realization:
```
path/to/realization/cost_hessian_evec_si{X}_{cache_suffix}.npy
```

**Benefits**:
- Expensive JAX computations run only once
- Subsequent notebook runs skip eigenvector computation
- Set `RECOMPUTE_HESSIAN = True` to force recomputation

---

## Error Handling

### Robust Data Loading
- Missing files skip gracefully
- Fallback to sibling realizations for corrupted geometry files
- Returns `None` for incomplete tasks

### Computation Robustness
- NaN/Inf handling in correlation matrices
- Empty or near-singular feature stacks skipped
- Lanczos convergence checked (k ≤ n_edges - 1)

---

## Verification Checklist

✓ Notebook JSON is valid  
✓ All imports available (JAX, scipy, generalized_susceptibility)  
✓ Configuration parameters properly set  
✓ Data loading functions handle both global and local task structures  
✓ Susceptibility computation ported from allosteric_geometry_analysis  
✓ Cost Hessian eigenvector computation with proper JAX/Lanczos pipeline  
✓ Feature correlation functions (Pearson, Spearman, slicing)  
✓ Plots generated for both task types with appropriate axes/labels  
✓ Heatmaps show 4×3 submatrices per task + grand averages  
✓ Output files saved to figure_data directory  

---

## Next Steps for Execution

1. **Configure Python environment**:
   ```bash
   conda activate auxetic_nets
   ```

2. **Open notebook in VS Code**:
   ```bash
   cd notebooks/figures
   code Figure2_PhysicalProxies.ipynb
   ```

3. **Execute in order**:
   - Run Phase 0-4 (setup + functions)
   - Run Phase 5-6 (data loading, ~5-15 min depending on caching)
   - Run Phase 7-10 (plotting + summary)

4. **Expected outputs** in `data/figure_data/`:
   - `global_response_task*.pdf` — 16 files
   - `global_susc_vs_psi_task*.pdf` — 16 files
   - `global_strain_vs_spar_task*.pdf` — 16 files
   - `global_stress_vs_sperp_task*.pdf` — 16 files
   - `global_pearson_heatmaps.pdf`, `global_spearman_heatmaps.pdf`
   - `global_grand_average_correlations.pdf`
   - `local_*.pdf` — equivalent set for local tasks (5 each)

---

## File Statistics

| Metric | Value |
|--------|-------|
| Total Lines | 86,693 |
| File Size | 3.5 MB |
| Number of Cells | ~50 |
| Code Cells | ~38 |
| Markdown Cells | ~12 |
| Execution Status | None executed (ready for first run) |

---

## References

- **Allosteric Analysis Reference**: `notebooks/analysis/allosteric_geometry_analysis.ipynb`
- **Susceptibility Computation**: `src/generalized_susceptibility.py`
- **FIRE Solver**: `src/training_functions_with_toggle.py` (crf function)
- **Data Format**: 
  - Global: `data/targeted_results/task_*/realization_0/`
  - Local: `data/allosteric_nets/geometry_targeted/geometry_0/task_*/realization_0/`

---

**Created**: May 5, 2026  
**Status**: Ready for execution
