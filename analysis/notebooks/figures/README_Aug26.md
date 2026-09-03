# Aug-26 figure set

Implements `docs/figureDescription_Aug26.md`. Two notebooks + one shared module,
split so heavy computation runs once and plotting iterates fast:

| file | role |
|---|---|
| `aug26_common.py` | shared loaders, physics wrappers, discovery, plotting helpers |
| `NewFiguresAug26_calc.ipynb` | **computes** everything → `figure_data/aug26_figs/{allosteric,auxetic}/*.npz` |
| `NewFiguresAug26_plot.ipynb` | **reads** those `.npz` → Figures 1–5 (`figure_data/aug26_figs/figN_aug26.pdf`) |

## Data source

`data/allosteric_nets_aug` (post-2026-08-20 pipeline) and the `jax` auxetic tree
`data/auxetic_nets_aug/<family> (targeted default)`. Set in `aug26_common.py`
(`ALLO_DATA_DIR`, `AUX_DATA_DIR`).

## Running

1. Open `NewFiguresAug26_calc.ipynb`. Adjust the **USER KNOBS** cell
   (`MAX_REAL_PER_TASK_*`, `DO_FIGx`, `REP_*`, `C.CFG.*`). Every knob also honours an
   `AUG26_<NAME>` environment variable, so the notebook can be batch-executed:
   ```
   AUG26_MAX_REAL_PER_TASK_ALLO=1 AUG26_DO_FIG2=0 \
     jupyter nbconvert --to notebook --execute \
     --ExecutePreprocessor.kernel_name=auxetic_nets \
     analysis/notebooks/figures/NewFiguresAug26_calc.ipynb
   ```
2. Run top-to-bottom. Sections skip work whose `.npz` already exists unless
   `FORCE_RECOMPUTE`.
3. Open `NewFiguresAug26_plot.ipynb`, set the global toggles (`POOL` — Fig 3b /
   Fig 5d only; `NOCOMP_HESSIAN`; `FIG2A_HESSIAN`), run. **Fig 2 c/d/g/h are
   fixed to the full converged *targeted* pool** (`fig2{c,d}_targeted.npz`).

## Cost & caveats

* **`lammps` must be importable** in the calc kernel for a *faithful* recompute —
  the local pull of `allosteric_nets_aug` has no `timestep_sweep.npz`, so Fig 2/3/4
  quantities are recomputed from `stiffnesses_traj.npy`. Without LAMMPS the notebook
  falls back to JAX-FIRE: initial-stiffness losses still match to <1 %, but near
  convergence the two solvers land at slightly different equilibria (the
  loss-faithfulness cell prints the discrepancy). Elastic-Hessian and susceptibility
  panels are robust to this; absolute cost-Hessian magnitudes are not.
* **Per-subtask cost Hessian** (Fig 3a/b) is the expensive part — autodiff
  through the whole FIRE ramp, twice, Lanczos each. Raise `C.CFG.k_cost_eigs`
  only for final runs. (Fig 2a no longer uses the cost Hessian — it now shows the
  rest-configuration *elastic* spectrum, which is cheap.)
* **`fig2b` / `2f`** now store **one** trajectory — the deepest subtask — vs the
  actual imposed strain magnitude (`before` / `best` / `*_strain` / `strain_shallow`),
  matching the reworked plot. The auxetic `before` curve uses `stiffness_trajectory[0]`
  when present, else best-K only. The plot keeps an old per-subtask (`sub0_*`)
  fallback, so stale `fig2b.npz` render without a re-run — delete them (or set
  `FORCE_RECOMPUTE`) to pick up the new x-axis. `fig2a` / 2e (rest elastic spectrum
  vs loss) and 2h (avoided-crossing trajectory-position histogram) are implemented.
* **Figure 4** is an adaptation of `Fig3Maker050526.ipynb` (spectrum vs strain +
  incremental-displacement/mode overlaps + network) using the repo's current physics
  wrappers, assembled into one matplotlib figure.
