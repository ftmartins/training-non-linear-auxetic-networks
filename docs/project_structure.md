# Project Structure and Dependency Model

## Motivation

Scientific computing projects tend to accumulate code in a single `src/` folder
that mixes simulation physics, training logic, and analysis scripts. As the
project grows, notebooks start importing from runners, runners import from
analysis helpers, and everything depends on everything. This document describes
the four-layer structure used here and the rules that keep it clean.

---

## The Four Layers

```
base/          shared data structures and physics primitives
training/      cluster training: optimisation loops, runners, checkpointing
analysis/      post-training analysis: observables, figures, notebooks
data/          raw and processed outputs — never deleted, never imported from
```

### Dependency rule (strict)

```
analysis  →  base
training  →  base
analysis  ✗  training      (with one intentional exception, see below)
training  ✗  analysis
*         ✗  data          (data is read from disk, never imported as a module)
```

A layer may only import from layers *below* it in this hierarchy.
`base` imports nothing from the project — only standard library and third-party
packages.

---

## Layer Descriptions

### `base/`

Contains everything that is both:
- **physics / domain knowledge** (not tied to training or analysis workflow), and
- **used by more than one of the other layers**.

Typical residents:
- Core data structures (`ElasticNetwork`)
- Configuration and physical constants (`config.py`)
- Network construction utilities (`network_utils.py`)
- Physics simulation primitives — energy functions, minimizers, quasistatic
  trajectory computation, JAX-differentiable solvers (`simulate.py`)
- Elasticity tensor and moduli (`elasticity_tensor.py`)
- Plotting style (`plot_config.py`)

**The test:** if you find yourself writing the same simulation function in both a
training script and an analysis notebook, it belongs in `base/`.

### `training/`

Contains everything needed to run optimisation on a compute cluster.

```
training/
  src/       Python modules: loss functions, GD loops, task/checkpoint managers
  runners/   entry-point scripts and SLURM submission scripts
```

`training/src/` imports physics from `base/` and adds the training-specific
layer on top: loss functions, gradient computation, learning-rate schedules,
checkpoint I/O.  It does **not** import from `analysis/`.

The runners are thin wrappers that parse command-line arguments and call
`training/src/` functions. They may import `task_generator` (which encodes what
was trained and with what parameters) — this is the only place where
training-specific configuration is allowed to propagate outward.

### `analysis/`

Contains everything needed to interpret trained results.

```
analysis/
  *.py           observable modules (susceptibility, hessian, modes, mechanics,
                 trajectory, cost_utils, data_io, plotting)
  compute_*.py   cluster scripts that produce cached data files
  notebooks/
    figures/          publication figure notebooks (load pre-computed data)
    generate_data/    local notebooks that compute and save trajectory / Hessian data
```

`analysis/` imports from `base/` for shared physics.  It does **not** import
from `training/src/` — if an analysis module needs a simulation function, that
function belongs in `base/`, not `training/`.

**Intentional exception:** the cluster scripts `compute_*.py` import
`generate_task_config` from `training.src.task_generator`.  This is acceptable
because these scripts are explicitly post-processing trained outputs and need to
know the task parameterisation.  Core analysis modules (`susceptibility.py`,
`hessian.py`, etc.) do not do this.

### `data/`

Stored outputs from training runs and any external datasets.  Nothing in
`data/` is ever `import`ed — it is only read from disk via `analysis/data_io.py`
or equivalent I/O functions.  The directory is never deleted or modified by
analysis code (analysis writes new derived files alongside the originals).

---

## How to place a new file

| Question | Answer |
|---|---|
| Is it a data structure or physics function used by both training and analysis? | `base/` |
| Is it part of the optimisation loop, loss, or gradient? | `training/src/` |
| Is it a SLURM script or cluster entry point? | `training/runners/` |
| Is it an observable, figure, or post-training computation? | `analysis/` |
| Is it raw or processed numerical output? | `data/` (never a Python module) |

---

## Importing conventions

```python
# Inside base/ — relative imports only
from .elastic_network import ElasticNetwork
from .config import FORCE_TOL

# Inside training/src/ — base via absolute, intra-package via relative
from base.simulate import fire_minimize_network, crf
from .checkpoint_manager import save_checkpoint

# Inside analysis/ — base via absolute, never training.*
from base.simulate import compute_quasistatic_trajectory_auxetic
from base.elastic_network import ElasticNetwork
from .susceptibility import compute_susceptibilities

# Runners — may reach into training.src and base; set sys.path to project root first
from base.config import DATA_DIR
from training.src.task_generator import generate_task_config
```

---

## What this model prevents

| Anti-pattern | Prevented by |
|---|---|
| Analysis notebook imports training runner | `analysis ✗ training` rule |
| Training loop reimplements trajectory code | shared physics lives in `base/` |
| Physics function duplicated in three places | single home in `base/simulate.py` |
| Deleting a data file breaks a module import | `data/` is never imported |
| Circular imports between analysis and training | one-way dependency graph |
