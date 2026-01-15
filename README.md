# Ensemble Training for Auxetic Networks

This directory contains scripts for running ensemble training of auxetic mechanical networks using quadratic springs.

## Overview

Trains 500 networks (10 tasks × 50 realizations) to achieve specific negative Poisson ratios (auxetic behavior):

- **Task**: Each task trains on 2 compression strains and 2 target Poisson ratios simultaneously
- **Realization**: Each realization uses different random initial stiffnesses
- **Network**: ~100 nodes with quadratic springs, optimized via gradient descent with FIRE minimization

## Installation

### Quick Setup (Conda - Recommended)

```bash
# Automated setup
bash setup_environment.sh

# Or manual setup
conda env create -f environment_mech_diffrax.yml
conda activate mech_diffrax

# Compile Cython extensions
cd ../instruments/
python setup_fire_minimizer_memview_cython.py build_ext --inplace
cd ../ensemble_training/
```

### Alternative Setup (Pip)

```bash
pip install -r requirements.txt
cd ../instruments/
python setup_fire_minimizer_memview_cython.py build_ext --inplace
cd ../ensemble_training/
```

For detailed installation instructions and troubleshooting, see [SETUP.md](SETUP.md).

## Quick Start

### 1. Test a single job

```bash
cd /Users/felipetm/Desktop/PhD/Research/Code/[1]Projects/energy_landscaping_minimizer/ensemble_training

# Run one training job with verbose output
python ensemble_runner.py --mode single --task 0 --realization 0 --verbose
```

### 2. Check status

```bash
# See how many jobs are complete
python ensemble_runner.py --mode status
```

### 3. Run full ensemble

```bash
# Run all 500 jobs sequentially (resumes automatically)
python ensemble_runner.py --mode sequential
```

### 4. Analyze results

```python
from data_loader import print_ensemble_summary, create_ensemble_dataframe
import pandas as pd

# Print summary statistics
print_ensemble_summary()

# Get DataFrame for custom analysis
df = create_ensemble_dataframe()
print(df.head())

# Export to CSV
from data_loader import export_results_to_csv
export_results_to_csv()
```

## File Structure

```
ensemble_training/
├── config.py                # Configuration constants
├── network_utils.py        # Network creation utilities
├── task_generator.py       # Task configuration generation
├── checkpoint_manager.py   # Checkpointing and resume logic
├── ensemble_runner.py      # Main execution script
├── data_loader.py          # Result loading and analysis
├── README.md               # This file
└── results/                # Training outputs (created automatically)
    ├── task_00/
    │   ├── realization_00/
    │   │   ├── history.pkl
    │   │   ├── final_network.pkl
    │   │   ├── task_config.json
    │   │   └── training_complete.txt
    │   ├── realization_01/
    │   └── ...
    └── task_01/
        └── ...
```

## Configuration

Edit `config.py` to modify:

- **Network parameters**: Number of nodes (default: 100), force type (quadratic)
- **Task configuration**: Number of tasks (10), realizations (50)
- **Compression/Poisson pools**: Which values to sample from
- **Training hyperparameters**:
  - `LEARNING_RATE = 1e-2`
  - `N_STEPS = 100,000`
  - `N_STRAIN_STEPS = 20`
  - `FORCE_TOL = 1e-6`
  - `VMIN = 1e-3, VMAX = 1e2` (stiffness bounds)
- **Parallelization**: `N_JOBS_OUTER = 4, N_JOBS_INNER = 2`

## How It Works

### Task Generation

Each task seed (0-9) deterministically selects:
- 2 compression strains from `[0.01, 0.02, ..., 0.09]`
- 2 Poisson ratios from `[-0.1, -0.25, -0.3, -0.5, -0.8, -1.0]`

Example:
```python
from task_generator import generate_task_config
config = generate_task_config(task_seed=0)
# {'task_seed': 0, 'packing_seed': 0,
#  'compression_strains': [0.08, 0.03],
#  'target_poisson_ratios': [-0.25, -0.5]}
```

### Network Creation

Each task has a unique network topology (via `packing_seed = task_seed`):
1. Generate sphere packing with 100 nodes
2. Extract contact network
3. Remove degree-1 (dangling) nodes
4. Identify boundary nodes (top, bottom, left, right)

### Stiffness Initialization

Each realization has different initial stiffnesses:
- Drawn from log-uniform distribution: `exp(uniform(log(1e-6), log(1)))`
- Reproducible via `realization_seed` (0-49)

### Training

Uses batch gradient descent to optimize stiffnesses:
1. For each iteration:
   - Minimize network energy (FIRE algorithm)
   - Compute quasistatic trajectories for each compression
   - Compute Poisson ratios (lateral strain / vertical strain)
   - Compute loss = MSE between actual and target Poisson ratios
   - Compute gradient via finite differences (parallelized)
   - Update stiffnesses with gradient descent
2. Clamp stiffnesses to `[VMIN, VMAX]`
3. Repeat for 100,000 steps or until convergence

## Dependencies

**Python packages** (should be installed):
- numpy
- scipy
- jax
- networkx
- matplotlib
- pandas
- joblib
- tqdm

**Internal modules**:
- `../instruments/elastic_network.py` - ElasticNetwork class
- `../instruments/fire_minimize_memview_cy.so` - Cython FIRE minimizer
- `../production/training_functions_with_toggle.py` - Training functions
- `../../cl_mech_repo/physical_learning/packing_utils.py` - Packing class

## Performance

- **Per job runtime**: 1-2 hours (depends on convergence)
- **Total runtime (sequential)**: 500-1000 hours (20-40 days)
- **Storage**: ~7.5-30 GB total
- **Memory per job**: 50-150 MB

## Troubleshooting

### FIRE minimization doesn't converge

**Symptoms**: `AssertionError` in quasistatic trajectory computation

**Solutions**:
1. Increase `FORCE_TOL` in `config.py` (e.g., `1e-6` → `1e-5`)
2. Check network topology (degree-1 nodes removed?)
3. Verify boundary node detection

### Training produces NaN

**Symptoms**: Stiffnesses become NaN, training stops early

**Solutions**:
1. Reduce `LEARNING_RATE` (e.g., `1e-2` → `1e-3`)
2. Check stiffness bounds (`VMIN`, `VMAX`)
3. Verify gradient computation isn't exploding

### Import errors (packing_utils, cmocean, etc.)

**Symptoms**: `ModuleNotFoundError` when running scripts

**Solutions**:
1. Check that `cl_mech_repo` path is correct in `config.py`
2. Install missing dependencies: `pip install cmocean`
3. Ensure Cython FIRE minimizer is compiled

### Disk space issues

**Symptoms**: `IOError` when saving results

**Solutions**:
1. Monitor disk usage: `df -h`
2. Compress old results: `tar -czf results.tar.gz results/`
3. Delete incomplete jobs: find partial results in `results/` and remove

## Checkpointing and Resume

### Automatic Checkpointing

The system automatically:
- ✓ Saves checkpoints during training
- ✓ Resumes from checkpoint if job crashes
- ✓ Skips already completed jobs
- ✓ Removes checkpoint after successful completion

### Local Resume

```bash
# Check what's left to do
python ensemble_runner.py --mode status

# Resume incomplete jobs (default behavior)
python ensemble_runner.py --mode sequential --resume

# Start from scratch (ignores existing results)
python ensemble_runner.py --mode sequential --no-resume
```

### HPC Cluster (SLURM)

For running on an HPC cluster, see [SLURM_GUIDE.md](SLURM_GUIDE.md).

**Quick start:**
```bash
# Submit all 500 jobs
sbatch submit_ensemble_slurm.sh

# Check progress
bash resubmit_incomplete.sh

# Resubmit failed jobs
sbatch --array=$(cat incomplete_jobs.txt) submit_ensemble_slurm.sh
```

Each job automatically resumes from checkpoint if interrupted.

## Advanced Usage

### Run specific range of tasks

```python
# Edit config.py temporarily
N_TASKS = 5  # Only run tasks 0-4

# Then run
python ensemble_runner.py --mode sequential
```

### Parallel execution (HPC)

For cluster environments, submit individual jobs:

```bash
#!/bin/bash
#SBATCH --array=0-499
#SBATCH --time=3:00:00
#SBATCH --mem=4G

TASK=$((SLURM_ARRAY_TASK_ID / 50))
REAL=$((SLURM_ARRAY_TASK_ID % 50))

python ensemble_runner.py --mode single --task $TASK --realization $REAL
```

### Custom analysis

```python
from data_loader import load_all_results
import numpy as np
import matplotlib.pyplot as plt

# Load all results
results = load_all_results()

# Extract final losses
final_losses = [r['history']['loss'][-1] for r in results.values()]

# Plot distribution
plt.hist(final_losses, bins=50)
plt.xlabel('Final Loss')
plt.ylabel('Count')
plt.yscale('log')
plt.savefig('loss_distribution.pdf')
```

## Citation

If you use this code, please cite the associated publication (add citation when available).

## Contact

For questions or issues, please contact the research group or open an issue in the repository.
# training-non-linear-auxetic-networks
