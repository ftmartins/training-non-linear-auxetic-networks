#!/usr/bin/env python3
"""
Auxetic network trainer, coupled-learning (contrastive Hebbian, free/clamped)
variant — single-network script for SLURM array submission.

Same task/geometry definitions as the gradient-based auxetic pipeline
(base.network_utils.create_auxetic_network, training.src.task_generator /
targeted_task_generator — unchanged), but the update rule is borrowed from
allosteric_trainer.py's coupled-learning scheme instead of computing
d(loss)/d(stiffness) via implicit differentiation through the equilibrium
solve. That adjoint solve is what was crashing (or, with a relaxed
regularizer, hanging) on the singular/near-singular tangent-stiffness matrix
that appears whenever a node's edges collapse toward vmin (a real floppy
mode — see base/simulate.py's crf_bwd). Coupled learning never needs it: the
gradient direction is estimated entirely from two *forward* FIRE relaxations
(free run, and a nudged "clamped" run) plus a local per-edge stretch-
difference rule, exactly like allosteric's learning_update — the update path
that has never once crashed in this whole investigation.

Free run: top/bottom boundary clamped at the task's compression strain,
left/right free — this is just the existing quasistatic compression, run to
equilibrium once per (compression_strain, target_poisson_ratio) pair the
task defines (1 or 2 pairs).

Clamped run: same top/bottom clamp, PLUS left/right pinned symmetrically to
a width nudged (by --eta) from the free run's own observed width toward the
width that would give exactly the target Poisson ratio.

See training/lammps_auxetic.py:compute_free_and_clamped_auxetic_lammps for
the physics, training/lammps_utils.py:strain_network_auxetic_clamped for the
LAMMPS mechanics, and /Users/fmartins/.claude/plans/floofy-hopping-
feigenbaum.md for the full design writeup.

Output layout:
  <output_dir>/{targeted|general}/task_<tid>/realization_<rid>/
    tasks.txt         — not written (task is fully determined by tid; see
                         task_config saved via task_config.json instead)
    stiffnesses.npy, loss.npy, stiffnesses_traj*.npy, best_*

Training runs for `training_steps` (see --training-steps); to train longer,
re-invoke with a larger --training-steps (resumes from the existing
checkpoint). Learning rate starts at --learning-rate and is scaled down by
training.src.lr_schedule every 1000 steps if loss has plateaued or
overshot — see that module for details.
"""

import argparse
import json
import os
import sys
import shutil

import numpy as np
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from base.config import FORCE_TYPE, BOUNDARY_MARGIN, FORCE_TOL, PACKING_PARAMS, NETWORK_TYPE
from base.network_utils import create_auxetic_network
from training.src.task_generator import (
    generate_task_config, compute_target_extensions, get_n_nodes, get_n_strain_steps,
)
from training.src.targeted_task_generator import get_targeted_task_config, TARGETED_TASKS
import base.config as _base_config
import training.lammps_auxetic as la
from training.src.run_provenance import (
    save_run_provenance,
    save_training_meta,
    save_code_snapshot,
    has_critical_mismatch,
    HyperparameterMismatchError,
    DEFAULT_CRITICAL_KEYS,
)
from training.src import lr_schedule
from training.src.good_realizations import get_realization_seed

_CODE_SNAPSHOT_FILES = [
    __file__,
    la.__file__,
    os.path.join(os.path.dirname(la.__file__), 'lammps_utils.py'),
    os.path.join(_ROOT, 'training', 'src', 'run_provenance.py'),
]

# ── Constants ─────────────────────────────────────────────────────────────────
# Same stiffness bounds as the rest of the auxetic pipeline (screening,
# GD-based training) — not allosteric's K_MIN/K_MAX, which were tuned for a
# different network scale.
K_MIN = 1e-4
K_MAX = 1e2
ETA = 1.0
LEARNING_RATE = 10.0
SOLVER = 'lammps'  # only backend implemented for Phase 1

N_TARGETED_TASKS = len(TARGETED_TASKS)
N_REALIZATIONS = 5
N_TRAINING_STEPS = 1_000

_REALIZATION_BASE = 3_000_000


def realization_rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(_REALIZATION_BASE + seed)


# ── Geometry / task setup (reuses the gradient-based pipeline's own code) ─────

def build_network_and_task(tid, targeted):
    """
    Returns (network, boundary_dict, task_config) — same construction the
    gradient-based auxetic pipeline uses (base.network_utils.create_auxetic_network,
    training.src.task_generator / targeted_task_generator), unchanged.
    """
    if targeted:
        task_config = get_targeted_task_config(tid)
    else:
        task_config = generate_task_config(tid)
        task_config = dict(task_config)
        task_config['n_nodes'] = get_n_nodes(tid)
        task_config['n_strain_steps'] = get_n_strain_steps(tid)

    network, boundary_dict = create_auxetic_network(
        n_nodes=task_config['n_nodes'], packing_seed=task_config['packing_seed'],
        force_type=FORCE_TYPE, boundary_margin=BOUNDARY_MARGIN,
        central_force=PACKING_PARAMS['central'], network_type=NETWORK_TYPE,
    )
    return network, boundary_dict, task_config


# ── Coupled-learning update rule ───────────────────────────────────────────────

def learning_update_auxetic(nodes_free, nodes_clamped, edges, rest_lengths, stiffnesses, eta):
    """
    One (compression_strain, target_poisson_ratio) pair's contribution to the
    stiffness update — same shape as allosteric_trainer.learning_update,
    adapted from incidence-matrix edge vectors to an explicit edge list
    (ElasticNetwork already stores edges/rest_lengths directly, no incidence
    matrix needed).
    """
    len_free = np.linalg.norm(nodes_free[edges[:, 1]] - nodes_free[edges[:, 0]], axis=1)
    len_clamped = np.linalg.norm(nodes_clamped[edges[:, 1]] - nodes_clamped[edges[:, 0]], axis=1)
    dV_free = len_free - rest_lengths
    dV_clamped = len_clamped - rest_lengths
    factors = (dV_free - dV_clamped) * dV_free
    return (1.0 / eta) * stiffnesses * factors


# ── Resume: find the latest NaN-free stiffness state ──────────────────────────

def load_resume_state(output_path):
    """Return (stiffnesses, loss_history, start_step) if a prior run exists, else None."""
    stiff_path = os.path.join(output_path, 'stiffnesses.npy')
    if not os.path.exists(stiff_path):
        return None

    loss_path = os.path.join(output_path, 'loss.npy')
    loss_history = np.load(loss_path) if os.path.exists(loss_path) else np.array([])

    stiff = np.load(stiff_path)
    if not np.any(np.isnan(stiff)):
        start_step = len(loss_history)
        print(f"  Resume: found clean stiffnesses at step {start_step}.")
        return stiff, loss_history, start_step

    ckpt_stiff_path = os.path.join(output_path, 'stiffnesses_ckpt.npy')
    ckpt_step_path = os.path.join(output_path, 'ckpt_step.txt')
    if os.path.exists(ckpt_stiff_path) and os.path.exists(ckpt_step_path):
        ckpt_stiff = np.load(ckpt_stiff_path)
        ckpt_step = int(np.loadtxt(ckpt_step_path))
        print(f"  Resume: stiffnesses.npy contains NaNs; "
              f"rolling back to checkpoint at step {ckpt_step}.")
        return ckpt_stiff, loss_history[:ckpt_step], ckpt_step

    print("  Resume: stiffnesses.npy contains NaNs and no clean checkpoint found; "
          "starting fresh.")
    return None


# ── Core training loop ────────────────────────────────────────────────────────

def _run_training_loop(network, boundary_dict, task_config, eta, learning_rate, n_steps,
                        output_path, loss_history=None, step_offset=0,
                        best_stiffnesses=None, best_loss=np.inf):
    if loss_history is None:
        loss_history = np.array([])
    if best_stiffnesses is None:
        best_stiffnesses = network.stiffnesses.copy()

    compression_strains = task_config['compression_strains']
    target_poisson_ratios = task_config['target_poisson_ratios']
    n_pairs = len(compression_strains)
    n_strain_steps = task_config['n_strain_steps']

    top_nodes = boundary_dict['top']
    bottom_nodes = boundary_dict['bottom']
    left_nodes = boundary_dict['left']
    right_nodes = boundary_dict['right']

    positions = network.positions
    edges = network.edges
    rest_lengths = network.rest_lengths

    best_updated = False
    update_mag = np.nan

    pbar = tqdm(range(n_steps), desc=f'(loss=nan, best={best_loss:.4e}), update_mag=nan')
    for j in pbar:
        stiffnesses = network.stiffnesses
        total_delta_K = np.zeros_like(stiffnesses)
        total_loss = 0.0

        for cs, target_nu in zip(compression_strains, target_poisson_ratios):
            nodes_free, nodes_clamped, observed_nu_free = la.compute_free_and_clamped_auxetic_lammps(
                positions, edges, stiffnesses, top_nodes, bottom_nodes, left_nodes, right_nodes,
                cs, target_nu, eta, n_steps=n_strain_steps, tol=FORCE_TOL,
            )
            total_delta_K = total_delta_K + learning_update_auxetic(
                nodes_free, nodes_clamped, edges, rest_lengths, stiffnesses, eta)
            total_loss += (observed_nu_free - target_nu) ** 2
        total_loss /= n_pairs

        lr_scale, _ = lr_schedule.lr_scale_for_step(loss_history)
        current_lr = learning_rate * lr_scale
        measured_stiffnesses = stiffnesses
        new_stiffnesses = np.clip(stiffnesses + current_lr * total_delta_K, K_MIN, K_MAX)
        update_mag = np.mean(np.log10(np.abs(current_lr * total_delta_K) + 1e-300))
        network.stiffnesses = new_stiffnesses

        loss_history = np.append(loss_history, total_loss)
        if total_loss < best_loss and not np.any(np.isnan(new_stiffnesses)):
            best_loss = total_loss
            best_stiffnesses = measured_stiffnesses.copy()
            best_updated = True

        pbar.set_description(
            f'(loss={total_loss:.4e}, best={best_loss:.4e}), '
            f'update_mag={update_mag:.4e}, lr_scale={lr_scale:.3g}')

        global_step = step_offset + j + 1
        if global_step % 50 == 0:
            print(f"  step {global_step}: loss={total_loss:.4e}  best={best_loss:.4e}")
            np.save(os.path.join(output_path, 'stiffnesses.npy'), new_stiffnesses)
            np.save(os.path.join(output_path, 'loss.npy'), loss_history)
            if not np.any(np.isnan(new_stiffnesses)):
                np.save(os.path.join(output_path, 'stiffnesses_ckpt.npy'), new_stiffnesses)
                np.savetxt(os.path.join(output_path, 'ckpt_step.txt'), [global_step], fmt='%d')
            if best_updated:
                np.save(os.path.join(output_path, 'best_stiffnesses.npy'), best_stiffnesses)
                np.savetxt(os.path.join(output_path, 'best_loss.txt'), [best_loss])
                best_updated = False
            traj_path = os.path.join(output_path, 'stiffnesses_traj.npy')
            steps_path = os.path.join(output_path, 'stiffnesses_traj_steps.npy')
            if os.path.exists(traj_path) and os.path.exists(steps_path):
                traj = np.vstack([np.load(traj_path), measured_stiffnesses])
                steps = np.append(np.load(steps_path), len(loss_history) - 1)
            else:
                traj = measured_stiffnesses[np.newaxis, :]
                steps = np.array([len(loss_history) - 1])
            np.save(traj_path, traj)
            np.save(steps_path, steps)

        if total_loss / loss_history[0] < 8e-5:
            print(f"  Early stop at step {global_step}: converged.")
            break

    return loss_history, network.stiffnesses, best_stiffnesses, best_loss


def check_success(loss_history):
    if len(loss_history) == 0:
        return False
    ratio = np.min(loss_history) / loss_history[0]
    print(f"  Success check: min_loss/loss[0] = {ratio:.4e} ({'PASS' if ratio < 0.01 else 'FAIL'})")
    return ratio < 0.01


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train one auxetic network via coupled learning (free/clamped, LAMMPS).')
    parser.add_argument('--task-id', type=int, required=True,
                        help=f'Task index (0 to {N_TARGETED_TASKS - 1} targeted, '
                             f'0 to {_base_config.N_TASKS - 1} general)')
    parser.add_argument('--realization-id', type=int, required=True,
                        help=f'Realization index (0 to {N_REALIZATIONS - 1})')
    parser.add_argument('--training-steps', type=int, default=N_TRAINING_STEPS)
    parser.add_argument('--output-dir', type=str,
                        default='/data2/shared/felipetm/auxetic_cl_nets')
    parser.add_argument('--targeted-ensemble', action='store_true')
    parser.add_argument('--eta', type=float, default=ETA,
                        help='Coupled-learning nudge strength (default: %(default)s)')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow this run to replace previously recorded solver/learning_rate/'
                             'k_min/k_max/eta in training_meta.json instead of failing on mismatch')
    args = parser.parse_args()

    tid = args.task_id
    rid = args.realization_id
    training_steps = args.training_steps
    output_dir = args.output_dir
    targeted = args.targeted_ensemble
    eta = args.eta
    learning_rate = args.learning_rate
    overwrite = args.overwrite

    mode_tag = 'targeted' if targeted else 'general'
    print(f"=== Auxetic coupled-learning trainer: {mode_tag}, task={tid}, realization={rid} ===")

    _seed_kind = 'auxetic_cl_targeted' if targeted else 'auxetic_cl_general'
    screened_seed = get_realization_seed(_seed_kind, tid, rid)

    output_path = os.path.join(output_dir, mode_tag, f'task_{tid}', f'realization_{rid}')
    os.makedirs(output_path, exist_ok=True)

    _CRITICAL_KEYS = DEFAULT_CRITICAL_KEYS | {'eta'}
    current_hparams = {
        'solver': SOLVER,
        'learning_rate': learning_rate,
        'k_min': K_MIN,
        'k_max': K_MAX,
        'eta': eta,
        'n_training_steps': training_steps,
        'realization_seed': screened_seed,
    }
    if has_critical_mismatch(output_path, current_hparams, critical_keys=_CRITICAL_KEYS):
        if not overwrite:
            raise HyperparameterMismatchError(
                f"{output_path}/training_meta.json already recorded different solver/"
                f"learning_rate/k_min/k_max/eta/realization_seed than this run's. Resuming would "
                f"mix incompatible settings into one training trajectory. Pass --overwrite to wipe "
                f"it and restart from scratch under the new hyperparameters."
            )
        print(f"  --overwrite set: recorded hyperparameters differ from this run's — wiping "
              f"{output_path} and restarting from scratch under the new hyperparameters.")
        shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)

    save_training_meta(output_path, current_hparams, critical_keys=_CRITICAL_KEYS)
    save_run_provenance(output_path, extra=current_hparams)
    save_code_snapshot(output_path, _CODE_SNAPSHOT_FILES)

    network, boundary_dict, task_config = build_network_and_task(tid, targeted)
    with open(os.path.join(output_path, 'task_config.json'), 'w') as fh:
        json.dump({k: (v if not isinstance(v, np.ndarray) else v.tolist())
                   for k, v in task_config.items()}, fh, indent=2)

    resume = load_resume_state(output_path)
    if resume is not None:
        stiffnesses, loss_history, start_step = resume
        network.stiffnesses = stiffnesses
        print(f"  Stiffnesses: [{stiffnesses.min():.4g}, {stiffnesses.max():.4g}] "
              f"(resumed from step {start_step})")
    else:
        rrng = realization_rng(screened_seed)
        network.stiffnesses = rrng.uniform(K_MIN, K_MAX, size=len(network.edges))
        loss_history = np.array([])
        start_step = 0
        print(f"  Stiffnesses: [{network.stiffnesses.min():.4g}, {network.stiffnesses.max():.4g}]")

    best_path = os.path.join(output_path, 'best_stiffnesses.npy')
    bloss_path = os.path.join(output_path, 'best_loss.txt')
    if os.path.exists(best_path) and os.path.exists(bloss_path):
        best_stiffnesses = np.load(best_path)
        best_loss = float(np.loadtxt(bloss_path))
        print(f"  Best state: loss={best_loss:.4e} (loaded from disk)")
    else:
        best_stiffnesses = None
        best_loss = np.inf

    print(f"  Training steps: {training_steps:,}")
    remaining_steps = max(0, training_steps - start_step)
    if remaining_steps > 0:
        loss_history, stiffnesses, best_stiffnesses, best_loss = _run_training_loop(
            network, boundary_dict, task_config, eta, learning_rate, remaining_steps,
            output_path, loss_history=loss_history, step_offset=start_step,
            best_stiffnesses=best_stiffnesses, best_loss=best_loss,
        )
        network.stiffnesses = stiffnesses

    success = check_success(loss_history)
    print("\nTraining succeeded." if success else "\nTraining did not reach the success threshold.")

    np.save(os.path.join(output_path, 'stiffnesses.npy'), network.stiffnesses)
    np.save(os.path.join(output_path, 'loss.npy'), loss_history)
    if best_stiffnesses is not None:
        np.save(os.path.join(output_path, 'best_stiffnesses.npy'), best_stiffnesses)
        np.savetxt(os.path.join(output_path, 'best_loss.txt'), [best_loss])


if __name__ == '__main__':
    main()
