"""
Post-training timestep-sweep analysis.

Pipeline (per selected training step):
  select_loss_threshold_steps -> recompute stiffness-based loss (training's own
  evaluation path) -> recompute trajectory + loss_from_trajectory -> reduced
  elastic Hessian eigenpairs at `n_hessian_traj_steps` linearly-spaced points
  along that trajectory (0 -> full compression_strain / actuation displacement),
  not just the endpoint. Applies to both `sweep_auxetic` and `sweep_allosteric`.

Cost Hessian (eigenpairs of the loss w.r.t. stiffnesses) is only evaluated at
the first and last selected step (before/after training), since it is much
more expensive than the elastic Hessian.

The full per-subtask/per-task trajectory (all frames, not just the
`n_hessian_traj_steps` sparse points the elastic Hessian is evaluated at) is
also saved, since it costs little next to the eigenvector arrays already kept
and lets downstream consumers (e.g. participation-ratio analyses, which need
displacement between arbitrary consecutive frames) index into it directly
instead of recomputing the trajectory.

Two physics engines, matching what each trainer actually uses:
  - `sweep_auxetic`    — auxetic/"global" networks, JAX quasistatic solver
                         (analysis.cost_utils, same as ensemble_runner.py /
                         targeted_ensemble_runner.py).
  - `sweep_allosteric` — allosteric/"local" clamp networks, LAMMPS
                         (training.runners.allosteric_trainer, same engine
                         allosteric_trainer.py trains with).
"""

import copy
import time

import numpy as np

from .cost_utils import compute_trajectory, loss_from_trajectory, compute_cost_hessian
from .hessian import compute_hessian_spectrum


# ---------------------------------------------------------------------------
# Timestep selection (ported from analysis/notebooks/figures/NewFiguresJune.ipynb)
# ---------------------------------------------------------------------------

def select_loss_threshold_steps(loss, n_steps=50, eps_min=1e-8):
    """Return training step indices at log-spaced loss thresholds, guaranteed monotone.

    Works on the running-minimum subsequence (times when loss sets a new record low)
    so L[selected] is strictly decreasing. t_star (argmin) is always the last step.
    """
    L = np.asarray(loss)
    t_star = int(np.argmin(L))
    L_d = L[:t_star + 1]
    L_min = max(float(L_d[-1]), 1e-300)

    # Build strictly-decreasing running-minimum subsequence (t=0 always included)
    nm_idx = [0]
    run_min = float(L_d[0])
    for t in range(1, len(L_d)):
        if float(L_d[t]) < run_min:
            run_min = float(L_d[t])
            nm_idx.append(t)
    nm_idx = np.array(nm_idx)
    L_nm  = L_d[nm_idx]
    rel   = (L_nm - L_min) / L_min

    if len(rel) == 0 or rel[0] <= 0:
        return np.array([t_star])

    thresholds = np.logspace(np.log10(rel[0] + 1e-300), np.log10(eps_min), n_steps)
    selected_nm = []; ptr = 0
    for th in thresholds:
        hits = np.where((np.arange(len(rel)) >= ptr) & (rel <= th))[0]
        if hits.size > 0:
            selected_nm.append(int(hits[0]))
            ptr = hits[0] + 1

    idx = np.unique(np.append(nm_idx[selected_nm], t_star))
    return idx


# ---------------------------------------------------------------------------
# Constrained-DOF conventions (which node DOFs are excluded from the reduced
# elastic Hessian), matching NewFiguresJune.ipynb's free_dofs_global/_local.
# ---------------------------------------------------------------------------

def global_constrained_nodes(boundary):
    return np.union1d(
        np.asarray(boundary['top'], dtype=int),
        np.asarray(boundary['bottom'], dtype=int),
    )


LOCAL_CONSTRAINED_NODES = np.array([0, 1, 2, 3])


# ---------------------------------------------------------------------------
# Auxetic ("global") sweep
# ---------------------------------------------------------------------------

def recompute_stiffness_loss(net, target_poisson_ratios, top, bottom, left, right,
                             compression_strains, n_strain_steps, force_type, tol,
                             gradient_method):
    """
    Recompute loss for the current net.positions/net.stiffnesses with the same
    solver the training run actually used — this is what makes it a genuine
    "training's own evaluation path" check instead of an independent cross-check.

    gradient_method: 'newton' -> compute_ift_gradient (Newton IFT), matching
        finish_training_GD_auxetic_batch(method='newton'). Anything else
        ('fire', 'parallel', 'jax') -> poisson_loss_batch_parallel (Cython FIRE
        quasistatic + finite-difference loss); 'fire'/'parallel' training uses
        this exact function, and 'jax' training relaxes with the same Cython
        FIRE minimizer (just autodiffs the loss instead), so this is the
        closest non-autodiff reconstruction available for it.
    """
    from base.simulate import compute_ift_gradient
    from training.src.training_functions import poisson_loss_batch_parallel

    if gradient_method == 'newton':
        mse, _ = compute_ift_gradient(
            net, compression_strains=compression_strains, target_poissons=target_poisson_ratios,
            top_nodes=top, bottom_nodes=bottom, left_nodes=left, right_nodes=right,
            tol=tol, n_strain_steps=n_strain_steps,
        )
    else:
        mse, _ = poisson_loss_batch_parallel(
            net, target_poisson_ratios, top, bottom, left, right,
            compression_strains, n_strain_steps=n_strain_steps,
            force_type=force_type, tol=tol,
        )
    return float(mse)


def sweep_auxetic(network, task_config, boundary, stiffness_traj, positions_traj, loss_traj,
                  n_thresh_steps=50, eps_min=1e-8, n_traj_steps=100, k_eigs=10,
                  n_hessian_traj_steps=20,
                  force_type='quadratic', n_strain_steps=100, tol=1e-10,
                  gradient_method='newton', verbose=True):
    """
    Run the full timestep-sweep analysis for one auxetic training result.

    Parameters
    ----------
    network : ElasticNetwork        topology (edges, rest_lengths); positions
                                     and stiffnesses are overwritten per
                                     selected step from positions_traj/stiffness_traj.
    task_config : dict              must have 'compression_strains' and
                                     'target_poisson_ratios' (equal-length lists).
    boundary : dict                 keys 'top', 'bottom', 'left', 'right'.
    stiffness_traj : (T, E) array   full per-step stiffness trajectory.
    positions_traj : sequence of (N, 2) arrays, length T
                                     full per-step positions trajectory
                                     (history['positions']) — must be indexed
                                     identically to stiffness_traj/loss_traj so
                                     that step t's stiffness is evaluated at
                                     step t's actual starting positions, not
                                     e.g. the final network's positions.
    loss_traj : (T,) array          full per-step stored loss (same array used
                                     to select timesteps).
    gradient_method : str           'newton', 'fire', 'parallel', or 'jax' — the
                                     gradient method the training run actually
                                     used (see finish_training_GD_auxetic_batch/
                                     finish_training_GD_auxetic_batch_jax).
                                     Selects the loss solver used to recompute
                                     recomputed_stiffness_loss/recomputed_trajectory_loss
                                     so the sweep matches training's own evaluation
                                     path instead of always using FIRE.
    n_hessian_traj_steps : int      number of linearly-spaced points along each
                                     recomputed compression trajectory (0 -> full
                                     compression_strain) at which the elastic
                                     Hessian spectrum is evaluated.
    k_eigs : int                    number of top (largest positive / algebraic)
                                     cost-Hessian eigenpairs to compute. Does NOT
                                     affect the elastic Hessian, which always
                                     returns its full spectrum (all modes) — the
                                     elastic Hessian is dense-solved via
                                     np.linalg.eigh regardless of truncation, so
                                     truncating it saves no compute, only storage.

    Returns
    -------
    dict of numpy arrays, ready for np.savez_compressed. Includes the full
    per-subtask, per-selected-step trajectory (`trajectory_positions`) — cheap
    to store next to the (n_free_dofs, n_free_dofs) eigenvector arrays already
    kept — so downstream analyses that need displacement between arbitrary
    frames (e.g. participation ratio) can index into it directly instead of
    recomputing the trajectory.
    """
    compression_strains   = list(task_config['compression_strains'])
    target_poisson_ratios = list(task_config['target_poisson_ratios'])
    top, bottom, left, right = boundary['top'], boundary['bottom'], boundary['left'], boundary['right']
    constrained_nodes = global_constrained_nodes(boundary)
    n_sub = len(compression_strains)

    # 'newton' training loss <-> compute_ift_gradient; everything else <-> FIRE —
    # mirrors finish_training_GD_auxetic_batch's own method dispatch.
    traj_method = 'newton' if gradient_method == 'newton' else 'fire'

    t_indices = select_loss_threshold_steps(loss_traj, n_thresh_steps, eps_min)
    n_t = len(t_indices)
    stored_loss = np.asarray(loss_traj)[t_indices]

    # Linearly-spaced trajectory indices (0 -> n_traj_steps-1) at which the
    # elastic Hessian is evaluated; same for every selected timestep/subtask
    # since n_traj_steps is fixed for the whole sweep.
    traj_hess_idx = np.unique(np.round(
        np.linspace(0, n_strain_steps - 1, n_hessian_traj_steps)
    ).astype(int))

    recomputed_stiffness_loss  = np.full(n_t, np.nan)
    recomputed_trajectory_loss = np.full(n_t, np.nan)
    subtask_trajectory_loss    = np.full((n_t, n_sub), np.nan)

    elastic_eigvals_list = []   # (n_sub,) list of (n_hess_steps, n_free_dofs) arrays, per t — full spectrum
    elastic_eigvecs_list = []   # (n_sub,) list of (n_hess_steps, n_free_dofs, n_free_dofs) arrays, per t
    traj_positions_list  = []   # (n_sub,) list of (n_strain_steps, N, 2) arrays, per t — full trajectory

    net = copy.deepcopy(network)

    for i, t in enumerate(t_indices):
        k_t = np.asarray(stiffness_traj[t], dtype=float)
        net.stiffnesses = k_t
        net.positions = np.asarray(positions_traj[t], dtype=float)

        if verbose:
            print(f"  [auxetic sweep] step {i+1}/{n_t} (t={t}) ...", flush=True)

        # (a) stiffness-based loss — the exact training-time evaluation path
        mse = recompute_stiffness_loss(
            net, target_poisson_ratios, top, bottom, left, right,
            compression_strains, n_strain_steps, force_type, tol, gradient_method,
        )
        recomputed_stiffness_loss[i] = mse

        # (b) trajectory-based loss + (c) elastic Hessian along the trajectory, per subtask
        eigvals_this_t = []
        eigvecs_this_t = []
        traj_pos_this_t = []
        sub_losses = np.full(n_sub, np.nan)
        for s, (cs, tp) in enumerate(zip(compression_strains, target_poisson_ratios)):
            traj = compute_trajectory(k_t, net, boundary, cs, n_steps=n_strain_steps,
                                      force_type=force_type, tol=tol, method=traj_method)
            sub_losses[s] = loss_from_trajectory(traj, cs, tp, boundary)

            vals_along_traj = []
            vecs_along_traj = []
            for j in traj_hess_idx:
                # n_modes=None -> full spectrum; the dense eigh solve below is
                # unconditional, so truncating here would save no compute.
                vals, vecs, _ = compute_hessian_spectrum(
                    traj[j], net.edges, k_t, net.rest_lengths,
                    constrained_nodes=constrained_nodes, n_modes=None)
                vals_along_traj.append(vals)
                vecs_along_traj.append(vecs)
            eigvals_this_t.append(np.stack(vals_along_traj))   # (n_hess_steps, n_free_modes)
            eigvecs_this_t.append(np.stack(vecs_along_traj))   # (n_hess_steps, n_free_dofs, n_free_modes)
            # Full-resolution positions (not just at traj_hess_idx) so downstream
            # analyses (e.g. participation ratio, which needs displacement between
            # arbitrary consecutive frames) can index into the trajectory themselves
            # without recomputing it.
            traj_pos_this_t.append(np.stack([np.asarray(p) for p in traj]))  # (n_strain_steps, N, 2)

        subtask_trajectory_loss[i]    = sub_losses
        recomputed_trajectory_loss[i] = float(np.mean(sub_losses))
        elastic_eigvals_list.append(np.stack(eigvals_this_t))
        elastic_eigvecs_list.append(np.stack(eigvecs_this_t))
        traj_positions_list.append(np.stack(traj_pos_this_t))   # (n_sub, n_strain_steps, N, 2)

    elastic_hessian_eigvals = np.stack(elastic_eigvals_list)   # (n_t, n_sub, n_hess_steps, n_free_dofs) — full spectrum
    elastic_hessian_eigvecs = np.stack(elastic_eigvecs_list)   # (n_t, n_sub, n_hess_steps, n_free_dofs, n_free_dofs)
    trajectory_positions    = np.stack(traj_positions_list)    # (n_t, n_sub, n_strain_steps, N, 2) — full trajectory, all selected steps

    # Cost Hessian only before (first selected step) and after (last selected step)
    cost_before_vals, cost_before_vecs = [], []
    cost_after_vals,  cost_after_vecs  = [], []
    for t, vals_out, vecs_out, tag in (
        (t_indices[0],  cost_before_vals, cost_before_vecs, 'before'),
        (t_indices[-1], cost_after_vals,  cost_after_vecs,  'after'),
    ):
        k_t = np.asarray(stiffness_traj[t], dtype=float)
        net.stiffnesses = k_t
        net.positions = np.asarray(positions_traj[t], dtype=float)
        for s, (cs, tp) in enumerate(zip(compression_strains, target_poisson_ratios)):
            if verbose:
                print(f"  [auxetic sweep] cost Hessian ({tag}, subtask {s}) ...", flush=True)
            vals, vecs = compute_cost_hessian(
                net, cs, tp, boundary, k_eigs=k_eigs,
                force_type=force_type, n_strain_steps=n_strain_steps, verbose=verbose)
            vals_out.append(vals)
            vecs_out.append(vecs)

    return {
        't_indices': t_indices,
        'stored_loss': stored_loss,
        'recomputed_stiffness_loss': recomputed_stiffness_loss,
        'recomputed_trajectory_loss': recomputed_trajectory_loss,
        'subtask_trajectory_loss': subtask_trajectory_loss,
        'elastic_hessian_eigvals': elastic_hessian_eigvals,
        'elastic_hessian_eigvecs': elastic_hessian_eigvecs,
        'elastic_hessian_traj_indices': traj_hess_idx,
        'elastic_hessian_traj_strain_frac': traj_hess_idx / (n_strain_steps - 1),
        'elastic_hessian_traj_strain': np.outer(
            compression_strains, traj_hess_idx / (n_strain_steps - 1)),  # (n_sub, n_hess_steps)
        'trajectory_positions': trajectory_positions,   # (n_t, n_sub, n_strain_steps, N, 2) — full trajectory
        'trajectory_strain_frac': np.arange(n_strain_steps) / (n_strain_steps - 1),   # (n_strain_steps,)
        'cost_hessian_before_eigvals': np.stack(cost_before_vals),
        'cost_hessian_before_eigvecs': np.stack(cost_before_vecs),
        'cost_hessian_after_eigvals': np.stack(cost_after_vals),
        'cost_hessian_after_eigvecs': np.stack(cost_after_vecs),
        'compression_strains': np.asarray(compression_strains, dtype=float),
        'target_poisson_ratios': np.asarray(target_poisson_ratios, dtype=float),
    }


# ---------------------------------------------------------------------------
# Allosteric ("local") sweep — LAMMPS-based, matches allosteric_trainer.py
# ---------------------------------------------------------------------------

def _select_local_threshold_indices(steps, mse1, mse2, n_thresh_steps, eps_min):
    """Checkpoint-aligned timestep selection for allosteric results.

    `steps` (1-indexed global step numbers) indexes into the sparse
    `stiffnesses_traj.npy` checkpoints; the full-resolution loss (mse1+mse2)
    is aligned to those checkpoints via `steps - 1`, exactly as
    NewFiguresJune.ipynb's load_local_stiffness_trajectory does.
    """
    loss_full = np.asarray(mse1) + np.asarray(mse2)
    idx = np.clip(np.asarray(steps, dtype=int) - 1, 0, len(loss_full) - 1)
    loss_ckpt = loss_full[idx]
    t_indices = select_loss_threshold_steps(loss_ckpt, n_thresh_steps, eps_min)
    return t_indices, loss_ckpt


def sweep_allosteric(nodes, incidence_matrix, eq_lengths, task_config,
                     stiffness_traj, steps, mse1, mse2,
                     n_thresh_steps=50, eps_min=1e-8, k_eigs=10,
                     n_hessian_traj_steps=20, use_clamped_trajectory=True,
                     cost_hessian_fn=None, verbose=True):
    """
    Run the full timestep-sweep analysis for one allosteric training result.

    Parameters
    ----------
    nodes, incidence_matrix, eq_lengths : network geometry (rest state).
    task_config : dict with 'tod', 'tod2', 'dinputdistance', 'dinputdistance2',
                  'nsteps', 'nsteps2' (see training.runners.allosteric_trainer).
    stiffness_traj : (T_ckpt, E) array   checkpointed stiffnesses (every 500 steps).
    steps : (T_ckpt,) array              1-indexed global step number per checkpoint.
    mse1, mse2 : (T_full,) arrays        full-resolution per-step stored losses.
    n_hessian_traj_steps : int      number of linearly-spaced points along each
                                     task's actuation trajectory (input node pulled
                                     from 0 -> full displacement) at which the
                                     elastic Hessian spectrum is evaluated. task1
                                     and task2 have different `nsteps`, so this is
                                     clamped to min(n_hessian_traj_steps, nsteps,
                                     nsteps2) to keep both tasks' arrays the same
                                     length.
    use_clamped_trajectory : bool   If True (default, matches the original
                                     behavior of this function), the elastic
                                     Hessian and saved trajectory are sampled
                                     along the *clamped* (output-spring) run.
                                     If False, the clamped run is skipped
                                     entirely (evaluate_actuation is called
                                     with compute_clamped=False) and the
                                     *free* run's trajectory is used instead —
                                     appropriate for trainers (e.g. a
                                     gradient-descent trainer) whose loss only
                                     ever depends on the free run, since mse
                                     never depends on the clamped run either way.
    cost_hessian_fn : callable or None
                                     Function computing the cost-Hessian eigenpairs
                                     at one checkpoint: cost_hessian_fn(nodes,
                                     incidence_matrix, task_config, t, stiffness_traj,
                                     k_eigs=..., verbose=...) -> (eigenvalues, eigenvectors).
                                     Defaults to _compute_cost_hessian_jax: exact
                                     autodiff (jax.grad for the base gradient,
                                     forward-over-reverse jax.jvp(jax.grad(...)) for
                                     each Lanczos HVP) through training.jax_actuation's
                                     differentiable JAX-FIRE ramp (training.jax_actuation.
                                     FREE_CRF) — one physics-ramp-equivalent pass per
                                     gradient/HVP instead of _compute_cost_hessian_lammps's
                                     O(n_edges) finite-difference physics calls per
                                     gradient, repeated every Lanczos iteration.
                                     Deliberately hardcoded to jax_fire regardless of
                                     DEFAULT_SOLVER / whichever solver the training run
                                     actually used (only jax_fire's custom_vjp is
                                     differentiable — LAMMPS isn't) — this is the one
                                     piece of the sweep that intentionally does NOT
                                     mirror training's own evaluation path; the elastic
                                     Hessian and stiffness/trajectory loss recompute
                                     below still go through evaluate_actuation and so
                                     still respect DEFAULT_SOLVER. Pass
                                     cost_hessian_fn=_compute_cost_hessian_lammps
                                     explicitly for a solver-matched (or LAMMPS-exact)
                                     finite-difference cross-check instead.
    k_eigs : int                    number of top (largest positive / algebraic)
                                     cost-Hessian eigenpairs to compute. Does NOT
                                     affect the elastic Hessian, which always
                                     returns its full spectrum (all modes) — the
                                     elastic Hessian is dense-solved via
                                     np.linalg.eigh regardless of truncation, so
                                     truncating it saves no compute, only storage.

    Returns
    -------
    dict of numpy arrays, ready for np.savez_compressed. Includes the full
    per-task, per-selected-step actuation trajectory (`trajectory_positions_task1`/
    `_task2`, kept separate since nsteps can differ between the two tasks) so
    downstream analyses that need displacement between arbitrary frames (e.g.
    participation ratio) can index into it directly instead of recomputing it.
    """
    from training.runners.allosteric_trainer import evaluate_actuation

    tod, tod2 = task_config['tod'], task_config['tod2']
    dinputdistance, dinputdistance2 = task_config['dinputdistance'], task_config['dinputdistance2']
    nsteps, nsteps2 = task_config['nsteps'], task_config['nsteps2']
    dx  = dinputdistance  / nsteps
    dx2 = dinputdistance2 / nsteps2
    edges = _incidence_to_edges(incidence_matrix)

    t_indices, loss_ckpt = _select_local_threshold_indices(
        steps, mse1, mse2, n_thresh_steps, eps_min)
    n_t = len(t_indices)
    stored_loss = loss_ckpt[t_indices]

    # Frame indices along each task's own actuation trajectory (0 -> nsteps-1)
    # at which the elastic Hessian is evaluated. nsteps differs between task1
    # and task2, so indices are computed per-task, but the *count* is shared
    # (clamped to the smaller of the two) so both tasks' arrays stack cleanly.
    n_hess = min(n_hessian_traj_steps, nsteps, nsteps2)
    traj_hess_idx_1 = np.round(np.linspace(0, nsteps  - 1, n_hess)).astype(int)
    traj_hess_idx_2 = np.round(np.linspace(0, nsteps2 - 1, n_hess)).astype(int)
    traj_hess_idx_per_task = [traj_hess_idx_1, traj_hess_idx_2]

    recomputed_stiffness_loss  = np.full((n_t, 2), np.nan)   # [:, 0]=task1, [:, 1]=task2
    recomputed_trajectory_loss = np.full((n_t, 2), np.nan)
    elastic_eigvals_list = []   # per t: (2, n_hess, n_free_dofs) — full spectrum
    elastic_eigvecs_list = []   # per t: (2, n_hess, n_free_dofs, n_free_dofs)
    traj_pos_task1_list  = []   # per t: (nsteps, N, 2) — full task1 trajectory
    traj_pos_task2_list  = []   # per t: (nsteps2, N, 2) — full task2 trajectory

    for i, t in enumerate(t_indices):
        k_t = np.asarray(stiffness_traj[t], dtype=float)
        if verbose:
            print(f"  [allosteric sweep] step {i+1}/{n_t} (checkpoint idx={t}) ...", flush=True)

        eigvals_this_t = []
        eigvecs_this_t = []
        traj_pos_this_t = []
        for task_idx, (tgt, dxi, nsi) in enumerate(
                ((tod, dx, nsteps), (tod2, dx2, nsteps2))):
            mse, nodes_free, nodes_clamped, frames = evaluate_actuation(
                nodes, incidence_matrix, k_t, tgt, dxi, nsi, return_trajectory=True,
                compute_clamped=use_clamped_trajectory)
            recomputed_stiffness_loss[i, task_idx]  = mse
            # "loss from trajectory" here is the same quantity (mse is already
            # evaluated at the recomputed equilibrium, i.e. end of the LAMMPS
            # quasistatic trajectory) — kept as a separate series for parity
            # with the auxetic pipeline's independent stiffness-vs-trajectory check.
            recomputed_trajectory_loss[i, task_idx] = mse

            vals_along_traj = []
            vecs_along_traj = []
            for j in traj_hess_idx_per_task[task_idx]:
                # n_modes=None -> full spectrum; the dense eigh solve below is
                # unconditional, so truncating here would save no compute.
                vals, vecs, _ = compute_hessian_spectrum(
                    frames[j], edges, k_t, eq_lengths,
                    constrained_nodes=LOCAL_CONSTRAINED_NODES, n_modes=None)
                vals_along_traj.append(vals)
                vecs_along_traj.append(vecs)
            eigvals_this_t.append(np.stack(vals_along_traj))   # (n_hess, n_free_dofs)
            eigvecs_this_t.append(np.stack(vecs_along_traj))   # (n_hess, n_free_dofs, n_free_dofs)
            # Full-resolution positions (not just at traj_hess_idx_per_task) so
            # downstream analyses (e.g. participation ratio) can index into the
            # trajectory themselves without recomputing it.
            traj_pos_this_t.append(np.stack([np.asarray(p) for p in frames]))

        elastic_eigvals_list.append(np.stack(eigvals_this_t))
        elastic_eigvecs_list.append(np.stack(eigvecs_this_t))
        traj_pos_task1_list.append(traj_pos_this_t[0])   # (nsteps, N, 2)
        traj_pos_task2_list.append(traj_pos_this_t[1])   # (nsteps2, N, 2)

    elastic_hessian_eigvals = np.stack(elastic_eigvals_list)   # (n_t, 2, n_hess, n_free_dofs) — full spectrum
    elastic_hessian_eigvecs = np.stack(elastic_eigvecs_list)   # (n_t, 2, n_hess, n_free_dofs, n_free_dofs)
    trajectory_positions_task1 = np.stack(traj_pos_task1_list)   # (n_t, nsteps, N, 2)
    trajectory_positions_task2 = np.stack(traj_pos_task2_list)   # (n_t, nsteps2, N, 2)

    # Per-task frame indices / actuation progress at the sampled Hessian points.
    # Frame index j (0-indexed) corresponds to cumulative input-node
    # displacement dx*(j+1) (see training.lammps_utils.strain_network).
    elastic_hessian_traj_indices = np.stack(traj_hess_idx_per_task)   # (2, n_hess)
    elastic_hessian_traj_strain_frac = np.stack([
        (traj_hess_idx_1 + 1) / nsteps,
        (traj_hess_idx_2 + 1) / nsteps2,
    ])   # (2, n_hess)
    elastic_hessian_traj_strain = np.stack([
        dx  * (traj_hess_idx_1 + 1),
        dx2 * (traj_hess_idx_2 + 1),
    ])   # (2, n_hess) — actual cumulative input-node displacement

    if cost_hessian_fn is None:
        cost_hessian_fn = _compute_cost_hessian_jax

    cost_before = cost_hessian_fn(
        nodes, incidence_matrix, task_config, t_indices[0], stiffness_traj,
        k_eigs=k_eigs, verbose=verbose)
    cost_after = cost_hessian_fn(
        nodes, incidence_matrix, task_config, t_indices[-1], stiffness_traj,
        k_eigs=k_eigs, verbose=verbose)

    return {
        't_indices': t_indices,
        'checkpoint_steps': np.asarray(steps)[t_indices],
        'stored_loss': stored_loss,
        'recomputed_stiffness_loss': recomputed_stiffness_loss,
        'recomputed_trajectory_loss': recomputed_trajectory_loss,
        'elastic_hessian_eigvals': elastic_hessian_eigvals,
        'elastic_hessian_eigvecs': elastic_hessian_eigvecs,
        'elastic_hessian_traj_indices': elastic_hessian_traj_indices,
        'elastic_hessian_traj_strain_frac': elastic_hessian_traj_strain_frac,
        'elastic_hessian_traj_strain': elastic_hessian_traj_strain,
        'trajectory_positions_task1': trajectory_positions_task1,   # (n_t, nsteps, N, 2)
        'trajectory_positions_task2': trajectory_positions_task2,   # (n_t, nsteps2, N, 2)
        'trajectory_strain_frac_task1': (np.arange(nsteps)  + 1) / nsteps,
        'trajectory_strain_frac_task2': (np.arange(nsteps2) + 1) / nsteps2,
        'cost_hessian_before_eigvals': cost_before[0],
        'cost_hessian_before_eigvecs': cost_before[1],
        'cost_hessian_after_eigvals': cost_after[0],
        'cost_hessian_after_eigvecs': cost_after[1],
    }


def _incidence_to_edges(incidence_matrix):
    from training.lammps_utils import incidence_to_edges
    return incidence_to_edges(incidence_matrix)


def _compute_cost_hessian_lammps(nodes, incidence_matrix, task_config, t, stiffness_traj,
                                 k_eigs=10, hvp_epsilon=1e-3, grad_epsilon=1e-4,
                                 verbose=True):
    """
    Top-k eigenpairs of the combined (task1+task2) LAMMPS loss Hessian w.r.t.
    stiffnesses, via finite-difference gradient + finite-difference HVP + Lanczos.

    Mirrors analysis/cost_utils.compute_cost_hessian's structure, but the loss
    function calls LAMMPS (training.runners.allosteric_trainer.evaluate_actuation)
    instead of JAX, since the allosteric physics engine isn't autodiff-differentiable.
    This is expensive: each HVP costs ~4 LAMMPS actuation-pairs (2 for the forward
    gradient, x2 tasks); keep k_eigs modest.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh
    from training.runners.allosteric_trainer import evaluate_actuation

    tod, tod2 = task_config['tod'], task_config['tod2']
    dinputdistance, dinputdistance2 = task_config['dinputdistance'], task_config['dinputdistance2']
    nsteps, nsteps2 = task_config['nsteps'], task_config['nsteps2']
    dx, dx2 = dinputdistance / nsteps, dinputdistance2 / nsteps2

    base_k = np.asarray(stiffness_traj[t], dtype=float)
    n_edges = len(base_k)

    def loss_fn(k):
        # compute_clamped=False: mse never depends on the clamped run (see
        # evaluate_actuation's docstring), so skipping it here is a pure
        # speedup — every eigsh iteration costs 2*n_edges loss_fn calls.
        mse1, _, _  = evaluate_actuation(nodes, incidence_matrix, k, tod,  dx,  nsteps,
                                         compute_clamped=False)
        mse2, _, _  = evaluate_actuation(nodes, incidence_matrix, k, tod2, dx2, nsteps2,
                                         compute_clamped=False)
        return 0.5 * (mse1 + mse2)

    def grad_fn(k):
        g = np.zeros(n_edges)
        for e in range(n_edges):
            k_plus, k_minus = k.copy(), k.copy()
            k_plus[e]  += grad_epsilon
            k_minus[e] -= grad_epsilon
            g[e] = (loss_fn(k_plus) - loss_fn(k_minus)) / (2 * grad_epsilon)
        return g

    if verbose:
        print(f"  [allosteric sweep] cost Hessian: base gradient ({n_edges} edges) ...", flush=True)
    t0 = time.time()
    g0 = grad_fn(base_k)
    if verbose:
        print(f"  [allosteric sweep] base gradient done in {time.time()-t0:.1f}s", flush=True)

    def hvp(v):
        v = np.asarray(v, dtype=float)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-14:
            return np.zeros_like(v)
        g_fwd = grad_fn(base_k + hvp_epsilon * v / norm_v)
        return norm_v * (g_fwd - g0) / hvp_epsilon

    k = min(k_eigs, n_edges - 1)
    H_op = LinearOperator((n_edges, n_edges), matvec=hvp, dtype=float)
    if verbose:
        print(f"  [allosteric sweep] cost Hessian: eigsh (k={k}) ...", flush=True)
    # 'LA' = largest algebraic, i.e. the k most positive eigenvalues.
    eigenvalues, eigenvectors = eigsh(H_op, k=k, which='LA')
    order = np.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]


def _compute_cost_hessian_jax(nodes, incidence_matrix, task_config, t, stiffness_traj,
                              k_eigs=10, verbose=True):
    """
    Top-k eigenpairs of the combined (task1+task2) loss Hessian w.r.t.
    stiffnesses, via exact JAX autodiff instead of _compute_cost_hessian_lammps's
    finite differences.

    Builds the mse1+mse2 loss as a single JAX-traced scalar using
    training.jax_actuation.strain_network_jax_final_traced (which keeps the
    whole FIRE ramp a jnp value, unlike evaluate_actuation, whose numpy
    conversion breaks the autodiff graph and is why the LAMMPS-recipe version
    needs finite differences at all). The base gradient is one jax.grad call;
    each Lanczos HVP is one forward-over-reverse jax.jvp(jax.grad(...), (k,),
    (v,)) call — one physics-ramp-equivalent pass each, vs. _compute_cost_
    hessian_lammps's O(n_edges) physics calls per gradient (repeated every
    Lanczos iteration).

    Always uses training.jax_actuation.FREE_CRF (jax_fire), regardless of
    DEFAULT_SOLVER / whatever solver the training run actually used — see
    sweep_allosteric's cost_hessian_fn docstring for why (only jax_fire is
    differentiable; this is the one part of the sweep that intentionally
    doesn't mirror training's own evaluation path).

    Validated against finite differences on the production geometry_targeted/
    task_0/realization_0 network (n_edges=253): jax.grad matches central
    differences on sampled edges to ~1e-3-1e-6 relative error; the jax.jvp HVP
    matches an independent finite-difference-of-the-exact-gradient reference
    to ~2e-7 relative error (cosine similarity 1.000000). A vmap-batched
    finite-difference alternative (computing all edges' perturbations in one
    batched call instead of this exact-gradient approach) was also tried and
    discarded — vmap-ing the ~150-step FIRE ramp (each step containing an
    internal jax.lax.fori_loop(max_steps=1_000_000) with jax.lax.cond
    branches) blows up XLA compile time; a 16-edge batch didn't finish
    compiling after 40+ minutes.
    """
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from scipy.sparse.linalg import LinearOperator, eigsh
    import training.jax_actuation as jx_act

    tod, tod2 = task_config['tod'], task_config['tod2']
    dinputdistance, dinputdistance2 = task_config['dinputdistance'], task_config['dinputdistance2']
    nsteps, nsteps2 = task_config['nsteps'], task_config['nsteps2']
    dx, dx2 = dinputdistance / nsteps, dinputdistance2 / nsteps2

    base_k = np.asarray(stiffness_traj[t], dtype=float)
    n_edges = len(base_k)
    edges = _incidence_to_edges(incidence_matrix)
    rest_lengths = np.linalg.norm(nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1)

    def scalar_loss(k_jax):
        pos1 = jx_act.strain_network_jax_final_traced(
            jx_act.FREE_CRF, nodes, edges, rest_lengths, k_jax, 0, 1, dx=dx, nsteps=nsteps
        ).reshape(-1, 2)
        pos2 = jx_act.strain_network_jax_final_traced(
            jx_act.FREE_CRF, nodes, edges, rest_lengths, k_jax, 0, 1, dx=dx2, nsteps=nsteps2
        ).reshape(-1, 2)
        mse1 = (jnp.linalg.norm(pos1[2] - pos1[3]) - tod) ** 2
        mse2 = (jnp.linalg.norm(pos2[2] - pos2[3]) - tod2) ** 2
        return 0.5 * (mse1 + mse2)

    @jax.jit
    def hvp_jax(k_jax, v_jax):
        _, Hv = jax.jvp(jax.grad(scalar_loss), (k_jax,), (v_jax,))
        return Hv

    k0_jax = jnp.asarray(base_k, dtype=jnp.float64)

    def hvp(v):
        return np.asarray(hvp_jax(k0_jax, jnp.asarray(v, dtype=jnp.float64)))

    k = min(k_eigs, n_edges - 1)
    H_op = LinearOperator((n_edges, n_edges), matvec=hvp, dtype=float)
    if verbose:
        print(f"  [allosteric sweep] cost Hessian (JAX autodiff, jax_fire): "
              f"eigsh (k={k}) ...", flush=True)
    eigenvalues, eigenvectors = eigsh(H_op, k=k, which='LA')
    order = np.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]


# ---------------------------------------------------------------------------
# Save/load
# ---------------------------------------------------------------------------

def save_sweep_results(path, **arrays):
    np.savez_compressed(path, **arrays)


def load_sweep_results(path):
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}
