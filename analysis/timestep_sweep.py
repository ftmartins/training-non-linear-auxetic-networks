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

def sweep_auxetic(network, task_config, boundary, stiffness_traj, loss_traj,
                  n_thresh_steps=50, eps_min=1e-8, n_traj_steps=100, k_eigs=10,
                  n_hessian_traj_steps=20,
                  force_type='quadratic', n_strain_steps=100, tol=1e-10,
                  verbose=True):
    """
    Run the full timestep-sweep analysis for one auxetic training result.

    Parameters
    ----------
    network : ElasticNetwork        topology (edges, rest_lengths); stiffnesses
                                     are overwritten per selected step.
    task_config : dict              must have 'compression_strains' and
                                     'target_poisson_ratios' (equal-length lists).
    boundary : dict                 keys 'top', 'bottom', 'left', 'right'.
    stiffness_traj : (T, E) array   full per-step stiffness trajectory.
    loss_traj : (T,) array          full per-step stored loss (same array used
                                     to select timesteps).
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
    dict of numpy arrays, ready for np.savez_compressed.
    """
    from training.src.training_functions import poisson_loss_batch_parallel

    compression_strains   = list(task_config['compression_strains'])
    target_poisson_ratios = list(task_config['target_poisson_ratios'])
    top, bottom, left, right = boundary['top'], boundary['bottom'], boundary['left'], boundary['right']
    constrained_nodes = global_constrained_nodes(boundary)
    n_sub = len(compression_strains)

    t_indices = select_loss_threshold_steps(loss_traj, n_thresh_steps, eps_min)
    n_t = len(t_indices)
    stored_loss = np.asarray(loss_traj)[t_indices]

    # Linearly-spaced trajectory indices (0 -> n_traj_steps-1) at which the
    # elastic Hessian is evaluated; same for every selected timestep/subtask
    # since n_traj_steps is fixed for the whole sweep.
    traj_hess_idx = np.unique(np.round(
        np.linspace(0, n_traj_steps - 1, n_hessian_traj_steps)
    ).astype(int))

    recomputed_stiffness_loss  = np.full(n_t, np.nan)
    recomputed_trajectory_loss = np.full(n_t, np.nan)
    subtask_trajectory_loss    = np.full((n_t, n_sub), np.nan)

    elastic_eigvals_list = []   # (n_sub,) list of (n_hess_steps, n_free_dofs) arrays, per t — full spectrum
    elastic_eigvecs_list = []   # (n_sub,) list of (n_hess_steps, n_free_dofs, n_free_dofs) arrays, per t

    net = copy.deepcopy(network)

    for i, t in enumerate(t_indices):
        k_t = np.asarray(stiffness_traj[t], dtype=float)
        net.stiffnesses = k_t

        if verbose:
            print(f"  [auxetic sweep] step {i+1}/{n_t} (t={t}) ...", flush=True)

        # (a) stiffness-based loss — the exact training-time evaluation path
        mse, _ = poisson_loss_batch_parallel(
            net, target_poisson_ratios, top, bottom, left, right,
            compression_strains, n_strain_steps=n_strain_steps,
            force_type=force_type, tol=tol,
        )
        recomputed_stiffness_loss[i] = float(mse)

        # (b) trajectory-based loss + (c) elastic Hessian along the trajectory, per subtask
        eigvals_this_t = []
        eigvecs_this_t = []
        sub_losses = np.full(n_sub, np.nan)
        for s, (cs, tp) in enumerate(zip(compression_strains, target_poisson_ratios)):
            traj = compute_trajectory(k_t, net, boundary, cs, n_steps=n_traj_steps,
                                      force_type=force_type, tol=tol)
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

        subtask_trajectory_loss[i]    = sub_losses
        recomputed_trajectory_loss[i] = float(np.mean(sub_losses))
        elastic_eigvals_list.append(np.stack(eigvals_this_t))
        elastic_eigvecs_list.append(np.stack(eigvecs_this_t))

    elastic_hessian_eigvals = np.stack(elastic_eigvals_list)   # (n_t, n_sub, n_hess_steps, n_free_dofs) — full spectrum
    elastic_hessian_eigvecs = np.stack(elastic_eigvecs_list)   # (n_t, n_sub, n_hess_steps, n_free_dofs, n_free_dofs)

    # Cost Hessian only before (first selected step) and after (last selected step)
    cost_before_vals, cost_before_vecs = [], []
    cost_after_vals,  cost_after_vecs  = [], []
    for t, vals_out, vecs_out, tag in (
        (t_indices[0],  cost_before_vals, cost_before_vecs, 'before'),
        (t_indices[-1], cost_after_vals,  cost_after_vecs,  'after'),
    ):
        k_t = np.asarray(stiffness_traj[t], dtype=float)
        net.stiffnesses = k_t
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
        'elastic_hessian_traj_strain_frac': traj_hess_idx / (n_traj_steps - 1),
        'elastic_hessian_traj_strain': np.outer(
            compression_strains, traj_hess_idx / (n_traj_steps - 1)),  # (n_sub, n_hess_steps)
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
                     n_hessian_traj_steps=20, verbose=True):
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
    k_eigs : int                    number of top (largest positive / algebraic)
                                     cost-Hessian eigenpairs to compute. Does NOT
                                     affect the elastic Hessian, which always
                                     returns its full spectrum (all modes) — the
                                     elastic Hessian is dense-solved via
                                     np.linalg.eigh regardless of truncation, so
                                     truncating it saves no compute, only storage.

    Returns
    -------
    dict of numpy arrays, ready for np.savez_compressed.
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

    for i, t in enumerate(t_indices):
        k_t = np.asarray(stiffness_traj[t], dtype=float)
        if verbose:
            print(f"  [allosteric sweep] step {i+1}/{n_t} (checkpoint idx={t}) ...", flush=True)

        eigvals_this_t = []
        eigvecs_this_t = []
        for task_idx, (tgt, dxi, nsi) in enumerate(
                ((tod, dx, nsteps), (tod2, dx2, nsteps2))):
            mse, nodes_free, nodes_clamped, frames_clamped = evaluate_actuation(
                nodes, incidence_matrix, k_t, tgt, dxi, nsi, return_trajectory=True)
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
                    frames_clamped[j], edges, k_t, eq_lengths,
                    constrained_nodes=LOCAL_CONSTRAINED_NODES, n_modes=None)
                vals_along_traj.append(vals)
                vecs_along_traj.append(vecs)
            eigvals_this_t.append(np.stack(vals_along_traj))   # (n_hess, n_free_dofs)
            eigvecs_this_t.append(np.stack(vecs_along_traj))   # (n_hess, n_free_dofs, n_free_dofs)

        elastic_eigvals_list.append(np.stack(eigvals_this_t))
        elastic_eigvecs_list.append(np.stack(eigvecs_this_t))

    elastic_hessian_eigvals = np.stack(elastic_eigvals_list)   # (n_t, 2, n_hess, n_free_dofs) — full spectrum
    elastic_hessian_eigvecs = np.stack(elastic_eigvecs_list)   # (n_t, 2, n_hess, n_free_dofs, n_free_dofs)

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

    cost_before = _compute_cost_hessian_lammps(
        nodes, incidence_matrix, task_config, t_indices[0], stiffness_traj,
        k_eigs=k_eigs, verbose=verbose)
    cost_after = _compute_cost_hessian_lammps(
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
        mse1, _, _  = evaluate_actuation(nodes, incidence_matrix, k, tod,  dx,  nsteps)
        mse2, _, _  = evaluate_actuation(nodes, incidence_matrix, k, tod2, dx2, nsteps2)
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


# ---------------------------------------------------------------------------
# Save/load
# ---------------------------------------------------------------------------

def save_sweep_results(path, **arrays):
    np.savez_compressed(path, **arrays)


def load_sweep_results(path):
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}
