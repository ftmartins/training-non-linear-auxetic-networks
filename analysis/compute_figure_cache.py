#!/usr/bin/env python3
"""
compute_figure_cache.py — populate .figure_notebook_cache/ for AllFigures.ipynb

Modes
-----
  global_nt       --task T --real R
  local_nt        --geom G --task T --real R
  global_rep      [--task T --real R]     (default: task 13, real 0)
  local_rep       [--task T --real R]     (default: task  0, real 0)
  modesens_local  --geom G --task T --real R
  modesens_global --task T --real R

All parameters mirror AllFigures.ipynb exactly so cache files are compatible.
"""

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.stats import spearmanr

# ── Parameters (must match AllFigures.ipynb Cell 2) ─────────────────────────
BOUNDARY_MARGIN   = 0.02
FORCE_TYPE        = 'quadratic'
N_STRAIN_STEPS    = 100
N_EQUIL           = 1
N_COST_EIGS       = 5
HVP_EPSILON       = 1e-4
LOCAL_MAX_STRAIN  = 1.0
N_TRAJ_STEPS_BULK = 200   # matches current training standard (config.N_STRAIN_STEPS)
MID_FRAME_BULK    = 100  # halfway through the trajectory

DEFAULT_GLOBAL_TASK = 13
DEFAULT_GLOBAL_REAL = 0
DEFAULT_LOCAL_TASK  = 0
DEFAULT_LOCAL_REAL  = 0

# ── Cluster data paths ───────────────────────────────────────────────────────
# Root directory that contains task_XX/ subdirectories for the non-targeted ensemble.
ENSEMBLE_TRAINING_ROOT = Path('/data2/shared/felipetm/auxetic_networks/ensemble_training_new_sqr')
# Subdirectory within ENSEMBLE_TRAINING_ROOT where task_XX/ folders live.
# Set to '' if task_XX/ folders are directly inside ENSEMBLE_TRAINING_ROOT.
ENSEMBLE_RESULTS_SUBDIR = ''

TARGETED_ROOT   = Path('/data2/shared/felipetm/auxetic_networks/targeted_results_sqr')
ALLOSTERIC_ROOT = Path('/data2/shared/felipetm/allosteric_nets')

# ── Lazy globals set after imports ───────────────────────────────────────────
_crf_diff = None   # differentiable FIRE solver, built once


def _setup_imports(repo_root: Path):
    """Add src/ to sys.path and import JAX + project modules."""
    import importlib
    src = str(repo_root / 'src')
    if src not in sys.path:
        sys.path.insert(0, src)

    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)

    from elastic_network import ElasticNetwork
    from network_utils import get_square_boundary_nodes
    from generalized_susceptibility import (
        compute_physical_hessian_strained,
        compute_constrained_hessian_inverse,
        compute_full_jacobian_matrixwise,
        susceptibilities_from_jacobian,
    )
    from training_functions_with_toggle import (
        crf as _crf,
        compute_quasistatic_trajectory_auxetic,
        compute_poisson_ratio_single_jax,
        make_compute_response_fire,
    )

    return dict(
        jax=jax, jnp=jnp,
        ElasticNetwork=ElasticNetwork,
        get_square_boundary_nodes=get_square_boundary_nodes,
        compute_physical_hessian_strained=compute_physical_hessian_strained,
        compute_constrained_hessian_inverse=compute_constrained_hessian_inverse,
        compute_full_jacobian_matrixwise=compute_full_jacobian_matrixwise,
        susceptibilities_from_jacobian=susceptibilities_from_jacobian,
        _crf=_crf,
        compute_quasistatic_trajectory_auxetic=compute_quasistatic_trajectory_auxetic,
        compute_poisson_ratio_single_jax=compute_poisson_ratio_single_jax,
        make_compute_response_fire=make_compute_response_fire,
    )


# ── Paths ─────────────────────────────────────────────────────────────────────
def build_paths(repo_root=None, cache_dir=None):
    """Build all data and cache paths.

    By default uses the cluster globals above (ENSEMBLE_TRAINING_ROOT, etc.).
    Pass repo_root to fall back to repo-local data/ layout (local development).

    Cache directories are co-located with their data by default:
      cache_results        → <results_dir>/.figure_notebook_cache/
      cache_targeted       → <targeted_dir>/.figure_notebook_cache/
      cache_allosteric     → <allosteric_dir>/.figure_notebook_cache/
      cache_allosteric_tgt → <allosteric_dir>/geometry_targeted/.figure_notebook_cache/

    Pass cache_dir to override all four with a single directory.
    """
    if repo_root is not None:
        r = Path(repo_root)
        results_dir    = r / 'data' / 'results'
        targeted_dir   = r / 'data' / 'targeted_results'
        allosteric_dir = r / 'data' / 'allosteric_nets'
    else:
        results_dir = (ENSEMBLE_TRAINING_ROOT / ENSEMBLE_RESULTS_SUBDIR
                       if ENSEMBLE_RESULTS_SUBDIR else ENSEMBLE_TRAINING_ROOT)
        targeted_dir   = TARGETED_ROOT
        allosteric_dir = ALLOSTERIC_ROOT

    def _cache(data_dir):
        return Path(cache_dir) if cache_dir else Path(data_dir) / '.figure_notebook_cache'

    return {
        'results':             results_dir,
        'targeted':            targeted_dir,
        'allosteric':          allosteric_dir,
        'allosteric_tgt':      allosteric_dir / 'geometry_targeted',
        'cache_results':       _cache(results_dir),
        'cache_targeted':      _cache(targeted_dir),
        'cache_allosteric':    _cache(allosteric_dir),
        'cache_allosteric_tgt': _cache(allosteric_dir / 'geometry_targeted'),
    }


# ── Cache helpers ─────────────────────────────────────────────────────────────
def cache_exists(cache_dir, key):
    return (Path(cache_dir) / f'{key}.npz').exists()


def save_cache(cache_dir, key, **arrays):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez(Path(cache_dir) / f'{key}.npz', **arrays)


# ── Geometry helpers ──────────────────────────────────────────────────────────
def incmat_to_edges(incmat):
    edges = np.zeros((len(incmat), 2), dtype=int)
    for e, row in enumerate(incmat):
        cols = np.where(np.abs(row) > 0.5)[0]
        edges[e, 0] = int(cols[0])
        edges[e, 1] = int(cols[1])
    return edges


def free_dofs_local(N):
    return np.arange(4, 2 * N)


def free_dofs_global(N, top_nodes, bottom_nodes):
    constrained = np.unique(np.concatenate([
        np.asarray(top_nodes, dtype=int),
        np.asarray(bottom_nodes, dtype=int),
    ]))
    free_nodes = np.setdiff1d(np.arange(N), constrained)
    return np.sort(np.concatenate([2 * free_nodes, 2 * free_nodes + 1]))


def build_incidence_matrix(edges, n_nodes):
    E = len(edges)
    B = np.zeros((E, n_nodes), dtype=float)
    for e, (i, j) in enumerate(edges):
        B[e, i] = +1.0
        B[e, j] = -1.0
    return B


def bond_strain_stress(positions, edges, k, L0):
    d      = positions[edges[:, 0]] - positions[edges[:, 1]]
    ell    = np.linalg.norm(d, axis=1)
    strain = (ell - L0) / L0
    return strain, k * strain


# ── Data loading ──────────────────────────────────────────────────────────────
def _sibling_realization_dirs(task_parent, own_dir):
    if not task_parent.is_dir():
        return
    yield own_dir
    for rd in sorted(task_parent.iterdir()):
        if rd.is_dir() and rd != own_dir:
            yield rd


def _geometry_realization_dirs(geom_dir, own_task_parent, own_dir):
    yield own_dir
    for rd in sorted(own_task_parent.iterdir()):
        if rd.is_dir() and rd != own_dir:
            yield rd
    for td in sorted(geom_dir.iterdir()):
        if td.is_dir() and td != own_task_parent:
            for rd in sorted(td.iterdir()):
                if rd.is_dir():
                    yield rd


def load_global_network(task_seed, real_seed, data_dir, mods):
    ElasticNetwork = mods['ElasticNetwork']
    get_square_boundary_nodes = mods['get_square_boundary_nodes']

    task_parent = Path(data_dir) / f'task_{task_seed:02d}'
    own         = task_parent / f'realization_{real_seed:02d}'

    with open(own / 'final_network.pkl', 'rb') as fh:
        net = pickle.load(fh)
    stiffnesses = np.array(net['stiffnesses'], dtype=float)

    positions    = np.array(net['positions'],    dtype=float) if net.get('positions')    is not None else None
    edges        = np.array(net['edges'],        dtype=int)   if net.get('edges')        is not None else None
    rest_lengths = np.array(net['rest_lengths'], dtype=float) if net.get('rest_lengths') is not None else None

    if positions is None or edges is None:
        for rd in _sibling_realization_dirs(task_parent, own):
            if rd == own:
                continue
            try:
                with open(rd / 'final_network.pkl', 'rb') as fh:
                    alt = pickle.load(fh)
                if positions    is None and alt.get('positions')    is not None:
                    positions    = np.array(alt['positions'],    dtype=float)
                if edges        is None and alt.get('edges')        is not None:
                    edges        = np.array(alt['edges'],        dtype=int)
                if rest_lengths is None and alt.get('rest_lengths') is not None:
                    rest_lengths = np.array(alt['rest_lengths'], dtype=float)
                if positions is not None and edges is not None:
                    break
            except (FileNotFoundError, KeyError):
                continue

    if positions is None:
        raise RuntimeError(f'No positions for global task {task_seed}')
    if edges is None:
        raise RuntimeError(f'No edges for global task {task_seed}')
    if rest_lengths is None:
        d = positions[edges[:, 0]] - positions[edges[:, 1]]
        rest_lengths = np.linalg.norm(d, axis=1)

    top, bottom, left, right = get_square_boundary_nodes(positions, BOUNDARY_MARGIN)
    boundary = dict(top=top, bottom=bottom, left=left, right=right)

    config = None
    for rd in _sibling_realization_dirs(task_parent, own):
        cfg = rd / 'task_config.json'
        if cfg.exists():
            with open(cfg) as fh:
                config = json.load(fh)
            if config:
                break
    if config is None:
        raise RuntimeError(f'No task_config.json for global task {task_seed}')

    network = ElasticNetwork(positions, edges, rest_lengths, stiffnesses)
    return network, boundary, config


def load_local_network(geometry_id, task_id, real_id, data_dir, mods):
    if geometry_id is None:
        task_parent = Path(data_dir) / f'task_{task_id}'
        geom_dir    = None
    else:
        task_parent = Path(data_dir) / f'geometry_{geometry_id}' / f'task_{task_id}'
        geom_dir    = task_parent.parent
    task_dir = task_parent / f'realization_{real_id}'

    if not (task_dir / 'stiffnesses.npy').exists():
        return None
    try:
        stiffnesses = np.abs(np.load(task_dir / 'stiffnesses.npy').astype(float))
    except Exception:
        return None

    if not (task_dir / 'eq_lengths.npy').exists():
        return None
    try:
        L0 = np.load(task_dir / 'eq_lengths.npy').astype(float)
    except Exception:
        return None

    nodes = incmat = None
    geom_iter = (
        _geometry_realization_dirs(geom_dir, task_parent, task_dir)
        if geom_dir is not None
        else _sibling_realization_dirs(task_parent, task_dir)
    )
    for rd in geom_iter:
        if nodes is None and (rd / 'nodes.npy').exists():
            try:
                nodes = np.load(rd / 'nodes.npy').astype(float)
            except Exception:
                pass
        if incmat is None and (rd / 'incidence_matrix.npy').exists():
            try:
                incmat = np.load(rd / 'incidence_matrix.npy')
            except Exception:
                pass
        if nodes is not None and incmat is not None:
            break

    task_vals = None
    for rd in _sibling_realization_dirs(task_parent, task_dir):
        if (rd / 'tasks.txt').exists():
            try:
                task_vals = np.loadtxt(rd / 'tasks.txt')
            except Exception:
                pass
        if task_vals is not None:
            break

    if nodes is None or incmat is None or task_vals is None:
        return None

    edges  = incmat_to_edges(incmat)
    N      = nodes.shape[0]
    fdofs  = free_dofs_local(N)
    target = float(np.abs(task_vals[2]))

    return dict(
        nodes=nodes, edges=edges, incmat=incmat,
        stiffnesses=stiffnesses, eq_lengths=L0,
        strain_input=LOCAL_MAX_STRAIN, target_output=target,
        free_dofs=fdofs, N=N,
        geometry_id=geometry_id, task_id=task_id, real_id=real_id,
    )


def load_local_targeted(task_id, real_id, paths, mods):
    return load_local_network(None, task_id, real_id, paths['allosteric_tgt'], mods)


# ── Trajectory computation ────────────────────────────────────────────────────
def compute_global_traj(network, compression_strain, boundary, mods, n_steps=N_STRAIN_STEPS):
    net_copy = copy.deepcopy(network)
    return mods['compute_quasistatic_trajectory_auxetic'](
        net_copy, compression_strain,
        boundary['top'], boundary['bottom'],
        n_steps=n_steps, verbose=False,
        force_type=FORCE_TYPE, tol=1e-9,
    )


def compute_local_traj(net_dict, mods, n_steps=N_STRAIN_STEPS):
    nodes0   = net_dict['nodes']
    edges_j  = mods['jnp'].asarray(net_dict['edges'],      dtype=mods['jnp'].int32)
    eq_j     = mods['jnp'].asarray(net_dict['eq_lengths'])
    k_j      = mods['jnp'].asarray(net_dict['stiffnesses'])
    _crf     = mods['_crf']
    jnp      = mods['jnp']

    x0,  y0  = float(nodes0[0, 0]), float(nodes0[0, 1])
    x1i, y1i = float(nodes0[1, 0]), float(nodes0[1, 1])
    d0        = float(np.linalg.norm(nodes0[1] - nodes0[0]))
    src_dof   = jnp.array([0, 2, 1, 3], dtype=jnp.int32)

    traj = [nodes0.copy()]
    pos  = jnp.asarray(nodes0.flatten())
    for step in range(n_steps):
        target  = (step + 1) / n_steps * LOCAL_MAX_STRAIN
        imposed = jnp.array([x0, x1i + target * d0, y0, y1i])
        for _ in range(N_EQUIL):
            pos = _crf(k_j, edges_j, eq_j, pos, src_dof, imposed)
        traj.append(np.array(pos).reshape(-1, 2))
    return traj


# ── Susceptibilities ──────────────────────────────────────────────────────────
def compute_susceptibilities(positions, edges, k, L0, constrained_nodes, mods):
    H_inv = mods['compute_constrained_hessian_inverse'](
        positions, edges, k, L0, constrained_nodes)
    _, Hjac_parts, _ = mods['compute_full_jacobian_matrixwise'](
        positions, edges, k, L0, H_ff_inv=None, H_full_inv=H_inv)
    s_par, s_perp, s_eq, s_tot = mods['susceptibilities_from_jacobian'](Hjac_parts)
    return (np.asarray(s_par, dtype=float), np.asarray(s_perp, dtype=float),
            np.asarray(s_eq,  dtype=float), np.asarray(s_tot,  dtype=float))


def compute_s_shift(positions, edges, k, L0, fdofs, mods):
    positions = np.asarray(positions, dtype=float)
    n_nodes   = positions.shape[0]
    E         = len(edges)

    B      = build_incidence_matrix(edges, n_nodes)
    disps  = B @ positions
    ells   = np.linalg.norm(disps, axis=1)
    nhats  = disps / ells[:, None]
    stretch = ells - np.asarray(L0)

    J = np.zeros((2 * n_nodes, E), dtype=float)
    J[0::2] = -B.T * (stretch * nhats[:, 0])
    J[1::2] = -B.T * (stretch * nhats[:, 1])

    H_full   = mods['compute_physical_hessian_strained'](k, L0, edges, positions)
    H_ff     = H_full[np.ix_(fdofs, fdofs)]
    H_ff_inv = np.linalg.inv(H_ff)

    S = H_ff_inv @ J[fdofs]
    return np.linalg.norm(S, axis=0)


# ── Cost Hessian eigenvectors ─────────────────────────────────────────────────
def _get_crf_diff(mods):
    global _crf_diff
    if _crf_diff is None:
        _crf_diff = mods['make_compute_response_fire'](
            d=2, force_type=FORCE_TYPE, max_steps=500_000, tol=1e-8)
    return _crf_diff


def compute_global_cost_evec(network, compression_strain, target_poisson, boundary,
                              mods, n_steps=N_STRAIN_STEPS, k_eigs=N_COST_EIGS):
    jax = mods['jax']
    jnp = mods['jnp']
    crf_diff = _get_crf_diff(mods)

    n_edges = len(network.stiffnesses)
    base_k  = np.array(network.stiffnesses, dtype=float)
    top     = jnp.asarray(np.array(boundary['top'],         dtype=np.int32))
    bot     = jnp.asarray(np.array(boundary['bottom'],      dtype=np.int32))
    left    = jnp.asarray(np.array(boundary['left'],        dtype=np.int32))
    right   = jnp.asarray(np.array(boundary['right'],       dtype=np.int32))
    edges_j = jnp.asarray(np.array(network.edges,           dtype=np.int32))
    restl_j = jnp.asarray(np.array(network.rest_lengths,    dtype=np.float64))
    pos_j   = jnp.asarray(np.array(network.positions,       dtype=np.float64).flatten())

    def loss_fn(k_jax):
        pr = mods['compute_poisson_ratio_single_jax'](
            crf_diff, k_jax, edges_j, restl_j, pos_j,
            top, bot, left, right, compression_strain, n_steps)
        return (pr - target_poisson) ** 2

    grad_fn = jax.jit(jax.grad(loss_fn))
    g0 = np.array(grad_fn(jnp.asarray(base_k)))

    def hvp(v):
        v  = np.asarray(v, dtype=float)
        nv = float(np.linalg.norm(v))
        if nv < 1e-14:
            return np.zeros(n_edges)
        gf = np.array(grad_fn(jnp.asarray(base_k + HVP_EPSILON * v / nv)))
        return nv * (gf - g0) / HVP_EPSILON

    H_op = LinearOperator((n_edges, n_edges), matvec=hvp, dtype=float)
    k    = min(k_eigs, n_edges - 1)
    evals, evecs = eigsh(H_op, k=k, which='LM')
    return np.abs(evecs[:, np.argmax(evals)])


def compute_local_cost_evec(net_dict, mods, k_eigs=N_COST_EIGS):
    jax = mods['jax']
    jnp = mods['jnp']
    crf_diff = _get_crf_diff(mods)

    positions    = net_dict['nodes']
    edges        = net_dict['edges']
    stiffnesses  = net_dict['stiffnesses']
    rest_lengths = net_dict['eq_lengths']
    target_out   = net_dict['target_output']
    n_steps      = N_STRAIN_STEPS

    base_k  = jnp.asarray(np.abs(np.array(stiffnesses, dtype=float)))
    edges_j = jnp.asarray(np.array(edges,        dtype=np.int32))
    restl_j = jnp.asarray(np.array(rest_lengths, dtype=np.float64))
    pos_j   = jnp.asarray(np.array(positions,    dtype=np.float64).flatten())
    E       = int(base_k.shape[0])

    x0,  y0  = float(positions[0, 0]), float(positions[0, 1])
    x1i, y1i = float(positions[1, 0]), float(positions[1, 1])
    d0        = float(np.linalg.norm(positions[1] - positions[0]))
    d23_0     = float(np.linalg.norm(positions[3] - positions[2]))
    src_dof   = jnp.array([0, 2, 1, 3], dtype=jnp.int32)

    def loss_fn(k):
        cur = pos_j
        for step in range(n_steps):
            frac    = (step + 1) / n_steps
            imposed = jnp.array([x0, x1i + frac * LOCAL_MAX_STRAIN * d0, y0, y1i])
            cur     = crf_diff(k, edges_j, restl_j, cur, src_dof, imposed)
        eq  = cur.reshape(-1, 2)
        d23 = jnp.linalg.norm(eq[3] - eq[2])
        return ((d23 - d23_0) / d23_0 - target_out) ** 2

    grad_fn = jax.jit(jax.grad(loss_fn))
    g0 = np.array(grad_fn(base_k))

    def hvp(v):
        v  = np.asarray(v, dtype=float)
        nv = float(np.linalg.norm(v))
        if nv < 1e-14:
            return np.zeros(E)
        gf = np.array(grad_fn(base_k + HVP_EPSILON * v / nv))
        return nv * (gf - g0) / HVP_EPSILON

    H_op = LinearOperator((E, E), matvec=hvp, dtype=float)
    k    = min(k_eigs, E - 1)
    _, evecs = eigsh(H_op, k=k, which='LA', maxiter=300, tol=1e-5)
    return np.abs(evecs[:, -1])


# ── Hessian mode analysis ─────────────────────────────────────────────────────
def hessian_modes(positions, stiffnesses, eq_lengths, edges, fdofs, mods):
    H      = mods['compute_physical_hessian_strained'](stiffnesses, eq_lengths, edges, positions)
    H_free = H[np.ix_(fdofs, fdofs)]
    vals, vecs = np.linalg.eigh(H_free)
    return vals, vecs.T


def compute_analysis(traj, stiffnesses, eq_lengths, edges, fdofs, mods, verbose=True):
    T = len(traj)
    M = len(fdofs)
    spectrum = np.zeros((T, M))
    overlaps = np.zeros((T - 1, M))

    for t in range(T):
        vals, vecs_T = hessian_modes(traj[t], stiffnesses, eq_lengths, edges, fdofs, mods)
        spectrum[t]  = vals
        if t < T - 1:
            dr  = (traj[t + 1] - traj[t]).ravel()[fdofs]
            nrm = float(np.linalg.norm(dr))
            if nrm > 1e-12:
                overlaps[t] = vecs_T @ (dr / nrm)
        if verbose and (t + 1) % 25 == 0:
            print(f'    frame {t + 1}/{T}', flush=True)

    return spectrum, overlaps


def compute_mode_sensitivity(stiffnesses_np, pos_t, pos_t1, fdofs_np, edges_np, eq_lengths_np, mods):
    jax = mods['jax']
    jnp = mods['jnp']
    N   = pos_t.shape[0]
    E   = len(stiffnesses_np)

    edges_a = np.asarray(edges_np, dtype=int)
    B       = build_incidence_matrix(edges_a, N)
    disps   = B @ pos_t
    ells    = np.linalg.norm(disps, axis=1)
    nhats   = disps / ells[:, None]
    fs      = 1.0 - np.asarray(eq_lengths_np) / ells

    dr  = (pos_t1 - pos_t).ravel()[fdofs_np]
    nrm = float(np.linalg.norm(dr))
    if nrm < 1e-12:
        return np.zeros((len(fdofs_np), E))
    dr_hat = dr / nrm

    nhats_j  = jnp.array(nhats)
    fs_j     = jnp.array(fs)
    B_j      = jnp.array(B)
    dr_hat_j = jnp.array(dr_hat)
    fdofs_a  = np.asarray(fdofs_np, dtype=int)

    def overlaps_fn(k):
        odiags1 = jnp.einsum('im,in->imn', nhats_j, nhats_j)
        odiags  = (odiags1
                   + jnp.einsum('i,mn->imn', fs_j, jnp.eye(2))
                   - jnp.einsum('i,imn->imn', fs_j, odiags1))
        H_flat = jnp.einsum('i,ia,ib,imn->ambn',
                             k, B_j, B_j, odiags).reshape(2 * N, 2 * N)
        H_free = H_flat[fdofs_a[:, None], fdofs_a[None, :]]
        _, vecs = jnp.linalg.eigh(H_free)
        return vecs.T @ dr_hat_j

    jac = jax.jacobian(overlaps_fn)(jnp.array(stiffnesses_np, dtype=jnp.float64))
    return np.array(jac)


# ── Mode runners ──────────────────────────────────────────────────────────────
def run_global_nt(paths, task_id, real_id, mods):
    key = f'global_nt_t{task_id:02d}_r{real_id:02d}'
    if cache_exists(paths['cache_results'], key):
        print(f'  {key}: already cached, skipping.')
        return

    real_dir = paths['results'] / f'task_{task_id:02d}' / f'realization_{real_id:02d}'
    if not real_dir.is_dir():
        print(f'  {key}: directory not found, skipping.')
        return
    if not (real_dir / 'training_complete.txt').exists():
        print(f'  {key}: training not complete, skipping.')
        return

    try:
        network, boundary, config = load_global_network(task_id, real_id, paths['results'], mods)
    except Exception as e:
        print(f'  {key}: load error — {e}')
        return

    comp_strains   = config['compression_strains']
    target_poisons = config['target_poisson_ratios']
    n_sub          = len(comp_strains)
    save_dict      = {'n_subtasks': np.array(n_sub), 'stiffnesses': network.stiffnesses}
    ok             = False

    for si, (cs, tp) in enumerate(zip(comp_strains, target_poisons)):
        print(f'  {key}/si{si} cs={cs:.3f}...', flush=True)
        try:
            traj        = compute_global_traj(network, cs, boundary, mods)
            pos_end     = np.array(traj[-1])
            constrained = np.unique(np.concatenate([boundary['top'], boundary['bottom']]))
            fdofs       = free_dofs_global(pos_end.shape[0], boundary['top'], boundary['bottom'])

            s_par, s_perp, s_eq, s_tot = compute_susceptibilities(
                pos_end, network.edges, network.stiffnesses, network.rest_lengths,
                constrained, mods)
            s_shift  = compute_s_shift(pos_end, network.edges, network.stiffnesses,
                                       network.rest_lengths, fdofs, mods)
            b_strain, b_stress = bond_strain_stress(
                pos_end, network.edges, network.stiffnesses, network.rest_lengths)
            psi = compute_global_cost_evec(network, cs, tp, boundary, mods)

            rho_sh,  _ = spearmanr(psi, np.abs(s_shift))
            rho_tot, _ = spearmanr(psi, np.abs(s_tot))
            print(f'    rho_shift={rho_sh:.3f}  rho_tot={rho_tot:.3f}', flush=True)

            save_dict.update({
                f'si{si}_s_par':             s_par,
                f'si{si}_s_perp':            s_perp,
                f'si{si}_s_eq':              s_eq,
                f'si{si}_s_tot':             s_tot,
                f'si{si}_s_shift':           s_shift,
                f'si{si}_bond_strain':       b_strain,
                f'si{si}_bond_stress':       b_stress,
                f'si{si}_psi':               psi,
                f'si{si}_compression_strain': np.array(cs),
            })
            ok = True
        except Exception as e:
            print(f'  {key}/si{si}: FAILED — {e}')

    if ok:
        save_cache(paths['cache_results'], key, **save_dict)
        print(f'  {key}: saved.')


def run_local_nt(paths, geom_id, task_id, real_id, mods):
    key = f'local_nt_g{geom_id}_t{task_id}_r{real_id}'
    if cache_exists(paths['cache_allosteric'], key):
        print(f'  {key}: already cached, skipping.')
        return

    try:
        net = load_local_network(geom_id, task_id, real_id, paths['allosteric'], mods)
    except Exception as e:
        print(f'  {key}: load error — {e}')
        return
    if net is None:
        print(f'  {key}: data incomplete, skipping.')
        return

    print(f'  {key} target={net["target_output"]:.3f}...', flush=True)
    try:
        traj              = compute_local_traj(net, mods)
        pos_end           = np.array(traj[-1])
        fdofs             = net['free_dofs']
        constrained_nodes = np.array([0, 1], dtype=int)

        s_par, s_perp, s_eq, s_tot = compute_susceptibilities(
            pos_end, net['edges'], net['stiffnesses'], net['eq_lengths'],
            constrained_nodes, mods)
        s_shift  = compute_s_shift(pos_end, net['edges'], net['stiffnesses'],
                                   net['eq_lengths'], fdofs, mods)
        b_strain, b_stress = bond_strain_stress(
            pos_end, net['edges'], net['stiffnesses'], net['eq_lengths'])
        psi = compute_local_cost_evec(net, mods)

        rho, _ = spearmanr(psi, np.abs(s_shift))
        print(f'    rho_shift={rho:.3f}', flush=True)

        save_cache(paths['cache_allosteric'], key,
                   s_par=s_par, s_perp=s_perp, s_eq=s_eq, s_tot=s_tot,
                   s_shift=s_shift, bond_strain=b_strain, bond_stress=b_stress,
                   stiffnesses=net['stiffnesses'], psi=psi,
                   target_output=np.array(net['target_output']))
        print(f'  {key}: saved.')
    except Exception as e:
        print(f'  {key}: FAILED — {e}')


def run_global_rep(paths, task_id, real_id, mods):
    key = f'global_rep_t{task_id:02d}_r{real_id:02d}'
    if cache_exists(paths['cache_targeted'], key):
        print(f'  {key}: already cached, skipping.')
        return

    print(f'  Loading global representative task_{task_id:02d}/real_{real_id:02d}...', flush=True)
    network, boundary, config = load_global_network(task_id, real_id, paths['targeted'], mods)
    comp_strains = config['compression_strains']
    target_pois  = config['target_poisson_ratios']

    fdofs       = free_dofs_global(network.positions.shape[0], boundary['top'], boundary['bottom'])
    constrained = np.unique(np.concatenate([boundary['top'], boundary['bottom']]))

    sub_trajs, sub_spectra, sub_overlaps, sub_psi = [], [], [], []
    sub_s_shift, sub_s_par, sub_s_perp, sub_s_eq, sub_s_tot = [], [], [], [], []

    for si, (cs, tp) in enumerate(zip(comp_strains, target_pois)):
        print(f'  Subtask {si}: cs={cs:.3f}  tp={tp:.3f}', flush=True)
        traj = compute_global_traj(network, cs, boundary, mods)
        T    = len(traj)

        spectrum, overlaps = compute_analysis(
            traj, network.stiffnesses, network.rest_lengths, network.edges, fdofs, mods)
        psi_si = compute_global_cost_evec(network, cs, tp, boundary, mods)

        ss_shift, ss_par, ss_perp, ss_eq, ss_tot = [], [], [], [], []
        for t_idx in range(T):
            pos = np.array(traj[t_idx])
            sp, spe, seq, st = compute_susceptibilities(
                pos, network.edges, network.stiffnesses, network.rest_lengths, constrained, mods)
            ssh = compute_s_shift(pos, network.edges, network.stiffnesses,
                                  network.rest_lengths, fdofs, mods)
            ss_par.append(sp); ss_perp.append(spe); ss_eq.append(seq)
            ss_tot.append(st); ss_shift.append(ssh)

        sub_trajs.append(np.array([np.array(p) for p in traj]))
        sub_spectra.append(spectrum)
        sub_overlaps.append(overlaps)
        sub_psi.append(psi_si)
        sub_s_shift.append(np.array(ss_shift))
        sub_s_par.append(np.array(ss_par))
        sub_s_perp.append(np.array(ss_perp))
        sub_s_eq.append(np.array(ss_eq))
        sub_s_tot.append(np.array(ss_tot))

    save_dict = dict(
        comp_strains=np.array(comp_strains),
        target_poissons=np.array(target_pois),
        positions0=network.positions,
        stiffnesses=network.stiffnesses,
        rest_lengths=network.rest_lengths,
        edges=network.edges,
        fdofs=fdofs,
        constrained=constrained,
        top=boundary['top'], bottom=boundary['bottom'],
        left=boundary['left'], right=boundary['right'],
    )
    for si in range(len(comp_strains)):
        save_dict.update({
            f'si{si}_traj':     sub_trajs[si],
            f'si{si}_spectrum': sub_spectra[si],
            f'si{si}_overlaps': sub_overlaps[si],
            f'si{si}_psi':      sub_psi[si],
            f'si{si}_s_shift':  sub_s_shift[si],
            f'si{si}_s_par':    sub_s_par[si],
            f'si{si}_s_perp':   sub_s_perp[si],
            f'si{si}_s_eq':     sub_s_eq[si],
            f'si{si}_s_tot':    sub_s_tot[si],
        })
    save_cache(paths['cache_targeted'], key, **save_dict)
    print(f'  {key}: saved ({len(comp_strains)} subtask(s), {len(network.edges)} edges).')


def run_local_rep(paths, task_id, real_id, mods):
    key = f'local_rep_tgt_t{task_id}_r{real_id}'
    if cache_exists(paths['cache_allosteric_tgt'], key):
        print(f'  {key}: already cached, skipping.')
        return

    print(f'  Loading local representative task_{task_id}/real_{real_id}...', flush=True)
    net_l = load_local_targeted(task_id, real_id, paths, mods)
    if net_l is None:
        raise RuntimeError(f'Local representative network not found (task {task_id}, real {real_id})')

    fdofs_l       = net_l['free_dofs']
    constrained_l = np.array([0, 1], dtype=int)

    traj_l             = compute_local_traj(net_l, mods)
    T_l                = len(traj_l)
    spectrum_l, overlaps_l = compute_analysis(
        traj_l, net_l['stiffnesses'], net_l['eq_lengths'], net_l['edges'], fdofs_l, mods)
    psi_l = compute_local_cost_evec(net_l, mods)

    ss_shift, ss_par, ss_perp, ss_eq, ss_tot = [], [], [], [], []
    for t_idx in range(T_l):
        pos = np.array(traj_l[t_idx])
        sp, spe, seq, st = compute_susceptibilities(
            pos, net_l['edges'], net_l['stiffnesses'], net_l['eq_lengths'], constrained_l, mods)
        ssh = compute_s_shift(pos, net_l['edges'], net_l['stiffnesses'],
                              net_l['eq_lengths'], fdofs_l, mods)
        ss_par.append(sp); ss_perp.append(spe); ss_eq.append(seq)
        ss_tot.append(st); ss_shift.append(ssh)

    save_cache(paths['cache_allosteric_tgt'], key,
               nodes=net_l['nodes'],
               stiffnesses=net_l['stiffnesses'],
               eq_lengths=net_l['eq_lengths'],
               edges=net_l['edges'],
               fdofs=fdofs_l,
               constrained=constrained_l,
               target_output=np.array(net_l['target_output']),
               traj=np.array([np.array(p) for p in traj_l]),
               spectrum=spectrum_l,
               overlaps=overlaps_l,
               psi=psi_l,
               s_shift_traj=np.array(ss_shift),
               s_par_traj=np.array(ss_par),
               s_perp_traj=np.array(ss_perp),
               s_eq_traj=np.array(ss_eq),
               s_tot_traj=np.array(ss_tot))
    print(f'  {key}: saved ({len(net_l["edges"])} edges, {T_l} frames).')


def run_modesens_local(paths, geom_id, task_id, real_id, mods):
    key = f'modesens_local_g{geom_id}_t{task_id}_r{real_id}'
    if cache_exists(paths['cache_allosteric'], key):
        print(f'  {key}: already cached, skipping.')
        return

    net = load_local_network(geom_id, task_id, real_id, paths['allosteric'], mods)
    if net is None:
        print(f'  {key}: data incomplete, skipping.')
        return

    print(f'  {key}...', flush=True)
    try:
        traj  = compute_local_traj(net, mods, n_steps=N_TRAJ_STEPS_BULK)
        pos_t = np.array(traj[MID_FRAME_BULK])
        pos_t1 = np.array(traj[MID_FRAME_BULK + 1])
        ssh   = compute_s_shift(pos_t, net['edges'], net['stiffnesses'],
                                net['eq_lengths'], net['free_dofs'], mods)
        sens  = compute_mode_sensitivity(net['stiffnesses'], pos_t, pos_t1,
                                         net['free_dofs'], net['edges'], net['eq_lengths'], mods)
        norms = np.linalg.norm(sens, axis=0)
        rho, _ = spearmanr(ssh, norms)
        save_cache(paths['cache_allosteric'], key, rho=np.array(rho), sshift=ssh, col_norms=norms)
        print(f'  {key}: rho={rho:.3f}, saved.')
    except Exception as e:
        print(f'  {key}: FAILED — {e}')


def run_modesens_global(paths, task_id, real_id, mods):
    key = f'modesens_global_t{task_id:02d}_r{real_id:02d}'
    if cache_exists(paths['cache_results'], key):
        print(f'  {key}: already cached, skipping.')
        return

    real_dir = paths['results'] / f'task_{task_id:02d}' / f'realization_{real_id:02d}'
    if not (real_dir / 'final_network.pkl').exists():
        print(f'  {key}: final_network.pkl not found, skipping.')
        return

    try:
        network, boundary, config = load_global_network(task_id, real_id, paths['results'], mods)
    except Exception as e:
        print(f'  {key}: load error — {e}')
        return

    cs0    = config['compression_strains'][0]
    fdofs  = free_dofs_global(network.positions.shape[0], boundary['top'], boundary['bottom'])

    print(f'  {key} cs={cs0:.3f}...', flush=True)
    try:
        traj  = compute_global_traj(network, cs0, boundary, mods, n_steps=N_TRAJ_STEPS_BULK)
        pos_t = np.array(traj[MID_FRAME_BULK])
        pos_t1 = np.array(traj[MID_FRAME_BULK + 1])
        ssh   = compute_s_shift(pos_t, network.edges, network.stiffnesses,
                                network.rest_lengths, fdofs, mods)
        sens  = compute_mode_sensitivity(network.stiffnesses, pos_t, pos_t1,
                                         fdofs, network.edges, network.rest_lengths, mods)
        norms = np.linalg.norm(sens, axis=0)
        rho, _ = spearmanr(ssh, norms)
        save_cache(paths['cache_results'], key, rho=np.array(rho), sshift=ssh, col_norms=norms)
        print(f'  {key}: rho={rho:.3f}, saved.')
    except Exception as e:
        print(f'  {key}: FAILED — {e}')


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Generate .figure_notebook_cache/ entries for AllFigures.ipynb')
    parser.add_argument('--mode', required=True,
                        choices=['global_nt', 'local_nt', 'global_rep', 'local_rep',
                                 'modesens_local', 'modesens_global'],
                        help='Which cache type to compute')
    parser.add_argument('--task', type=int, default=None)
    parser.add_argument('--real', type=int, default=None)
    parser.add_argument('--geom', type=int, default=None)
    parser.add_argument('--repo-root', type=str, default=None,
                        help='Path to repository root (default: two levels up from this file)')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Override cache directory path')
    args = parser.parse_args()

    repo_root = Path(args.repo_root) if args.repo_root else None
    paths     = build_paths(repo_root, args.cache_dir)
    # src/ imports always come from the repo where this script lives
    src_root  = Path(args.repo_root) if args.repo_root else Path(__file__).resolve().parents[1]
    mods      = _setup_imports(src_root)

    print(f'Mode               : {args.mode}')
    print(f'cache_results      : {paths["cache_results"]}')
    print(f'cache_targeted     : {paths["cache_targeted"]}')
    print(f'cache_allosteric   : {paths["cache_allosteric"]}')
    print(f'cache_allosteric_tgt: {paths["cache_allosteric_tgt"]}')
    print(f'JAX devices        : {mods["jax"].devices()}')
    print()

    if args.mode == 'global_nt':
        if args.task is None or args.real is None:
            parser.error('--mode global_nt requires --task and --real')
        run_global_nt(paths, args.task, args.real, mods)

    elif args.mode == 'local_nt':
        if args.geom is None or args.task is None or args.real is None:
            parser.error('--mode local_nt requires --geom, --task, and --real')
        run_local_nt(paths, args.geom, args.task, args.real, mods)

    elif args.mode == 'global_rep':
        task = args.task if args.task is not None else DEFAULT_GLOBAL_TASK
        real = args.real if args.real is not None else DEFAULT_GLOBAL_REAL
        run_global_rep(paths, task, real, mods)

    elif args.mode == 'local_rep':
        task = args.task if args.task is not None else DEFAULT_LOCAL_TASK
        real = args.real if args.real is not None else DEFAULT_LOCAL_REAL
        run_local_rep(paths, task, real, mods)

    elif args.mode == 'modesens_local':
        if args.geom is None or args.task is None or args.real is None:
            parser.error('--mode modesens_local requires --geom, --task, and --real')
        run_modesens_local(paths, args.geom, args.task, args.real, mods)

    elif args.mode == 'modesens_global':
        if args.task is None or args.real is None:
            parser.error('--mode modesens_global requires --task and --real')
        run_modesens_global(paths, args.task, args.real, mods)


if __name__ == '__main__':
    main()
