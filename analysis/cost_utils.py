"""
Cost function utilities: loss evaluation and cost Hessian eigenvectors.

Provides a unified pipeline:
  compute_trajectory → loss_from_trajectory → compute_cost_hessian

The trajectory step separates FIRE minimization (expensive, Cython)
from the loss evaluation (cheap, differentiable), enabling JAX autodiff
through the loss w.r.t. stiffnesses K.
"""

import time
import numpy as np

import jax
import jax.numpy as jnp
from scipy.sparse.linalg import LinearOperator, eigsh

jax.config.update("jax_enable_x64", True)

from base.simulate import compute_poisson_ratio_single_jax, crf as _crf


# ---------------------------------------------------------------------------
# Trajectory given K
# ---------------------------------------------------------------------------

def compute_trajectory(stiffnesses, network, boundary, compression_strain,
                       n_steps=100, force_type='quadratic', tol=1e-6, method='fire'):
    """
    Compute quasistatic trajectory for given stiffness vector K.

    This is a thin wrapper around trajectory.compute_auxetic_trajectory that
    accepts K explicitly (useful when differentiating w.r.t. K via JAX).

    Parameters
    ----------
    stiffnesses : (E,) array   spring stiffnesses K
    network : ElasticNetwork   (topology + positions; stiffnesses overridden)
    boundary : dict with keys 'top', 'bottom', 'left', 'right'
    compression_strain : float
    n_steps : int
    force_type : str
    tol : float
    method : str   'fire' (Cython FIRE, default) or 'newton' (Newton-Raphson) —
                   pass the same method the training run used so the recomputed
                   trajectory uses the same solver as the stored loss.

    Returns
    -------
    list of (N, 2) position arrays, length n_steps
    """
    import copy
    from .trajectory import compute_auxetic_trajectory
    net = copy.copy(network)
    net.stiffnesses = np.asarray(stiffnesses, dtype=float)
    return compute_auxetic_trajectory(net, compression_strain, boundary,
                                      n_steps=n_steps, force_type=force_type, tol=tol,
                                      method=method)


# ---------------------------------------------------------------------------
# Loss from trajectory
# ---------------------------------------------------------------------------

def loss_from_trajectory(trajectory, compression_strain, target_poisson, boundary):
    """
    MSE Poisson-ratio loss evaluated on a pre-computed trajectory.

    Parameters
    ----------
    trajectory : list of (N, 2) position arrays
    compression_strain : float   (negative, e.g. -0.08)
    target_poisson : float
    boundary : dict with keys 'top', 'bottom', 'left', 'right'

    Returns
    -------
    loss : float   (attained_poisson - target_poisson)^2
    """
    attained = _poisson_from_trajectory(trajectory, compression_strain, boundary)
    return float((attained - target_poisson) ** 2)


def _poisson_from_trajectory(trajectory, compression_strain, boundary):
    """Estimate Poisson ratio from the endpoint of the trajectory."""
    top    = boundary['top']
    bottom = boundary['bottom']
    left   = boundary['left']
    right  = boundary['right']

    pos0 = np.asarray(trajectory[0])
    posf = np.asarray(trajectory[-1])

    H0 = pos0[top, 1].mean() - pos0[bottom, 1].mean()
    Hf = posf[top, 1].mean() - posf[bottom, 1].mean()
    W0 = pos0[right, 0].mean() - pos0[left, 0].mean()
    Wf = posf[right, 0].mean() - posf[left, 0].mean()

    eps_yy = (Hf - H0) / H0
    eps_xx = (Wf - W0) / W0

    if abs(eps_yy) < 1e-12:
        return 0.0
    return -eps_xx / eps_yy


# ---------------------------------------------------------------------------
# Cost Hessian
# ---------------------------------------------------------------------------

def compute_cost_hessian(network, compression_strain, target_poisson, boundary,
                         k_eigs=None, hvp_epsilon=1e-4, force_type='quadratic',
                         n_strain_steps=100, verbose=True):
    """
    Top-k eigenvalues/vectors of the loss Hessian w.r.t. spring stiffnesses K.

    Loss = (attained_poisson(K) - target_poisson)^2

    Uses JAX autodiff for gradient + finite-difference HVP + Lanczos (eigsh).

    Parameters
    ----------
    network : ElasticNetwork   (stiffnesses are the operating point)
    compression_strain : float
    target_poisson : float
    boundary : dict with keys 'top', 'bottom', 'left', 'right'
    k_eigs : int or None    number of top (largest positive / algebraic) eigenvalues
                             to return; None → all (n_edges - 1)
    hvp_epsilon : float     finite-difference step for HVP
    force_type : str
    n_strain_steps : int
    verbose : bool

    Returns
    -------
    eigenvalues  : (k,) sorted ascending — the k largest algebraic (most positive)
                   eigenvalues, not the k largest by absolute magnitude.
    eigenvectors : (n_edges, k)
    """
    n_edges = len(network.stiffnesses)
    base_k  = np.array(network.stiffnesses, dtype=float)

    edges_jax        = jnp.asarray(np.array(network.edges,        dtype=np.int32))
    rest_lengths_jax = jnp.asarray(np.array(network.rest_lengths, dtype=np.float64))
    positions_jax    = jnp.asarray(np.array(network.positions,    dtype=np.float64).flatten())
    top    = boundary['top']
    bottom = boundary['bottom']
    left   = boundary['left']
    right  = boundary['right']

    def loss_jax(k_jax):
        pr = compute_poisson_ratio_single_jax(
            _crf, k_jax, edges_jax, rest_lengths_jax, positions_jax,
            top, bottom, left, right,
            compression_strain, n_strain_steps,
        )
        return (pr - target_poisson) ** 2

    grad_loss = jax.jit(jax.grad(loss_jax))

    if verbose:
        print(f"  Computing base gradient ({n_edges} edges) ...", flush=True)
    t0 = time.time()
    g0 = np.array(grad_loss(jnp.asarray(base_k)))
    if verbose:
        print(f"  Base gradient done in {time.time()-t0:.1f}s", flush=True)

    hvp_count = [0]

    def hvp(v):
        v = np.asarray(v, dtype=float)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-14:
            return np.zeros_like(v)
        hvp_count[0] += 1
        g_fwd = np.array(grad_loss(jnp.asarray(base_k + hvp_epsilon * v / norm_v)))
        return norm_v * (g_fwd - g0) / hvp_epsilon

    k = (n_edges - 1) if k_eigs is None else min(k_eigs, n_edges - 1)
    H_op = LinearOperator((n_edges, n_edges), matvec=hvp, dtype=float)
    if verbose:
        print(f"  Starting eigsh (k={k}) ...", flush=True)
    t_eig = time.time()
    # 'LA' = largest algebraic, i.e. the k most positive eigenvalues (not the k
    # largest in absolute value, which 'LM' would also pull from large-negative
    # curvature directions).
    eigenvalues, eigenvectors = eigsh(H_op, k=k, which='LA')
    if verbose:
        print(f"  eigsh done in {time.time()-t_eig:.1f}s ({hvp_count[0]} HVPs)", flush=True)

    order = np.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]
