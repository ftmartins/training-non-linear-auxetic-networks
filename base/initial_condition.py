"""
Initial condition: the initial bond stiffnesses of a realization, a pure function of
(task kind, realization seed, network size, stiffness bounds). Every trainer, the
screening scripts and the analysis code draw through here, so the draw has one definition.

Each draw reproduces, bit for bit, the formula the trainers used before this module:
  * auxetic gradient descent (task_seed < 20): exp of a uniform draw in log space;
    task_seed >= 20: uniform draw in linear space between exp(log bounds);
  * allosteric and auxetic contrastive learning: RandomState(REALIZATION_BASE + seed)
    .uniform(k_min, k_max).
"""
import numpy as np

from base.config import STIFFNESS_LOG_MIN, STIFFNESS_LOG_MAX

REALIZATION_BASE = 3_000_000   # separates realization seeds from geometry (1e6) and task (2e6) seeds


def auxetic_gradient(task_seed, realization_seed, n_edges):
    """Gradient-descent auxetic trainers (ensemble_runner, targeted_ensemble_runner)."""
    rng = np.random.RandomState(realization_seed)
    if task_seed < 20:
        return np.exp(rng.uniform(STIFFNESS_LOG_MIN, STIFFNESS_LOG_MAX, size=n_edges))
    return rng.uniform(np.exp(STIFFNESS_LOG_MIN), np.exp(STIFFNESS_LOG_MAX), size=n_edges)


def realization_rng(realization_seed):
    return np.random.RandomState(REALIZATION_BASE + realization_seed)


def contrastive(realization_seed, n_edges, k_min, k_max):
    """Allosteric and auxetic contrastive-learning trainers."""
    return realization_rng(int(realization_seed)).uniform(k_min, k_max, size=n_edges)
