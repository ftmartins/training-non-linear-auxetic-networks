"""Mode overlap analysis: cosine projections and participation ratios."""

import numpy as np


def mode_overlaps(trajectory, eigenvectors):
    """
    Compute cosine overlap of successive displacements with elastic eigenmodes.

    Parameters
    ----------
    trajectory : list of (N, 2) or (2N,) arrays, length T+1
    eigenvectors : (2N, M) array of mode shapes (columns = modes)

    Returns
    -------
    overlaps : (T, M) array
        overlaps[t, k] = |cos θ| = |dr_t · v_k| / (||dr_t|| * ||v_k||)
        where dr_t = trajectory[t+1] - trajectory[t].
    """
    traj = [np.asarray(x).ravel() for x in trajectory]
    evecs = np.asarray(eigenvectors)                     # (2N, M)
    evec_norms = np.linalg.norm(evecs, axis=0)           # (M,)
    evecs_unit = evecs / np.maximum(evec_norms, 1e-15)   # (2N, M)

    T = len(traj) - 1
    overlaps = np.zeros((T, evecs.shape[1]))
    for t in range(T):
        dr = traj[t + 1] - traj[t]
        norm_dr = np.linalg.norm(dr)
        if norm_dr < 1e-15:
            continue
        dr_unit = dr / norm_dr
        overlaps[t] = np.abs(dr_unit @ evecs_unit)       # (M,)

    return overlaps


def participation_ratio(overlaps):
    """
    Participation ratio per mode across trajectory steps.

    PR_k = (sum_t |overlap[t,k]|)^2 / (T * sum_t overlap[t,k]^2)

    Returns (M,) array.
    """
    overlaps = np.asarray(overlaps)
    T = overlaps.shape[0]
    numerator = np.sum(np.abs(overlaps), axis=0) ** 2
    denominator = T * np.sum(overlaps ** 2, axis=0)
    return np.where(denominator > 0, numerator / denominator, 0.0)
