"""
Learning-rate schedule as a pure function of a saved loss trajectory.

`lr_scale_for_step` never persists anything: it replays a plateau/overshoot
classifier over the loss values already written to disk for other reasons
(`history['loss']` for the auxetic/moduli training loops, the combined
mse1/mse2 trajectory for allosteric) and returns the multiplier that should
be applied on top of the run's fixed nominal `learning_rate`. Because it's
recomputed from data that's already checkpointed/saved, it needs no new
persisted state, survives resumes for free, and is fully reproducible after
the fact by anyone re-running it over the saved loss array.
"""
import numpy as np


def classify_window(loss_window, plateau_tol=0.05, overshoot_ratio=1.3):
    """
    Classify one window of consecutive loss values as:
      - 'overshoot': contains a non-finite value, or the loss rose above
        overshoot_ratio times the window's starting value at any point.
      - 'plateau': finite throughout, but didn't drop by at least
        plateau_tol (relative) from start to end of the window.
      - 'improving': neither of the above.

    Args:
        loss_window: 1D array-like of loss values, in step order.
        plateau_tol: Minimum relative drop (start to end) to count as
            'improving' rather than 'plateau'.
        overshoot_ratio: Loss rising above this multiple of the window's
            starting value, anywhere in the window, counts as 'overshoot'.

    Returns:
        'overshoot', 'plateau', or 'improving'.
    """
    arr = np.asarray(loss_window, dtype=float)
    if arr.size == 0:
        return 'plateau'
    if not np.all(np.isfinite(arr)):
        return 'overshoot'
    first, last = arr[0], arr[-1]
    if arr.max() > first * overshoot_ratio:
        return 'overshoot'
    if last < first * (1 - plateau_tol):
        return 'improving'
    return 'plateau'


def lr_scale_for_step(loss_trajectory, window=1000, decay=0.5, min_scale=1e-3,
                       plateau_tol=0.05, overshoot_ratio=1.3):
    """
    Replay classify_window over completed `window`-length chunks of
    loss_trajectory, halving a running scale factor on 'overshoot' or
    'plateau' (never increasing it on 'improving'). Pure function of the
    trajectory — safe to call every step, safe to recompute identically
    after a resume.

    Args:
        loss_trajectory: 1D array-like of loss values recorded so far,
            in step order (e.g. history['loss'], or (mse1+mse2)/2).
        window: Chunk size the trajectory is classified over.
        decay: Multiplier applied to the running scale on 'overshoot'/
            'plateau'.
        min_scale: Lower clamp on the returned scale, so a persistently
            stuck run doesn't decay to a step size that can never move.
        plateau_tol, overshoot_ratio: Passed through to classify_window.

    Returns:
        (scale, last_status) — last_status is the classification of the
        most recently completed window, or None if loss_trajectory has
        fewer than `window` entries yet.
    """
    arr = np.asarray(loss_trajectory, dtype=float)
    n_complete_windows = len(arr) // window

    scale = 1.0
    last_status = None
    for i in range(n_complete_windows):
        chunk = arr[i * window:(i + 1) * window]
        last_status = classify_window(chunk, plateau_tol=plateau_tol,
                                       overshoot_ratio=overshoot_ratio)
        if last_status in ('overshoot', 'plateau'):
            scale = max(scale * decay, min_scale)

    return scale, last_status
