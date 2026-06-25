"""Figure-making utilities for analysis notebooks."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.colors import LogNorm, Normalize

from base.plot_config import (
    apply_style,
    plot_network,
    ensure_all_spines,
    latexify_ticks,
    subtask_color,
    subtask_label,
    COLORS,
    SUBTASK_COLORS,
    MARKERS,
    network_cmap,
)

__all__ = [
    'apply_style', 'plot_network', 'ensure_all_spines', 'latexify_ticks',
    'subtask_color', 'subtask_label', 'COLORS', 'SUBTASK_COLORS', 'MARKERS',
    'network_cmap',
    'plot_susceptibility_scatter',
    'plot_mode_overlap_heatmap',
    'plot_trajectory_snapshot',
]


def plot_susceptibility_scatter(s_tot, cost_evec, ax, color='steelblue',
                                alpha=0.5, log_scale=True, **kwargs):
    """
    Scatter plot of |s_tot| vs |cost eigenvector| per edge.

    Parameters
    ----------
    s_tot : (E,) susceptibility values
    cost_evec : (E,) cost Hessian eigenvector values
    ax : matplotlib Axes
    log_scale : bool  use log–log axes
    """
    x = np.abs(cost_evec)
    y = np.abs(s_tot)
    ax.scatter(x, y, color=color, alpha=alpha, **kwargs)
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel(r'$|\psi_e|$')
    ax.set_ylabel(r'$|s_\mathrm{tot}|$')
    return ax


def plot_mode_overlap_heatmap(overlaps, spectrum=None, ax=None, cmap='viridis',
                              vmax=None, aspect='auto', **kwargs):
    """
    Heatmap of mode overlaps (T-1, M).

    Parameters
    ----------
    overlaps : (T, M) array
    spectrum : (T, M) eigenvalue array (optional, for x-axis labelling)
    ax : matplotlib Axes or None (creates new figure)
    """
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(overlaps.T, aspect=aspect, origin='lower', cmap=cmap,
                   vmin=0, vmax=(vmax or overlaps.max()), **kwargs)
    ax.set_xlabel('Trajectory step')
    ax.set_ylabel('Mode index')
    plt.colorbar(im, ax=ax, label='|overlap|')
    return ax, im


def plot_trajectory_snapshot(positions, edges, stiffnesses=None, ax=None,
                             cmap=None, vmin=None, vmax=None, **kwargs):
    """
    Draw a network snapshot (positions + edges), optionally colored by stiffness.

    Parameters
    ----------
    positions : (N, 2)
    edges : (E, 2)
    stiffnesses : (E,) or None  (if None, edges drawn in uniform grey)
    ax : matplotlib Axes or None
    """
    if ax is None:
        fig, ax = plt.subplots()

    if stiffnesses is not None:
        sm = plot_network(positions, edges, stiffnesses, ax,
                          cmap=cmap, vmin=vmin, vmax=vmax,
                          show_colorbar=False, **kwargs)
    else:
        segs = positions[edges]          # (E, 2, 2)
        lc = mc.LineCollection(segs, colors='grey', linewidths=0.8)
        ax.add_collection(lc)
        ax.scatter(positions[:, 0], positions[:, 1], s=4, c='k', zorder=3)
        ax.set_aspect('equal')
        ax.autoscale_view()

    return ax
