"""
Plotting configuration for ensemble training figures.

Usage:
    from plot_config import apply_style, COLORS, MARKERS, network_cmap, ...
    apply_style()  # call once at notebook start
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.colors import LogNorm, Normalize
import numpy as np

# =============================================================================
# cmocean import (fallback to built-in if not installed)
# =============================================================================
try:
    import cmocean
    network_cmap = cmocean.cm.matter_r
except ImportError:
    network_cmap = plt.cm.YlOrBr_r  # reasonable fallback

# =============================================================================
# COLOR PALETTE  — high-contrast, colorblind-friendly
# =============================================================================
COLORS = {
    'subtask_0': '#0072B2',   # blue
    'subtask_1': '#D55E00',   # vermillion
    'subtask_2': '#009E73',   # bluish green
    'subtask_3': '#CC79A7',   # reddish purple
    'subtask_4': '#E69F00',   # orange
    'train_star': '#CC0000',  # red for training point markers
    'gray':      '#999999',
    'black':     '#000000',
}

SUBTASK_COLORS = [COLORS[f'subtask_{i}'] for i in range(5)]

# =============================================================================
# MARKERS
# =============================================================================
MARKERS = {
    'strain':  'o',
    'stress':  's',
    'default': 'o',
}

# =============================================================================
# GLOBAL MATPLOTLIB STYLE
# =============================================================================
def apply_style():
    """Apply clean scientific style globally."""
    style = {
        # Font
        'font.family':       'sans-serif',
        'font.sans-serif':   ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size':         10,
        'axes.labelsize':    15,
        'axes.titlesize':    13,
        'xtick.labelsize':   15,
        'ytick.labelsize':   15,
        'legend.fontsize':   11,
        # Use matplotlib mathtext (do not require external LaTeX)
        'text.usetex':        False,

        # Lines / markers
        'lines.linewidth':   1.5,
        'lines.markersize':  4,

        # Axes
        'axes.linewidth':    0.8,
        'axes.spines.top':   True,
        'axes.spines.right': True,

        # Ticks
        'xtick.direction':   'in',
        'ytick.direction':   'in',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,

        # Grid — off by default, enable per-figure
        'axes.grid':         False,

        # Legend
        'legend.frameon':       True,
        'legend.framealpha':    0.9,
        'legend.edgecolor':     '0.8',
        'legend.borderpad':     0.4,
        'legend.handlelength':  1.5,

        # Figure
        'figure.dpi':       150,
        'savefig.dpi':      300,
        'savefig.bbox':     'tight',
        'savefig.pad_inches': 0.05,

        # Image / colorbar
        'image.cmap': 'viridis',
    }
    mpl.rcParams.update(style)


# =============================================================================
# AXIS HELPERS
# =============================================================================
def ensure_all_spines(ax):
    """Ensure all four axis spines are visible and ticks appear on both sides."""
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.tick_params(which='both', top=True, right=True, direction='in')


def latexify_ticks(ax):
    """Format major tick labels as mathtext so they render as LaTeX-style numbers.

    Uses a FuncFormatter to wrap numeric labels in $...$ so matplotlib's mathtext
    renders them consistently with axis labels.
    """
    from matplotlib.ticker import FuncFormatter

    fmt = FuncFormatter(lambda x, pos: f'${x:g}$')
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)


# =============================================================================
# NETWORK PLOTTING HELPER
# =============================================================================
def plot_network(network, edge_values, ax, cmap=None, vmin=None, vmax=None,
                 use_log=False, linewidth=2.0, node_size=8, node_color='k',
                 show_colorbar=True, colorbar_label=None):
    """
    Draw network with edges colored by a per-edge scalar.

    Parameters
    ----------
    network : ElasticNetwork
    edge_values : (E,) array
    ax : matplotlib Axes
    cmap : colormap (default: network_cmap = cmocean matter_r)
    vmin, vmax : colorbar limits (auto if None)
    use_log : use LogNorm for coloring
    show_colorbar : add colorbar
    colorbar_label : text for colorbar label

    Returns
    -------
    sm : ScalarMappable (for external colorbar control)
    """
    if cmap is None:
        cmap = network_cmap

    positions = np.asarray(network.positions)
    edges = np.asarray(network.edges)
    values = np.asarray(edge_values, dtype=float)

    if vmin is None:
        vmin = values.min()
    if vmax is None:
        vmax = values.max()

    if use_log:
        norm = LogNorm(vmin=max(vmin, 1e-10), vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        if vmin < 0 and vmax > 0:
            # Center diverging colormap at zero
            norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Build line segments
    segments = []
    for i, j in edges:
        segments.append([positions[i], positions[j]])

    lc = mc.LineCollection(segments, cmap=cmap, norm=norm, linewidths=linewidth)
    lc.set_array(values)
    ax.add_collection(lc)

    # Nodes
    ax.scatter(positions[:, 0], positions[:, 1], c=node_color, s=node_size,
               zorder=3, linewidths=0)

    ax.set_aspect('equal')

    # Auto-limits with small pad
    pad = 0.03
    xmin, xmax = positions[:, 0].min(), positions[:, 0].max()
    ymin, ymax = positions[:, 1].min(), positions[:, 1].max()
    dx = (xmax - xmin) * pad
    dy = (ymax - ymin) * pad
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(ymin - dy, ymax + dy)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    if show_colorbar:
        cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        if colorbar_label:
            cb.set_label(colorbar_label)

    return sm


# =============================================================================
# UTILITY: subtask color / label helpers
# =============================================================================
def subtask_color(idx):
    """Return color for subtask index (wraps around)."""
    return SUBTASK_COLORS[idx % len(SUBTASK_COLORS)]


def subtask_label(idx, task_config=None):
    """
    Human-readable subtask label.
    If task_config supplied, includes strain/poisson values.
    """
    if task_config is not None:
        cs = task_config['compression_strains'][idx]
        tp = task_config['target_poisson_ratios'][idx]
        return rf'Subtask {idx} ($\epsilon$={cs:.2f}, $\nu^*$={tp:.2f})'
    return f'Subtask {idx}'
