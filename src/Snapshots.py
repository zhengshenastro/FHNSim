"""
Snapshots.py
------------
Time-series snapshot plotting utilities for FHN simulations.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_time_series_snapshots(analyser, sim, n_snap=15, filename=None, title=None):
    """
    Plot a grid of time-evolution snapshots for a given simulation.

    Parameters
    ----------
    analyser : FHNAnalyser instance
    sim      : FHNSimulation instance
    n_snap   : number of snapshots (default: 15)
    filename : optional save path
    title    : figure title
    """

    # Select indices uniformly in time
    indices = np.linspace(
        0,
        len(sim.t_history) - 1,
        n_snap,
        dtype=int
    )

    # Grid shape (auto)
    ncols = int(np.ceil(np.sqrt(n_snap)))
    nrows = int(np.ceil(n_snap / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))

    # Flatten axes for easy indexing
    axes = np.array(axes).reshape(-1)

    if title is None:
        title = f"{analyser.model.__class__.__name__} — Time Evolution"

    fig.suptitle(title, fontsize=14)

    for i, idx in enumerate(indices):
        analyser.plot_snapshot(t_idx=idx, ax=axes[i])
        axes[i].set_title(f"t={sim.t_history[idx]:.1f}", fontsize=8)

    # Remove unused axes
    for j in range(len(indices), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=150)

    plt.show()