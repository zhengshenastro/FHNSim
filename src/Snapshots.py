"""
Snapshots.py
------------
Time-series snapshot plotting utilities for FHN simulations.

Update (Scheme A):
- Use one unified color scale across all snapshots (global vmin/vmax)
- Optionally use robust percentiles to avoid outliers dominating the scale
- Use a single shared colorbar for the whole figure
- Keep backwards compatibility with the old call signature
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_time_series_snapshots(
    analyser,
    sim,
    n_snap: int = 15,
    filename: str | None = None,
    title: str | None = None,
    *,
    field: str = "u",
    global_scale: str = "history",
    robust: float | None = None,
    cmap: str = "RdBu_r",
    aspect: str = "equal",
):
    """
    Plot a grid of time-evolution snapshots for a given simulation with a unified scale.

    Parameters
    ----------
    analyser : FHNAnalyser instance (only used for model name in title)
    sim      : FHNSimulation instance
    n_snap   : number of snapshots (default: 15)
    filename : optional save path
    title    : figure title

    Keyword-only
    ------------
    field : {"u","v"}
        Which field to plot.
    global_scale : {"history","selected"}
        "history"  -> vmin/vmax computed from the entire saved history
        "selected" -> vmin/vmax computed only from the selected snapshot frames
    robust : float or None
        If not None, use percentile clipping for vmin/vmax:
          vmin = p[robust], vmax = p[100-robust]
        Example: robust=0.5 -> [0.5%, 99.5%]
    cmap : matplotlib colormap name
    aspect : imshow aspect ("equal" recommended for square domains)
    """
    if field not in ("u", "v"):
        raise ValueError("field must be 'u' or 'v'")

    data_all = sim.u_history if field == "u" else sim.v_history
    if data_all is None or len(data_all) == 0:
        raise ValueError("Simulation has no stored history (u_history/v_history is empty).")

    # Select indices uniformly in time
    indices = np.linspace(0, len(sim.t_history) - 1, n_snap, dtype=int)

    # Decide which data is used to compute the global scale
    if global_scale == "history":
        data_for_scale = data_all
    elif global_scale == "selected":
        data_for_scale = data_all[indices]
    else:
        raise ValueError("global_scale must be 'history' or 'selected'.")

    # Compute global vmin/vmax
    if robust is None:
        vmin = float(np.min(data_for_scale))
        vmax = float(np.max(data_for_scale))
    else:
        if not (0.0 < robust < 50.0):
            raise ValueError("robust must be between 0 and 50 (exclusive), e.g. 0.5 or 1.0.")
        lo, hi = robust, 100.0 - robust
        vmin, vmax = np.percentile(data_for_scale, [lo, hi]).astype(float)

    # Grid shape (auto)
    ncols = int(np.ceil(np.sqrt(n_snap)))
    nrows = int(np.ceil(n_snap / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)

    if title is None:
        title = f"{analyser.model.__class__.__name__} — {field}(x,y) Time Evolution"
    fig.suptitle(title, fontsize=14)

    last_im = None
    for i, idx in enumerate(indices):
        ax = axes[i]
        frame = data_all[idx]

        last_im = ax.imshow(
            frame.T,
            origin="lower",
            extent=[0, sim.sizex, 0, sim.sizey],
            cmap=cmap,
            aspect=aspect,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"t={sim.t_history[idx]:.1f}", fontsize=8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # Remove unused axes
    for j in range(len(indices), len(axes)):
        fig.delaxes(axes[j])

    # Single shared colorbar
    # Leave space on the right for a dedicated colorbar
    fig.subplots_adjust(right=0.90)

    # Dedicated colorbar axis: [left, bottom, width, height]
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.70])

    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label(field)


    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=150, bbox_inches="tight")

    plt.show()