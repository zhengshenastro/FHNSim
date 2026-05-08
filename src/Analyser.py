"""
FHNAnalyser.py
--------------
Post-simulation analysis and plotting for the 2D FHN system.
Works with any completed FHNSimulation object.

Covers Steps 2-4 of the project:
  - Linear stability / Jacobian
  - Dispersion relation
  - 2D snapshot plots
  - Space-time slice plots
  - Mass conservation diagnostics
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from FHNmodel import RegularFHN, MassConservedFHN


class FHNAnalyser:
    """
    Parameters
    ----------
    simulation : completed FHNSimulation instance
    """

    def __init__(self, simulation):
        self.sim = simulation
        self.model = simulation.model

    # ── Step 2: Linear stability ─────────────────────────────────────────────

    def steady_state(self):
        """Find homogeneous steady state (u*, v*) numerically."""
        m = self.model

        def equations(vars):
            u, v = vars
            return [m.f(u, v), m.g(u, v)]

        u_star, v_star = fsolve(equations, [0.0, 0.0])
        return u_star, v_star

    def jacobian(self, u0, v0):
        """
        Jacobian of (f, ε·g) at steady state (u0, v0).

            J = [[ 1 - u0²,    -1       ],
                 [ ε,           -ε·b    ]]
        """
        m = self.model
        return np.array([
            [1 - u0**2,       -1           ],
            [m.epsilon,       -m.epsilon * m.b]
        ])

    def dispersion_relation(self, u0=None, v0=None, k_max=5.0, n_k=300):
        """
        Compute max real eigenvalue (growth rate) vs wavenumber k.

        Regular FHN       : M(k) = J - diag(Du·k², Dv·k²)
        Mass-conserved    : M(k) = k²·J - diag(Du·k⁴, Dv·k⁴)

        Returns
        -------
        k_vals      : wavenumber array
        growth_rates: max Re(λ) at each k
        """
        if u0 is None or v0 is None:
            u0, v0 = self.steady_state()

        m = self.model
        J = self.jacobian(u0, v0)
        k_vals = np.linspace(0, k_max, n_k)
        growth_rates = []

        for k in k_vals:
            k2 = k**2
            k4 = k**4

            if isinstance(m, RegularFHN):
                M = J - np.diag([m.Du * k2, m.Dv * k2])
            elif isinstance(m, MassConservedFHN):
                M = k2 * J - np.diag([m.Du * k4, m.Dv * k4])

            eigvals = np.linalg.eigvals(M)
            growth_rates.append(np.max(eigvals.real))

        return k_vals, np.array(growth_rates)

    # ── Step 3 & 4: Diagnostics ──────────────────────────────────────────────

    def check_mass_conservation(self):
        """Return total mass at each saved step."""
        return self.sim.mass()

    # ── Plotting ─────────────────────────────────────────────────────────────

    def plot_snapshot(self, field='u', t_idx=-1, ax=None):
        """
        2D heatmap of u or v at a given saved time index.

        Parameters
        ----------
        field : 'u' or 'v'
        t_idx : index into saved history (-1 = final state)
        """
        data = self.sim.u_history if field == 'u' else self.sim.v_history
        t = self.sim.t_history[t_idx]

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))

        im = ax.imshow(
            data[t_idx].T,
            origin='lower',
            extent=[0, self.sim.sizex, 0, self.sim.sizey],
            cmap='RdBu_r',
            aspect='equal'
        )
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{self.model.__class__.__name__} — {field}(x,y,t={t:.1f})')
        plt.colorbar(im, ax=ax)
        return ax

    def plot_spacetime_slice(self, field='u', y_idx=None, ax=None):
        """
        Space-time (x, t) heatmap along a fixed y-slice.
        Useful for tracking wave propagation.

        Parameters
        ----------
        field : 'u' or 'v'
        y_idx : y grid index for the slice (default: midpoint)
        """
        data = self.sim.u_history if field == 'u' else self.sim.v_history
        if y_idx is None:
            y_idx = self.sim.resy // 2

        slice_data = data[:, :, y_idx]          # shape: (n_saved, resx)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        im = ax.imshow(
            slice_data,
            aspect='auto',
            extent=[0, self.sim.sizex,
                    self.sim.t_history[-1], 0],
            cmap='RdBu_r'
        )
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_title(
            f'{self.model.__class__.__name__} — {field}(x, y={y_idx}, t)'
        )
        plt.colorbar(im, ax=ax)
        return ax

    def plot_dispersion(self, ax=None):
        """Plot growth rate Re(λ) vs wavenumber k."""
        k, gr = self.dispersion_relation()

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(k, gr, lw=2)
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_xlabel('Wavenumber k')
        ax.set_ylabel('Growth rate Re(λ)')
        ax.set_title(f'Dispersion — {self.model.__class__.__name__}')
        return ax

    def plot_mass(self, ax=None):
        """Show mass conservation (or drift) over time."""
        mass = self.check_mass_conservation()

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 3))

        ax.plot(self.sim.t_history, mass, lw=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Total mass ∫∫u dA')
        ax.set_title(f'Mass — {self.model.__class__.__name__}')
        return ax

    def plot_comparison(self, other):
        """
        Side-by-side comparison of this simulation against another.

        Parameters
        ----------
        other : FHNAnalyser instance for the second model
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle('Regular vs Mass-Conserved FHN (2D Spectral)', fontsize=13)

        self.plot_snapshot(ax=axes[0, 0])
        other.plot_snapshot(ax=axes[1, 0])

        self.plot_dispersion(ax=axes[0, 1])
        other.plot_dispersion(ax=axes[1, 1])

        self.plot_mass(ax=axes[0, 2])
        other.plot_mass(ax=axes[1, 2])

        plt.tight_layout()
        return fig