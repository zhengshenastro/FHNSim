"""
Analyser.py
-----------
Post-simulation analysis and plotting for the 2D FHN system.
Works with any completed FHNSimulation object.

Plots
-----
  plot_snapshot            : 2D heatmap of u or v at a saved time index
  plot_spacetime_slice     : (x,t) heatmap along a fixed y-slice
  plot_dispersion          : growth rate Re(λ) vs wavenumber k
  plot_mass                : total mass ∫∫u dA over time
  plot_power_spectrum      : radially averaged |û(k)|² — dominant wavelength
  plot_dominant_wavenumber : k*(t) over time — coarsening vs band locking
  plot_nullclines          : f=0 and g=0 curves in (u,v) phase plane
  plot_phase_portrait      : u(t) vs v(t) at a single grid point

  — Dynamics comparison (wave speed / instability / robustness) —
  plot_wave_speed          : space-time slice + 1D front profiles + fitted speed
  plot_instability_onset   : dispersion with unstable band shaded and λ* annotated
  plot_robustness          : static method — snapshot grid + k* vs swept param

  — Panel methods —
  plot_comparison          : 4x3 panel for two models
  plot_phase_plane         : nullclines + phase portraits for two models
  plot_dynamics_comparison : 3-row panel covering all three dynamics questions
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
from scipy.optimize import fsolve
from FHNmodel import RegularFHN, MassConservedFHN


class FHNAnalyser:
    """
    Parameters
    ----------
    simulation : completed FHNSimulation instance
    """

    def __init__(self, simulation):
        self.sim   = simulation
        self.model = simulation.model

    # ── Linear stability ──────────────────────────────────────────────────────

    def steady_state(self):
        """Find homogeneous steady state (u*, v*) numerically."""
        m = self.model

        def equations(vars):
            u, v = vars
            return [m.f(u, v), m.g(u, v)]

        u_star, v_star = fsolve(equations, [0.0, 0.0])
        return u_star, v_star

    def jacobian(self, u0, v0):
        """Jacobian of (f, ε·g) at (u0, v0)."""
        m = self.model
        return np.array([
            [1 - u0**2,  -1              ],
            [m.epsilon,  -m.epsilon * m.b]
        ])

    def dispersion_relation(self, u0=None, v0=None, k_max=5.0, n_k=300):
        """
        Max real eigenvalue (growth rate) vs wavenumber k.

        Regular FHN     : M(k) = J − diag(Du·k², Dv·k²)
        Mass-conserved  : M(k) = k²·J − diag(Du·k⁴, Dv·k⁴)

        Returns k_vals, growth_rates.
        """
        if u0 is None or v0 is None:
            u0, v0 = self.steady_state()

        m  = self.model
        J  = self.jacobian(u0, v0)
        k_vals = np.linspace(0, k_max, n_k)
        growth_rates = []

        for k in k_vals:
            k2, k4 = k**2, k**4
            if isinstance(m, RegularFHN):
                M = J - np.diag([m.Du * k2, m.Dv * k2])
            elif isinstance(m, MassConservedFHN):
                M = k2 * J - np.diag([m.Du * k4, m.Dv * k4])
            growth_rates.append(np.max(np.linalg.eigvals(M).real))

        return k_vals, np.array(growth_rates)

    # ── Spectral diagnostics ──────────────────────────────────────────────────

    def radial_power_spectrum(self, t_idx=-1):
        """
        Radially averaged power spectrum |û(k)|² at a saved time index.

        Collapses the 2D Fourier plane onto 1D by binning by |k|.
        Peak identifies the dominant spatial wavelength λ* = 2π/k*.

        Returns k_bins, power.
        """
        u      = self.sim.u_history[t_idx]
        u_hat  = fft.fftn(u)
        p2d    = np.abs(fft.fftshift(u_hat))**2

        kx = fft.fftshift(fft.fftfreq(self.sim.resx, d=self.sim.dx)) * 2 * np.pi
        ky = fft.fftshift(fft.fftfreq(self.sim.resy, d=self.sim.dy)) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K = np.sqrt(KX**2 + KY**2)

        k_max  = min(kx.max(), ky.max())
        n_bins = 80
        k_bins = np.linspace(0, k_max, n_bins + 1)
        power  = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (K >= k_bins[i]) & (K < k_bins[i + 1])
            if mask.any():
                power[i] = p2d[mask].mean()

        return 0.5 * (k_bins[:-1] + k_bins[1:]), power

    def dominant_wavenumber(self):
        """
        Track dominant wavenumber k*(t) over all saved frames.

        Regular FHN     : k* stabilises quickly (Turing band is fixed).
        Mass-Conserved  : k* drifts as t^(-1/3) during Cahn-Hilliard coarsening.

        Returns t, k_stars.
        """
        k_stars = []
        for t_idx in range(len(self.sim.t_history)):
            k_bins, power = self.radial_power_spectrum(t_idx)
            power[0] = 0                           # suppress DC
            k_stars.append(k_bins[np.argmax(power)])
        return self.sim.t_history, np.array(k_stars)

    # ── Phase-plane diagnostics ───────────────────────────────────────────────

    def nullclines(self, u_range=(-2.5, 2.5), n=400):
        """
        Nullcline curves in (u,v) phase space.

        f(u,v)=0  →  v = u − u³/3        (u-nullcline, cubic)
        g(u,v)=0  →  v = (u + a)/b       (v-nullcline, linear)

        Returns u_vals, v_f_null, v_g_null.
        """
        m        = self.model
        u_vals   = np.linspace(*u_range, n)
        v_f_null = u_vals - u_vals**3 / 3
        v_g_null = (u_vals + m.a) / m.b
        return u_vals, v_f_null, v_g_null

    # ── Mass diagnostic ───────────────────────────────────────────────────────

    def check_mass_conservation(self):
        """
        Total mass ∫∫u dA at each saved step.

        Mass-Conserved : constant to floating-point precision.
        Regular FHN    : drifts freely — not conserved by design.
        """
        return self.sim.mass()

    # ── Wave speed and shape ──────────────────────────────────────────────────

    def estimate_wave_speed(self, field='u', y_idx=None):
        """
        Estimate wave propagation speed from the space-time slice.

        Tracks the steepest-gradient position along x at each saved time,
        then fits a line.  Slope = wave speed c.

        Regular FHN     : clear nonzero slope — fronts propagate.
        Mass-Conserved  : slope ≈ 0 — domain walls drift but do not travel.

        Returns t, front_x, speed.
        """
        data  = self.sim.u_history if field == 'u' else self.sim.v_history
        y_idx = y_idx or self.sim.resy // 2
        x     = np.linspace(0, self.sim.sizex, self.sim.resx)
        t     = self.sim.t_history

        front_x = []
        for frame in data:
            grad = np.abs(np.gradient(frame[:, y_idx], x))
            front_x.append(x[np.argmax(grad)])
        front_x = np.array(front_x)

        i0    = len(t) // 5                        # skip early transient
        coeff = np.polyfit(t[i0:], front_x[i0:], 1)
        speed = coeff[0]

        return t, front_x, speed

    def plot_wave_speed(self, ax_st=None, ax_front=None):
        """
        Two-panel wave-speed diagram.

        Left  : space-time (x,t) slice with fitted front trajectory overlaid.
                Diagonal stripes → propagating wave; slope = speed c.
        Right : 1D cross-sections at early / mid / final time.
                Sharp, translating profiles → Regular FHN.
                Stationary, broadening profiles → Mass-Conserved FHN.
        """
        if ax_st is None or ax_front is None:
            _, (ax_st, ax_front) = plt.subplots(1, 2, figsize=(12, 4))

        self.plot_spacetime_slice(ax=ax_st)

        t, front_x, speed = self.estimate_wave_speed()
        i0  = len(t) // 5
        fit = np.polyval(np.polyfit(t[i0:], front_x[i0:], 1), t)
        ax_st.plot(fit, t, 'r--', lw=1.5, label=f'c = {speed:.3f} L/t')
        ax_st.legend(fontsize=9)

        y_idx   = self.sim.resy // 2
        x       = np.linspace(0, self.sim.sizex, self.sim.resx)
        indices = [0, len(self.sim.t_history) // 2, -1]
        for idx in indices:
            u1d = self.sim.u_history[idx, :, y_idx]
            ax_front.plot(x, u1d, lw=1.5,
                          label=f't = {self.sim.t_history[idx]:.0f}')

        ax_front.set_xlabel('x')
        ax_front.set_ylabel('u(x)')
        ax_front.set_title(f'Front profiles — {self.model.__class__.__name__}')
        ax_front.legend(fontsize=9)

        return ax_st, ax_front

    # ── Instability onset and pattern wavelength ──────────────────────────────

    def plot_instability_onset(self, ax=None):
        """
        Dispersion relation annotated with the unstable band and λ*.

        Regular FHN : shades Re(λ) > 0 band in red; marks k* and λ* = 2π/k*.
                      The width of the band sets how many wavelengths are
                      simultaneously unstable (narrow → single-mode pattern).

        MC-FHN      : no band crosses zero (Type-II, k=0 always neutral);
                      marks the least-damped k for comparison.  The steep k⁴
                      fall-off means fine-scale modes are heavily suppressed.
        """
        k, gr = self.dispersion_relation()

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))

        ax.plot(k, gr, lw=2, color='steelblue')
        ax.axhline(0, color='k', lw=0.8, ls='--')

        unstable = gr > 0
        if unstable.any():
            ax.fill_between(k, gr, 0, where=unstable,
                            alpha=0.25, color='red', label='Unstable band')
            k_star   = k[np.argmax(gr)]
            lam_star = 2 * np.pi / k_star if k_star > 0 else np.inf
            ax.axvline(k_star, color='red', lw=1.2, ls=':',
                       label=f'k* = {k_star:.2f}  →  λ* = {lam_star:.1f}')
        else:
            gr_copy     = gr.copy(); gr_copy[0] = -np.inf
            k_least     = k[np.argmax(gr_copy)]
            lam_least   = 2 * np.pi / k_least if k_least > 0 else np.inf
            ax.axvline(k_least, color='orange', lw=1.2, ls=':',
                       label=f'least-damped k = {k_least:.2f}  (λ = {lam_least:.1f})')

        ax.set_xlabel('Wavenumber k')
        ax.set_ylabel('Growth rate Re(λ)')
        ax.set_title(f'Instability onset — {self.model.__class__.__name__}')
        ax.legend(fontsize=9)
        return ax

    # ── Robustness to parameter changes ──────────────────────────────────────

    @staticmethod
    def plot_robustness(sweep_results, param_name, fig=None):
        """
        Visualise a parameter sweep (run in main) as a snapshot grid + k* plot.

        Parameters
        ----------
        sweep_results : list of dicts, each with:
                          'param_value' : numeric value of the swept parameter
                          'sim'         : completed FHNSimulation
                          'label'       : string for subplot title
        param_name    : x-axis label for the k* vs param plot
        fig           : optional existing Figure; created if None

        Top row   : final u(x,y) snapshot for each parameter value.
                    Shows visually how patterns coarsen / sharpen / vanish.
        Bottom    : dominant k* at final time vs parameter value.
                    k* → 0 means pattern wavelength diverges (pattern collapse).
        """
        n = len(sweep_results)

        if fig is None:
            fig = plt.figure(figsize=(3.5 * n, 7))

        gs         = fig.add_gridspec(2, n, height_ratios=[3, 1],
                                      hspace=0.45, wspace=0.3)
        axes_snap  = [fig.add_subplot(gs[0, i]) for i in range(n)]
        ax_kstar   = fig.add_subplot(gs[1, :])

        param_vals, k_stars = [], []

        for i, res in enumerate(sweep_results):
            sim   = res['sim']
            label = res['label']
            pval  = res['param_value']

            # Snapshot
            axes_snap[i].imshow(
                sim.u_history[-1].T, origin='lower',
                extent=[0, sim.sizex, 0, sim.sizey],
                cmap='RdBu_r', aspect='equal'
            )
            axes_snap[i].set_title(label, fontsize=9)
            axes_snap[i].axis('off')

            # Dominant k* from final power spectrum
            ana          = FHNAnalyser(sim)
            k_bin, power = ana.radial_power_spectrum(t_idx=-1)
            power[0]     = 0                       # suppress DC
            k_stars.append(k_bin[np.argmax(power)])
            param_vals.append(pval)

        ax_kstar.plot(param_vals, k_stars, 'o-', lw=2)
        ax_kstar.set_xlabel(param_name)
        ax_kstar.set_ylabel('Dominant k*')
        ax_kstar.set_title(f'Pattern wavelength vs {param_name}  '
                           f'(k*→0 means pattern collapses)')

        return fig

    # ── Individual baseline plots ─────────────────────────────────────────────

    def plot_snapshot(self, field='u', t_idx=-1, ax=None):
        """2D heatmap of u or v at a given saved time index."""
        data = self.sim.u_history if field == 'u' else self.sim.v_history
        t    = self.sim.t_history[t_idx]

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))

        im = ax.imshow(
            data[t_idx].T, origin='lower',
            extent=[0, self.sim.sizex, 0, self.sim.sizey],
            cmap='RdBu_r', aspect='equal'
        )
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(f'{self.model.__class__.__name__} — {field}(x,y,t={t:.1f})')
        plt.colorbar(im, ax=ax)
        return ax

    def plot_spacetime_slice(self, field='u', y_idx=None, ax=None):
        """
        Space-time (x,t) heatmap along a fixed y-slice.

        Diagonal stripes → wave propagation (Regular FHN).
        Vertical or blurry features → stationary domains (Mass-Conserved).
        """
        data  = self.sim.u_history if field == 'u' else self.sim.v_history
        y_idx = y_idx or self.sim.resy // 2

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        im = ax.imshow(
            data[:, :, y_idx], aspect='auto',
            extent=[0, self.sim.sizex, self.sim.t_history[-1], 0],
            cmap='RdBu_r'
        )
        ax.set_xlabel('x'); ax.set_ylabel('t')
        ax.set_title(f'{self.model.__class__.__name__} — {field}(x,y={y_idx},t)')
        plt.colorbar(im, ax=ax)
        return ax

    def plot_dispersion(self, ax=None):
        """Growth rate Re(λ) vs wavenumber k (unannotated version)."""
        k, gr = self.dispersion_relation()

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))

        ax.plot(k, gr, lw=2)
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_xlabel('Wavenumber k')
        ax.set_ylabel('Growth rate Re(λ)')
        ax.set_title(f'Dispersion — {self.model.__class__.__name__}')
        return ax

    def plot_mass(self, ax=None):
        """Total mass ∫∫u dA over time."""
        mass = self.check_mass_conservation()

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 3))

        ax.plot(self.sim.t_history, mass, lw=2)
        ax.set_xlabel('Time'); ax.set_ylabel('Total mass ∫∫u dA')
        ax.set_title(f'Mass — {self.model.__class__.__name__}')
        return ax

    def plot_power_spectrum(self, t_idx=-1, ax=None):
        """Radially averaged |û(k)|² with peak k* annotated."""
        k_bins, power = self.radial_power_spectrum(t_idx)
        t = self.sim.t_history[t_idx]

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))

        ax.plot(k_bins, power, lw=2)
        k_star = k_bins[np.argmax(power[1:]) + 1]
        ax.axvline(k_star, color='r', ls='--', lw=1,
                   label=f'k* = {k_star:.2f}  (λ* = {2*np.pi/k_star:.1f})')
        ax.set_xlabel('Wavenumber k  [rad / length]')
        ax.set_ylabel('Mean power |û|²')
        ax.set_title(f'Power spectrum — {self.model.__class__.__name__}  (t={t:.0f})')
        ax.legend(fontsize=9)
        return ax

    def plot_dominant_wavenumber(self, ax=None):
        """k*(t) over time — plateau vs coarsening drift."""
        t, k_stars = self.dominant_wavenumber()

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))

        ax.plot(t, k_stars, lw=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Dominant wavenumber k*')
        ax.set_title(f'Coarsening — {self.model.__class__.__name__}')
        return ax

    def plot_nullclines(self, ax=None):
        """f=0 and g=0 nullclines in (u,v) phase space."""
        u_vals, v_f, v_g = self.nullclines()
        u_star, v_star   = self.steady_state()

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))

        ax.plot(u_vals, v_f, lw=2, label='f = 0  (u-nullcline)')
        ax.plot(u_vals, v_g, lw=2, label='g = 0  (v-nullcline)')
        ax.plot(u_star, v_star, 'ko', ms=7,
                label=f'SS ({u_star:.2f}, {v_star:.2f})')
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.0, 2.0)
        ax.set_xlabel('u'); ax.set_ylabel('v')
        ax.set_title(f'Nullclines — {self.model.__class__.__name__}')
        ax.legend(fontsize=9)
        return ax

    def plot_phase_portrait(self, ix=None, iy=None, ax=None):
        """(u,v) time trace at a single grid point, coloured by time."""
        ix = ix or self.sim.resx // 2
        iy = iy or self.sim.resy // 2

        u_trace = self.sim.u_history[:, ix, iy]
        v_trace = self.sim.v_history[:, ix, iy]

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))

        sc = ax.scatter(u_trace, v_trace, c=self.sim.t_history,
                        cmap='viridis', s=6)
        plt.colorbar(sc, ax=ax, label='time')
        ax.set_xlabel('u'); ax.set_ylabel('v')
        ax.set_title(f'Phase portrait — {self.model.__class__.__name__}'
                     f'  (ix={ix}, iy={iy})')
        return ax

    # ── Panel methods ─────────────────────────────────────────────────────────

    def plot_comparison(self, other):
        """
        4×3 panel: snapshot | dispersion | mass
                   space-time | power spectrum | dominant k*
        for this model (rows 0-1) and other (rows 2-3).
        """
        fig, axes = plt.subplots(4, 3, figsize=(16, 20))
        fig.suptitle('Regular vs Mass-Conserved FHN (2D Spectral)', fontsize=14)

        for row, ana in enumerate([self, other]):
            r = row * 2
            ana.plot_snapshot(ax=axes[r, 0])
            ana.plot_dispersion(ax=axes[r, 1])
            ana.plot_mass(ax=axes[r, 2])

            r = row * 2 + 1
            ana.plot_spacetime_slice(ax=axes[r, 0])
            ana.plot_power_spectrum(ax=axes[r, 1])
            ana.plot_dominant_wavenumber(ax=axes[r, 2])

        plt.tight_layout()
        return fig

    def plot_phase_plane(self, other):
        """1×4 panel: nullclines + phase portraits for both models."""
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        fig.suptitle('Phase plane — Regular vs Mass-Conserved FHN', fontsize=13)

        self.plot_nullclines(ax=axes[0])
        self.plot_phase_portrait(ax=axes[1])
        other.plot_nullclines(ax=axes[2])
        other.plot_phase_portrait(ax=axes[3])

        plt.tight_layout()
        return fig

    def plot_dynamics_comparison(self, other,
                                 reg_sweep, mc_sweep,
                                 sweep_param_name):
        """
        3-row panel covering all three dynamics questions.

        Row 0 — Wave speed and shape
            Col 0-1 : Regular FHN   space-time + front profiles
            Col 2-3 : MC-FHN        space-time + front profiles

        Row 1 — Onset of instability and pattern wavelength
            Col 0-1 : Regular FHN   annotated dispersion + power spectrum
            Col 2-3 : MC-FHN        annotated dispersion + power spectrum

        Row 2 — Robustness to parameter changes
            Full-width : snapshot grid + k* vs swept parameter
            (two sub-figures, one per model, stacked)

        Parameters
        ----------
        other            : FHNAnalyser for the second model
        reg_sweep        : list of sweep_result dicts for Regular FHN
        mc_sweep         : list of sweep_result dicts for MC-FHN
        sweep_param_name : parameter name string (e.g. 'Du')
        """
        # ── Row 0 & 1: 4-column layout ────────────────────────────────────
        fig_top, axes = plt.subplots(2, 4, figsize=(20, 9))
        fig_top.suptitle(
            'Dynamics comparison — Regular vs Mass-Conserved FHN', fontsize=14)

        # Row 0: wave speed
        self.plot_wave_speed(ax_st=axes[0, 0], ax_front=axes[0, 1])
        other.plot_wave_speed(ax_st=axes[0, 2], ax_front=axes[0, 3])
        axes[0, 0].set_title(f'RegularFHN — space-time', fontsize=10)
        axes[0, 2].set_title(f'MC-FHN — space-time', fontsize=10)

        # Row 1: instability onset
        self.plot_instability_onset(ax=axes[1, 0])
        self.plot_power_spectrum(ax=axes[1, 1])
        other.plot_instability_onset(ax=axes[1, 2])
        other.plot_power_spectrum(ax=axes[1, 3])

        fig_top.tight_layout()

        # ── Row 2: robustness (separate figures, one per model) ───────────
        fig_reg = FHNAnalyser.plot_robustness(
            reg_sweep, sweep_param_name)
        fig_reg.suptitle(
            f'Robustness — RegularFHN  (sweep: {sweep_param_name})',
            fontsize=12)

        fig_mc = FHNAnalyser.plot_robustness(
            mc_sweep, sweep_param_name)
        fig_mc.suptitle(
            f'Robustness — MassConservedFHN  (sweep: {sweep_param_name})',
            fontsize=12)

        return fig_top, fig_reg, fig_mc