"""
FHNSimulation.py
----------------
2D spectral simulation of the FitzHugh-Nagumo system.

Inherits Torus2D for grid setup and spectral operators.
Accepts any FHNBase subclass as the physical model.

Mirrors the ISF class structure:
  ISF.BuildSchrodinger  →  FHNSimulation.BuildSpectralMasks
  ISF.SchroedingerFlow  →  FHNSimulation.SpectralStep
"""

import numpy as np
import scipy.fft as fft

from Torus import Torus2D
from FHNmodel import RegularFHN, MassConservedFHN


class FHNSimulation(Torus2D):
    """
    Spectral time-stepping for the 2D FHN system on a periodic domain.

    Parameters
    ----------
    model       : RegularFHN or MassConservedFHN instance
    sizex/sizey : physical domain lengths
    resx/resy   : grid resolution (number of points)
    dt          : time step
    save_every  : store state every n steps
    """

    def __init__(self, model, sizex, sizey, resx, resy,
                 dt=0.05, save_every=20):
        super().__init__(sizex, sizey, resx, resy)

        self.model = model
        self.dt = dt
        self.save_every = save_every

        # Fields (set by set_initial_conditions)
        self.u = None
        self.v = None

        # History storage
        self.u_history = []
        self.v_history = []
        self.t_history = []

        # Spectral masks (set by BuildSpectralMasks)
        self.linear_mask_u = None
        self.linear_mask_v = None

    # ── Analogous to ISF.BuildSchrodinger ───────────────────────────────────

    def BuildSpectralMasks(self):
        """
        Precompute the linear (diffusion) part of each equation in Fourier
        space.  The nonlinear reaction terms f and g are handled explicitly
        in real space each step (pseudo-spectral approach).

        Regular FHN      : linear part is  -Du·k²  and  -Dv·k²
        Mass-conserved   : linear part is  -Du·k⁴  and  -Dv·k⁴
          (the extra -∇² wrapping raises the diffusion to 4th order)

        Using an integrating-factor (exponential) approach for the linear
        term improves stability over naive Euler — important for the stiff
        k⁴ operator in the mass-conserved model.
        """
        m = self.model

        if isinstance(m, RegularFHN):
            # exp(-Du·k²·dt)  and  exp(-Dv·k²·dt)
            self.linear_mask_u = np.exp(-m.Du * self.k2 * self.dt)
            self.linear_mask_v = np.exp(-m.Dv * self.k2 * self.dt)

        elif isinstance(m, MassConservedFHN):
            # exp(-Du·k⁴·dt)  and  exp(-Dv·k⁴·dt)
            self.linear_mask_u = np.exp(-m.Du * self.k4 * self.dt)
            self.linear_mask_v = np.exp(-m.Dv * self.k4 * self.dt)

        else:
            raise TypeError(f"Unknown model type: {type(m)}")

    # ── Analogous to ISF.SchroedingerFlow ───────────────────────────────────

    def SpectralStep(self):
        """
        Advance u and v by one time step using a pseudo-spectral
        exponential-time-differencing (ETD) scheme.

        Linear diffusion  : treated exactly via spectral masks (BuildSpectralMasks)
        Nonlinear reaction: treated explicitly in real space

        Regular FHN:
            û_{n+1} = mask_u · ( û_n  +  dt · f̂(u,v) )
            v̂_{n+1} = mask_v · ( v̂_n  +  dt · ε·ĝ(u,v) )

        Mass-conserved FHN:
            û_{n+1} = mask_u · ( û_n  +  dt · k²·f̂(u,v) )
            v̂_{n+1} = mask_v · ( v̂_n  +  dt · ε·k²·ĝ(u,v) )
            (the k² factor applies the -∇² wrapping to the reaction term)
        """
        m = self.model

        # Transform current fields to Fourier space
        u_hat = fft.fftn(self.u)
        v_hat = fft.fftn(self.v)

        # Evaluate nonlinear terms in real space, then transform
        f_hat = fft.fftn(m.f(self.u, self.v))
        g_hat = fft.fftn(m.g(self.u, self.v))

        if isinstance(m, RegularFHN):
            # eqs. 8 & 9 in Fourier space
            u_hat_new = self.linear_mask_u * (u_hat + self.dt * f_hat)
            v_hat_new = self.linear_mask_v * (v_hat + self.dt * m.epsilon * g_hat)

        elif isinstance(m, MassConservedFHN):
            # eqs. 12 & 13 in Fourier space
            # -∇²[f + Du∇²u]  →  k²·f̂  (linear Du·k⁴ part absorbed into mask)
            u_hat_new = self.linear_mask_u * (u_hat + self.dt * self.k2 * f_hat)
            v_hat_new = self.linear_mask_v * (v_hat + self.dt * m.epsilon * self.k2 * g_hat)

        # Transform back to real space
        self.u = np.real(fft.ifftn(u_hat_new))
        self.v = np.real(fft.ifftn(v_hat_new))

    # ── Initial conditions ───────────────────────────────────────────────────

    def set_initial_conditions(self, u0=None, v0=None, init_type="random", seed=42):
        """
        Set initial fields.  Default: small random perturbation around (0,0).
        Custom arrays must match (resx, resy).
        """
        rng = np.random.default_rng(seed)
        shape = (self.resx, self.resy)
        x = self.px
        y = self.py

        if u0 is not None and v0 is not None:
            self.u = u0
            self.v = v0
            return

        if init_type == "random":
            self.u = 0.1 * rng.standard_normal(shape)
            self.v = 0.1 * rng.standard_normal(shape)

        elif init_type == "pulse":
            cx, cy = self.sizex / 2, self.sizey / 2
            r2 = (x - cx) ** 2 + (y - cy) ** 2
            self.u = np.exp(-r2 / 5.0)
            self.v = np.zeros_like(self.u)

        elif init_type == "front":
            self.u = np.where(x < self.sizex / 2, 1.0, -1.0)
            self.v = np.zeros_like(self.u)

        elif init_type == "sin":
            k = 2 * np.pi / self.sizex * 4
            self.u = 0.1 * np.sin(k * x)
            self.v = np.zeros_like(self.u)

        elif init_type == "biased":
            self.u = 0.5 + 0.1 * rng.standard_normal(shape)
            self.v = 0.0 + 0.1 * rng.standard_normal(shape)

        elif init_type == "spiral_cg":
            # Spiral-seeding IC (cross-gradient):
            # u has an x-gradient, v has a y-gradient. This creates a phase defect near the center.
            cx, cy = self.sizex / 2, self.sizey / 2
            A = 0.02  # try 0.005 ~ 0.05 depending on parameters
            self.u = A * (x - cx)
            self.v = A * (y - cy)

        elif init_type == "spiral_bw":
            # Spiral-seeding IC (broken wavefront):
            # Left half is excited; a notch breaks the front so the tips curl into a spiral.
            self.u = np.zeros(shape)
            self.v = np.zeros(shape)

        elif init_type == "spiral_pf":
            # Spiral-seeding IC (phase field):
            # u = u* + A cos(theta), v = v* + A sin(theta)
            cx, cy = self.sizex / 2, self.sizey / 2
            theta = np.arctan2(y - cy, x - cx)
            A = 0.2  # try 0.05 ~ 0.5
            self.u = A * np.cos(theta)
            self.v = A * np.sin(theta)

            # excite left half-plane
            self.u = np.where(x < self.sizex / 2, 1.0, 0.0)

            # notch (gap) near the center to create wave tips
            cx, cy = self.resx // 2, self.resy // 2
            wx = max(2, self.resx // 50)  # notch half-width in x (grid units)
            wy = max(6, self.resy // 10)  # notch half-height in y (grid units)
            self.u[cx - wx: cx + wx, cy - wy: cy + wy] = 0.0

        else:
            raise ValueError(f"Unknown init_type: {init_type}")


            self.u = np.sin(theta + r / 5)
            self.v = np.cos(theta + r / 5)


        #self.u = u0 if u0 is not None else 0.1 * rng.standard_normal(shape)
        #self.v = v0 if v0 is not None else 0.1 * rng.standard_normal(shape)

    # ── Time-stepping loop ───────────────────────────────────────────────────

    def run(self, T):
        """
        Run the simulation for total time T.

        Parameters
        ----------
        T : total simulation time
        """
        if self.linear_mask_u is None:
            self.BuildSpectralMasks()

        if self.u is None:
            self.set_initial_conditions()

        n_steps = int(T / self.dt)

        # ── NEW: store initial condition (t = 0) ───────────────────────────────
        self.u_history = [self.u.copy()]
        self.v_history = [self.v.copy()]
        self.t_history = [0.0]

        for step in range(n_steps):
            self.SpectralStep()

            if step % self.save_every == 0:
                self.u_history.append(self.u.copy())
                self.v_history.append(self.v.copy())
                self.t_history.append(step * self.dt)

        self.u_history = np.array(self.u_history)
        self.v_history = np.array(self.v_history)
        self.t_history = np.array(self.t_history)
        print(f"Done: {self.model} | {n_steps} steps")

    # ── Mass diagnostic ──────────────────────────────────────────────────────

    def mass(self):
        """Total mass ∫∫u dA at each saved time step."""
        return self.u_history.sum(axis=(1, 2)) * self.dx * self.dy