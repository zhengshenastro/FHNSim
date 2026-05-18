"""
FHNSimulation_grid2d.py
-----------------------
2D pseudo-spectral simulation of the FitzHugh–Nagumo (FHN) system
on a rectangular domain with **Neumann (zero-flux) boundary conditions**.

Grid:
  - Uses Grid2D (DCT-II / IDCT) so fields remain real and boundaries are reflecting.

Model:
  - Expects a model compatible with the "Simulation (1) + FHNmodel (2)" design:
      * model.build_masks(k2, k4, dt) -> (mask_u, mask_v)
      * model.spectral_update(u_hat, v_hat, f_hat, g_hat, k2, dt) -> (u_hat_new, v_hat_new)
    (Masks are typically stashed on the model as model.masks.)

Initial conditions:
  - Ports the richer init_type options from the original Simulation.py while
    keeping backwards compatibility with the older signature where the 3rd
    positional argument is `seed`.

Snapshot interface:
  - Keeps u_history, v_history, t_history as numpy arrays, compatible with the
    existing FHNAnalyser.plot_snapshot and Snapshots.py utilities.
"""

from __future__ import annotations

import numpy as np

from Grid2D import Grid2D  # make sure your file/module is named Grid2D.py
from FHNmodel import RegularFHN, MassConservedFHN


class FHNSimulation(Grid2D):
    """
    Parameters
    ----------
    model       : RegularFHN or MassConservedFHN instance (or compatible FHNBase)
    sizex/sizey : physical domain lengths
    resx/resy   : grid resolution (number of points)
    dt          : time step
    save_every  : store state every n steps
    store_initial : whether to store the initial condition as the first frame
    """

    def __init__(
        self,
        model,
        sizex: float,
        sizey: float,
        resx: int,
        resy: int,
        dt: float = 0.05,
        save_every: int = 20,
        store_initial: bool = True,
    ):
        super().__init__(sizex, sizey, resx, resy)

        self.model = model
        self.dt = float(dt)
        self.save_every = int(save_every)
        self.store_initial = bool(store_initial)

        # Fields (set by set_initial_conditions)
        self.u: np.ndarray | None = None
        self.v: np.ndarray | None = None

        # History storage
        self.u_history = []
        self.v_history = []
        self.t_history = []

        # Optional "legacy" attrs retained for compatibility (not used by model-driven update)
        self.linear_mask_u = None
        self.linear_mask_v = None

    # ── Linear masks / integrating factors ──────────────────────────────────

    def BuildSpectralMasks(self) -> None:
        """
        Ask the model to build its integrating-factor masks from k^2, k^4 and dt.
        """
        masks = self.model.build_masks(self.k2, self.k4, self.dt)
        # Stash on the model so its spectral_update can access them (matches your Simulation (1) pattern)
        self.model.masks = masks

    # ── One pseudo-spectral ETD step (DCT-based) ─────────────────────────────

    def SpectralStep(self) -> None:
        """
        Advance u and v by one time step using:
          - DCT for spectral transforms (Neumann BCs)
          - model.spectral_update for the ETD/integrating-factor update
        """
        if self.u is None or self.v is None:
            raise RuntimeError("Fields not initialised. Call set_initial_conditions() first.")

        # Transform current fields to DCT space
        u_hat = self.dct(self.u)
        v_hat = self.dct(self.v)

        # Nonlinear terms in real space, then DCT
        f_hat = self.dct(self.model.f(self.u, self.v))
        g_hat = self.dct(self.model.g(self.u, self.v))

        # Model-driven update in spectral space
        u_hat_new, v_hat_new = self.model.spectral_update(
            u_hat, v_hat, f_hat, g_hat, self.k2, self.dt
        )

        # Back to real space
        self.u = self.idct(u_hat_new)
        self.v = self.idct(v_hat_new)

    # ── Initial conditions (ported from original Simulation.py) ──────────────

    def set_initial_conditions(
        self,
        u0: np.ndarray | None = None,
        v0: np.ndarray | None = None,
        seed: int = 42,
        init_type: str = "random",
    ) -> None:
        """
        Set initial fields.

        Backwards-compatible signature:
            set_initial_conditions(u0=None, v0=None, seed=42, init_type="random")

        - If u0 and v0 are provided: use them directly (must match (resx, resy)).
        - Otherwise choose an init_type.

        Supported init_type:
            "random", "pulse", "front", "sin", "biased",
            "spiral_cg", "spiral_bw", "spiral_pf"
        """
        rng = np.random.default_rng(seed)
        shape = (self.resx, self.resy)
        x = self.px
        y = self.py

        if (u0 is not None) and (v0 is not None):
            u0 = np.asarray(u0)
            v0 = np.asarray(v0)
            if u0.shape != shape or v0.shape != shape:
                raise ValueError(f"Custom u0/v0 must have shape {shape}, got {u0.shape} and {v0.shape}.")
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
            # Step-like excited half-plane
            self.u = np.where(x < self.sizex / 2, 1.0, -1.0)
            self.v = np.zeros_like(self.u)

        elif init_type == "sin":
            # Single-mode sinusoidal perturbation
            k = 2 * np.pi / self.sizex * 4
            self.u = 0.1 * np.sin(k * x)
            self.v = np.zeros_like(self.u)

        elif init_type == "biased":
            self.u = 0.5 + 0.1 * rng.standard_normal(shape)
            self.v = 0.0 + 0.1 * rng.standard_normal(shape)

        elif init_type == "spiral_cg":
            # Spiral-seeding IC (cross-gradient): creates a phase defect near the center.
            cx, cy = self.sizex / 2, self.sizey / 2
            A = 0.02  # try 0.005 ~ 0.05 depending on parameters
            self.u = A * (x - cx)
            self.v = A * (y - cy)

        elif init_type == "spiral_bw":
            # Spiral-seeding IC (broken wavefront):
            # Left half excited; a notch breaks the front so the tips curl into a spiral.
            self.u = np.where(x < self.sizex / 2, 1.0, 0.0)
            self.v = np.zeros_like(self.u)

            # notch (gap) near the center to create wave tips
            cx, cy = self.resx // 2, self.resy // 2
            wx = max(2, self.resx // 50)   # notch half-width (grid units)
            wy = max(6, self.resy // 10)   # notch half-height (grid units)
            self.u[cx - wx: cx + wx, cy - wy: cy + wy] = 0.0

        elif init_type == "spiral_pf":
            # Spiral-seeding IC (phase field):
            # u = A cos(theta), v = A sin(theta), then excite half-plane and notch.
            cx, cy = self.sizex / 2, self.sizey / 2
            theta = np.arctan2(y - cy, x - cx)
            A = 0.2  # try 0.05 ~ 0.5
            self.u = A * np.cos(theta)
            self.v = A * np.sin(theta)

            # excite left half-plane
            self.u = np.where(x < self.sizex / 2, 1.0, 0.0)

            # notch (gap) near the center to create wave tips
            cx_i, cy_i = self.resx // 2, self.resy // 2
            wx = max(2, self.resx // 50)
            wy = max(6, self.resy // 10)
            self.u[cx_i - wx: cx_i + wx, cy_i - wy: cy_i + wy] = 0.0

        else:
            raise ValueError(f"Unknown init_type: {init_type}")

    # ── Time-stepping loop ───────────────────────────────────────────────────

    def run(self, T: float) -> None:
        """
        Run the simulation for total time T.

        Parameters
        ----------
        T : total simulation time
        """
        if getattr(self.model, "masks", None) is None:
            self.BuildSpectralMasks()

        if self.u is None or self.v is None:
            self.set_initial_conditions()

        n_steps = int(T / self.dt)

        # Optional: store the initial condition at t=0 for snapshot utilities
        if self.store_initial:
            self.u_history = [self.u.copy()]
            self.v_history = [self.v.copy()]
            self.t_history = [0.0]
            start_step = 1
        else:
            self.u_history = []
            self.v_history = []
            self.t_history = []
            start_step = 0

        for step in range(start_step, n_steps + 1):
            # If store_initial=True, step starts at 1 and represents time = step*dt
            self.SpectralStep()

            if (step % self.save_every) == 0:
                self.u_history.append(self.u.copy())
                self.v_history.append(self.v.copy())
                self.t_history.append(step * self.dt)

        self.u_history = np.array(self.u_history)
        self.v_history = np.array(self.v_history)
        self.t_history = np.array(self.t_history)

        print(f"Done: {self.model} | {n_steps} steps")

    # ── Mass diagnostic ──────────────────────────────────────────────────────

    def mass(self) -> np.ndarray:
        """Total mass ∫∫u dA at each saved time step."""
        if len(self.u_history) == 0:
            return np.array([])
        return self.u_history.sum(axis=(1, 2)) * self.dx * self.dy
