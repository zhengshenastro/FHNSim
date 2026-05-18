"""
FHNSimulation_grid2d_jax.py
---------------------------
JAX-optimised 2D pseudo-spectral simulation of the FitzHugh–Nagumo (FHN) system
on a rectangular domain with **Neumann (zero-flux) boundary conditions**.

This is the JAX analogue of:
  - Simulation (1): model-driven ETD structure (integrating factor)
  - Grid2D: DCT-II basis (reflecting / Neumann BCs)

Design goals (per request)
--------------------------
1) Use Grid2D (Neumann) geometry: k_n = n*pi/L and DCT/IDCT transforms.
2) Provide the richer initial conditions (init_type) from the original Simulation.py.
3) Preserve snapshot interface compatibility:
      u_history, v_history, t_history are NumPy arrays after run().
4) JAX optimisation "in the style of Simulation_jax.py":
      - JIT-compiled time loop via lax.fori_loop
      - Preallocated history arrays
      - float64 enabled (recommended for pattern-forming PDEs)

Notes / Requirements
--------------------
- Requires JAX installed. If not available, import will raise.
- Uses jax.scipy.fft.dct/idct (separable 2D DCT-II).
- Model classes are assumed to be RegularFHN or MassConservedFHN (for type check)
  and to provide parameters a,b,epsilon,Du,Dv. We do NOT call model.build_masks()
  nor model.spectral_update(); we compute masks and updates in JAX for speed.

Files expected in your project
------------------------------
- Grid2D.py providing class Grid2D with geometry arrays:
    sizex,sizey,resx,resy,dx,dy,px,py,k2,k4
  (We inherit Grid2D for geometry only.)
- FHNmodel.py providing RegularFHN and MassConservedFHN
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)  # IMPORTANT: before importing jnp
import jax.numpy as jnp
from jax import lax

from Grid2D import Grid2D
from FHNmodel import RegularFHN, MassConservedFHN


@dataclass(frozen=True)
class _ModelParams:
    a: float
    b: float
    epsilon: float
    Du: float
    Dv: float
    mass_conserved: bool


def _dct2(x: jnp.ndarray) -> jnp.ndarray:
    """2D DCT-II with orthonormal normalisation (separable)."""
    # Import here to keep module import light
    from jax.scipy.fft import dct
    y = dct(x, type=2, norm="ortho", axis=0)
    y = dct(y, type=2, norm="ortho", axis=1)
    return y


def _idct2(x_hat: jnp.ndarray) -> jnp.ndarray:
    """Inverse of DCT-II using IDCT with matching conventions (separable)."""
    from jax.scipy.fft import idct
    y = idct(x_hat, type=2, norm="ortho", axis=0)
    y = idct(y, type=2, norm="ortho", axis=1)
    return y


class FHNSimulation(Grid2D):
    """
    JAX pseudo-spectral time-stepping for the 2D FHN system with Neumann BCs.

    Parameters
    ----------
    model       : RegularFHN or MassConservedFHN instance
    sizex/sizey : physical domain lengths
    resx/resy   : grid resolution (number of points)
    dt          : time step
    save_every  : store state every n steps
    store_initial : store t=0 initial condition in histories
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

        # Fields (NumPy on host)
        self.u: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None

        # History (NumPy on host, filled after run)
        self.u_history: Optional[np.ndarray] = None
        self.v_history: Optional[np.ndarray] = None
        self.t_history: Optional[np.ndarray] = None

        # Cached JAX arrays
        self._k2_j: Optional[jnp.ndarray] = None
        self._mask_u_j: Optional[jnp.ndarray] = None
        self._mask_v_j: Optional[jnp.ndarray] = None
        self._mp: Optional[_ModelParams] = None

    # ── Model params & masks ────────────────────────────────────────────────

    def _model_params(self) -> _ModelParams:
        m = self.model
        mass_conserved = isinstance(m, MassConservedFHN)
        return _ModelParams(
            a=float(m.a),
            b=float(m.b),
            epsilon=float(m.epsilon),
            Du=float(m.Du),
            Dv=float(m.Dv),
            mass_conserved=mass_conserved,
        )

    def BuildSpectralMasks(self) -> None:
        """
        Build integrating-factor masks in JAX for the linear diffusion operators.
        """
        mp = self._model_params()
        self._mp = mp

        k2 = jnp.array(self.k2, dtype=jnp.float64)
        self._k2_j = k2

        dt = jnp.float64(self.dt)
        Du = jnp.float64(mp.Du)
        Dv = jnp.float64(mp.Dv)

        if mp.mass_conserved:
            k4 = k2 * k2
            self._mask_u_j = jnp.exp(-Du * k4 * dt)
            self._mask_v_j = jnp.exp(-Dv * k4 * dt)
        else:
            self._mask_u_j = jnp.exp(-Du * k2 * dt)
            self._mask_v_j = jnp.exp(-Dv * k2 * dt)

    # ── One JAX step (DCT-based ETD) ────────────────────────────────────────

    @staticmethod
    def _jax_step(u, v, *, a, b, epsilon, dt, k2, mask_u, mask_v, mass_conserved: bool):
        """
        One ETD / integrating-factor step in DCT space.

        Regular FHN:
          û_{n+1} = mask_u * ( û_n + dt * f̂ )
          v̂_{n+1} = mask_v * ( v̂_n + dt * ε * ĝ )

        Mass-conserved FHN:
          û_{n+1} = mask_u * ( û_n + dt * k² * f̂ )
          v̂_{n+1} = mask_v * ( v̂_n + dt * ε * k² * ĝ )
        """
        f_uv = u - (u ** 3) / 3.0 - v
        g_uv = u + a - b * v

        u_hat = _dct2(u)
        v_hat = _dct2(v)
        f_hat = _dct2(f_uv)
        g_hat = _dct2(g_uv)

        if mass_conserved:
            u_hat_new = mask_u * (u_hat + dt * k2 * f_hat)
            v_hat_new = mask_v * (v_hat + dt * epsilon * k2 * g_hat)
        else:
            u_hat_new = mask_u * (u_hat + dt * f_hat)
            v_hat_new = mask_v * (v_hat + dt * epsilon * g_hat)

        u_new = _idct2(u_hat_new)
        v_new = _idct2(v_hat_new)
        return u_new, v_new

    # ── Initial conditions (ported) ─────────────────────────────────────────

    def set_initial_conditions(
        self,
        u0: np.ndarray | None = None,
        v0: np.ndarray | None = None,
        seed: int = 0,
        init_type: str = "random",
    ) -> None:
        """
        Set initial fields.

        Backwards-compatible signature:
            set_initial_conditions(u0=None, v0=None, seed=42, init_type="random")

        Supported init_type:
            "random", "pulse", "front", "sin", "biased",
            "spiral_cg", "spiral_bw", "spiral_pf"
        """
        rng = np.random.default_rng(seed)
        shape = (self.resx, self.resy)
        x = self.px
        y = self.py

        if (u0 is not None) and (v0 is not None):
            u0 = np.asarray(u0, dtype=np.float64)
            v0 = np.asarray(v0, dtype=np.float64)
            if u0.shape != shape or v0.shape != shape:
                raise ValueError(f"Custom u0/v0 must have shape {shape}, got {u0.shape} and {v0.shape}.")
            self.u = u0
            self.v = v0
            return

        if init_type == "random":
            self.u = 0.1 * rng.standard_normal(shape)
            #self.u = -1 + 0.1 * rng.standard_normal(shape)
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
            cx, cy = self.sizex / 2, self.sizey / 2
            A = 0.02
            self.u = A * (x - cx)
            self.v = A * (y - cy)

        elif init_type == "spiral_bw":
            self.u = np.where(x < self.sizex / 2, 1.0, 0.0)
            self.v = np.zeros_like(self.u)
            cx, cy = self.resx // 2, self.resy // 2
            wx = max(2, self.resx // 50)
            wy = max(6, self.resy // 10)
            self.u[cx - wx: cx + wx, cy - wy: cy + wy] = 0.0

        elif init_type == "spiral_pf":
            cx, cy = self.sizex / 2, self.sizey / 2
            theta = np.arctan2(y - cy, x - cx)
            A = 0.2
            self.u = A * np.cos(theta)
            self.v = A * np.sin(theta)
            self.u = np.where(x < self.sizex / 2, 1.0, 0.0)
            cx_i, cy_i = self.resx // 2, self.resy // 2
            wx = max(2, self.resx // 50)
            wy = max(6, self.resy // 10)
            self.u[cx_i - wx: cx_i + wx, cy_i - wy: cy_i + wy] = 0.0

        else:
            raise ValueError(f"Unknown init_type: {init_type}")

        # Ensure float64
        self.u = np.asarray(self.u, dtype=np.float64)
        self.v = np.asarray(self.v, dtype=np.float64)

    # ── Run loop (JAX) ──────────────────────────────────────────────────────

    def run(self, T: float) -> None:
        """
        Run the simulation for total time T, using a JIT-compiled JAX loop.

        After completion, histories are stored as NumPy arrays:
          self.u_history, self.v_history, self.t_history
        """
        if self._mask_u_j is None or self._k2_j is None or self._mp is None:
            self.BuildSpectralMasks()

        if self.u is None or self.v is None:
            self.set_initial_conditions()

        mp = self._mp
        n_steps = int(np.floor(T / self.dt))
        save_every = max(1, self.save_every)

        if self.store_initial:
            n_saves = 1 + (n_steps // save_every)
        else:
            n_saves = (n_steps // save_every)

        # Device arrays
        u = jnp.array(self.u, dtype=jnp.float64)
        v = jnp.array(self.v, dtype=jnp.float64)

        # Preallocate histories on device
        u_hist = jnp.zeros((n_saves, self.resx, self.resy), dtype=jnp.float64)
        v_hist = jnp.zeros((n_saves, self.resx, self.resy), dtype=jnp.float64)

        # Store initial condition if requested
        if self.store_initial:
            u_hist = u_hist.at[0].set(u)
            v_hist = v_hist.at[0].set(v)
            start_step = 1
            save_idx0 = jnp.int32(0)
        else:
            start_step = 1
            save_idx0 = jnp.int32(-1)  # first save will bump to 0

        step_kwargs = dict(
            a=jnp.float64(mp.a),
            b=jnp.float64(mp.b),
            epsilon=jnp.float64(mp.epsilon),
            dt=jnp.float64(self.dt),
            k2=self._k2_j,
            mask_u=self._mask_u_j,
            mask_v=self._mask_v_j,
            mass_conserved=bool(mp.mass_conserved),
        )

        step_fn = jax.jit(lambda uu, vv: FHNSimulation._jax_step(uu, vv, **step_kwargs))

        def body(step, carry):
            uu, vv, sidx, uh, vh = carry
            uu, vv = step_fn(uu, vv)

            do_save = (step % save_every) == 0

            def save_fn(args):
                uu_, vv_, sidx_, uh_, vh_ = args
                sidx2 = sidx_ + 1
                uh2 = uh_.at[sidx2].set(uu_)
                vh2 = vh_.at[sidx2].set(vv_)
                return uu_, vv_, sidx2, uh2, vh2

            uu, vv, sidx, uh, vh = lax.cond(
                do_save, save_fn, lambda x: x, (uu, vv, sidx, uh, vh)
            )
            return (uu, vv, sidx, uh, vh)

        uu, vv, sidx, u_hist, v_hist = lax.fori_loop(
            start_step, n_steps + 1, body, (u, v, save_idx0, u_hist, v_hist)
        )

        # Move to host (NumPy), keep snapshot interface
        self.u_history = np.asarray(jax.device_get(u_hist), dtype=np.float64)
        self.v_history = np.asarray(jax.device_get(v_hist), dtype=np.float64)

        # Build t_history on host (deterministic, lightweight)
        self.t_history = np.zeros((n_saves,), dtype=np.float64)
        if self.store_initial:
            if n_saves > 1:
                self.t_history[1:] = (np.arange(1, n_saves) * save_every) * self.dt
        else:
            if n_saves > 0:
                self.t_history[:] = (np.arange(1, n_saves + 1) * save_every) * self.dt

        # Update terminal state on host
        self.u = self.u_history[-1].copy()
        self.v = self.v_history[-1].copy()

        print(f"Done: {self.model} | {n_steps} steps | backend=jax | grid=Grid2D(DCT/Neumann)")

    # ── Mass diagnostic ──────────────────────────────────────────────────────

    def mass(self) -> np.ndarray:
        """Total mass ∫∫u dA at each saved time step."""
        if self.u_history is None:
            return np.array([])
        return self.u_history.sum(axis=(1, 2)) * self.dx * self.dy
