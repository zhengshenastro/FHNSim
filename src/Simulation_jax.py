
"""
Simulation_jax.py
-----------------
FHN reaction–diffusion simulation on a 2D periodic domain (torus), using Fourier pseudo-spectral methods.

This file supports two backends:

1) backend="numpy"
   - Uses NumPy arrays and SciPy FFT (scipy.fft.fftn/ifftn).

2) backend="jax"
   - Two JAX modes:
     a) jax_mode="xla" (fast): uses jax.numpy.fft.* (XLA FFT on CPU/GPU).
     b) jax_mode="scipy_callback" (Route B / deterministic matching): uses
        SciPy FFT *inside* a JAX loop via jax.pure_callback so that the FFT
        implementation matches the NumPy/SciPy backend as closely as possible.

Why "scipy_callback"?
- Pattern-forming PDEs can amplify tiny floating-point differences (even in float64)
  into qualitatively different late-time patterns.
- If you need NumPy vs JAX runs to match frame-by-frame, you must match the FFT
  *implementation*, not just the PDE discretization.
- This mode prioritizes reproducibility/consistency over speed.

Notes
-----
- Periodic boundary conditions are enforced by construction via FFT on a torus.
- Default dtype is float64.
- Compatible with NumPy 2.0 (no np.array(..., copy=False) hard errors).

Dependencies
------------
- numpy
- scipy
- jax (optional, only if backend="jax")
- Torus.py providing Torus2D with k-space arrays (kx, ky, k2, k4) and spatial grids (px, py)
- FHNmodel.py providing RegularFHN and MassConservedFHN with parameters (a,b,epsilon,Du,Dv)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.fft as spfft

from Torus import Torus2D
from FHNmodel import RegularFHN, MassConservedFHN

# JAX is optional
try:
    import jax
    jax.config.update("jax_enable_x64", True)  # IMPORTANT: before importing jnp
    import jax.numpy as jnp
    from jax import lax
    _JAX_AVAILABLE = True
except Exception:
    jax = None
    jnp = None
    lax = None
    _JAX_AVAILABLE = False


@dataclass(frozen=True)
class _ModelParams:
    a: float
    b: float
    epsilon: float
    Du: float
    Dv: float
    mass_conserved: bool


class FHNSimulation(Torus2D):
    """
    Fourier pseudo-spectral simulation of FHN on a 2D periodic domain.
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
        backend: str = "numpy",
        dtype=np.float64,
        dealias: bool = False,
        jax_mode: str = "scipy_callback",  # "scipy_callback" (Route B) or "xla" (fast)
    ):
        super().__init__(sizex, sizey, resx, resy)

        self.model = model
        self.dt = float(dt)
        self.save_every = int(save_every)
        self.backend = str(backend).lower().strip()
        self.dtype = dtype
        self.dealias = bool(dealias)
        self.jax_mode = str(jax_mode).lower().strip()

        if self.backend not in {"numpy", "jax"}:
            raise ValueError('backend must be "numpy" or "jax"')
        if self.backend == "jax":
            if not _JAX_AVAILABLE:
                raise ImportError('backend="jax" requested but JAX could not be imported.')
            if self.jax_mode not in {"xla", "scipy_callback"}:
                raise ValueError('jax_mode must be "xla" or "scipy_callback"')

        # Fields
        self.u: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None

        # History (NumPy)
        self.u_history: Optional[np.ndarray] = None
        self.v_history: Optional[np.ndarray] = None
        self.t_history: Optional[np.ndarray] = None

        # NumPy spectral masks (integrating factor)
        self.linear_mask_u: Optional[np.ndarray] = None
        self.linear_mask_v: Optional[np.ndarray] = None
        self._dealias_mask_np: Optional[np.ndarray] = None

        # JAX cached arrays
        self._k2_j: Optional["jnp.ndarray"] = None
        self._mask_u_j: Optional["jnp.ndarray"] = None
        self._mask_v_j: Optional["jnp.ndarray"] = None
        self._dealias_mask_j: Optional["jnp.ndarray"] = None

        # Route-B callback payload
        self._np_step_payload = None  # set in BuildSpectralMasks()

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

    # ── Dealias masks ──────────────────────────────────────────────────────

    def _build_dealias_mask_numpy_rect_23(self) -> None:
        """Standard 2/3 rectangular rule (keeps modes with |kx| and |ky| below cutoff)."""
        kx = self.kx
        ky = self.ky
        kx_max = np.max(np.abs(kx))
        ky_max = np.max(np.abs(ky))
        cutoff_x = (2.0 / 3.0) * kx_max
        cutoff_y = (2.0 / 3.0) * ky_max
        mask = (np.abs(kx) <= cutoff_x) & (np.abs(ky) <= cutoff_y)
        self._dealias_mask_np = mask.astype(self.dtype)

    # ── BuildSpectralMasks ──────────────────────────────────────────────────

    def BuildSpectralMasks(self) -> None:
        """
        Build integrating-factor masks and prepare payloads for each backend.
        """
        m = self.model
        mp = self._model_params()

        # NumPy masks (float64 default)
        if isinstance(m, RegularFHN):
            self.linear_mask_u = np.exp(-mp.Du * self.k2 * self.dt).astype(self.dtype)
            self.linear_mask_v = np.exp(-mp.Dv * self.k2 * self.dt).astype(self.dtype)
        elif isinstance(m, MassConservedFHN):
            self.linear_mask_u = np.exp(-mp.Du * self.k4 * self.dt).astype(self.dtype)
            self.linear_mask_v = np.exp(-mp.Dv * self.k4 * self.dt).astype(self.dtype)
        else:
            raise TypeError(f"Unknown model type: {type(m)}")

        if self.dealias and (self._dealias_mask_np is None):
            self._build_dealias_mask_numpy_rect_23()

        # Payload used by Route-B SciPy callback (kept as NumPy arrays)
        self._np_step_payload = dict(
            a=mp.a,
            b=mp.b,
            epsilon=mp.epsilon,
            dt=float(self.dt),
            k2=np.asarray(self.k2, dtype=np.float64),
            linear_mask_u=np.asarray(self.linear_mask_u, dtype=np.complex128),  # allow complex multiply
            linear_mask_v=np.asarray(self.linear_mask_v, dtype=np.complex128),
            mass_conserved=bool(mp.mass_conserved),
            dealias_mask=(np.asarray(self._dealias_mask_np, dtype=np.complex128) if self.dealias else None),
        )

        # JAX masks (for xla mode)
        if self.backend == "jax":
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

            if self.dealias:
                self._dealias_mask_j = jnp.array(self._dealias_mask_np, dtype=jnp.float64)

    # ── NumPy one-step ─────────────────────────────────────────────────────

    def SpectralStep(self) -> None:
        """One time step using NumPy + SciPy FFT (pseudo-spectral)."""
        m = self.model
        mp = self._model_params()

        u_hat = spfft.fftn(self.u)
        v_hat = spfft.fftn(self.v)

        f_uv = self.u - (self.u ** 3) / 3.0 - self.v
        g_uv = self.u + mp.a - mp.b * self.v

        f_hat = spfft.fftn(f_uv)
        g_hat = spfft.fftn(g_uv)

        if mp.mass_conserved:
            u_hat_new = self.linear_mask_u * (u_hat + self.dt * (self.k2 * f_hat))
            v_hat_new = self.linear_mask_v * (v_hat + self.dt * (mp.epsilon * self.k2 * g_hat))
        else:
            u_hat_new = self.linear_mask_u * (u_hat + self.dt * f_hat)
            v_hat_new = self.linear_mask_v * (v_hat + self.dt * (mp.epsilon * g_hat))

        if self.dealias:
            u_hat_new = u_hat_new * self._dealias_mask_np
            v_hat_new = v_hat_new * self._dealias_mask_np

        self.u = np.real(spfft.ifftn(u_hat_new)).astype(self.dtype, copy=False)
        self.v = np.real(spfft.ifftn(v_hat_new)).astype(self.dtype, copy=False)

    # ── JAX fast step (XLA FFT) ─────────────────────────────────────────────

    @staticmethod
    def _jax_step_xla(u, v, *, a, b, epsilon, dt, k2, mask_u, mask_v, mass_conserved: bool, dealias_mask):
        f_uv = u - (u ** 3) / 3.0 - v
        g_uv = u + a - b * v

        u_hat = jnp.fft.fftn(u, axes=(0, 1))
        v_hat = jnp.fft.fftn(v, axes=(0, 1))
        f_hat = jnp.fft.fftn(f_uv, axes=(0, 1))
        g_hat = jnp.fft.fftn(g_uv, axes=(0, 1))

        if mass_conserved:
            u_hat_new = mask_u * (u_hat + dt * k2 * f_hat)
            v_hat_new = mask_v * (v_hat + dt * epsilon * k2 * g_hat)
        else:
            u_hat_new = mask_u * (u_hat + dt * f_hat)
            v_hat_new = mask_v * (v_hat + dt * epsilon * g_hat)

        if dealias_mask is not None:
            u_hat_new = u_hat_new * dealias_mask
            v_hat_new = v_hat_new * dealias_mask

        u_new = jnp.real(jnp.fft.ifftn(u_hat_new, axes=(0, 1)))
        v_new = jnp.real(jnp.fft.ifftn(v_hat_new, axes=(0, 1)))
        return u_new, v_new

    # ── JAX Route-B step: SciPy FFT callback ────────────────────────────────

    def _np_step_scipy(self, u_np: np.ndarray, v_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Host-side step using SciPy FFT, intended to match NumPy backend behavior.
        """
        p = self._np_step_payload
        a = p["a"]
        b = p["b"]
        eps = p["epsilon"]
        dt = p["dt"]
        k2 = p["k2"]
        mask_u = p["linear_mask_u"]
        mask_v = p["linear_mask_v"]
        mass_conserved = p["mass_conserved"]
        dealias_mask = p["dealias_mask"]

        # Ensure float64 inputs on host
        u_np = np.asarray(u_np, dtype=np.float64)
        v_np = np.asarray(v_np, dtype=np.float64)

        u_hat = spfft.fftn(u_np)
        v_hat = spfft.fftn(v_np)

        f_uv = u_np - (u_np ** 3) / 3.0 - v_np
        g_uv = u_np + a - b * v_np

        f_hat = spfft.fftn(f_uv)
        g_hat = spfft.fftn(g_uv)

        if mass_conserved:
            u_hat_new = mask_u * (u_hat + dt * (k2 * f_hat))
            v_hat_new = mask_v * (v_hat + dt * (eps * k2 * g_hat))
        else:
            u_hat_new = mask_u * (u_hat + dt * f_hat)
            v_hat_new = mask_v * (v_hat + dt * (eps * g_hat))

        if dealias_mask is not None:
            u_hat_new = u_hat_new * dealias_mask
            v_hat_new = v_hat_new * dealias_mask

        u_new = np.real(spfft.ifftn(u_hat_new)).astype(np.float64, copy=False)
        v_new = np.real(spfft.ifftn(v_hat_new)).astype(np.float64, copy=False)
        return u_new, v_new

    def _jax_step_scipy_callback(self, u, v):
        """
        JAX wrapper around _np_step_scipy using pure_callback.
        """
        # Output shape/dtype must be declared
        out_spec_u = jax.ShapeDtypeStruct(u.shape, jnp.float64)
        out_spec_v = jax.ShapeDtypeStruct(v.shape, jnp.float64)

        def _cb(u_in, v_in):
            u_out, v_out = self._np_step_scipy(u_in, v_in)
            return (u_out, v_out)

        u_new, v_new = jax.pure_callback(_cb, (out_spec_u, out_spec_v), u, v)
        return u_new, v_new

    # ── Initial conditions ──────────────────────────────────────────────────

    def set_initial_conditions(self, u0=None, v0=None, init_type="random", seed=42):
        rng = np.random.default_rng(seed)
        shape = (self.resx, self.resy)
        x = self.px
        y = self.py

        if u0 is not None and v0 is not None:
            self.u = np.asarray(u0).astype(self.dtype, copy=False)
            self.v = np.asarray(v0).astype(self.dtype, copy=False)
            return

        if init_type == "random":
            self.u = (0.1 * rng.standard_normal(shape)).astype(self.dtype)
            self.v = (0.1 * rng.standard_normal(shape)).astype(self.dtype)

        elif init_type == "zeros":
            self.u = np.zeros(shape)
            self.v = np.zeros(shape)

        elif init_type == "pulse":
            cx, cy = self.sizex / 2, self.sizey / 2
            r2 = (x - cx) ** 2 + (y - cy) ** 2
            self.u = np.exp(-r2 / 5.0).astype(self.dtype)
            self.v = np.zeros_like(self.u)
        elif init_type == "front":
            self.u = np.where(x < self.sizex / 2, 1.0, -1.0).astype(self.dtype)
            self.v = np.zeros_like(self.u)
        elif init_type == "sin":
            k = 2 * np.pi / self.sizex * 4
            self.u = (0.1 * np.sin(k * x)).astype(self.dtype)
            self.v = np.zeros_like(self.u)
        elif init_type == "biased":
            self.u = (0.5 + 0.1 * rng.standard_normal(shape)).astype(self.dtype)
            self.v = (0.0 + 0.1 * rng.standard_normal(shape)).astype(self.dtype)

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

            # excite left half-plane
            self.u = np.where(x < self.sizex / 2, 1.0, 0.0)

            # notch (gap) near the center to create wave tips
            cx, cy = self.resx // 2, self.resy // 2
            wx = max(2, self.resx // 50)  # notch half-width in x (grid units)
            wy = max(6, self.resy // 10)  # notch half-height in y (grid units)
            self.u[cx - wx: cx + wx, cy - wy: cy + wy] = 0.0

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

    # ── Run loops ──────────────────────────────────────────────────────────

    def _run_numpy(self, T: float):
        if self.linear_mask_u is None:
            self.BuildSpectralMasks()
        if self.u is None:
            self.set_initial_conditions()

        n_steps = int(np.floor(T / self.dt))
        save_every = max(1, self.save_every)
        n_saves = 1 + (n_steps // save_every)

        u_hist = np.empty((n_saves, self.resx, self.resy), dtype=self.dtype)
        v_hist = np.empty((n_saves, self.resx, self.resy), dtype=self.dtype)
        t_hist = np.empty((n_saves,), dtype=np.float64)

        u_hist[0] = self.u
        v_hist[0] = self.v
        t_hist[0] = 0.0

        save_idx = 0
        for step in range(1, n_steps + 1):
            self.SpectralStep()
            if step % save_every == 0:
                save_idx += 1
                u_hist[save_idx] = self.u
                v_hist[save_idx] = self.v
                t_hist[save_idx] = step * self.dt

        self.u_history = u_hist
        self.v_history = v_hist
        self.t_history = t_hist

    def _run_jax(self, T: float):
        if self.linear_mask_u is None:
            self.BuildSpectralMasks()
        if self.u is None:
            self.set_initial_conditions()

        mp = self._model_params()
        n_steps = int(np.floor(T / self.dt))
        save_every = max(1, self.save_every)
        n_saves = 1 + (n_steps // save_every)

        # Work in float64 inside JAX
        u = jnp.array(self.u, dtype=jnp.float64)
        v = jnp.array(self.v, dtype=jnp.float64)

        u_hist = jnp.zeros((n_saves, self.resx, self.resy), dtype=jnp.float64)
        v_hist = jnp.zeros((n_saves, self.resx, self.resy), dtype=jnp.float64)
        u_hist = u_hist.at[0].set(u)
        v_hist = v_hist.at[0].set(v)

        if self.jax_mode == "xla":
            if self._mask_u_j is None or self._k2_j is None:
                # ensure jax masks exist
                self.backend = "jax"
                self.BuildSpectralMasks()

            step_kwargs = dict(
                a=jnp.float64(mp.a),
                b=jnp.float64(mp.b),
                epsilon=jnp.float64(mp.epsilon),
                dt=jnp.float64(self.dt),
                k2=self._k2_j,
                mask_u=self._mask_u_j,
                mask_v=self._mask_v_j,
                mass_conserved=bool(mp.mass_conserved),
                dealias_mask=(self._dealias_mask_j if self.dealias else None),
            )
            step_fn = jax.jit(lambda uu, vv: self._jax_step_xla(uu, vv, **step_kwargs))
            do_step = lambda uu, vv: step_fn(uu, vv)
        else:
            # Route B: SciPy FFT callback (match NumPy)
            # We still jit the loop body; callbacks execute on host.
            do_step = lambda uu, vv: self._jax_step_scipy_callback(uu, vv)

        def body(step, carry):
            uu, vv, sidx, uh, vh = carry
            uu, vv = do_step(uu, vv)
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
            1, n_steps + 1, body, (u, v, jnp.int32(0), u_hist, v_hist)
        )

        # Host conversion (NumPy 2.0 safe)
        self.u_history = np.asarray(jax.device_get(u_hist)).astype(self.dtype, copy=False)
        self.v_history = np.asarray(jax.device_get(v_hist)).astype(self.dtype, copy=False)

        t_hist = np.zeros((n_saves,), dtype=np.float64)
        if n_saves > 1:
            t_hist[1:] = (np.arange(1, n_saves) * save_every) * self.dt
        self.t_history = t_hist

        self.u = self.u_history[-1]
        self.v = self.v_history[-1]

    def run(self, T: float):
        if self.backend == "numpy":
            self._run_numpy(T)
        else:
            self._run_jax(T)

        n_steps = int(np.floor(T / self.dt))
        print(
            f"Done: {self.model} | {n_steps} steps | backend={self.backend}"
            f" | jax_mode={self.jax_mode if self.backend=='jax' else '-'}"
            f" | dealias={self.dealias}"
        )

    def mass(self):
        return self.u_history.sum(axis=(1, 2)) * self.dx * self.dy
