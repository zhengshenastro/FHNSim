"""
Grid2D.py
---------
2D rectangular grid with Neumann (zero-flux) boundary conditions.
Replaces Torus2D (periodic) for the spiral wave simulations.

Key difference from Torus2D:
  - Uses Discrete Cosine Transform (DCT-II) instead of FFT
  - Wavenumbers: k_n = n*pi/L  (not 2*pi*n/L)
  - No periodic wrapping — boundaries are reflecting, not connected
  - Fields stay real throughout (no complex arithmetic)

The DCT-II basis functions cos(n*pi*x/L) satisfy:
    d/dx [cos(n*pi*x/L)] = 0  at x=0 and x=L
which is exactly the Neumann condition.
"""

import numpy as np
import scipy.fft as fft


class Grid2D:
    def __init__(self, sizex, sizey, resx, resy):
        # Domain size
        self.sizex = sizex
        self.sizey = sizey

        # Grid resolution
        self.resx = round(resx)
        self.resy = round(resy)

        # Grid spacing
        self.dx = self.sizex / self.resx
        self.dy = self.sizey / self.resy

        # 1D index arrays
        self.ix = np.arange(self.resx)
        self.iy = np.arange(self.resy)

        # 2D index arrays
        self.iix, self.iiy = np.meshgrid(self.ix, self.iy, indexing='ij')

        # Physical coordinates (cell centres)
        self.px = (self.iix + 0.5) * self.dx
        self.py = (self.iiy + 0.5) * self.dy

        # ── DCT wavenumbers ──────────────────────────────────────────────────
        # For DCT-II on [0, L] with N points:
        #   k_n = n * pi / L,  n = 0, 1, ..., N-1
        # These are the eigenvalues of the Neumann Laplacian:
        #   ∇² cos(k_n * x) = -k_n² * cos(k_n * x)
        kx_1d = np.arange(self.resx) * np.pi / self.sizex
        ky_1d = np.arange(self.resy) * np.pi / self.sizey
        self.kx, self.ky = np.meshgrid(kx_1d, ky_1d, indexing='ij')

        # Precompute k^2 and k^4 for Laplacian and bilaplacian
        self.k2 = self.kx**2 + self.ky**2
        self.k4 = self.k2**2

    # ── DCT transform pair ───────────────────────────────────────────────────

    def dct(self, f):
        """
        Forward 2D DCT-II with orthonormal normalisation.
        Real input → real output.
        """
        return fft.dctn(f, type=2, norm='ortho')

    def idct(self, f_hat):
        """
        Inverse 2D DCT-II (= DCT-III) with orthonormal normalisation.
        Real input → real output.
        """
        return fft.idctn(f_hat, type=2, norm='ortho')

    # ── Spectral operators ───────────────────────────────────────────────────

    def spectral_laplacian(self, f):
        """
        ∇²f via DCT:  f̂ → -k² * f̂ → IDCT
        """
        return self.idct(-self.k2 * self.dct(f))

    def spectral_bilaplacian(self, f):
        """
        ∇⁴f via DCT:  f̂ → k⁴ * f̂ → IDCT
        """
        return self.idct(self.k4 * self.dct(f))

    # ── Poisson solver ───────────────────────────────────────────────────────

    def poisson_solve(self, f):
        """
        Solves ∇²q = f with Neumann BCs (∂q/∂n = 0).

        In DCT space:  -k² * q̂ = f̂
                        q̂ = -f̂ / k²

        The (0,0) mode (mean) is set to zero to enforce uniqueness.
        """
        f_hat = self.dct(f)
        denom = self.k2.copy()
        denom[0, 0] = 1.0            # avoid division by zero
        q_hat = -f_hat / denom
        q_hat[0, 0] = 0.0            # zero mean — Neumann problems are unique up to a constant
        return self.idct(q_hat)

    # ── Finite difference operators (for reference / validation) ────────────

    def grad(self, f):
        """
        Forward finite difference gradient.
        One-sided at boundaries (no flux condition).
        """
        gx = np.zeros_like(f)
        gy = np.zeros_like(f)

        # Interior and right/top boundary: forward difference
        gx[:-1, :] = (f[1:, :] - f[:-1, :]) / self.dx
        gy[:, :-1] = (f[:, 1:] - f[:, :-1]) / self.dy

        # Neumann: zero flux — boundary gradient stays zero (already initialised)
        return gx, gy

    def div(self, vx, vy):
        """
        Backward finite difference divergence with Neumann BCs.
        """
        f = np.zeros_like(vx)

        # Interior
        f[1:, :] += (vx[1:, :] - vx[:-1, :]) / self.dx
        f[:, 1:] += (vy[:, 1:] - vy[:, :-1]) / self.dy

        return f
    