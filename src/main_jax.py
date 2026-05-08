import matplotlib.pyplot as plt
import time

import numpy as np

from FHNmodel import RegularFHN, MassConservedFHN
from Simulation_jax import FHNSimulation
from Analyser import FHNAnalyser
from Snapshots import plot_time_series_snapshots

# ── Parameters ────────────────────────────────────────────────────────────────
INIT_TYPE = "random"

#Turing
#PARAMS = dict(a=0.025, b=1.26, epsilon=.5, Du=1, Dv=5)
#Oscilation to stripe conservation
PARAMS = dict(a=0.025, b=1.1, epsilon=.5, Du=1, Dv=5)
#Breathing stripe
#PARAMS = dict(a=0.025, b=1.8, epsilon=.5, Du=1, Dv=5)
#spiral
#PARAMS = dict(a=0.1, b=1, epsilon=.005, Du=1, Dv=1)
GRID   = dict(sizex=50.0, sizey=50.0, resx=500, resy=500)
T      = 200.0
n_snap = 49

# ── Instantiate models ────────────────────────────────────────────────────────
regular   = RegularFHN(**PARAMS)
conserved = MassConservedFHN(**PARAMS)

# ── Set up simulations (backend="jax") ───────────────────────────────────────
# backend:
#   "numpy" : NumPy arrays + SciPy FFT
#   "jax"   : JAX/XLA accelerated backend
#
# jax_mode (only used when backend="jax"):
#   "xla"
#       Fast JAX-native FFT (CPU/GPU via XLA).
#       Best performance, but long-time pattern formation may differ slightly
#       from SciPy/NumPy due to FFT implementation and floating-point ordering.
#
#   "scipy_callback"
#       Uses SciPy FFT inside the JAX loop (Route B).
#       Much slower, but reproduces the original NumPy/SciPy spectral evolution
#       much more faithfully for debugging / validation.
#
# dtype:
#   np.float64 : recommended for stability and reproducibility
#   np.float32 : faster and lower memory usage, but may introduce stronger
#                numerical asymmetry / mode-selection artifacts in long runs
#sim_reg = FHNSimulation(regular, **GRID, dt=0.04, save_every=20, backend="jax", jax_mode="xla", dtype=np.float64)
#sim_mc  = FHNSimulation(conserved, **GRID, dt=0.04, save_every=20, backend="jax", jax_mode="xla", dtype=np.float64)


sim_reg = FHNSimulation(regular, **GRID, dt=0.04, save_every=20,  backend="numpy", dtype=np.float64)
sim_mc  = FHNSimulation(conserved, **GRID, dt=0.04, save_every=20,  backend="numpy",  dtype=np.float64)

sim_reg.set_initial_conditions(init_type=INIT_TYPE, seed=0)
sim_mc.set_initial_conditions(init_type=INIT_TYPE, seed=0)

sim_reg.BuildSpectralMasks()
sim_mc.BuildSpectralMasks()

# ── Run ───────────────────────────────────────────────────────────────────────
t0 = time.perf_counter()
sim_reg.run(T)
t1 = time.perf_counter()
sim_mc.run(T)
t2 = time.perf_counter()

u = sim_reg.u_history

print("\n=== Numerical Check (Regular FHN) ===")
print("u nan:", np.isnan(u).any(), "inf:", np.isinf(u).any())
print("u min/max:", np.nanmin(u), np.nanmax(u))

std_t = u.reshape(u.shape[0], -1).std(axis=1)
print("std first/last:", std_t[0], std_t[-1])
print("std last 5:", std_t[-5:])

umin = u.reshape(u.shape[0], -1).min(axis=1)
umax = u.reshape(u.shape[0], -1).max(axis=1)
print("umin last 5:", umin[-5:])
print("umax last 5:", umax[-5:])

du = np.abs(u[1:] - u[:-1]).reshape(u.shape[0]-1, -1).max(axis=1)
print("max frame-to-frame change last 10:", du[-10:])

print("\n=== Timing Report ===")
print(f"Regular FHN time        : {t1 - t0:.3f} s")
print(f"Mass-conserved FHN time : {t2 - t1:.3f} s")
print(f"Total runtime           : {t2 - t0:.3f} s")

# ── Analyse ───────────────────────────────────────────────────────────────────
ana_reg = FHNAnalyser(sim_reg)
ana_mc  = FHNAnalyser(sim_mc)

fig = ana_reg.plot_comparison(ana_mc)
plt.savefig('fhn_2d_comparison.png', dpi=150)
plt.show()

plot_time_series_snapshots(
    ana_reg, sim_reg, n_snap, filename="fhn_time_series_regular.png"
)

plot_time_series_snapshots(
    ana_mc, sim_mc, n_snap, filename="fhn_time_series_mc.png"
)