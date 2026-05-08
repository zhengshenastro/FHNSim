"""
main.py
-------
Example usage: run and compare both FHN variants in 2D.

File structure expected in the same folder:
    Torus2D.py
    FHNModels.py
    FHNSimulation.py
    FHNAnalyser.py
    main.py
"""

import matplotlib.pyplot as plt
import numpy as np
import time

from FHNmodel import RegularFHN, MassConservedFHN
from Simulation import FHNSimulation
from Analyser import FHNAnalyser
from Snapshots import plot_time_series_snapshots

# ── Parameters ────────────────────────────────────────────────────────────────
#INIT_TYPE = "random"
INIT_TYPE = "pulse"
#INIT_TYPE = "front"
#INIT_TYPE = "sin"
#INIT_TYPE = "biased"
#INIT_TYPE = "spiral"
#Excitable
PARAMS = dict(a=0.7, b=0.8, epsilon=.08, Du=1, Dv=5)
#Limit cycle
#PARAMS = dict(a=0.5, b=0.7, epsilon=.05, Du=1, Dv=5)
PARAMSRFHN = dict(a = 0.7, b =0.8, epsilon = 0.08, Du = 10000.0, Dv = 0.5)
GRID   = dict(sizex=50.0, sizey=50.0, resx=600, resy=600)
#T      = 1251.0
T      = 200.0
n_snap=50

# ── Instantiate models ────────────────────────────────────────────────────────
regular   = RegularFHN(**PARAMS)
conserved = MassConservedFHN(**PARAMS)

# ── Set up simulations ────────────────────────────────────────────────────────
sim_reg = FHNSimulation(regular,   **GRID, dt=0.04, save_every=20)
sim_mc  = FHNSimulation(conserved, **GRID, dt=0.04, save_every=20)

# Same initial conditions for fair comparison
sim_reg.set_initial_conditions(init_type=INIT_TYPE, seed=0)
sim_mc.set_initial_conditions(init_type=INIT_TYPE, seed=0)

# Precompute spectral masks
sim_reg.BuildSpectralMasks()
sim_mc.BuildSpectralMasks()

# ── Run ───────────────────────────────────────────────────────────────────────
t0 = time.perf_counter()
sim_reg.run(T)
t1 = time.perf_counter()
t2 = time.perf_counter()
sim_mc.run(T)
t3 = time.perf_counter()
time_reg = t1 - t0
time_mc  = t3 - t2
time_total = t3 - t0
print("\n=== Timing Report ===")
print(f"Regular FHN time        : {time_reg:.3f} s")
print(f"Mass-conserved FHN time: {time_mc:.3f} s")
print(f"Total runtime          : {time_total:.3f} s")

# ── Analyse ───────────────────────────────────────────────────────────────────
ana_reg = FHNAnalyser(sim_reg)
ana_mc  = FHNAnalyser(sim_mc)

fig = ana_reg.plot_comparison(ana_mc)
plt.savefig('fhn_2d_comparison.png', dpi=150)
plt.show()

# ── Time-series snapshots ───────────────────────────────────────────────

plot_time_series_snapshots(
    ana_reg,
    sim_reg,
    n_snap,
    filename="fhn_time_series_regular.png"
)

plot_time_series_snapshots(
    ana_mc,
    sim_mc,
    n_snap,
    filename="fhn_time_series_mc.png"
)

