"""main_jax_grid2d_match_uploaded_main.py
-------------------------------------
This is the "fully-updated" (Grid2D + JAX) main script, but configured to
match the *uploaded* main.py's parameters and initial conditions as closely
as possible.

Important:
- The uploaded main.py uses Torus2D (periodic) + FFT, while this script uses
  Grid2D (Neumann/reflecting) + DCT. Even with identical (a,b,epsilon,Du,Dv),
  grid sizes, dt, and random seed, the boundary condition / spectral basis
  differs and can lead to different late-time patterns.

Outputs mirror uploaded main.py:
  fhn_comparison.png
  fhn_phase_plane.png
  fhn_dynamics.png
  fhn_robustness_reg.png
  fhn_robustness_mc.png
"""

import numpy as np
import time
import matplotlib.pyplot as plt

from FHNmodel import RegularFHN, MassConservedFHN

# JAX-only Neumann (DCT) simulation
from Simulation_jax import FHNSimulation

from Analyser import FHNAnalyser
from Snapshots import plot_time_series_snapshots


# ── Shared grid and time settings (MATCH uploaded main.py) ────────────────────
GRID = dict(sizex=50.0, sizey=50.0, resx=256, resy=256)
T, DT, SAVE_EVERY = 500.0, 0.05, 20

n_snap = 49  # number of frames in time-series snapshot grid

# ── Base model parameters (MATCH uploaded main.py) ────────────────────────────

PARAMS = dict(a=0.1, b=1.0, epsilon=0.005, Du=1.0, Dv=1.0)
#Stationary Pattern
#PARAMS = dict(a=-0.08, b=1.26, epsilon=.5, Du=1, Dv=5) #negative hexagon
#PARAMS = dict(a=0.08, b=1.26, epsilon=.05, Du=1, Dv=5) #positive hexagon
#PARAMS = dict(a=0.08, b=1.26, epsilon=.5, Du=1, Dv=5) #positive hexagon

#Turing
#PARAMS = dict(a=0.025, b=1.26, epsilon=.5, Du=1, Dv=5)
#Oscilation to stripe conservation
#PARAMS = dict(a=0.025, b=1.1, epsilon=.5, Du=1, Dv=5)
#Breathing stripe
#PARAMS = dict(a=0.025, b=1.8, epsilon=.5, Du=1, Dv=5)
#spiral
#PARAMS = dict(a=0.1, b=1, epsilon=.005, Du=1, Dv=1)


# ── Initial conditions (MATCH uploaded main.py) ───────────────────────────────
# Uploaded main.py calls: sim.set_initial_conditions(seed=seed)
# which uses default "small random perturbation around (0,0)".
# In our Grid2D-JAX simulation, this corresponds to init_type="random".
INIT_TYPE = "random"
SEED = 0

# ── Helper: build, run, and return a simulation ───────────────────────────────
def run_sim(model, seed=SEED, init_type=INIT_TYPE, T_run=T):
    sim = FHNSimulation(model, **GRID, dt=DT, save_every=SAVE_EVERY, store_initial=True)
    sim.set_initial_conditions(seed=SEED, init_type=init_type)
    sim.BuildSpectralMasks()
    sim.run(T_run)
    return sim


# ── Main simulations ──────────────────────────────────────────────────────────
print("Running Regular FHN...")
t0 = time.perf_counter()
sim_reg = run_sim(RegularFHN(**PARAMS))
t1 = time.perf_counter()

print("Running Mass-Conserved FHN...")
sim_mc = run_sim(MassConservedFHN(**PARAMS))
t2 = time.perf_counter()

print("\n=== Timing Report (main runs) ===")
print(f"Regular FHN time        : {t1 - t0:.3f} s")
print(f"Mass-conserved FHN time : {t2 - t1:.3f} s")
print(f"Total runtime           : {t2 - t0:.3f} s")

# ── Mass diagnostics (same prints as uploaded main.py, but BCs differ) ─────────
mc_mass  = sim_mc.mass()
reg_mass = sim_reg.mass()

print(f"\nMassConservedFHN | initial: {mc_mass[0]:.6f} | "
      f"final: {mc_mass[-1]:.6f} | "
      f"drift: {abs(mc_mass[-1] - mc_mass[0]):.2e}")

settled = reg_mass[reg_mass.size // 2:]
print("RegularFHN       | mass not conserved by design")
print(f"                 | settled mean: {settled.mean():.4f} "
      f"± {settled.std():.4f}")

# ── Analysers ─────────────────────────────────────────────────────────────────
ana_reg = FHNAnalyser(sim_reg)
ana_mc  = FHNAnalyser(sim_mc)

# ── Parameter sweep for robustness (MATCH uploaded main.py) ───────────────────
T_SWEEP = 2001.0

print("\nRunning Regular FHN Du sweep...")
reg_Du_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
reg_sweep = []
for Du in reg_Du_values:
    params = {**PARAMS, "Du": Du}
    s = run_sim(RegularFHN(**params), T_run=T_SWEEP)
    reg_sweep.append({"param_value": Du, "sim": s, "label": f"Du = {Du}"})

print("Running Mass-Conserved FHN Du sweep...")
mc_Du_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
mc_sweep = []
for Du in mc_Du_values:
    params = {**PARAMS, "Du": Du}
    s = run_sim(MassConservedFHN(**params), T_run=T_SWEEP)
    mc_sweep.append({"param_value": Du, "sim": s, "label": f"Du = {Du}"})


# ── Plots (MATCH uploaded main.py) ────────────────────────────────────────────

# 1) Baseline 4x3 comparison
fig_cmp = ana_reg.plot_comparison(ana_mc)
fig_cmp.savefig("fhn_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved: fhn_comparison.png")

# 2) Phase plane
fig_phase = ana_reg.plot_phase_plane(ana_mc)
fig_phase.savefig("fhn_phase_plane.png", dpi=150, bbox_inches="tight")
print("Saved: fhn_phase_plane.png")

# 3) Full dynamics comparison (wave speed / instability / robustness)
fig_dyn, fig_rob_reg, fig_rob_mc = ana_reg.plot_dynamics_comparison(
    other=ana_mc,
    reg_sweep=reg_sweep,
    mc_sweep=mc_sweep,
    sweep_param_name="Du",
)

fig_dyn.savefig("fhn_dynamics.png", dpi=150, bbox_inches="tight")
fig_rob_reg.savefig("fhn_robustness_reg.png", dpi=150, bbox_inches="tight")
fig_rob_mc.savefig("fhn_robustness_mc.png", dpi=150, bbox_inches="tight")
print("Saved: fhn_dynamics.png")
print("Saved: fhn_robustness_reg.png")
print("Saved: fhn_robustness_mc.png")


plot_time_series_snapshots(
    ana_reg,
    sim_reg,
    n_snap,
    filename="fhn_time_series_regular.png",
    field="u",
    global_scale="history",   # 全历史统一色标
    robust=0.5,               # 可选：0.5%~99.5%裁剪；不想裁剪就删掉这行
)

plot_time_series_snapshots(
    ana_mc,
    sim_mc,
    n_snap,
    filename="fhn_time_series_mc.png",
    field="u",
    global_scale="history",
    robust=0.5,
)