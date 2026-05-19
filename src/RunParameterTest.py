import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from scipy.optimize import brentq
import warnings

warnings.filterwarnings("ignore")

# ======================================================
# INPUT PARAMETERS (edit ONLY this block)
# ======================================================
PARAMS = {
    "a": 0.05,
    "b": 1.26,
    "eps": 0.1,
    "Du": 1.0,
    "Dv": 5.0,
}

# ======================================================
# RESOLUTION / RANGES
# ======================================================
RES = {
    # phase plane
    "phase_grid": 28,
    "phase_xlim": (-2.0, 2.0),
    "phase_ylim": (-2.0, 2.0),

    # (a,b) parameter plane
    "ab_xlim": (-1.5, 1.5),
    "ab_ylim": (0.0, 3.0),
    "boundary_n_u": 5000,
    "boundary_u_eps": 1e-4,

    # k grids
    "k_max": 10.0,
    "n_k": 2200,         # dense for complex-plane scatter
    "k2_max": None,
    "n_k2": 1600,

    # annotation / scatter styling
    "scatter_s": 6,
    "special_s": 70,
    "special_edge_lw": 0.7,
    "ann_font": 9,
    "ann_scale": 22,     # base offset magnitude in "offset points"
}


# ======================================================
# MODEL / LINEAR ALGEBRA
# ======================================================
def f_u(u, v):
    return u - u**3 / 3.0 - v


def g_v(u, v, a, b, eps):
    return eps * (u + a - b * v)


def u_null(u):
    return u - u**3 / 3.0


def v_null(u, a, b):
    return (u + a) / b


def jac(u, v, eps, b):
    return np.array([[1.0 - u**2, -1.0],
                     [eps,        -eps * b]])


def jac_k(u, v, eps, b, Du, Dv, k):
    J = jac(u, v, eps, b).copy()
    k2 = k**2
    J[0, 0] -= Du * k2
    J[1, 1] -= Dv * k2
    return J


def trace_J(u, eps, b):
    return (1.0 - u**2) - eps * b


def det_J(u, eps, b):
    return (1.0 - u**2) * (-eps * b) + eps


def trace_Jk(u, v, eps, b, Du, Dv, k2):
    return trace_J(u, eps, b) - k2 * (Du + Dv)


def det_Jk(u, v, eps, b, Du, Dv, k2):
    fu = 1.0 - u**2
    gv = -eps * b
    return Du * Dv * k2**2 - (Du * gv + Dv * fu) * k2 + det_J(u, eps, b)


def stab_label_of_fp(p, eps, b):
    ev = np.linalg.eigvals(jac(p[0], p[1], eps, b))
    re = ev.real
    if np.max(re) < -1e-9:
        return "Stable", "green"
    if np.min(re) > 1e-9:
        return "Unstable", "red"
    return "Saddle", "magenta"


def lambda_max_from_tau_delta(tau, delta):
    """
    λ = (τ ± sqrt(τ^2 - 4Δ))/2, choose branch with larger Re.
    tau, delta may be arrays (real). Return complex array.
    """
    tau = np.asarray(tau)
    delta = np.asarray(delta)
    disc = (tau.astype(complex))**2 - 4.0 * (delta.astype(complex))
    root = np.sqrt(disc)
    lam1 = 0.5 * (tau + root)
    lam2 = 0.5 * (tau - root)
    return np.where(lam1.real >= lam2.real, lam1, lam2)


# ======================================================
# FIXED POINTS
# ======================================================
def find_fps(a, b):
    def diff(u):
        return u_null(u) - v_null(u, a, b)

    u_arr = np.linspace(-3.0, 3.0, 25000)
    vals = diff(u_arr)
    fps = []
    for i in range(len(vals) - 1):
        if vals[i] * vals[i + 1] < 0:
            try:
                u_fp = brentq(diff, u_arr[i], u_arr[i + 1], xtol=1e-12)
                fps.append(np.array([u_fp, u_null(u_fp)]))
            except Exception:
                pass
    return fps


# ======================================================
# PARAMETER-SPACE BOUNDARIES (smooth)
# ======================================================
def param_curves_exact(eps, b_min=0.05, b_max=5.0, u_eps=1e-4, n_u=5000):
    u = np.linspace(-1.0 + u_eps, 1.0 - u_eps, n_u)
    un = u_null(u)

    bH = (1.0 - u**2) / eps
    aH = bH * un - u
    mH = np.isfinite(bH) & (bH >= b_min) & (bH <= b_max)
    aH, bH = aH[mH], bH[mH]

    bS = 1.0 / (1.0 - u**2)
    aS = bS * un - u
    mS = np.isfinite(bS) & (bS >= b_min) & (bS <= b_max)
    aS, bS = aS[mS], bS[mS]

    return aH, bH, aS, bS


# ======================================================
# PER-FP HELPERS
# ======================================================
def sigma_curve_for_fp(p, eps, b, Du, Dv, kv):
    """
    Regular (rFHN) dispersion: σ_reg(k) = max Re(eigs(J_k)).
    """
    u, v = p
    sig = np.empty_like(kv, dtype=float)
    for i, k in enumerate(kv):
        ev = np.linalg.eigvals(jac_k(u, v, eps, b, Du, Dv, k))
        sig[i] = np.max(ev.real)
    return sig


def find_kc_from_sigma(kv, sig, allow_k0=True):
    """
    Find first sign-change crossing from negative to positive.
    If allow_k0=False, ignore the very first segment starting at k=0.
    """
    start = 0 if allow_k0 else 1
    idx = np.where((sig[start:-1] < 0.0) & (sig[start+1:] > 0.0))[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0] + start
    x0, x1 = kv[i], kv[i + 1]
    y0, y1 = sig[i], sig[i + 1]
    if np.isfinite(y0) and np.isfinite(y1) and (y1 != y0):
        return x0 - y0 * (x1 - x0) / (y1 - y0)
    return x0


def dominant_k_from_sigma(kv, sig):
    j = int(np.nanargmax(sig))
    return kv[j], sig[j], j


# ======================================================
# SMART ANNOTATION OFFSETS (reduce overlap)
# ======================================================
def smart_offset(z, scale=22):
    x = float(np.real(z))
    y = float(np.imag(z))
    dx = scale if x >= 0 else -scale
    dy = scale if y >= 0 else -scale
    return dx, dy


def smart_offset_three(z0, z1, z2, base=22):
    d0 = smart_offset(z0, base)
    d1 = smart_offset(z1, base)
    d2 = smart_offset(z2, base)
    d1 = (int(d1[0] * 1.10), int(-d1[1] * 0.95))
    d2 = (int(-d2[0] * 0.95), int(d2[1] * 1.10))
    return d0, d1, d2


# ======================================================
# PLOTTING (ONE SINGLE FIGURE)
# Row 0: phase | parameter
# Row i: dispersion | tau/delta | λ_max scatter | k^2 λ_max scatter (MC)
# ======================================================
def plot_one_figure(a, b, eps, Du, Dv):
    fps = find_fps(a, b)
    nfp = len(fps)

    nrows = 1 + nfp
    fig = plt.figure(figsize=(24.0, 4.2 + 3.3 * nfp))
    gs = gridspec.GridSpec(nrows, 4, figure=fig, wspace=0.35, hspace=0.60)

    # ---------- Row 0 ----------
    ax_phase = fig.add_subplot(gs[0, 0])
    ax_param = fig.add_subplot(gs[0, 1:4])

    # Phase space
    NQ = RES["phase_grid"]
    ug = np.linspace(RES["phase_xlim"][0], RES["phase_xlim"][1], NQ)
    vg = np.linspace(RES["phase_ylim"][0], RES["phase_ylim"][1], NQ)
    U, V = np.meshgrid(ug, vg)

    dU = f_u(U, V)
    dV = g_v(U, V, a, b, eps)
    spd = np.sqrt(dU**2 + dV**2)
    spd[spd < 1e-12] = 1e-12
    log_spd = np.log1p(spd)
    norm = Normalize(vmin=log_spd.min(), vmax=log_spd.max())
    ax_phase.quiver(U, V, dU / spd, dV / spd, log_spd, cmap="cool", norm=norm, pivot="mid", alpha=0.9)

    uu = np.linspace(RES["phase_xlim"][0] - 0.2, RES["phase_xlim"][1] + 0.2, 900)
    ax_phase.plot(uu, u_null(uu), "k-", lw=2, label="u-nullcline")
    ax_phase.plot(uu, v_null(uu, a, b), "k--", lw=2, label="v-nullcline")

    for i, p in enumerate(fps, 1):
        lab, col = stab_label_of_fp(p, eps, b)
        ax_phase.plot(p[0], p[1], "o", ms=8.5, color=col, mec="k", mew=0.5)
        ax_phase.annotate(f"FP{i}", (p[0], p[1]), textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax_phase.set_title(f"Phase space (found {nfp} FP)", fontweight="bold")
    ax_phase.set_xlabel("u")
    ax_phase.set_ylabel("v")
    ax_phase.set_xlim(*RES["phase_xlim"])
    ax_phase.set_ylim(*RES["phase_ylim"])
    ax_phase.grid(True, alpha=0.25)
    ax_phase.legend(fontsize=8, loc="lower left")

    # Parameter space
    aH, bH, aS, bS = param_curves_exact(
        eps,
        b_min=0.05, b_max=5.0,
        u_eps=RES["boundary_u_eps"],
        n_u=RES["boundary_n_u"],
    )
    if len(aH):
        ax_param.plot(aH, bH, "-", lw=2, label="Hopf boundary")
    if len(aS):
        ax_param.plot(aS, bS, "--", lw=2, label="Saddle-Node boundary")
    ax_param.plot(a, b, "ko", ms=8.5)
    ax_param.annotate("(a,b) now", (a, b), textcoords="offset points", xytext=(7, 5), fontsize=9)

    ax_param.set_title("Parameter space (a-b)", fontweight="bold")
    ax_param.set_xlabel("a")
    ax_param.set_ylabel("b")
    ax_param.set_xlim(*RES["ab_xlim"])
    ax_param.set_ylim(*RES["ab_ylim"])
    ax_param.grid(True, alpha=0.25)
    ax_param.legend(fontsize=9)

    # Common grids
    kv = np.linspace(0.0, RES["k_max"], RES["n_k"])
    kv2 = kv**2
    k2v = np.linspace(
        0.0,
        (RES["k2_max"] if RES["k2_max"] is not None else RES["k_max"]**2),
        RES["n_k2"]
    )

    # ---------- FP rows ----------
    for r, p in enumerate(fps, start=1):
        fp_id = r
        u, v = p
        lab, col = stab_label_of_fp(p, eps, b)

        ax_disp = fig.add_subplot(gs[r, 0])
        ax_trdt = fig.add_subplot(gs[r, 1])
        ax_cplx = fig.add_subplot(gs[r, 2])
        ax_k2cx = fig.add_subplot(gs[r, 3])

        # =========================
        # Regular dispersion σ_reg(k)
        # =========================
        sig = sigma_curve_for_fp(p, eps, b, Du, Dv, kv)
        kc_reg = find_kc_from_sigma(kv, sig, allow_k0=True)
        kdom_reg, smax_reg, j_dom_reg = dominant_k_from_sigma(kv, sig)

        # =========================
        # Mass-conserved "dispersion" proxy:
        # σ_mc(k) = k^2 σ_reg(k)
        # (Same J(k), growth multiplied by k^2)
        # =========================
        sig_mc = kv2 * sig
        # k=0 is always 0 in MC, so a "crossing" definition is the same for k>0;
        # we can ignore the segment at k=0 to avoid trivial behavior.
        kc_mc = find_kc_from_sigma(kv, sig_mc, allow_k0=False)
        kdom_mc, smax_mc, j_dom_mc = dominant_k_from_sigma(kv, sig_mc)

        # Plot σ_reg(k)
        ax_disp.axhline(0, color="k", lw=1, alpha=0.6)
        ax_disp.plot(kv, sig, lw=2)
        ax_disp.plot([kdom_reg], [smax_reg], "o", ms=7)
        ax_disp.annotate(
            f"rFHN: k_dom={kdom_reg:.4g}\nσ_max={smax_reg:.4g}",
            (kdom_reg, smax_reg),
            textcoords="offset points", xytext=(10, 10), fontsize=9
        )
        if np.isfinite(kc_reg):
            ax_disp.plot([kc_reg], [0.0], "s", ms=7)
            ax_disp.annotate(
                f"rFHN: k_c={kc_reg:.4g}",
                (kc_reg, 0.0),
                textcoords="offset points", xytext=(10, -22), fontsize=9
            )

        ax_disp.set_title(f"FP{fp_id} dispersion σ_reg(k) — {lab}", fontweight="bold", color=col)
        ax_disp.set_xlabel("k")
        ax_disp.set_ylabel("σ = max Re(λ)")
        ax_disp.set_xlim(0, RES["k_max"])
        ax_disp.grid(True, alpha=0.25)

        # τ, Δ vs k²
        tau2 = trace_Jk(u, v, eps, b, Du, Dv, k2v)
        del2 = det_Jk(u, v, eps, b, Du, Dv, k2v)
        ax_trdt.axhline(0, color="k", lw=1, alpha=0.6)
        ax_trdt.plot(k2v, tau2, lw=2, label="τ=Tr(Jk)")
        ax_trdt.plot(k2v, del2, lw=2, label="Δ=Det(Jk)")
        ax_trdt.set_title("τ, Δ vs k²", fontweight="bold")
        ax_trdt.set_xlabel("k²")
        ax_trdt.grid(True, alpha=0.25)
        ax_trdt.legend(fontsize=8)

        # λ_max(k) complex scatter colored by k (regular)
        tau_k = trace_Jk(u, v, eps, b, Du, Dv, kv2)
        del_k = det_Jk(u, v, eps, b, Du, Dv, kv2)
        lam = lambda_max_from_tau_delta(tau_k, del_k)

        c_norm = Normalize(vmin=float(kv.min()), vmax=float(kv.max()))
        sc1 = ax_cplx.scatter(
            lam.real, lam.imag,
            c=kv, s=RES["scatter_s"], cmap="viridis", norm=c_norm,
            linewidths=0
        )
        cbar1 = fig.colorbar(sc1, ax=ax_cplx, fraction=0.046, pad=0.02)
        cbar1.set_label("k", rotation=90)

        # special indices (regular)
        j0 = 0
        j_c_reg = None
        if np.isfinite(kc_reg):
            j_c_reg = int(np.argmin(np.abs(kv - kc_reg)))

        # offsets (regular): k=0, k_dom_reg, k_c_reg
        if j_c_reg is None:
            d0, d1, _ = smart_offset_three(lam[j0], lam[j_dom_reg], lam[j_dom_reg], base=RES["ann_scale"])
            d2 = None
        else:
            d0, d1, d2 = smart_offset_three(lam[j0], lam[j_dom_reg], lam[j_c_reg], base=RES["ann_scale"])

        arrow = dict(arrowstyle="->", lw=0.8, alpha=0.9)

        # plot + annotate k=0 (regular)
        ax_cplx.scatter(
            [lam.real[j0]], [lam.imag[j0]],
            s=RES["special_s"], marker="o",
            edgecolors="k", linewidths=RES["special_edge_lw"]
        )
        ax_cplx.annotate(
            f"k=0\nλ={lam[j0].real:+.4g}{lam[j0].imag:+.4g}i",
            (lam.real[j0], lam.imag[j0]),
            textcoords="offset points", xytext=d0,
            fontsize=RES["ann_font"], arrowprops=arrow
        )

        # plot + annotate k_dom (regular)
        ax_cplx.scatter(
            [lam.real[j_dom_reg]], [lam.imag[j_dom_reg]],
            s=RES["special_s"], marker="o",
            edgecolors="k", linewidths=RES["special_edge_lw"]
        )
        ax_cplx.annotate(
            f"rFHN k_dom={kdom_reg:.4g}\nλ={lam[j_dom_reg].real:+.4g}{lam[j_dom_reg].imag:+.4g}i\nσ={sig[j_dom_reg]:+.4g}",
            (lam.real[j_dom_reg], lam.imag[j_dom_reg]),
            textcoords="offset points", xytext=d1,
            fontsize=RES["ann_font"], arrowprops=arrow
        )

        # plot + annotate k_c (regular)
        if j_c_reg is not None:
            ax_cplx.scatter(
                [lam.real[j_c_reg]], [lam.imag[j_c_reg]],
                s=RES["special_s"], marker="s",
                edgecolors="k", linewidths=RES["special_edge_lw"]
            )
            ax_cplx.annotate(
                f"rFHN k_c={kc_reg:.4g}\nλ={lam[j_c_reg].real:+.4g}{lam[j_c_reg].imag:+.4g}i\nσ≈0",
                (lam.real[j_c_reg], lam.imag[j_c_reg]),
                textcoords="offset points", xytext=d2,
                fontsize=RES["ann_font"], arrowprops=arrow
            )

        # legend (explicit)
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                   markeredgecolor="k", markersize=7, label="k=0 (special)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                   markeredgecolor="k", markersize=8, label="k_dom (special)"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="none",
                   markeredgecolor="k", markersize=8, label="k_c (special)"),
        ]
        ax_cplx.legend(handles=handles, fontsize=8, loc="best")
        ax_cplx.set_title("rFHN: λ_max(k) on complex plane", fontweight="bold")
        ax_cplx.set_xlabel("Re(λ)")
        ax_cplx.set_ylabel("Im(λ)")
        ax_cplx.grid(True, alpha=0.25)

        # =========================
        # MC: k²λ_max(k) scatter (and MC-specific k_dom)
        # =========================
        k2lam = (kv2.astype(complex)) * lam
        sc2 = ax_k2cx.scatter(
            k2lam.real, k2lam.imag,
            c=kv, s=RES["scatter_s"], cmap="viridis", norm=c_norm,
            linewidths=0
        )
        cbar2 = fig.colorbar(sc2, ax=ax_k2cx, fraction=0.046, pad=0.02)
        cbar2.set_label("k", rotation=90)

        # MC special indices
        j_c_mc = None
        if np.isfinite(kc_mc):
            j_c_mc = int(np.argmin(np.abs(kv - kc_mc)))

        # offsets for MC labels: k=0, k_dom_mc, k_c_mc
        if j_c_mc is None:
            e0, e1, _ = smart_offset_three(k2lam[j0], k2lam[j_dom_mc], k2lam[j_dom_mc], base=RES["ann_scale"])
            e2 = None
        else:
            e0, e1, e2 = smart_offset_three(k2lam[j0], k2lam[j_dom_mc], k2lam[j_c_mc], base=RES["ann_scale"])

        # mark + annotate k=0 (MC)
        ax_k2cx.scatter(
            [k2lam.real[j0]], [k2lam.imag[j0]],
            s=RES["special_s"], marker="o",
            edgecolors="k", linewidths=RES["special_edge_lw"]
        )
        ax_k2cx.annotate(
            f"k=0\nk²λ={k2lam[j0].real:+.4g}{k2lam[j0].imag:+.4g}i",
            (k2lam.real[j0], k2lam.imag[j0]),
            textcoords="offset points", xytext=e0,
            fontsize=RES["ann_font"], arrowprops=arrow
        )

        # mark + annotate k_dom (MC)
        ax_k2cx.scatter(
            [k2lam.real[j_dom_mc]], [k2lam.imag[j_dom_mc]],
            s=RES["special_s"], marker="o",
            edgecolors="k", linewidths=RES["special_edge_lw"]
        )
        ax_k2cx.annotate(
            f"MC k_dom={kdom_mc:.4g}\nσ_mc,max={smax_mc:+.4g}\nk²λ={k2lam[j_dom_mc].real:+.4g}{k2lam[j_dom_mc].imag:+.4g}i",
            (k2lam.real[j_dom_mc], k2lam.imag[j_dom_mc]),
            textcoords="offset points", xytext=e1,
            fontsize=RES["ann_font"], arrowprops=arrow
        )

        # mark + annotate k_c (MC)
        if j_c_mc is not None:
            ax_k2cx.scatter(
                [k2lam.real[j_c_mc]], [k2lam.imag[j_c_mc]],
                s=RES["special_s"], marker="s",
                edgecolors="k", linewidths=RES["special_edge_lw"]
            )
            ax_k2cx.annotate(
                f"MC k_c={kc_mc:.4g}\nk²λ={k2lam[j_c_mc].real:+.4g}{k2lam[j_c_mc].imag:+.4g}i",
                (k2lam.real[j_c_mc], k2lam.imag[j_c_mc]),
                textcoords="offset points", xytext=e2,
                fontsize=RES["ann_font"], arrowprops=arrow
            )

        ax_k2cx.legend(handles=handles, fontsize=8, loc="best")
        ax_k2cx.set_title("MC: k²·λ_max(k) on complex plane", fontweight="bold")
        ax_k2cx.set_xlabel("Re(k²λ)")
        ax_k2cx.set_ylabel("Im(k²λ)")
        ax_k2cx.grid(True, alpha=0.25)

    fig.suptitle(
        f"rFHN/MC linear stability | a={a}, b={b}, eps={eps}, Du={Du}, Dv={Dv}",
        fontsize=14, fontweight="bold"
    )
    return fig


def main():
    p = PARAMS
    print("\n=== RUNNING PARAMETERS ===")
    for key in ["a", "b", "eps", "Du", "Dv"]:
        print(f"  {key:<3} = {p[key]}")
    print("==========================\n")

    plot_one_figure(p["a"], p["b"], p["eps"], p["Du"], p["Dv"])
    plt.show()


if __name__ == "__main__":
    main()