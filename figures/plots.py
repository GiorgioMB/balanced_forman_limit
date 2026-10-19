import numpy as np
import sys
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

from bflib.bfsim import *
from bflib.paths import RES, FIG

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
    }
)

T_VALUES = [0.25, 0.5, 0.75, 1.0]
COLORS = {0.25: "#1b9e77", 0.5: "#d95f02", 0.75: "#7570b3", 1.0: "#e7298a"}
LAMS = [100, 200, 400, 800, 1600, 3200, 6400, 12800]

flat_edges = {dim: np.load(f"{RES}/flat_edges_d{dim}.npz") for dim in [2, 3]}
sq_const = json.load(open(f"{RES}/sq_const.json"))


def sn_pdf(w_vals, f):
    scale = np.sqrt(f * (1 - f) * (9 - 4 * f))
    skew = -np.sqrt(f / (2 - f)) / 3
    return 2 / scale * norm.pdf(w_vals / scale) * norm.cdf(skew * w_vals / scale)


# ---------- Fig: rate / variance / bias ----------
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.35))
for dim, marker, linestyle in [(2, "o", "-"), (3, "s", "--")]:
    for t in T_VALUES:
        f = float(f_lens(dim, t))
        psi_t = float(Psi(dim, t))
        rms = []
        lam_var = []
        bias_scaled = []
        bias_se = []
        for lam in LAMS:
            samples = flat_edges[dim][f"{dim}_{lam}_{t}"]
            bf = samples[:, 0]
            sq = samples[:, 1]
            rms.append(np.sqrt(((bf - psi_t) ** 2).mean()))
            lam_var.append(lam * bf.var())
            bias_scaled.append(
                np.sqrt(lam) * (bf.mean() - psi_t - (4 - 3 * f) / lam - sq.mean())
            )
            bias_se.append(np.sqrt(lam) * bf.std() / np.sqrt(len(bf)))

        lams = np.array(LAMS)
        ax[0].loglog(
            lams,
            rms,
            marker,
            ms=2.8,
            color=COLORS[t],
            mfc="none" if dim == 3 else COLORS[t],
            mew=0.7,
        )
        ax[0].loglog(
            lams,
            np.sqrt((W_var(f) + W_mean(f) ** 2) / lams),
            linestyle,
            color=COLORS[t],
            lw=0.7,
        )
        ax[1].semilogx(
            lams,
            np.array(lam_var) / W_var(f),
            marker + linestyle,
            ms=2.8,
            color=COLORS[t],
            mfc="none" if dim == 3 else COLORS[t],
            mew=0.7,
            lw=0.6,
        )
        ax[2].errorbar(
            lams * (1.04 if dim == 3 else 0.96),
            np.array(bias_scaled) / W_mean(f),
            yerr=2 * np.array(bias_se) / abs(W_mean(f)),
            fmt=marker,
            ms=2.8,
            color=COLORS[t],
            mfc="none" if dim == 3 else COLORS[t],
            mew=0.7,
            elinewidth=0.5,
            capsize=0,
        )

ax[0].set_xlabel(r"$\lambda$")
ax[0].set_ylabel(r"RMS of $\mathrm{BF}-\Psi_d(t)$")
ax[0].set_title(r"(a) rate $\lambda^{-1/2}$", fontsize=9)

ax[1].axhline(1, color="k", lw=0.6)
ax[1].set_ylim(0.8, 1.2)
ax[1].set_xlabel(r"$\lambda$")
ax[1].set_ylabel(r"$\lambda\,\mathrm{Var}(\mathrm{BF})\,/\,\mathrm{Var}(W)$")
ax[1].set_title(r"(b) variance", fontsize=9)

ax[2].set_xscale("log")
ax[2].axhline(1, color="k", lw=0.6)
ax[2].set_ylim(0, 2)
ax[2].set_xlabel(r"$\lambda$")
ax[2].set_ylabel(
    r"$\sqrt{\lambda}\,(\mathrm{bias}-\frac{4-3f}{\lambda}-\mathrm{E}\,S)\,/\,\mathrm{E}W$"
)
ax[2].set_title(r"(c) $\lambda^{-1/2}$ bias", fontsize=9)

from matplotlib.lines import Line2D

legend_handles = [
    Line2D([], [], color=COLORS[t], marker="o", ls="", ms=3, label=f"$t={t}$")
    for t in T_VALUES
] + [
    Line2D([], [], color="gray", marker="o", ls="-", ms=3, label="$d=2$"),
    Line2D([], [], color="gray", marker="s", mfc="none", ls="--", ms=3, label="$d=3$"),
]
ax[0].legend(
    handles=legend_handles,
    loc="lower left",
    frameon=False,
    ncol=2,
    handlelength=1.2,
    columnspacing=0.6,
)
fig.tight_layout(pad=0.3, w_pad=0.6)
fig.savefig(f"{FIG}/fig_rate.pdf")
plt.close(fig)


# ---------- Fig: CLT ----------
fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.3))
for panel, (dim, t) in enumerate([(2, 0.5), (3, 1.0)]):
    f = float(f_lens(dim, t))
    w_chunks = []
    for lam in [1600, 3200, 6400]:
        samples = flat_edges[dim][f"{dim}_{lam}_{t}"]
        bf = samples[:, 0]
        sq = samples[:, 1]
        w_chunks.append(np.sqrt(lam) * (bf - Psi(dim, t) - (4 - 3 * f) / lam - sq))
    w_samples = np.concatenate(w_chunks)

    ax[panel].hist(
        w_samples,
        bins=45,
        density=True,
        color="0.82",
        edgecolor="0.55",
        lw=0.35,
        label=f"simulation ($N={len(w_samples)}$)",
    )

    w_grid = np.linspace(-5, 5, 400)
    ax[panel].plot(w_grid, sn_pdf(w_grid, f), "k", lw=1.1, label="skew-normal limit $W$")
    ax[panel].plot(
        w_grid,
        norm.pdf(w_grid, 0, np.sqrt(W_var(f))),
        "k:",
        lw=0.8,
        label="$N(0,\\mathrm{Var}\\,W)$",
    )

    ax[panel].axvline(W_mean(f), color="k", lw=0.5, ls="--")
    ax[panel].set_xlim(-5, 5)
    ax[panel].set_ylim(0, 0.48)
    ax[panel].set_title(
        f"$d={dim}$, $t={t}$, $\\lambda\\in\\{{1600,3200,6400\\}}$", fontsize=8.5
    )
    ax[panel].set_xlabel(
        r"$\sqrt{\lambda}\,(\mathrm{BF}-\Psi_d(t)-S-(4-3f_d)/\lambda)$"
    )

ax[0].legend(frameon=False, loc="upper left", fontsize=6.5)
fig.tight_layout(pad=0.3)
fig.savefig(f"{FIG}/fig_clt.pdf")
plt.close(fig)


# ---------- Fig: 4-cycle term ----------
fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.2), sharey=False)
for panel, dim in enumerate([2, 3]):
    for t in T_VALUES:
        y = [
            LAMS[lam_idx] * flat_edges[dim][f"{dim}_{lam}_{t}"][:, 1].mean()
            for lam_idx, lam in enumerate(LAMS)
        ]
        ax[panel].semilogx(
            LAMS, y, "o-", ms=2.8, color=COLORS[t], lw=0.7, label=f"$t={t}$"
        )
        ax[panel].axhline(
            sq_const[f"{dim}_{t}"]["lim"], color=COLORS[t], ls=":", lw=0.9
        )

    ax[panel].set_xlabel(r"$\lambda$")
    ax[panel].set_ylabel(r"$\lambda\,\mathrm{E}\,S$")
    ax[panel].set_title(f"$d={dim}$", fontsize=9)

ax[0].legend(frameon=False, loc="lower right", ncol=2)
fig.tight_layout(pad=0.3)
fig.savefig(f"{FIG}/fig_sq.pdf")
plt.close(fig)


# ---------- Fig: global ----------
def phi_inf(bf_vals, d):
    omega_ratio = omega(d) / omega(d - 1)
    tau = np.array([brentq(lambda s: Psi(d, s) - bf_val, 0, 1) for bf_val in bf_vals])

    return d * omega_ratio / 3 * tau ** (d - 1) * (1 - tau**2 / 4) ** (-(d - 1) / 2)


fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3))
for panel, (dim, tag) in enumerate([(2, "n10000_lam400"), (3, "n25000_lam200")]):
    data = np.load(f"{RES}/global_d{dim}_{tag}.npz")
    bf = data["bf"]

    ax[panel].hist(
        bf,
        bins=60,
        density=True,
        color="0.8",
        edgecolor="0.55",
        lw=0.3,
        label=f"all $|E|={len(bf):,}$ edges",
    )

    y = np.linspace(Psi(dim, 1.0) + 1e-6, 1 - 1e-6, 300)
    ax[panel].plot(y, phi_inf(y, dim), "k", lw=1.1, label=r"$\varphi_\infty$")
    ax[panel].set_title(
        f'$d={dim}$, $n={int(data["n"])}$, $\\lambda={int(data["lam"])}$', fontsize=9
    )
    ax[panel].set_xlabel("BF curvature")
    ax[panel].legend(frameon=False, loc="upper right", fontsize=7)

global_summary = json.load(open(f"{RES}/global_summary.json"))
for dim, marker in [(2, "o"), (3, "s")]:
    global_rows = [row for row in global_summary if row["d"] == dim]
    lams = np.array([row["lam"] for row in global_rows])
    ax[2].loglog(
        lams,
        [row["maxdev"] for row in global_rows],
        marker + "-",
        color="C3",
        ms=3,
        lw=0.7,
        mfc="none" if dim == 3 else "C3",
    )
    ax[2].loglog(
        lams,
        [row["meandev"] for row in global_rows],
        marker + "-",
        color="C0",
        ms=3,
        lw=0.7,
        mfc="none" if dim == 3 else "C0",
    )
    ax[2].loglog(
        lams,
        [row["W1"] for row in global_rows],
        marker + "-",
        color="C2",
        ms=3,
        lw=0.7,
        mfc="none" if dim == 3 else "C2",
    )

lams = np.array([25, 400])
ax[2].loglog(lams, 1.9 * np.sqrt(np.log(1e4) / lams), "k--", lw=0.7)
ax[2].loglog(lams, 1.0 / np.sqrt(lams), "k:", lw=0.7)

ax[2].text(140, 0.75, r"$\max_e|\mathrm{BF}-\Psi|$", color="C3", fontsize=7)
ax[2].text(26, 0.062, r"mean$_e|\mathrm{BF}-\Psi|$", color="C0", fontsize=7)
ax[2].text(26, 0.0075, r"$W_1(\mu_n,\mu_\infty)$", color="C2", fontsize=7)

ax[2].set_xlabel(r"$\lambda$")
ax[2].set_title(
    r"dashed: $1.9\sqrt{\log n/\lambda}$; dotted: $\lambda^{-1/2}$", fontsize=7.5
)
fig.tight_layout(pad=0.3, w_pad=0.5)
fig.savefig(f"{FIG}/fig_global.pdf")
plt.close(fig)


# ---------- Fig: curved ----------
curved_summary = json.load(open(f"{RES}/curved_summary.json"))
CURVED_R = 0.5

CURVED_GEOS = {
    "S2": (2, 1.5, r"$S^2$ ($\rho=3/2$)", "C3", "o"),
    "H2": (2, -1.5, r"$H^2$ ($\rho=-3/2$)", "C0", "v"),
    "S3": (3, 3.2, r"$S^3$ ($\rho=16/5$)", "C1", "s"),
    "S2xR_S": (
        3,
        1.4,
        r"$S^2\!\times\!\mathbb{R}$, $v\in TS^2$ ($\rho=7/5$)",
        "C2",
        "D",
    ),
    "S2xR_R": (
        3,
        0.4,
        r"$S^2\!\times\!\mathbb{R}$, $v\in T\mathbb{R}$ ($\rho=2/5$)",
        "C4",
        "^",
    ),
}

fig, ax = plt.subplots(1, 1, figsize=(4.6, 2.9))
t_grid = np.linspace(0, 1, 200)
for geo, (d, rho, label, color, marker) in CURVED_GEOS.items():
    ax.plot(t_grid, 3 * kappa(d, t_grid) * rho * CURVED_R**2, color=color, lw=0.8)
    curved_rows = [row for row in curved_summary if row["geo"] == geo]
    ax.errorbar(
        [row["t"] for row in curved_rows],
        [row["diff"] for row in curved_rows],
        yerr=[2 * row["se"] for row in curved_rows],
        fmt=marker,
        color=color,
        ms=3.5,
        capsize=1.5,
        elinewidth=0.7,
        label=label,
    )
    ax.plot(
        [row["t"] for row in curved_rows],
        [row["exact"] for row in curved_rows],
        "x",
        color="k",
        ms=3.5,
        mew=0.7,
    )

ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel(r"$t=\delta/r$")
ax.set_ylabel(r"$\mathrm{E\,BF}_{M}-\mathrm{E\,BF}_{\mathrm{flat}}$")
ax.legend(frameon=False, fontsize=6.6, loc="lower left", bbox_to_anchor=(0.0, 0.0))
ax.set_title(
    r"$r=0.5$, $\lambda=1500$; lines: $3\kappa_d(t)\rho\,r^2$; $\times$: exact lens",
    fontsize=8,
)
fig.tight_layout(pad=0.3)
fig.savefig(f"{FIG}/fig_curved.pdf")
plt.close(fig)

print("ok")
