"""Figure S4: (a) plug-in Euler characteristic, (b) planted edges at fixed lambda, (c) two edges sharing a
vertex. Draws from the summaries of the clustering, degree and pairs experiments."""

import numpy as np

from ..pipeline import figure_style
from ..theory import Psi, pair_limit_correlation

NAME = "figure_s4"
TITLE = "Figure S4"
ITEMS = ["Figure S4"]
REQUIRES = ["clustering", "degree", "pairs"]


def run(ctx):
    pass


def analyse(ctx):
    plt = figure_style()
    clustering = ctx.load_summary("clustering")
    fixed = ctx.load_summary("degree")["fixed"]
    pairs = ctx.load_summary("pairs")
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3))
    labels = {"bolza": "Bolza ($g=2$)", "S2": "$S^2$", "T2": "$T^2$"}
    selected = [row for row in clustering if abs(row["r2_over_area"] - 0.0286) < 1e-3]
    for pos, row in enumerate(selected):
        ax[0].errorbar(pos, row["chi_hat"], yerr=2 * row["chi_se"], fmt="o", color="C0", capsize=2)
        ax[0].plot(pos, row["chi"], "kx", ms=7)
    ax[0].set_xlim(-0.5, len(selected) - 0.5)
    ax[0].set_xticks(range(len(selected)))
    ax[0].set_xticklabels([labels[row["surface"]] for row in selected])
    ax[0].set_ylabel(r"$\hat\chi$")
    ax[0].axhline(0, color="k", lw=0.4)
    ax[0].set_title("(a) Euler characteristic", fontsize=8.5)
    lam = np.array([row["lam"] for row in fixed])
    ax[1].errorbar(lam, [row["mean"] for row in fixed], yerr=[row["sd"] for row in fixed], fmt="o", color="C3",
                   capsize=2, label="mean $\\pm$ sd")
    ax[1].axhline(float(Psi(2, 0.75)), color="k", lw=0.6, ls="--", label=r"$\Psi_2(0.75)$")
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$\lambda$ (fixed)")
    ax[1].set_title(r"(b) fixed $\lambda$, $t=0.75$", fontsize=8.5)
    ax[1].legend(frameon=False)
    angles = [row["theta"] for row in pairs]
    ax[2].errorbar(angles, [row["corr"] for row in pairs], yerr=[2 * row["se"] for row in pairs], fmt="o", color="C0",
                   capsize=2, label="simulation")
    grid = np.linspace(min(angles), max(angles), 200)
    ax[2].plot(grid, [pair_limit_correlation(a) for a in grid], "k--", lw=0.8, label="Gaussian limit")
    ax[2].axhline(0, color="k", lw=0.4)
    ax[2].set(xlabel="angle between edges (deg)", ylabel="correlation")
    ax[2].legend(frameon=False)
    ax[2].set_title("(c) edges sharing a vertex", fontsize=8.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(ctx.figure("fig_more.pdf"))
    plt.close(fig)
