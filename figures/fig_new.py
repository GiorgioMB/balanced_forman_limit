import numpy as np
import json
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bflib.bfsim import Psi, f_lens, kappa
from bflib.paths import RESULTS, FIGS

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 9,
        "legend.fontsize": 7,
    }
)

results_dir = f"{RESULTS}/"
figure_name = sys.argv[1]


if figure_name == "e23":
    e2_summary = json.load(open(results_dir + "e2_summary.json"))
    e3_summary = json.load(open(results_dir + "e3_summary.json"))
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4))

    geo_labels = {"bolza": "Bolza (genus 2)", "S2": "$S^2$", "T2": "$T^2$"}
    for pos, x in enumerate(
        [entry for entry in e2_summary if abs(entry["r"] * entry["r"] / entry["A"] - 0.0286) < 1e-3]
    ):
        ax[0].errorbar(
            pos, x["chihat"], yerr=2 * x["chise"], fmt="o", color="C0", capsize=2
        )
        ax[0].plot(pos, x["chi"], "kx", ms=7)
    ax[0].set_xticks([0, 1, 2])
    ax[0].set_xticklabels(
        [
            geo_labels[entry["geo"]]
            for entry in e2_summary
            if abs(entry["r"] * entry["r"] / entry["A"] - 0.0286) < 1e-3
        ]
    )
    ax[0].set_ylabel(r"$\hat\chi$")
    ax[0].set_title(r"(a) Euler characteristic from mean clustering", fontsize=8.5)
    ax[0].axhline(0, color="k", lw=0.4)

    for t, c in [(0.75, "C3"), (1.0, "C1")]:
        rows = [
            entry
            for entry in e3_summary
            if entry["t"] == t and entry["mode"] == "fixed" and entry["dir"] == "par"
        ]
        radii = [entry["r"] for entry in rows]

        ax[1].errorbar(
            radii,
            [entry["diff"] for entry in rows],
            yerr=[2 * entry["se"] for entry in rows],
            fmt="o",
            color=c,
            capsize=2,
            label=f"parallel, $t={t}$",
        )
        ax[1].plot(radii, [entry["pred"] for entry in rows], "--", color=c, lw=0.8)
        ax[1].plot(radii, [entry["pred_folded"] for entry in rows], ":", color=c, lw=1)

        perp_rows = [entry for entry in e3_summary if entry["t"] == t and entry["dir"] == "perp"]
        ax[1].errorbar(
            [entry["r"] for entry in perp_rows],
            [entry["diff"] for entry in perp_rows],
            yerr=[2 * entry["se"] for entry in perp_rows],
            fmt="s",
            mfc="none",
            color=c,
            capsize=2,
        )

        adapt_rows = [entry for entry in e3_summary if entry["t"] == t and entry["mode"] == "adapt"]
        ax[1].errorbar(
            [entry["r"] for entry in adapt_rows],
            [entry["diff"] for entry in adapt_rows],
            yerr=[2 * entry["se"] for entry in adapt_rows],
            fmt="^",
            color=c,
            mfc="none",
            capsize=2,
        )

    ax[1].axhline(0, color="k", lw=0.4)
    ax[1].set_xlabel("$r$")
    ax[1].set_ylabel(r"$\mathrm{E\,BF}_\varrho-\mathrm{E\,BF}_{\rm flat}$")
    ax[1].set_title(r"(b) density gradient", fontsize=8)
    ax[1].legend(frameon=False, loc="lower left")

    fig.tight_layout(pad=0.3)
    fig.savefig(f"{FIGS}/fig_e2e3.pdf")


if figure_name == "e45":
    e4_summary = json.load(open(results_dir + "e4_summary.json"))
    e5_summary = json.load(open(results_dir + "e5_summary.json"))
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3))

    for pos, x in enumerate(e4_summary):
        ax[0].errorbar(pos, x["diff"], yerr=2 * x["se"], fmt="o", color="C0", capsize=2)
        ax[0].plot(pos, x["pred"], "kx", ms=6)
    ax[0].set_xticks(range(len(e4_summary)))
    ax[0].set_xticklabels(
        [
            ("C" if entry["dir"] == "circ" else "A") + f"\n{entry['r']}\n{entry['t']}"
            for entry in e4_summary
        ],
        fontsize=6,
    )
    ax[0].axhline(0, color="k", lw=0.4)
    ax[0].set_title("(a) cylinder, ambient distances", fontsize=8.5)
    ax[0].set_xlabel("dir (C/A), $r$, $t$", fontsize=7)

    fixed_lam = e5_summary["fixed"]
    lam = np.array([entry["lam"] for entry in fixed_lam])
    ax[1].errorbar(
        lam,
        [entry["mean"] for entry in fixed_lam],
        yerr=[entry["sd"] for entry in fixed_lam],
        fmt="o",
        color="C3",
        capsize=2,
        label="mean $\\pm$ sd",
    )
    ax[1].axhline(Psi(2, 0.75), color="k", lw=0.6, ls="--", label=r"$\Psi_2(0.75)$")
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$\lambda$ (fixed)")
    ax[1].set_title(r"(b) fixed $\lambda$, $t=0.75$", fontsize=8.5)
    ax[1].legend(frameon=False)

    bin_edges = np.array(e5_summary["knn"][0]["bins"])
    c = (bin_edges[:-1] + bin_edges[1:]) / 2

    for rec, line_color, label in zip(
        e5_summary["knn"],
        ["C0", "C2", "k"],
        ["union $k$NN, $k=20$", "union $k$NN, $k=50$", "RGG, $\\lambda=50$"],
    ):
        bin_means = [np.nan if value is None else value for value in rec["mean_by_bin"]]
        ax[2].plot(c, bin_means, "o-", ms=2.5, lw=0.7, color=line_color, label=label)

    t_grid = np.linspace(0, 1, 100)
    ax[2].plot(t_grid, Psi(2, t_grid), "r--", lw=0.8, label=r"$\Psi_2$")
    ax[2].set_xlabel(r"$t=d/r_k$")
    ax[2].legend(frameon=False, fontsize=6)
    ax[2].set_title("(c) $k$-NN graphs", fontsize=8.5)

    fig.tight_layout(pad=0.3)
    fig.savefig(f"{FIGS}/fig_e4e5.pdf")


if figure_name == "e6oq":
    e6_summary = json.load(open(results_dir + "e6_summary.json"))
    rows = json.load(open(results_dir + "oq1_pair.json"))
    oq2_summary = json.load(open(results_dir + "oq2_summary.json"))
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3))

    budgets = ["0", "50", "100", "200", "400"]
    for strategy, c, label in [
        ("random", "0.5", "random"),
        ("raw", "C3", "raw BF"),
        ("cen", "C0", "centred BF"),
        ("oracle", "C2", "corridor (oracle)"),
    ]:
        ax[0].plot(
            [int(budget) for budget in budgets],
            [e6_summary["rewire"][strategy][budget]["mean"] for budget in budgets],
            "o-",
            ms=3,
            color=c,
            label=label,
        )
    ax[0].set_xlabel("added edges")
    ax[0].set_ylabel(r"mean $R_{\rm eff}$")
    ax[0].legend(frameon=False, fontsize=6)
    ax[0].set_title("(a) rewiring two rooms", fontsize=8.5)

    angles = [entry["theta"] for entry in rows]
    ax[1].errorbar(
        angles,
        [entry["corr"] for entry in rows],
        yerr=[2 * entry["se"] for entry in rows],
        fmt="o",
        color="C0",
        capsize=2,
        label="simulation",
    )
    ax[1].plot(
        angles, [entry["theory"] for entry in rows], "k--", lw=0.8, label="Gaussian cell limit"
    )
    ax[1].axhline(0, color="k", lw=0.4)
    ax[1].set_xlabel(r"angle between edges (deg)")
    ax[1].set_ylabel("correlation")
    ax[1].legend(frameon=False)
    ax[1].set_title("(b) two edges sharing a vertex", fontsize=8.5)

    x = np.arange(len(oq2_summary))
    bar_width = 0.35
    ax[2].bar(
        x - bar_width / 2,
        [entry["BF"]["shift"] for entry in oq2_summary],
        bar_width,
        yerr=[2 * entry["BF"]["se"] for entry in oq2_summary],
        color="C3",
        label="BF",
        capsize=2,
    )
    ax[2].bar(
        x + bar_width / 2,
        [entry["ORC"]["shift"] for entry in oq2_summary],
        bar_width,
        yerr=[2 * entry["ORC"]["se"] for entry in oq2_summary],
        color="C0",
        label="Ollivier (data cloud)",
        capsize=2,
    )

    ax[2].plot(x - bar_width / 2, [entry["BF_pred"] for entry in oq2_summary], "kx")
    ax[2].plot(x + bar_width / 2, [entry["ORC_ollivier_ref"] for entry in oq2_summary], "k+")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(
        [
            "$S^2$",
            "$H^2$",
            r"$S^2\!\times\!\mathbb{R}$" + "\n$TS^2$",
            r"$S^2\!\times\!\mathbb{R}$" + "\n$T\mathbb{R}$",
        ],
        fontsize=6.5,
    )
    ax[2].axhline(0, color="k", lw=0.4)
    ax[2].legend(frameon=False, fontsize=6)
    ax[2].set_title("(c) curved minus flat shift", fontsize=8.5)

    fig.tight_layout(pad=0.3)
    fig.savefig(f"{FIGS}/fig_e6oq.pdf")
