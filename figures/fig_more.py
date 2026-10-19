"""fig_more.pdf: (a) Euler characteristic from mean clustering, (b) planted-edge BF at fixed lambda,
(c) correlation of BF on two edges sharing a vertex against the exact Gaussian limit of Thm joint.
Inputs: ../results/e2_summary.json, ../results/e5_summary.json, ../results/oq1_pair.json.
Panel (c) computes the limit with fig_7c_rerun.limit_corr; the 'theory' field of oq1_pair.json is not used.
"""

import numpy as np
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bflib.bfsim import Psi
from fig_7c_rerun import limit_corr
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
e2_summary = json.load(open(results_dir + "e2_summary.json"))
e5_summary = json.load(open(results_dir + "e5_summary.json"))
pair_corr = json.load(open(results_dir + "oq1_pair.json"))
T_PAIR = 0.75  # normalised length of both edges in run_oq1pair.py

fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3))

geo_labels = {"bolza": "Bolza ($g=2$)", "S2": "$S^2$", "T2": "$T^2$"}
selected = [
    entry
    for entry in e2_summary
    if abs(entry["r"] * entry["r"] / entry["A"] - 0.0286) < 1e-3
]
for pos, entry in enumerate(selected):
    ax[0].errorbar(pos, entry["chihat"], yerr=2 * entry["chise"], fmt="o", color="C0", capsize=2)
    ax[0].plot(pos, entry["chi"], "kx", ms=7)
ax[0].set_xticks(range(len(selected)))
ax[0].set_xticklabels([geo_labels[entry["geo"]] for entry in selected])
ax[0].set_ylabel(r"$\hat\chi$")
ax[0].axhline(0, color="k", lw=0.4)
ax[0].set_title("(a) Euler characteristic", fontsize=8.5)

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

angles = [entry["theta"] for entry in pair_corr]
ax[2].errorbar(
    angles,
    [entry["corr"] for entry in pair_corr],
    yerr=[2 * entry["se"] for entry in pair_corr],
    fmt="o",
    color="C0",
    capsize=2,
    label="simulation",
)
grid = np.linspace(min(angles), max(angles), 200)
ax[2].plot(
    grid,
    [limit_corr(angle, T_PAIR, T_PAIR) for angle in grid],
    "k--",
    lw=0.8,
    label="Gaussian limit",
)
ax[2].axhline(0, color="k", lw=0.4)
ax[2].set_xlabel("angle between edges (deg)")
ax[2].set_ylabel("correlation")
ax[2].legend(frameon=False)
ax[2].set_title("(c) edges sharing a vertex", fontsize=8.5)

fig.tight_layout(pad=0.3)
fig.savefig(f"{FIGS}/fig_more.pdf")
