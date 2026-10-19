import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import an_v1_lib as v1
from bflib.bfsim import Psi
from bflib.paths import RESULTS, FIGS

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
summary = json.load(open(f"{RESULTS}/e1_summary.json"))
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3))

fits = {}
for ref, d in [("flat2_l2000_r1.0", 2), ("flat3_l600_r1.0", 3), ("flat2_l500_r1.0", 2)]:
    fits[ref] = v1.bias_fit(v1.load(ref), d)

for name, ref, d, r, color, label in [
    ("S2_l2000_r0.5", "flat2_l2000_r1.0", 2, 0.5, "C3", "$S^2$ (Scal $=2$)"),
    ("H2_l2000_r0.5", "flat2_l2000_r1.0", 2, 0.5, "C0", "$H^2$ (Scal $=-2$)"),
    ("S3_l600_r0.6", "flat3_l600_r1.0", 3, 0.6, "C1", "$S^3$ (Scal $=6$)"),
]:
    scal_est = v1.scal_hat(v1.load(name), d, r, fits[ref])
    ax[0].hist(
        scal_est, bins=30, histtype="step", color=color, density=True, label=label
    )
ax[0].set_xlabel(r"$\widehat{\mathrm{Scal}}_i$")
ax[0].legend(frameon=False, fontsize=6.5)
ax[0].set_title("(a) per-vertex estimates", fontsize=8.5)

points = [
    (record["lr4"], record["rmse"]) for key, record in summary.items() if key.startswith("S2_l")
]
lr4, rmse = np.array(points).T
ax[1].loglog(lr4, rmse, "o", color="C3", ms=4, label="$S^2$")
for key, color in [
    ("S3_l600_r0.6", "C1"),
    ("S2xR_l600_r0.6", "C2"),
    ("H2_l2000_r0.5", "C0"),
]:
    ax[1].loglog(summary[key]["lr4"], summary[key]["rmse"], "s", color=color, ms=4)
x_line = np.array([3, 150])
ax[1].loglog(x_line, 11 * x_line**-0.5, "k--", lw=0.7)
ax[1].set_xlabel(r"$\lambda r^4$")
ax[1].set_ylabel("RMSE")
ax[1].set_title(r"(b) error $\propto(\lambda r^4)^{-1/2}$", fontsize=8.5)

ricci = summary["ricci_S2xR_l600_r0.6"]
Ric = np.array(ricci["Ric"])
Ric_se = np.array(ricci["Ric_se"])
labels = ["$R_{11}$", "$R_{22}$", "$R_{33}$", "$R_{12}$", "$R_{13}$", "$R_{23}$"]
entries = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
Ric_true = [1, 1, 0, 0, 0, 0]
ax[2].errorbar(
    range(6),
    [Ric[entry] for entry in entries],
    yerr=[2 * Ric_se[entry] for entry in entries],
    fmt="o",
    color="C2",
    ms=4,
    capsize=2,
)
ax[2].plot(range(6), Ric_true, "kx", ms=6)
ax[2].set_xticks(range(6))
ax[2].set_xticklabels(labels, fontsize=7)
ax[2].set_title(r"(c) Ricci tensor, $S^2\times\mathbb{R}$", fontsize=8.5)
ax[2].axhline(0, color="k", lw=0.4)

fig.tight_layout(pad=0.3)
fig.savefig(f"{FIGS}/fig_e1.pdf")
