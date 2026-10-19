import glob
import json
import sys

import numpy as np

from bflib.bfsim import Psi, f_lens, W_mean, W_var
from bflib.paths import RES

TS = [0.25, 0.5, 0.75, 1.0]
LAMS = [100, 200, 400, 800, 1600, 3200, 6400, 12800]

summary = {}
for dim in [2, 3]:
    samples = {}
    for t in TS:
        for lam in LAMS:
            files = sorted(glob.glob(f"{RES}/flat/f_{dim}_{lam}_{t}_*.npy"))
            samples[f"{dim}_{lam}_{t}"] = np.concatenate(
                [np.load(path) for path in files]
            )
    np.savez_compressed(f"{RES}/flat_edges_d{dim}.npz", **samples)

    for t in TS:
        f = float(f_lens(dim, t))
        psi = float(Psi(dim, t))
        lam_grid = np.array(LAMS, float)

        rms = []
        var_ratio = []
        z_scores = []
        lam_sq = []

        for lam in LAMS:
            sample = samples[f"{dim}_{lam}_{t}"]
            bf = sample[:, 0]
            sq = sample[:, 1]
            n_samples = len(bf)

            rms.append(np.sqrt(((bf - psi) ** 2).mean()))
            var_ratio.append(lam * bf.var(ddof=1) / W_var(f))

            centred = np.sqrt(lam) * (bf.mean() - psi - (4 - 3 * f) / lam - sq.mean())
            se = np.sqrt(lam) * np.std(bf - sq, ddof=1) / np.sqrt(n_samples)
            z_scores.append((centred - W_mean(f)) / se)

            lam_sq.append(lam * sq.mean())

        large_lam = lam_grid >= 400
        slope = np.polyfit(
            np.log(lam_grid[large_lam]), np.log(np.array(rms)[large_lam]), 1
        )[0]

        summary[f"{dim}_{t}"] = dict(
            slope=float(slope),
            varratio=[float(val) for val in var_ratio],
            z=[float(val) for val in z_scores],
            lamES=[float(val) for val in lam_sq],
            rms=[float(val) for val in rms],
        )

json.dump(summary, open(f"{RES}/flat_summary.json", "w"), indent=1)

slopes = [entry["slope"] for entry in summary.values()]
var_ratio = np.concatenate([entry["varratio"] for entry in summary.values()])
z_all = np.concatenate([np.array(entry["z"])[2:] for entry in summary.values()])

print("slopes", min(slopes), max(slopes), np.mean(slopes))
print("varratio", var_ratio.min(), var_ratio.max(), var_ratio.mean(), len(var_ratio))
print("z (lam>=400)", len(z_all), z_all.mean(), z_all.std(ddof=1), np.abs(z_all).max())

for cfg, entry in summary.items():
    print(
        cfg,
        "1600:", round(entry["lamES"][4], 2),
        "12800:", round(entry["lamES"][7], 2),
    )
