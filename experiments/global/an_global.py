"""Summaries for all-edge RGG experiments on T^d (fullgraph.c output)."""

import glob
import json
import sys

import numpy as np

from bflib.bfsim import Psi, f_lens
from bflib.paths import RES

rows = []
for d, n, lams in [
    (2, 10000, [25, 50, 100, 200, 400]),
    (3, 25000, [25, 50, 100, 200]),
]:
    for lam in lams:
        data = np.concatenate(
            [
                np.fromfile(path, dtype=np.float32).reshape(-1, 3)
                for path in sorted(glob.glob(f"{RES}/g_{d}_{n}_{lam}_*.bin"))
            ]
        ).astype(float)

        t, bf, sq = data[:, 0], data[:, 1], data[:, 2]
        dev = np.abs(bf - Psi(d, t))
        n_edges = len(bf)

        # W1 between empirical law of BF and law of Psi_d(T), T density d t^{d-1}: quantile coupling
        levels = (np.arange(n_edges) + 0.5) / n_edges
        limit_quantiles = np.sort(Psi(d, (1 - levels) ** (1 / d)))
        W1 = float(np.mean(np.abs(np.sort(bf) - limit_quantiles)))

        row = dict(
            d=d,
            n=n,
            lam=lam,
            E=n_edges,
            W1=W1,
            meandev=float(dev.mean()),
            maxdev=float(dev.max()),
            ratio=float(dev.max() / np.sqrt(np.log(n) / lam)),
            meanbf=float(bf.mean()),
            meanlim=float(-2 + 4.5 * f_lens(d, 1.0)),
            lam_max_sq=float(lam * sq[t >= 0.25].max()),
        )
        rows.append(row)
        print(row)

        if (d, lam) in [(2, 400), (3, 200)]:
            np.savez_compressed(
                f"{RES}/global_d{d}_n{n}_lam{lam}.npz",
                bf=bf.astype(np.float32),
                n=n,
                lam=lam,
            )

json.dump(rows, open(f"{RES}/global_summary.json", "w"), indent=1)
