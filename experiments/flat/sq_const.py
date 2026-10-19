import json
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from bflib.bfsim import omega, f_lens
from bflib.paths import RES

rng = np.random.default_rng(7)


def sample_lune(d, t, side, n_samples):
    centre_i = np.zeros(d)
    centre_i[0] = -t / 2
    centre_j = -centre_i
    center = centre_j if side == "j" else centre_i
    other = centre_i if side == "j" else centre_j

    batches = []
    while sum(len(batch) for batch in batches) < n_samples:
        pts = center + rng.uniform(-1, 1, (2 * n_samples, d))
        pts = pts[
            (((pts - center) ** 2).sum(1) <= 1) & (((pts - other) ** 2).sum(1) > 1)
        ]
        batches.append(pts)

    return np.concatenate(batches)[:n_samples]


results = {}
for d in [2, 3]:
    for t in [0.25, 0.5, 0.75, 1.0]:
        f = float(f_lens(d, t))
        vol_U = omega(d) * (1 - f)

        lune_j_pts = sample_lune(d, t, "j", 400000 if d == 2 else 600000)
        tree_j = cKDTree(lune_j_pts)
        lune_i_pts = sample_lune(d, t, "i", 40000)

        n_nbrs = np.asarray(tree_j.query_ball_point(lune_i_pts, 1.0, return_length=True))
        reach_vol = vol_U * n_nbrs / len(lune_j_pts)
        u_frac = (1 - f) * np.mean(n_nbrs > 0)

        # refine the max of v by local search from the best candidates (restricted to U_i)
        centre_i = np.zeros(d)
        centre_i[0] = -t / 2
        centre_j = -centre_i

        def neg_v(point):
            if ((point - centre_i) ** 2).sum() > 1 or ((point - centre_j) ** 2).sum() <= 1:
                return 0.0
            return (
                -vol_U * tree_j.query_ball_point(point, 1.0, return_length=True) / len(lune_j_pts)
            )

        starts = lune_i_pts[np.argsort(-reach_vol)[:5]]
        g_max = max(
            reach_vol.max(),
            max(
                -minimize(
                    neg_v,
                    start,
                    method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-6},
                ).fun
                for start in starts
            ),
        )
        g_max /= omega(d)

        results[f"{d}_{t}"] = dict(u=u_frac, g=g_max, lim=2 * u_frac / g_max)
        print(
            f"d={d} t={t}: u={u_frac:.4f} (1-f={1-f:.4f})  g={g_max:.4f}  2u/g={2*u_frac/g_max:.3f}",
            flush=True,
        )

json.dump(results, open(f"{RES}/sq_const.json", "w"), indent=1)
