"""Planted edge in R^2 with Poisson intensity n*rho(x), rho(x)=1+a*cos(2*pi*x1), midpoint at x1=0.25.
mode 'fixed': edge iff |x-y|<=r ; mode 'adapt': edge iff |x-y| <= r*(rho(x)rho(y))^(-1/4) (oracle).
"""

import numpy as np
from scipy.spatial import cKDTree

from bflib.bfsim import bf_core


def rho(points, a):
    return 1 + a * np.cos(2 * np.pi * points[:, 0])


def trial(rng, lam, r, t, a, direc, mode):
    n = lam / (np.pi * r * r)
    mid = np.array([0.25, 0.0])
    axis = np.array([1.0, 0.0]) if direc == "par" else np.array([0.0, 1.0])
    rmax = r * (1 - a) ** (-0.5) if mode == "adapt" else r

    if mode == "adapt":
        # choose p,q so that d(p,q) = t * r_pq exactly (solve by fixed point)
        dl = t * r
        for _ in range(30):
            p = mid - dl / 2 * axis
            q = mid + dl / 2 * axis
            rpq = r * (rho(p[None], a)[0] * rho(q[None], a)[0]) ** -0.25
            dl = t * rpq
    else:
        dl = t * r
        p = mid - dl / 2 * axis
        q = mid + dl / 2 * axis

    half_width = dl / 2 + rmax
    lo = mid - half_width
    hi = mid + half_width

    n_points = rng.poisson(n * (1 + a) * (2 * half_width) ** 2)
    points = rng.uniform(lo, hi, (n_points, 2))
    points = points[rng.random(n_points) < rho(points, a) / (1 + a)]
    rho_pts = rho(points, a)

    if mode == "adapt":
        rho_p = rho(p[None], a)[0]
        rho_q = rho(q[None], a)[0]
        in_p = np.linalg.norm(points - p, axis=1) <= r * (rho_pts * rho_p) ** -0.25
        in_q = np.linalg.norm(points - q, axis=1) <= r * (rho_pts * rho_q) ** -0.25
    else:
        in_p = np.linalg.norm(points - p, axis=1) <= r
        in_q = np.linalg.norm(points - q, axis=1) <= r

    in_Ui = in_p & ~in_q
    in_Uj = in_q & ~in_p
    Ui = points[in_Ui]
    Uj = points[in_Uj]

    cnt_k = np.zeros(len(Ui), int)
    cnt_w = np.zeros(len(Uj), int)
    if len(Ui) and len(Uj):
        close_pairs = cKDTree(Ui).sparse_distance_matrix(
            cKDTree(Uj), rmax, output_type="ndarray"
        )
        idx_i = close_pairs["i"]
        idx_j = close_pairs["j"]

        if mode == "adapt":
            adjacent = (
                close_pairs["v"]
                <= r * (rho_pts[in_Ui][idx_i] * rho_pts[in_Uj][idx_j]) ** -0.25
            )
        else:
            adjacent = close_pairs["v"] <= r

        cnt_k = np.bincount(idx_i[adjacent], minlength=len(Ui))
        cnt_w = np.bincount(idx_j[adjacent], minlength=len(Uj))

    return bf_core(in_p.sum(), in_q.sum(), (in_p & in_q).sum(), cnt_k, cnt_w)[0]
