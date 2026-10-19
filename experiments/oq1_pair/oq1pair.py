"""Joint law of BF on two edges e1=(x,j), e2=(x,k) sharing vertex x (flat R^2, unit r).
Theory: Gaussian limit of cell counts (Cov N(A),N(B) = n Vol(A cap B)) pushed through the linearisation.
"""

import numpy as np
from scipy.spatial import cKDTree

from bflib.bfsim import bf_core, f_lens


def pts(t, th):
    x = np.zeros(2)
    j = np.array([t, 0.0])
    k = t * np.array([np.cos(th), np.sin(th)])
    return x, j, k


def edge_bf(points, p, q, r=1.0):
    in_p = ((points - p) ** 2).sum(1) <= r * r
    in_q = ((points - q) ** 2).sum(1) <= r * r
    Ui = points[in_p & ~in_q]
    Uj = points[in_q & ~in_p]

    if len(Ui) and len(Uj):
        cnt_k = np.asarray(cKDTree(Uj).query_ball_point(Ui, r, return_length=True))
        cnt_w = np.asarray(cKDTree(Ui).query_ball_point(Uj, r, return_length=True))
    else:
        cnt_k = np.zeros(len(Ui), int)
        cnt_w = np.zeros(len(Uj), int)

    return bf_core(in_p.sum(), in_q.sum(), (in_p & in_q).sum(), cnt_k, cnt_w)[0]


def trial(rng, lam, t, th):
    x, j, k = pts(t, th)
    n = lam / np.pi

    lo = np.minimum.reduce([x, j, k]) - 1
    hi = np.maximum.reduce([x, j, k]) + 1
    n_points = rng.poisson(n * np.prod(hi - lo))
    points = rng.uniform(lo, hi, (n_points, 2))

    bf_e1 = edge_bf(np.vstack([points, k]), x, j)
    bf_e2 = edge_bf(np.vstack([points, j]), x, k)
    return bf_e1, bf_e2


def theory(t, th, M=400000, G=400000, seed=0):
    rng = np.random.default_rng(seed)
    x, j, k = pts(t, th)

    mc_points = rng.uniform(-1 - t, 1 + t, (M, 2))
    vol_factor = (2 + 2 * t) ** 2 / np.pi  # volume factor (normalised by omega_2 = pi)
    in_ball = lambda centre: ((mc_points - centre) ** 2).sum(1) <= 1
    Bx, Bj, Bk = in_ball(x), in_ball(j), in_ball(k)

    cells = [Bx & Bj, Bx, Bj, Bx & Bk, Bk]  # L1,Bx,Bj,L2,Bk
    cov = np.array(
        [
            [vol_factor * np.mean(cell_a & cell_b) for cell_b in cells]
            for cell_a in cells
        ]
    )

    gauss = np.random.default_rng(seed + 1).multivariate_normal(
        np.zeros(5), cov, size=G
    )

    f = float(f_lens(2, t))
    W1 = (
        3 * gauss[:, 0]
        - 1.5 * f * (gauss[:, 1] + gauss[:, 2])
        - 0.5 * f * np.abs(gauss[:, 1] - gauss[:, 2])
    )
    W2 = (
        3 * gauss[:, 3]
        - 1.5 * f * (gauss[:, 1] + gauss[:, 4])
        - 0.5 * f * np.abs(gauss[:, 1] - gauss[:, 4])
    )

    return float(np.corrcoef(W1, W2)[0, 1]), float(W1.var())
