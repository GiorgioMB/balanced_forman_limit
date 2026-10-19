"""Two rooms on a torus [0,2(1+L)) x [0,1): room A x in [0,1), room B x in [1+L, 2+L); two walls of
thickness L with a corridor of width w centred at y=1/2. Uniform points, Euclidean (periodic) RGG.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

Lw = 0.3
w = 0.1
r = 0.05
lam = 30.0
PX = 2 * (1 + Lw)


def in_domain(points):
    xcoord = points[:, 0] % PX
    ycoord = points[:, 1]
    wall = ((xcoord >= 1) & (xcoord < 1 + Lw)) | ((xcoord >= 2 + Lw) & (xcoord < PX))
    return ~wall | (np.abs(ycoord - 0.5) <= w / 2)


def build(seed):
    rng = np.random.default_rng(seed)
    density = lam / (np.pi * r * r)
    n_points = rng.poisson(density * PX)
    points = np.column_stack([rng.random(n_points) * PX, rng.random(n_points)])
    points = points[in_domain(points)]

    tree = cKDTree(points, boxsize=[PX, 1.0])
    edges = tree.query_pairs(r, output_type="ndarray")

    n = len(points)
    rows = np.r_[edges[:, 0], edges[:, 1]]
    cols = np.r_[edges[:, 1], edges[:, 0]]
    adj = csr_matrix((np.ones(2 * len(edges)), (rows, cols)), shape=(n, n))
    adj.sort_indices()

    return points, edges, adj


def region(X):
    xcoord = X[:, 0] % PX
    return np.where(
        xcoord < 1, 0, np.where(xcoord < 1 + Lw, 2, np.where(xcoord < 2 + Lw, 1, 2))
    )  # 0=A,1=B,2=corridor


def edge_len(X, P):
    disp = X[P[:, 0]] - X[P[:, 1]]
    disp[:, 0] -= PX * np.round(disp[:, 0] / PX)
    disp[:, 1] -= np.round(disp[:, 1])
    return np.linalg.norm(disp, axis=1)
