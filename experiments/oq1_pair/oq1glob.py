"""Global mean of centred BF on full RGGs: unit sphere S^2 (geodesic) vs flat torus with the same lambda."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from bflib.allbf import bf_edges
from bflib.bfsim import Psi


def graph(geo, lam, r, seed):
    rng = np.random.default_rng(seed)

    if geo == "S2":
        n_points = rng.poisson(lam * 4 / (r * r))
        gauss = rng.standard_normal((n_points, 3))
        points = gauss / np.linalg.norm(gauss, axis=1, keepdims=True)

        edges = cKDTree(points).query_pairs(2 * np.sin(r / 2), output_type="ndarray")

        cos_angle = (points[edges[:, 0]] * points[edges[:, 1]]).sum(1)
        t = np.arccos(np.clip(cos_angle, -1, 1)) / r
    else:  # torus with the same n: side 1, radius rT so that n*pi*rT^2 = lam
        n_points = rng.poisson(lam * 4 / (r * r))
        r_torus = np.sqrt(lam / (n_points * np.pi))
        points = rng.random((n_points, 2))

        edges = cKDTree(points, boxsize=1.0).query_pairs(r_torus, output_type="ndarray")

        disp = points[edges[:, 0]] - points[edges[:, 1]]
        disp -= np.round(disp)
        t = np.linalg.norm(disp, axis=1) / r_torus

    rows = np.r_[edges[:, 0], edges[:, 1]]
    cols = np.r_[edges[:, 1], edges[:, 0]]
    adj = csr_matrix(
        (np.ones(2 * len(edges)), (rows, cols)),
        shape=(n_points, n_points),
    )
    adj.sort_indices()

    return n_points, edges, adj, t


def rep(geo, lam, r, seed):
    n_points, edges, adj, t = graph(geo, lam, r, seed)
    bf_sq = bf_edges(adj, edges)
    return n_points, len(edges), float(np.mean(bf_sq[:, 0] - Psi(2, t)))
