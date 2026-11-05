"""Whole graphs: exact BF curvature of arbitrary edges, local clustering, and the geometric graphs on
the flat torus T^2, the unit sphere S^2 and the Bolza surface used in Sections 8.4 and S6."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from . import bolza


def csr_from_pairs(pairs, n):
    rows = np.r_[pairs[:, 0], pairs[:, 1]]
    cols = np.r_[pairs[:, 1], pairs[:, 0]]
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A.sort_indices()
    return A


def bf_edges(A, edges):
    """Exact BF curvature (Definition 2.1) of the given edges of a graph with symmetric CSR
    adjacency A. Returns an array of rows (BF, S)."""
    indptr, indices = A.indptr, A.indices
    n = A.shape[0]
    out = np.zeros((len(edges), 2))
    in_Ni, in_Nj = np.zeros(n, bool), np.zeros(n, bool)
    ind_Ui, ind_Uj = np.zeros(n), np.zeros(n)
    for e, (i, j) in enumerate(edges):
        N_i = indices[indptr[i] : indptr[i + 1]]
        N_j = indices[indptr[j] : indptr[j + 1]]
        d_i, d_j = len(N_i), len(N_j)
        if min(d_i, d_j) <= 1:
            continue
        in_Ni[N_i] = True
        in_Nj[N_j] = True
        tri = int(in_Nj[N_i].sum())
        U_i = N_i[~in_Nj[N_i]]
        U_i = U_i[U_i != j]
        U_j = N_j[~in_Ni[N_j]]
        U_j = U_j[U_j != i]
        four_cycle = 0.0
        if len(U_i) and len(U_j):
            ind_Uj[U_j] = 1
            ind_Ui[U_i] = 1
            cnt_k = A[U_i] @ ind_Uj
            cnt_w = A[U_j] @ ind_Ui
            gamma_max = max(cnt_k.max(), cnt_w.max())
            if gamma_max > 0:
                four_cycle = (np.count_nonzero(cnt_k) + np.count_nonzero(cnt_w)) / (gamma_max * max(d_i, d_j))
            ind_Uj[U_j] = 0
            ind_Ui[U_i] = 0
        in_Ni[N_i] = False
        in_Nj[N_j] = False
        out[e] = (2 / d_i + 2 / d_j - 2 + 2 * tri / max(d_i, d_j) + tri / min(d_i, d_j) + four_cycle, four_cycle)
    return out


def mean_local_clustering(pairs, n):
    """Mean local clustering over vertices of degree >= 2, the fraction of such vertices, mean degree."""
    A = csr_from_pairs(pairs, n)
    deg = np.asarray(A.sum(1)).ravel()
    triangles = np.asarray((A @ A).multiply(A).sum(1)).ravel() / 2
    has_two = deg >= 2
    clust = np.zeros(n)
    clust[has_two] = triangles[has_two] / (deg[has_two] * (deg[has_two] - 1) / 2)
    return clust[has_two].mean(), has_two.mean(), deg.mean()


AREA = {"T2": 1.0, "S2": 4 * np.pi, "bolza": 4 * np.pi}
EULER = {"T2": 0, "S2": 2, "bolza": -2}


def sample_surface(surface, rng, n):
    if surface == "T2":
        return rng.random((n, 2))
    if surface == "S2":
        gauss = rng.standard_normal((n, 3))
        return gauss / np.linalg.norm(gauss, axis=1, keepdims=True)
    if surface == "bolza":
        return bolza.sample(rng, n)
    raise ValueError(surface)


def surface_edges(surface, points, r):
    """Edges of the geodesic RGG with radius r on T^2 (unit torus), S^2 (unit sphere) or Bolza."""
    if surface == "T2":
        return cKDTree(points, boxsize=1.0).query_pairs(r, output_type="ndarray")
    if surface == "S2":
        return cKDTree(points).query_pairs(2 * np.sin(r / 2), output_type="ndarray")
    if surface == "bolza":
        return bolza.edges(points, r)
    raise ValueError(surface)
