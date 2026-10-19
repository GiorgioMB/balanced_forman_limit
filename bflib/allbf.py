"""Exact BF curvature for arbitrary edges of a graph given as symmetric CSR adjacency."""

import numpy as np


def bf_edges(A, E):
    indptr, indices = A.indptr, A.indices
    n_vertices = A.shape[0]

    results = np.zeros((len(E), 2))
    in_Ni = np.zeros(n_vertices, bool)
    in_Nj = np.zeros(n_vertices, bool)
    ind_Ui = np.zeros(n_vertices)
    ind_Uj = np.zeros(n_vertices)

    for edge_idx, (i, j) in enumerate(E):
        Ni = indices[indptr[i] : indptr[i + 1]]
        Nj = indices[indptr[j] : indptr[j + 1]]
        di, dj = len(Ni), len(Nj)
        if min(di, dj) <= 1:
            continue

        in_Ni[Ni] = True
        in_Nj[Nj] = True
        tri = int(in_Nj[Ni].sum())

        Ui = Ni[~in_Nj[Ni]]
        Ui = Ui[Ui != j]
        Uj = Nj[~in_Ni[Nj]]
        Uj = Uj[Uj != i]

        sq = 0.0
        if len(Ui) and len(Uj):
            ind_Uj[Uj] = 1
            ind_Ui[Ui] = 1

            cnt_k = A[Ui] @ ind_Uj
            cnt_w = A[Uj] @ ind_Ui
            gam = max(cnt_k.max(), cnt_w.max())
            if gam > 0:
                sq = (np.count_nonzero(cnt_k) + np.count_nonzero(cnt_w)) / (
                    gam * max(di, dj)
                )

            ind_Uj[Uj] = 0
            ind_Ui[Ui] = 0

        in_Ni[Ni] = False
        in_Nj[Nj] = False

        results[edge_idx] = (
            2 / di + 2 / dj - 2 + 2 * tri / max(di, dj) + tri / min(di, dj) + sq,
            sq,
        )

    return results
