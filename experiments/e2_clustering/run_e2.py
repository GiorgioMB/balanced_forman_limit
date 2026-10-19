import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree

from bflib.bfsim import *
from bflib.paths import RES

WORKER_STATE = {}


def init_worker(points, indptr, indices, r):
    WORKER_STATE["X"] = points
    WORKER_STATE["ip"] = indptr
    WORKER_STATE["ix"] = indices
    WORKER_STATE["r"] = r


def process_edges(edges):
    points = WORKER_STATE["X"]
    indptr = WORKER_STATE["ip"]
    indices = WORKER_STATE["ix"]
    r_sq = WORKER_STATE["r"] ** 2

    results = np.empty((len(edges), 3), np.float32)
    for edge_idx, (i, j) in enumerate(edges):
        Ni = indices[indptr[i] : indptr[i + 1]]
        Nj = indices[indptr[j] : indptr[j + 1]]
        n_common = np.intersect1d(Ni, Nj, assume_unique=True).size

        Ni_only = np.setdiff1d(Ni, Nj, assume_unique=True)
        Ni_only = Ni_only[Ni_only != j]
        Nj_only = np.setdiff1d(Nj, Ni, assume_unique=True)
        Nj_only = Nj_only[Nj_only != i]

        if Ni_only.size and Nj_only.size:
            disp = points[Ni_only][:, None, :] - points[Nj_only][None, :, :]
            disp -= np.round(disp)
            close = (disp * disp).sum(2) <= r_sq
            cnt_k = close.sum(1)
            cnt_w = close.sum(0)
        else:
            cnt_k = np.zeros(Ni_only.size, int)
            cnt_w = np.zeros(Nj_only.size, int)

        # bf_core expects counts excluding the planted pair: d_i = 1 + n_p  ->  n_p = |N_i| - 1
        bf, sq, _ = bf_core(Ni.size - 1, Nj.size - 1, n_common, cnt_k, cnt_w)

        disp_ij = points[i] - points[j]
        disp_ij -= np.round(disp_ij)
        t = np.sqrt((disp_ij * disp_ij).sum()) / WORKER_STATE["r"]
        results[edge_idx] = (t, bf, sq)

    return results


if __name__ == "__main__":
    d = int(sys.argv[1])
    n = int(sys.argv[2])
    lam = float(sys.argv[3])
    path = f"{RES}/e2_d{d}_n{n}_lam{int(lam)}.npz"
    if os.path.exists(path):
        sys.exit(0)

    rng = np.random.default_rng(int(1e6 * d + lam))
    r = (lam / (n * omega(d))) ** (1 / d)
    assert r < 0.25
    points = rng.random((n, d))

    tree = cKDTree(points, boxsize=1.0)
    neighbours = tree.query_ball_point(points, r)
    indptr = np.zeros(n + 1, np.int64)
    nbr_lists = []
    for i, nbrs in enumerate(neighbours):
        nbr_arr = np.array(sorted(nbrs), np.int64)
        nbr_arr = nbr_arr[nbr_arr != i]
        nbr_lists.append(nbr_arr)
        indptr[i + 1] = indptr[i] + nbr_arr.size
    indices = np.concatenate(nbr_lists)
    edges = np.array(
        [(i, j) for i in range(n) for j in nbr_lists[i] if j > i], np.int64
    )

    start_time = time.time()
    chunks = np.array_split(edges, 64)
    with Pool(4, initializer=init_worker, initargs=(points, indptr, indices, r)) as pool:
        results = np.concatenate(pool.map(process_edges, chunks))

    np.savez_compressed(
        path,
        t=results[:, 0],
        bf=results[:, 1],
        sq=results[:, 2],
        r=r,
        n=n,
        lam=lam,
        deg=np.diff(indptr),
    )
    print(
        path,
        "edges",
        len(edges),
        "time",
        round(time.time() - start_time, 1),
        flush=True,
    )
