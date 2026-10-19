"""Variance of F = sum_e (BF(e) - m_lam(t_e)) on the flat torus (Poisson model), as a function of lambda.
Per graph we store binned sums of BF-Psi and counts (20 bins in t) so that m_lam can be estimated by pooling.
"""

import glob
import os
import sys
import time
import zlib

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from bflib.allbf import bf_edges
from bflib.bfsim import Psi
from bflib.paths import RESULTS

lam = float(sys.argv[1])
n0 = int(sys.argv[2])
worker = int(os.environ["W"])
time_limit = float(os.environ.get("TL", "2.5"))

tag = f"var_l{int(lam)}_n{n0}"
batch = len(glob.glob(f"{RESULTS}/var/{tag}_w{worker}_b*"))
bins = np.linspace(0, 1, 21)

start_time = time.time()
records = []
rep_idx = 0
while time.time() - start_time < time_limit:
    rng = np.random.default_rng(
        zlib.crc32(tag.encode()) + worker * 10**6 + batch * 100 + rep_idx
    )
    rep_idx += 1

    n = rng.poisson(n0)
    r = np.sqrt(lam / (n0 * np.pi))
    points = rng.random((n, 2))
    edges = cKDTree(points, boxsize=1.0).query_pairs(r, output_type="ndarray")

    disp = points[edges[:, 0]] - points[edges[:, 1]]
    disp -= np.round(disp)
    t = np.linalg.norm(disp, axis=1) / r

    rows = np.r_[edges[:, 0], edges[:, 1]]
    cols = np.r_[edges[:, 1], edges[:, 0]]
    adj = csr_matrix((np.ones(2 * len(edges)), (rows, cols)), shape=(n, n))
    adj.sort_indices()

    centred_bf = bf_edges(adj, edges)[:, 0] - Psi(2, t)
    bin_idx = np.minimum(np.digitize(t, bins) - 1, 19)
    records.append(
        np.r_[
            n,
            len(edges),
            np.bincount(bin_idx, weights=centred_bf, minlength=20),
            np.bincount(bin_idx, minlength=20),
        ]
    )

np.save(f"{RESULTS}/var/{tag}_w{worker}_b{batch}.npy", np.array(records))
