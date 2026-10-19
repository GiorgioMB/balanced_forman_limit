# E5: (a) RGG on T^2 with lambda = c log n ; (c) union-kNN graphs.  Each call: build graph (seeded), process
# a slice of edges for worker W, save.  Graph rebuilt identically from seed each call.
import glob
import os
import sys
import time
import zlib

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from bflib.allbf import bf_edges
from bflib.paths import RESULTS

kind = sys.argv[1]
n = int(sys.argv[2])
param = float(sys.argv[3])

worker = int(os.environ["W"])
n_workers = 4
time_limit = float(os.environ.get("TL", "4.5"))

tag = f"{kind}_n{n}_p{param:g}"
data_dir = f"{RESULTS}/e5"

rng = np.random.default_rng(zlib.crc32(tag.encode()))
points = rng.random((n, 2))
tree = cKDTree(points, boxsize=1.0)

if kind == "rgg":
    r = np.sqrt(param / (n * np.pi))
    edges = tree.query_pairs(r, output_type="ndarray")
else:
    k = int(param)
    knn_dist, knn_idx = tree.query(points, k + 1)
    src = np.repeat(np.arange(n), k)
    dst = knn_idx[:, 1:].ravel()
    edges = np.unique(np.sort(np.column_stack([src, dst]), axis=1), axis=0)
    r = np.sqrt(k / (n * np.pi))

adj = csr_matrix(
    (
        np.ones(2 * len(edges)),
        (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
    ),
    shape=(n, n),
)
adj.sort_indices()

# edge order: random permutation (so partial completion = random subsample)
perm = np.random.default_rng(1).permutation(len(edges))
edges = edges[perm]

done_files = sorted(glob.glob(f"{data_dir}/{tag}_w{worker}_s*.npz"))
n_done = sum(len(np.load(path)["bf"]) for path in done_files)
my_edges = np.arange(worker, len(edges), n_workers)

start_time = time.time()
results = []
batches = []
pos = n_done
while pos < len(my_edges) and time.time() - start_time < time_limit:
    batch_idx = my_edges[pos : pos + 200]
    results.append(bf_edges(adj, edges[batch_idx]))
    batches.append(batch_idx)
    pos += len(batch_idx)

if results:
    batch_idx = np.concatenate(batches)
    disp = points[edges[batch_idx, 0]] - points[edges[batch_idx, 1]]
    disp -= np.round(disp)
    t_norm = np.linalg.norm(disp, axis=1) / r

    bf_sq = np.concatenate(results)
    np.savez(
        f"{data_dir}/{tag}_w{worker}_s{len(done_files)}.npz",
        bf=bf_sq[:, 0],
        sq=bf_sq[:, 1],
        t=t_norm,
        E=len(edges),
    )
