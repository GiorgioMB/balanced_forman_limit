import glob
import os
import time

import numpy as np

from e6lib import build
from bflib.allbf import bf_edges
from bflib.paths import RESULTS

worker = int(os.environ["W"])
time_limit = float(os.environ.get("TL", "4.5"))
data_dir = f"{RESULTS}/e6"

points, edges, adj = build(2026)
my_edges = np.arange(worker, len(edges), 4)
done_files = sorted(glob.glob(f"{data_dir}/bf_w{worker}_s*.npz"))
start = sum(len(np.load(path)["idx"]) for path in done_files)

start_time = time.time()
results = []
edge_idx = []
n_done = start
while n_done < len(my_edges) and time.time() - start_time < time_limit:
    batch_idx = my_edges[n_done : n_done + 300]
    results.append(bf_edges(adj, edges[batch_idx]))
    edge_idx.append(batch_idx)
    n_done += len(batch_idx)

if results:
    bf_sq = np.concatenate(results)
    np.savez(
        f"{data_dir}/bf_w{worker}_s{len(done_files)}.npz",
        idx=np.concatenate(edge_idx),
        bf=bf_sq[:, 0],
        sq=bf_sq[:, 1],
        E=len(edges),
    )
