# E1: Palm-vertex simulations. usage: python run_v1.py geo lam r nchunks [t_min] [max_edges]
import os
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np

from bflib.vsim import sample_local, adjacency, bf_vertex_edges
from bflib.paths import RESULTS

geo = sys.argv[1]
lam = float(sys.argv[2])
r = float(sys.argv[3])
n_chunks = int(sys.argv[4])
t_min = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
max_edges = int(sys.argv[6]) if len(sys.argv) > 6 else 200
vertices_per_chunk = int(os.environ.get("VPC", "25"))
data_dir = f"{RESULTS}/e1"


def chunk(chunk_id):
    path = f"{data_dir}/{geo}_l{int(lam)}_r{r}_c{chunk_id}.npz"
    if os.path.exists(path):
        return path

    rng = np.random.default_rng(zlib.crc32(f"{geo}{lam}{r}{chunk_id}".encode()))
    t_list = []
    bf_list = []
    sq_list = []
    dir_list = []
    vertex_ids = []
    degrees = []

    for vertex in range(vertices_per_chunk):
        local = sample_local(rng, geo, lam, r)
        adj = adjacency(local)
        nbrs = adj.indices[adj.indptr[0] : adj.indptr[1]]
        tangents = local["tan"][nbrs]
        t_norm = np.linalg.norm(tangents, axis=1) / r

        candidates = np.where(t_norm >= t_min)[0]
        pick = rng.choice(candidates, min(max_edges, len(candidates)), replace=False)
        edge_bf = bf_vertex_edges(adj, 0, nbrs[pick])

        t_list.append(t_norm[pick])
        bf_list.append(edge_bf[:, 0])
        sq_list.append(edge_bf[:, 1])
        dir_list.append(
            tangents[pick] / np.linalg.norm(tangents[pick], axis=1, keepdims=True)
        )
        vertex_ids.append(np.full(len(pick), chunk_id * vertices_per_chunk + vertex))
        degrees.append(len(nbrs))

    np.savez_compressed(
        path,
        t=np.concatenate(t_list),
        bf=np.concatenate(bf_list),
        sq=np.concatenate(sq_list),
        v=np.concatenate(dir_list),
        vid=np.concatenate(vertex_ids),
        deg=np.array(degrees),
    )
    return path


if __name__ == "__main__":
    start_time = time.time()
    worker = int(os.environ.get("W", "0"))
    n_workers = int(os.environ.get("NW", "1"))
    time_limit = float(os.environ.get("TL", "95"))

    for chunk_id in range(worker, n_chunks, n_workers):
        if time.time() - start_time > time_limit:
            break
        chunk(chunk_id)
