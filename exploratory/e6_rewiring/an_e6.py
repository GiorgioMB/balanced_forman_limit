import glob
import json
import os
import sys

import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import splu
from scipy.stats import spearmanr

from e6lib import build, region, edge_len, r, PX, Lw, w
from bflib.bfsim import Psi
from bflib.paths import RESULTS

points, edges, adj = build(2026)
n = len(points)

chunks = [np.load(path) for path in glob.glob(f"{RESULTS}/e6/bf_w*")]
edge_idx = np.concatenate([chunk["idx"] for chunk in chunks])
bf_values = np.concatenate([chunk["bf"] for chunk in chunks])
bf = np.zeros(len(edges))
bf[edge_idx] = bf_values

t = edge_len(points, edges) / r
regions = region(points)
xcoord = points[:, 0] % PX
# distance to the walls (in x) for room points
dist_wall = np.minimum.reduce(
    [
        np.abs(xcoord - 1),
        np.abs(xcoord - (1 + Lw)),
        np.abs(xcoord - (2 + Lw)),
        np.abs(xcoord - PX),
        np.abs(xcoord),
    ]
)

interior = (
    (regions[edges[:, 0]] < 2)
    & (regions[edges[:, 1]] < 2)
    & (dist_wall[edges[:, 0]] > 2 * r)
    & (dist_wall[edges[:, 1]] > 2 * r)
)
coef = np.polyfit(t[interior], bf[interior] - Psi(2, t[interior]), 3)
bf_centred = bf - Psi(2, t) - np.polyval(coef, t)

in_corridor = (regions[edges[:, 0]] == 2) | (regions[edges[:, 1]] == 2)
near_wall = (~in_corridor) & (
    (dist_wall[edges[:, 0]] <= r) | (dist_wall[edges[:, 1]] <= r)
)


def describe(score, frac=0.01):
    n_sel = int(frac * len(edges))
    selected = np.argsort(score)[:n_sel]
    return dict(
        k=n_sel,
        frac_corridor=float(in_corridor[selected].mean()),
        frac_wallface=float(near_wall[selected].mean()),
        frac_interior=float(interior[selected].mean()),
        mean_t=float(t[selected].mean()),
    )


mode = sys.argv[1] if len(sys.argv) > 1 else "describe"
if mode == "describe":
    summary = dict(
        n=n,
        E=len(edges),
        base_corridor=float(in_corridor.mean()),
        base_wallface=float(near_wall.mean()),
        spearman_raw_t=float(spearmanr(bf, t)[0]),
        spearman_cen_t=float(spearmanr(bf_centred, t)[0]),
        raw=describe(bf),
        cen=describe(bf_centred),
        raw_01=describe(bf, 0.001),
        cen_01=describe(bf_centred, 0.001),
        coef=coef.tolist(),
    )
    print(json.dumps(summary, indent=1))
    json.dump(summary, open(f"{RESULTS}/e6_describe.json", "w"), indent=1)
else:
    strategy = mode
    rng = np.random.default_rng(int(os.environ.get("SEED", "0")))
    indptr, indices = adj.indptr, adj.indices
    neighbours = [set(indices[indptr[i] : indptr[i + 1]]) for i in range(n)]

    if strategy != "none":
        order = {
            "raw": np.argsort(bf),
            "cen": np.argsort(bf_centred),
            "random": rng.permutation(len(edges)),
            "oracle": np.r_[
                rng.permutation(np.where(in_corridor)[0]),
                rng.permutation(np.where(~in_corridor)[0]),
            ],
        }[strategy]
    else:
        order = []

    pair_rng = np.random.default_rng(5)
    room_a = np.where(regions == 0)[0]
    room_b = np.where(regions == 1)[0]
    room_a = room_a[dist_wall[room_a] > 0.3]
    room_b = room_b[dist_wall[room_b] > 0.3]
    pairs = list(zip(pair_rng.choice(room_a, 30), pair_rng.choice(room_b, 30)))

    results = {}
    added = []
    order_pos = 0
    for budget in [0, 50, 100, 200, 400]:
        while len(added) < budget and order_pos < len(order):
            i, j = edges[order[order_pos]]
            order_pos += 1

            cands_i = list(neighbours[i] - neighbours[j] - {j})
            cands_j = list(neighbours[j] - neighbours[i] - {i})
            if not cands_i or not cands_j:
                continue

            for _ in range(20):
                new_end_i = cands_i[rng.integers(len(cands_i))]
                new_end_j = cands_j[rng.integers(len(cands_j))]
                if new_end_j not in neighbours[new_end_i] and new_end_i != new_end_j:
                    neighbours[new_end_i].add(new_end_j)
                    neighbours[new_end_j].add(new_end_i)
                    added.append((new_end_i, new_end_j))
                    break

        if added:
            all_edges = np.vstack([edges, np.array(added).reshape(-1, 2)])
        else:
            all_edges = edges
        rows = np.r_[all_edges[:, 0], all_edges[:, 1]]
        cols = np.r_[all_edges[:, 1], all_edges[:, 0]]
        adj_new = csr_matrix(
            (np.ones(2 * len(all_edges)), (rows, cols)),
            shape=(n, n),
        )

        laplacian = diags(np.asarray(adj_new.sum(1)).ravel()) - adj_new
        ground_vertex = 0
        keep = np.arange(1, n)
        lap_lu = splu(laplacian[keep][:, keep].tocsc())

        reff = []
        for src, dst in pairs:
            injection = np.zeros(n)
            injection[src] += 1
            injection[dst] -= 1
            potential = np.zeros(n)
            potential[keep] = lap_lu.solve(injection[keep])
            reff.append(potential[src] - potential[dst])

        added_arr = np.array(added).reshape(-1, 2)
        if len(added_arr):
            n_cross = int(np.sum(regions[added_arr[:, 0]] != regions[added_arr[:, 1]]))
        else:
            n_cross = 0

        results[budget] = dict(Reff=float(np.mean(reff)), cross_region_added=n_cross)

    json.dump(
        results,
        open(f'{RESULTS}/e6_rew_{strategy}_s{os.environ.get("SEED","0")}.json', "w"),
    )
    print(strategy, results)
