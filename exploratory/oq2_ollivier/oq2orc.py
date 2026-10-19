"""Per-edge comparison of BF and data-cloud Ollivier curvature (uniform measures on 1-hop neighbourhoods,
geodesic ground cost; kappa = 1 - W1/d(x,j)) on a selected edge of a planted vertex."""

import numpy as np
import ot

from bflib.vsim import sample_local, adjacency, bf_vertex_edges
from bflib.bfsim import Psi


def geod(geo, loc, idx_a, idx_b, r):
    emb = loc["emb"]

    if geo.startswith("flat"):
        return np.linalg.norm(emb[idx_a][:, None] - emb[idx_b][None], axis=2)

    if geo in ("S2", "S3"):
        return np.arccos(np.clip(emb[idx_a] @ emb[idx_b].T, -1, 1))

    if geo == "H2":
        tan = loc["tan"]
        rad = np.linalg.norm(tan, axis=1)
        unit = tan / np.where(rad > 0, rad, 1)[:, None]
        hyp = np.column_stack([np.cosh(rad), np.sinh(rad)[:, None] * unit])

        time_part = hyp[idx_a][:, None, 0] * hyp[idx_b][None, :, 0]
        space_part = (hyp[idx_a][:, None, 1:] * hyp[idx_b][None, :, 1:]).sum(2)
        mink = time_part - space_part
        return np.arccosh(np.maximum(mink, 1))

    if geo == "S2xR":
        sphere = emb[:, :3]
        height = emb[:, 3]
        sphere_dist = np.arccos(np.clip(sphere[idx_a] @ sphere[idx_b].T, -1, 1))
        height_diff = height[idx_a][:, None] - height[idx_b][None]
        return np.sqrt(sphere_dist**2 + height_diff**2)


def trial(rng, geo, lam, r, target):
    loc = sample_local(rng, geo, lam, r)
    adj = adjacency(loc)

    nbrs = adj.indices[adj.indptr[0] : adj.indptr[1]]
    tan = loc["tan"][nbrs]
    t_nbrs = np.linalg.norm(tan, axis=1) / r
    cand = np.where((t_nbrs >= 0.7) & (t_nbrs <= 0.8))[0]

    if target is None:
        pick = cand[rng.integers(len(cand))]
    else:
        alignment = np.abs(tan[cand] @ target) / np.linalg.norm(tan[cand], axis=1)
        pick = cand[np.argmax(alignment)]
    j = nbrs[pick]

    bf = bf_vertex_edges(adj, 0, np.array([j]))[0, 0]

    Nx = adj.indices[adj.indptr[0] : adj.indptr[1]]
    Nj = adj.indices[adj.indptr[j] : adj.indptr[j + 1]]
    cost = geod(geo, loc, Nx, Nj, r)
    dist = geod(geo, loc, np.array([0]), np.array([j]), r)[0, 0]
    mu_x = np.full(len(Nx), 1 / len(Nx))
    mu_j = np.full(len(Nj), 1 / len(Nj))
    w1 = ot.emd2(mu_x, mu_j, cost)

    return t_nbrs[pick], bf - Psi(loc["d"], t_nbrs[pick]), 1 - w1 / dist
