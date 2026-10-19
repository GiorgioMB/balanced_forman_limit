import os

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix

import bolza as B
from bflib.paths import RESULTS

GROUP_FILE = f"{RESULTS}/bolza_group_r0.6.npy"


def group():
    if not os.path.exists(GROUP_FILE):
        np.save(GROUP_FILE, np.array(B.group_elements(0.6, 4)))
    return np.load(GROUP_FILE)


def edges(geo, points, r):
    n = len(points)

    if geo == "T2":
        pairs = cKDTree(points, boxsize=1.0).query_pairs(r, output_type="ndarray")
        return pairs

    if geo == "S2":
        return cKDTree(points).query_pairs(2 * np.sin(r / 2), output_type="ndarray")

    if geo == "bolza":
        group_mats = group()
        images = []
        owner = []
        for g in group_mats:
            images.append((g @ points.T).T)
            owner.append(np.arange(n))
        image_points = np.concatenate(images)
        owner = np.concatenate(owner)

        tree = cKDTree(B.poincare(image_points))
        chord_rad = np.sinh(r / 2)
        neighbours = tree.query_ball_point(B.poincare(points), chord_rad * 1.0000001)
        src = np.repeat(np.arange(n), [len(nbrs) for nbrs in neighbours])
        img_idx = np.concatenate([np.array(nbrs, int) for nbrs in neighbours])

        minkowski_prod = points[src] * np.array([1.0, -1.0, -1.0]) * image_points[img_idx]
        cosh_dist = minkowski_prod.sum(1)  # = cosh d

        close = (cosh_dist <= np.cosh(r)) & (owner[img_idx] != src)
        src, dst = src[close], owner[img_idx[close]]
        pairs = np.unique(np.sort(np.column_stack([src, dst]), axis=1), axis=0)
        return pairs


def sample(geo, rng, n):
    if geo == "T2":
        return rng.random((n, 2))
    if geo == "S2":
        gauss = rng.standard_normal((n, 3))
        return gauss / np.linalg.norm(gauss, axis=1, keepdims=True)
    if geo == "bolza":
        return B.sample(rng, n)


AREA = {"T2": 1.0, "S2": 4 * np.pi, "bolza": 4 * np.pi}


def clustering(pairs, n):
    rows = np.r_[pairs[:, 0], pairs[:, 1]]
    cols = np.r_[pairs[:, 1], pairs[:, 0]]
    adj = csr_matrix((np.ones(2 * len(pairs)), (rows, cols)), shape=(n, n))

    deg = np.asarray(adj.sum(1)).ravel()
    tri = np.asarray((adj @ adj).multiply(adj).sum(1)).ravel() / 2

    has_two = deg >= 2
    clust = np.zeros(n)
    clust[has_two] = tri[has_two] / (deg[has_two] * (deg[has_two] - 1) / 2)
    return clust[has_two].mean(), has_two.mean(), deg.mean()


def rep(geo, lam, r, seed):
    rng = np.random.default_rng(seed)
    n = rng.poisson(lam * AREA[geo] / (np.pi * r * r))
    points = sample(geo, rng, n)
    pairs = edges(geo, points, r)
    return clustering(pairs, n)
