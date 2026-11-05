"""The Bolza surface (genus 2, curvature -1, area 4 pi) as the regular hyperbolic octagon with angles
pi/4 in the hyperboloid model, opposite sides paired by translations of length 2a, cosh a = 1 + sqrt 2
(Section S1). Geodesic distances are min_gamma d(x, gamma y) over the deck transformations gamma with
d(o, gamma o) <= 2R + r, where R is the circumradius, cosh R = (1 + sqrt 2)^2.
"""

from functools import lru_cache

import numpy as np
from scipy.spatial import cKDTree

A_HALF = np.arccosh(1 + np.sqrt(2))  # inradius a; also the injectivity radius
R_CIRC = np.arccosh((1 + np.sqrt(2)) ** 2)  # circumradius R
_MINKOWSKI = np.array([1.0, -1.0, -1.0])


def _boost(theta, s):
    c, s_ = np.cos(theta), np.sin(theta)
    rot = np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])
    bx = np.array([[np.cosh(s), np.sinh(s), 0], [np.sinh(s), np.cosh(s), 0], [0, 0, 1]])
    return rot @ bx @ rot.T


GENERATORS = [_boost(k * np.pi / 4, 2 * A_HALF) for k in range(8)]


def _distance_from_origin(mat):
    return np.arccosh(max(mat[0, 0], 1.0))


@lru_cache(maxsize=None)
def group_elements(r, word_length=4):
    """Deck transformations g with d(o, g o) <= 2R + r (including the identity)."""
    max_dist = 2 * R_CIRC + r
    seen = [np.eye(3)]
    frontier = [np.eye(3)]
    for _ in range(word_length):
        new = []
        for mat in frontier:
            for gen in GENERATORS:
                cand = mat @ gen
                if _distance_from_origin(cand) > max_dist + 2 * A_HALF:
                    continue
                if any(np.allclose(cand, prev, atol=1e-7) for prev in seen):
                    continue
                seen.append(cand)
                new.append(cand)
        frontier = new
    return tuple(m for m in seen if _distance_from_origin(m) <= max_dist)


def in_octagon(points):
    """Hyperboloid points inside the regular octagon (Klein model; vertices at angles (k+1/2) pi/4)."""
    klein = points[:, 1:] / points[:, :1]
    apothem = np.tanh(A_HALF)
    inside = np.ones(len(points), bool)
    for side in range(8):
        angle = side * np.pi / 4
        inside &= klein[:, 0] * np.cos(angle) + klein[:, 1] * np.sin(angle) <= apothem
    return inside


def sample(rng, n):
    """n uniform points of the Bolza surface (rejection from the circumscribed disc)."""
    batches = []
    while sum(len(b) for b in batches) < n:
        count = 2 * n
        rad = np.arccosh(1 + rng.random(count) * (np.cosh(R_CIRC) - 1))
        angle = 2 * np.pi * rng.random(count)
        pts = np.column_stack([np.cosh(rad), np.sinh(rad) * np.cos(angle), np.sinh(rad) * np.sin(angle)])
        batches.append(pts[in_octagon(pts)])
    return np.concatenate(batches)[:n]


def poincare(points):
    return points[:, 1:] / (1 + points[:, :1])


def edges(points, r):
    """Pairs {i, j} at geodesic distance <= r on the surface (requires r < injectivity radius)."""
    assert r < A_HALF, "the radius must be smaller than the injectivity radius"
    n = len(points)
    group = group_elements(round(float(r), 12))
    images = np.concatenate([(g @ points.T).T for g in group])
    owner = np.tile(np.arange(n), len(group))
    # Poincare-disc Euclidean distance <= sinh(d/2): a superset, filtered by the exact test
    neighbours = cKDTree(poincare(images)).query_ball_point(poincare(points), np.sinh(r / 2) * 1.0000001)
    src = np.repeat(np.arange(n), [len(nb) for nb in neighbours])
    img = np.concatenate([np.array(nb, int) for nb in neighbours])
    cosh_dist = (points[src] * _MINKOWSKI * images[img]).sum(1)
    close = (cosh_dist <= np.cosh(r)) & (owner[img] != src)
    pairs = np.sort(np.column_stack([src[close], owner[img[close]]]), axis=1)
    return np.unique(pairs, axis=0)
