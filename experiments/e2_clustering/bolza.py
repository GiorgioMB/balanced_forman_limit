"""Bolza surface (genus 2, curvature -1, area 4*pi) as the regular octagon with angles pi/4 in
the hyperboloid model, opposite sides paired by translations of length 2a, cosh a = 1+sqrt2.
"""

import numpy as np

A_HALF = np.arccosh(1 + np.sqrt(2))
R_CIRC = np.arccosh((1 + np.sqrt(2)) ** 2)


def boost(theta, s):
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot = np.array([[1, 0, 0], [0, cos_t, -sin_t], [0, sin_t, cos_t]])
    boost_x = np.array(
        [[np.cosh(s), np.sinh(s), 0], [np.sinh(s), np.cosh(s), 0], [0, 0, 1]]
    )
    return rot @ boost_x @ rot.T


GENS = [boost(k * np.pi / 4, 2 * A_HALF) for k in range(8)]
MINKOWSKI = np.diag([1.0, -1.0, -1.0])


def hdist0(mat):  # distance from origin o=(1,0,0) to mat o
    return np.arccosh(max(mat[0, 0], 1.0))


def group_elements(r, maxlen=5):
    """elements g whose tile g(Oct) lies within distance r of Oct (incl. identity)."""
    keep = {(): np.eye(3)}
    frontier = [np.eye(3)]
    seen = [np.eye(3)]
    max_dist = 2 * R_CIRC + r

    for _ in range(maxlen):
        new = []
        for mat in frontier:
            for gen in GENS:
                cand = mat @ gen
                if hdist0(cand) > max_dist + 2 * A_HALF:
                    continue
                if any(np.allclose(cand, prev, atol=1e-7) for prev in seen):
                    continue
                seen.append(cand)
                new.append(cand)
        frontier = new

    return [mat for mat in seen if hdist0(mat) <= max_dist]


def in_octagon(points):
    """points: hyperboloid points (N,3). Klein coords inside regular octagon (vertices at angles (k+1/2)pi/4)."""
    klein = points[:, 1:] / points[:, :1]
    apothem = np.tanh(A_HALF)

    inside = np.ones(len(points), bool)
    for side in range(8):
        normal_angle = side * np.pi / 4
        projection = klein[:, 0] * np.cos(normal_angle) + klein[:, 1] * np.sin(normal_angle)
        inside &= projection <= apothem
    return inside


def sample(rng, N):
    batches = []
    while sum(len(batch) for batch in batches) < N:
        n_cand = 2 * N
        rad = np.arccosh(1 + rng.random(n_cand) * (np.cosh(R_CIRC) - 1))
        angle = 2 * np.pi * rng.random(n_cand)
        points = np.column_stack(
            [np.cosh(rad), np.sinh(rad) * np.cos(angle), np.sinh(rad) * np.sin(angle)]
        )
        batches.append(points[in_octagon(points)])
    return np.concatenate(batches)[:N]


def poincare(X):
    return X[:, 1:] / (1 + X[:, :1])
