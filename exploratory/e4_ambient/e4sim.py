"""Planted edges on a cylinder of radius R (intrinsically flat), graph built with AMBIENT (chord)
distances in R^3.  Intrinsic coords (u,z), u = R*theta.  Edge direction 'circ' (along u) or 'axial'.
"""

import numpy as np
from scipy.spatial import cKDTree

from bflib.bfsim import bf_core


def emb(coords, R):
    angle = coords[:, 0] / R
    return np.column_stack([R * np.cos(angle), R * np.sin(angle), coords[:, 1]])


def chord_u(du, R):
    return 2 * R * np.sin(np.abs(du) / (2 * R))


def trial(rng, lam, r, t, R, direc, ambient=True):
    # intensity per unit intrinsic area; lam = n*pi*r^2 (flat normalisation)
    n = lam / (np.pi * r * r)

    if direc == "axial":
        p = np.array([0.0, -t * r / 2])
        q = np.array([0.0, t * r / 2])
    else:  # ambient distance t*r along the circle: solve chord(du)=t r
        du = 2 * R * np.arcsin(t * r / (2 * R)) if ambient else t * r
        p = np.array([-du / 2, 0.0])
        q = np.array([du / 2, 0.0])

    half_width = r * 1.2 + abs(q - p).max() / 2
    n_points = rng.poisson(n * (2 * half_width) ** 2)
    points = rng.uniform(-half_width, half_width, (n_points, 2))

    emb_points = emb(points, R)
    emb_p = emb(p[None], R)[0]
    emb_q = emb(q[None], R)[0]

    if ambient:
        in_p = np.linalg.norm(emb_points - emb_p, axis=1) <= r
        in_q = np.linalg.norm(emb_points - emb_q, axis=1) <= r
        Ui = emb_points[in_p & ~in_q]
        Uj = emb_points[in_q & ~in_p]
    else:
        in_p = np.linalg.norm(points - p, axis=1) <= r
        in_q = np.linalg.norm(points - q, axis=1) <= r
        Ui = points[in_p & ~in_q]
        Uj = points[in_q & ~in_p]

    if len(Ui) and len(Uj):
        cnt_k = np.asarray(cKDTree(Uj).query_ball_point(Ui, r, return_length=True))
        cnt_w = np.asarray(cKDTree(Ui).query_ball_point(Uj, r, return_length=True))
    else:
        cnt_k = np.zeros(len(Ui), int)
        cnt_w = np.zeros(len(Uj), int)

    return bf_core(in_p.sum(), in_q.sum(), (in_p & in_q).sum(), cnt_k, cnt_w)[0]


def lens_ratio(r, t, R, direc, M=3000):
    """exact ambient lens ratio by grid integration in intrinsic coords"""
    if direc == "axial":
        p = np.array([0.0, -t * r / 2])
        q = np.array([0.0, t * r / 2])
    else:
        du = 2 * R * np.arcsin(t * r / (2 * R))
        p = np.array([-du / 2, 0.0])
        q = np.array([du / 2, 0.0])

    half_width = r * 1.2 + abs(q - p).max() / 2
    grid = (np.arange(M) + 0.5) / M * 2 * half_width - half_width
    grid_u, grid_z = np.meshgrid(grid, grid, indexing="ij")

    def in_ball(center):
        return chord_u(grid_u - center[0], R) ** 2 + (grid_z - center[1]) ** 2 <= r * r

    in_ball_p = in_ball(p)
    return (in_ball_p & in_ball(q)).sum() / in_ball_p.sum()
