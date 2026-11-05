"""Exact Balanced Forman curvature of planted edges in Poisson random geometric graphs.

Only the points of B(p,r) U B(q,r) matter for the curvature of the edge {p,q} (eq. (4.1)).
For each lune point we count its neighbours in the opposite lune; by identity (2.3) these counts
give |sq_i|, |sq_j| and gamma_max. All curvatures are exact; nothing is approximated.

The order in which every sampler consumes random numbers is part of the reproducibility contract:
with the seeds of bflib.experiments it regenerates the numbers reported in the paper.
"""

import numpy as np
from scipy.integrate import quad
from scipy.spatial import cKDTree
from scipy.special import gamma

from .theory import omega


def bf_from_counts(n_p, n_q, n_L, cnt_k, cnt_w):
    """BF curvature of a planted edge from its local counts (Definition 2.1).

    n_p, n_q: Poisson points in B(p,r) and B(q,r) (the planted points excluded); n_L: points in the
    lens; cnt_k[k]: neighbours in U_q of the point k of U_p; cnt_w[w]: neighbours in U_p of w in U_q.
    Returns (BF, S, kink) with S the 4-cycle term and kink = (1/2)|tri||1/d_i - 1/d_j|, so that the
    symmetrised curvature of Example 5.14 is BF + kink.
    """
    d_i, d_j = 1 + n_p, 1 + n_q
    d_max, d_min = max(d_i, d_j), min(d_i, d_j)
    if d_min <= 1:
        return 0.0, 0.0, 0.0
    affine = 2 / d_i + 2 / d_j - 2
    triangle = 2 * n_L / d_max + n_L / d_min
    sq_i = int(np.count_nonzero(cnt_k))
    sq_j = int(np.count_nonzero(cnt_w))
    gamma_max = max(int(cnt_k.max()) if len(cnt_k) else 0, int(cnt_w.max()) if len(cnt_w) else 0)
    four_cycle = (sq_i + sq_j) / (gamma_max * d_max) if gamma_max > 0 else 0.0
    kink = 0.5 * n_L * abs(1 / d_i - 1 / d_j)
    return affine + triangle + four_cycle, four_cycle, kink


def _cross_counts(U_i, U_j, radius):
    if len(U_i) == 0 or len(U_j) == 0:
        return np.zeros(len(U_i), int), np.zeros(len(U_j), int)
    cnt_k = cKDTree(U_j).query_ball_point(U_i, radius, return_length=True)
    cnt_w = cKDTree(U_i).query_ball_point(U_j, radius, return_length=True)
    return np.asarray(cnt_k), np.asarray(cnt_w)


def _filtered_cross_counts(U_i, U_j, emb_i, emb_j, radius, exact):
    """Cross-lune counts from a superset query in an embedding, filtered by the exact test."""
    cnt_k, cnt_w = np.zeros(len(U_i), int), np.zeros(len(U_j), int)
    if len(U_i) and len(U_j):
        close = cKDTree(emb_i).sparse_distance_matrix(cKDTree(emb_j), radius, output_type="ndarray")
        keep = exact(U_i[close["i"]], U_j[close["j"]])
        cnt_k = np.bincount(close["i"][keep], minlength=len(U_i))
        cnt_w = np.bincount(close["j"][keep], minlength=len(U_j))
    return cnt_k, cnt_w


# ------------------------------------------------------------------------------- flat (torus, locally)
def flat_trial(rng, d, lam, t):
    """Planted edge of normalised length t in the flat Palm-Poisson model, at unit radius."""
    r = 1.0
    n = lam / omega(d)
    edge = t * r
    lo = np.array([-edge / 2 - r] + [-r] * (d - 1))
    hi = -lo
    points = rng.uniform(lo, hi, size=(rng.poisson(n * np.prod(hi - lo)), d))
    p = np.zeros(d)
    p[0] = -edge / 2
    q = -p
    in_p = ((points - p) ** 2).sum(1) <= r * r
    in_q = ((points - q) ** 2).sum(1) <= r * r
    cnt_k, cnt_w = _cross_counts(points[in_p & ~in_q], points[in_q & ~in_p], r)
    return bf_from_counts(in_p.sum(), in_q.sum(), (in_p & in_q).sum(), cnt_k, cnt_w)


# ------------------------------------------------------------------------------- round spheres S^k
def _cap_polar_angles(rng, k, amax, count):
    """Polar angles on S^k with density proportional to sin^{k-1} on [0, amax] (rejection)."""
    angles = np.empty(0)
    while len(angles) < count:
        n_draw = int(1.3 * (count - len(angles))) + 50
        cand = amax * rng.random(n_draw) ** (1 / k)
        accept = rng.random(n_draw) < (np.sin(cand) / np.where(cand > 0, cand, 1)) ** (k - 1)
        angles = np.concatenate([angles, cand[accept]])
    return angles[:count]


def _unit_vectors(rng, count, k):
    vecs = rng.standard_normal((count, k))
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


_cap_volume_cache = {}


def sphere_cap_volume(k, amax):
    key = (k, round(amax, 12))
    if key not in _cap_volume_cache:
        area = 2 * np.pi ** (k / 2) / gamma(k / 2)
        _cap_volume_cache[key] = area * quad(lambda a: np.sin(a) ** (k - 1), 0, amax)[0]
    return _cap_volume_cache[key]


def sphere_trial(rng, k, lam, t, r):
    """Planted edge on the unit sphere S^k (sectional curvature 1), radius r."""
    n = lam / (omega(k) * r**k)
    edge = t * r
    amax = r + edge / 2
    count = rng.poisson(n * sphere_cap_volume(k, amax))
    polar = _cap_polar_angles(rng, k, amax, count)
    dirs = _unit_vectors(rng, count, k)
    points = np.column_stack([np.cos(polar), np.sin(polar)[:, None] * dirs])
    p = np.zeros(k + 1)
    p[0], p[1] = np.cos(edge / 2), -np.sin(edge / 2)
    q = p.copy()
    q[1] = np.sin(edge / 2)
    cos_r = np.cos(r)
    in_p, in_q = points @ p >= cos_r, points @ q >= cos_r
    # geodesic balls on S^k are chordal balls of radius 2 sin(r/2): the k-d tree test is exact
    cnt_k, cnt_w = _cross_counts(points[in_p & ~in_q], points[in_q & ~in_p], 2 * np.sin(r / 2))
    return bf_from_counts(in_p.sum(), in_q.sum(), (in_p & in_q).sum(), cnt_k, cnt_w)


# ------------------------------------------------------------------------------- hyperbolic plane H^2
_MINKOWSKI = np.array([1.0, -1.0, -1.0])


def hyperbolic_trial(rng, lam, t, r):
    """Planted edge on H^2 (curvature -1), hyperboloid model, radius r."""
    n = lam / (omega(2) * r**2)
    edge = t * r
    amax = r + edge / 2
    count = rng.poisson(n * 2 * np.pi * (np.cosh(amax) - 1))
    radial = np.arccosh(1 + rng.random(count) * (np.cosh(amax) - 1))
    angle = 2 * np.pi * rng.random(count)
    points = np.column_stack([np.cosh(radial), np.sinh(radial) * np.cos(angle), np.sinh(radial) * np.sin(angle)])
    p = np.array([np.cosh(edge / 2), -np.sinh(edge / 2), 0.0])
    q = p.copy()
    q[1] = -q[1]
    cosh_r = np.cosh(r)
    in_p = (points * _MINKOWSKI) @ p <= cosh_r
    in_q = (points * _MINKOWSKI) @ q <= cosh_r
    in_Ui, in_Uj = in_p & ~in_q, in_q & ~in_p
    # normal coordinates at the midpoint: Euclidean distance there is at most the hyperbolic
    # distance (Rauch comparison), so the Euclidean query returns a superset
    emb_i = np.column_stack([radial[in_Ui] * np.cos(angle[in_Ui]), radial[in_Ui] * np.sin(angle[in_Ui])])
    emb_j = np.column_stack([radial[in_Uj] * np.cos(angle[in_Uj]), radial[in_Uj] * np.sin(angle[in_Uj])])
    exact = lambda a, b: (a * _MINKOWSKI * b).sum(1) <= cosh_r
    cnt_k, cnt_w = _filtered_cross_counts(points[in_Ui], points[in_Uj], emb_i, emb_j, r, exact)
    return bf_from_counts(in_p.sum(), in_q.sum(), (in_p & in_q).sum(), cnt_k, cnt_w)


# ------------------------------------------------------------------------------- S^2 x R
def s2xr_trial(rng, lam, t, r, direction):
    """Planted edge on S^2 x R (product metric), along the sphere ('S') or the line ('R')."""
    n = lam / (omega(3) * r**3)
    edge = t * r
    if direction == "R":
        amax, s_lo, s_hi = r, -r - edge / 2, r + edge / 2
        y_p = y_q = np.array([1.0, 0.0, 0.0])
        s_p, s_q = -edge / 2, edge / 2
    else:
        amax, s_lo, s_hi = r + edge / 2, -r, r
        y_p = np.array([np.cos(edge / 2), -np.sin(edge / 2), 0.0])
        y_q = y_p.copy()
        y_q[1] = -y_q[1]
        s_p = s_q = 0.0
    count = rng.poisson(n * 2 * np.pi * (1 - np.cos(amax)) * (s_hi - s_lo))
    polar = np.arccos(1 - rng.random(count) * (1 - np.cos(amax)))
    azimuth = 2 * np.pi * rng.random(count)
    sphere = np.column_stack([np.cos(polar), np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth)])
    heights = rng.uniform(s_lo, s_hi, count)

    def sq_dist(sph, hgt, y, s):
        return np.arccos(np.clip(sph @ y, -1, 1)) ** 2 + (hgt - s) ** 2

    in_p = sq_dist(sphere, heights, y_p, s_p) <= r * r
    in_q = sq_dist(sphere, heights, y_q, s_q) <= r * r
    in_Ui, in_Uj = in_p & ~in_q, in_q & ~in_p
    U_i = np.column_stack([sphere[in_Ui], heights[in_Ui]])
    U_j = np.column_stack([sphere[in_Uj], heights[in_Uj]])
    # (y, s) -> (y, s) in R^4 is 1-Lipschitz for the product metric: superset, then exact filter
    exact = lambda a, b: (
        np.arccos(np.clip((a[:, :3] * b[:, :3]).sum(1), -1, 1)) ** 2 + (a[:, 3] - b[:, 3]) ** 2 <= r * r
    )
    cnt_k, cnt_w = _filtered_cross_counts(U_i, U_j, U_i, U_j, r, exact)
    return bf_from_counts(in_p.sum(), in_q.sum(), (in_p & in_q).sum(), cnt_k, cnt_w)


# ------------------------------------------------------------------------------- density gradient
def cosine_density(points, amplitude):
    """varpi(x) = 1 + a cos(2 pi x_1), a density on T^2 (Proposition 5.8, Section S9)."""
    return 1 + amplitude * np.cos(2 * np.pi * points[:, 0])


def density_endpoints(r, t, midpoint=(0.25, 0.0)):
    """Endpoints of the planted edge: centred at x_1 = 1/4, where |grad log varpi| is largest,
    and parallel to the gradient."""
    mid = np.array(midpoint)
    axis = np.array([1.0, 0.0])
    return mid - t * r / 2 * axis, mid + t * r / 2 * axis


def density_trial(rng, lam, r, t, amplitude):
    """Planted edge in R^2 for the Poisson intensity n varpi(x), n = lam / (pi r^2), fixed radius r.

    This is the local (Palm) picture of the density varpi on T^2 around the edge.
    """
    n = lam / (np.pi * r * r)
    p, q = density_endpoints(r, t)
    mid = (p + q) / 2
    half_width = t * r / 2 + r
    lo, hi = mid - half_width, mid + half_width
    count = rng.poisson(n * (1 + amplitude) * (2 * half_width) ** 2)
    points = rng.uniform(lo, hi, (count, 2))
    points = points[rng.random(count) < cosine_density(points, amplitude) / (1 + amplitude)]
    in_p = np.linalg.norm(points - p, axis=1) <= r
    in_q = np.linalg.norm(points - q, axis=1) <= r
    cnt_k, cnt_w = _cross_counts(points[in_p & ~in_q], points[in_q & ~in_p], r)
    return bf_from_counts(in_p.sum(), in_q.sum(), (in_p & in_q).sum(), cnt_k, cnt_w)


# ------------------------------------------------------------------------------- two edges at a vertex
def pair_trial(rng, lam, t, theta):
    """BF curvature of the edges {x,j}, {x,k} with |j| = |k| = t at angle theta (flat, unit radius)."""
    x = np.zeros(2)
    j = np.array([t, 0.0])
    k = t * np.array([np.cos(theta), np.sin(theta)])
    lo = np.minimum.reduce([x, j, k]) - 1
    hi = np.maximum.reduce([x, j, k]) + 1
    points = rng.uniform(lo, hi, (rng.poisson(lam / np.pi * np.prod(hi - lo)), 2))

    def edge_bf(pts, a, b):
        in_a = ((pts - a) ** 2).sum(1) <= 1
        in_b = ((pts - b) ** 2).sum(1) <= 1
        cnt_k, cnt_w = _cross_counts(pts[in_a & ~in_b], pts[in_b & ~in_a], 1.0)
        return bf_from_counts(in_a.sum(), in_b.sum(), (in_a & in_b).sum(), cnt_k, cnt_w)[0]

    # the third planted point is an ordinary point for the other edge
    return edge_bf(np.vstack([points, k]), x, j), edge_bf(np.vstack([points, j]), x, k)
