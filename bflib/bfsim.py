"""Exact Balanced Forman curvature of a planted edge in Poisson random geometric graphs.
Geometries: flat R^d (torus, locally), round spheres S^k, hyperbolic plane H^2, S^2 x R.
Only points in B(p,r) U B(q,r) matter for BF of edge {p,q}."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import gamma, betainc
from scipy.integrate import quad


def omega(d):
    return np.pi ** (d / 2) / gamma(d / 2 + 1)


def f_lens(d, t):
    return betainc((d + 1) / 2, 0.5, 1 - np.asarray(t, float) ** 2 / 4)


def Psi(d, t):
    return -2 + 3 * f_lens(d, t)


def kappa(d, t):
    t = np.asarray(t, float)
    return omega(d - 1) / (6 * (d + 1) * omega(d)) * t * (1 - t**2 / 4) ** ((d + 1) / 2)


def W_mean(f):
    return -f * np.sqrt((1 - f) / np.pi)


def W_var(f):
    return (
        9 * f * (1 - f) ** 2
        + 4.5 * f**2 * (1 - f)
        + 0.5 * f**2 * (1 - f) * (1 - 2 / np.pi)
    )


def bf_core(n_p, n_q, n_L, cnt_k, cnt_w):
    """n_p,n_q: # Poisson pts in B(p,r), B(q,r) (planted pts excluded); n_L: # in lens;
    cnt_k[k]: # U_j-points adjacent to k in U_i; cnt_w[w]: # U_i-points adjacent to w in U_j.
    """
    d_i = 1 + n_p
    d_j = 1 + n_q
    dmax = max(d_i, d_j)
    dmin = min(d_i, d_j)
    if dmin <= 1:
        return 0.0, 0.0, 0.0

    aff = 2 / d_i + 2 / d_j - 2
    tri = 2 * n_L / dmax + n_L / dmin

    sq_i = int(np.count_nonzero(cnt_k))
    sq_j = int(np.count_nonzero(cnt_w))
    gam = max(
        int(cnt_k.max()) if len(cnt_k) else 0,
        int(cnt_w.max()) if len(cnt_w) else 0,
    )
    sq = (sq_i + sq_j) / (gam * dmax) if gam > 0 else 0.0

    return aff + tri + sq, sq, tri


def _cross_kd(Ui, Uj, rad):
    if len(Ui) == 0 or len(Uj) == 0:
        return np.zeros(len(Ui), int), np.zeros(len(Uj), int)

    cnt_k = cKDTree(Uj).query_ball_point(Ui, rad, return_length=True)
    cnt_w = cKDTree(Ui).query_ball_point(Uj, rad, return_length=True)
    return np.asarray(cnt_k), np.asarray(cnt_w)


def _cross_brute(Ui, Uj, adj_fn, chunk=2000):
    cnt_k = np.zeros(len(Ui), int)
    cnt_w = np.zeros(len(Uj), int)

    for start in range(0, len(Ui), chunk):
        block_adj = adj_fn(Ui[start : start + chunk], Uj)  # boolean matrix
        cnt_k[start : start + chunk] = block_adj.sum(1)
        cnt_w += block_adj.sum(0)

    return cnt_k, cnt_w


# ---------------- flat ----------------
def flat_trial(rng, d, lam, t):
    r = 1.0
    n = lam / omega(d)
    edge_len = t * r

    lo = np.array([-edge_len / 2 - r] + [-r] * (d - 1))
    hi = -lo
    n_points = rng.poisson(n * np.prod(hi - lo))
    points = rng.uniform(lo, hi, size=(n_points, d))

    p = np.zeros(d)
    p[0] = -edge_len / 2
    q = -p

    in_p = ((points - p) ** 2).sum(1) <= r * r
    in_q = ((points - q) ** 2).sum(1) <= r * r
    cnt_k, cnt_w = _cross_kd(points[in_p & ~in_q], points[in_q & ~in_p], r)

    return bf_core(in_p.sum(), in_q.sum(), (in_p & in_q).sum(), cnt_k, cnt_w)


# ---------------- spheres S^k (radius 1) ----------------
def _sphere_cap_sample(rng, k, amax, N):
    angles = np.empty(0)
    while len(angles) < N:
        n_draw = int(1.3 * (N - len(angles))) + 50
        cand = amax * rng.random(n_draw) ** (1 / k)
        accept = rng.random(n_draw) < (
            np.sin(cand) / np.where(cand > 0, cand, 1)
        ) ** (k - 1)
        angles = np.concatenate([angles, cand[accept]])
    return angles[:N]


def _unit(rng, N, k):
    vecs = rng.standard_normal((N, k))
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


_capvol_cache = {}


def sphere_capvol(k, amax):
    key = (k, round(amax, 12))
    if key not in _capvol_cache:
        sphere_area = 2 * np.pi ** (k / 2) / gamma(k / 2)
        _capvol_cache[key] = sphere_area * quad(
            lambda a: np.sin(a) ** (k - 1), 0, amax
        )[0]
    return _capvol_cache[key]


def sphere_trial(rng, k, lam, t, r):
    n = lam / (omega(k) * r**k)
    edge_len = t * r
    amax = r + edge_len / 2

    n_points = rng.poisson(n * sphere_capvol(k, amax))
    polar = _sphere_cap_sample(rng, k, amax, n_points)
    dirs = _unit(rng, n_points, k)
    points = np.column_stack([np.cos(polar), np.sin(polar)[:, None] * dirs])

    p = np.zeros(k + 1)
    p[0] = np.cos(edge_len / 2)
    p[1] = -np.sin(edge_len / 2)
    q = p.copy()
    q[1] = np.sin(edge_len / 2)

    cos_r = np.cos(r)
    in_p = points @ p >= cos_r
    in_q = points @ q >= cos_r
    cnt_k, cnt_w = _cross_kd(
        points[in_p & ~in_q], points[in_q & ~in_p], 2 * np.sin(r / 2)
    )

    return bf_core(in_p.sum(), in_q.sum(), (in_p & in_q).sum(), cnt_k, cnt_w)


# ---------------- hyperbolic plane H^2 (K=-1), hyperboloid model ----------------
def hyp_trial(rng, lam, t, r):
    n = lam / (omega(2) * r**2)
    edge_len = t * r
    amax = r + edge_len / 2

    n_points = rng.poisson(n * 2 * np.pi * (np.cosh(amax) - 1))
    radial = np.arccosh(1 + rng.random(n_points) * (np.cosh(amax) - 1))
    angle = 2 * np.pi * rng.random(n_points)
    points = np.column_stack(
        [
            np.cosh(radial),
            np.sinh(radial) * np.cos(angle),
            np.sinh(radial) * np.sin(angle),
        ]
    )

    p = np.array([np.cosh(edge_len / 2), -np.sinh(edge_len / 2), 0.0])
    q = p.copy()
    q[1] = -q[1]

    minkowski = np.array([1.0, -1.0, -1.0])
    cosh_r = np.cosh(r)
    in_p = (points * minkowski) @ p <= cosh_r
    in_q = (points * minkowski) @ q <= cosh_r

    in_Ui = in_p & ~in_q
    in_Uj = in_q & ~in_p
    Ui = points[in_Ui]
    Uj = points[in_Uj]

    # normal coordinates at m: Euclidean distance there is <= hyperbolic distance (K<=0), so it gives a superset
    normal_i = np.column_stack(
        [radial[in_Ui] * np.cos(angle[in_Ui]), radial[in_Ui] * np.sin(angle[in_Ui])]
    )
    normal_j = np.column_stack(
        [radial[in_Uj] * np.cos(angle[in_Uj]), radial[in_Uj] * np.sin(angle[in_Uj])]
    )

    cnt_k = np.zeros(len(Ui), int)
    cnt_w = np.zeros(len(Uj), int)
    if len(Ui) and len(Uj):
        close = cKDTree(normal_i).sparse_distance_matrix(
            cKDTree(normal_j), r, output_type="ndarray"
        )
        pair_i = close["i"]
        pair_j = close["j"]
        adjacent = (Ui[pair_i] * minkowski * Uj[pair_j]).sum(1) <= cosh_r
        cnt_k = np.bincount(pair_i[adjacent], minlength=len(Ui))
        cnt_w = np.bincount(pair_j[adjacent], minlength=len(Uj))

    return bf_core(in_p.sum(), in_q.sum(), (in_p & in_q).sum(), cnt_k, cnt_w)


# ---------------- S^2 x R (K=1 on S^2 factor) ----------------
def s2xr_trial(rng, lam, t, r, direction):
    n = lam / (omega(3) * r**3)
    edge_len = t * r

    if direction == "R":
        amax = r
        s_lo, s_hi = -r - edge_len / 2, r + edge_len / 2
        yp = yq = np.array([1.0, 0.0, 0.0])
        s_p, s_q = -edge_len / 2, edge_len / 2
    else:
        amax = r + edge_len / 2
        s_lo, s_hi = -r, r
        yp = np.array([np.cos(edge_len / 2), -np.sin(edge_len / 2), 0.0])
        yq = yp.copy()
        yq[1] = -yq[1]
        s_p = s_q = 0.0

    n_points = rng.poisson(n * 2 * np.pi * (1 - np.cos(amax)) * (s_hi - s_lo))
    polar = np.arccos(1 - rng.random(n_points) * (1 - np.cos(amax)))
    azimuth = 2 * np.pi * rng.random(n_points)
    sphere_pts = np.column_stack(
        [
            np.cos(polar),
            np.sin(polar) * np.cos(azimuth),
            np.sin(polar) * np.sin(azimuth),
        ]
    )
    heights = rng.uniform(s_lo, s_hi, n_points)


    def sq_dist(sph, hgt, y, s):
        return np.arccos(np.clip(sph @ y, -1, 1)) ** 2 + (hgt - s) ** 2

    in_p = sq_dist(sphere_pts, heights, yp, s_p) <= r * r
    in_q = sq_dist(sphere_pts, heights, yq, s_q) <= r * r

    in_Ui = in_p & ~in_q
    in_Uj = in_q & ~in_p
    Ui = np.column_stack([sphere_pts[in_Ui], heights[in_Ui]])
    Uj = np.column_stack([sphere_pts[in_Uj], heights[in_Uj]])

    cnt_k = np.zeros(len(Ui), int)
    cnt_w = np.zeros(len(Uj), int)
    if len(Ui) and len(Uj):
        close = cKDTree(Ui).sparse_distance_matrix(
            cKDTree(Uj), r, output_type="ndarray"
        )
        pair_i = close["i"]
        pair_j = close["j"]
        geo_sq_dist = (
            np.arccos(np.clip((Ui[pair_i, :3] * Uj[pair_j, :3]).sum(1), -1, 1)) ** 2
            + (Ui[pair_i, 3] - Uj[pair_j, 3]) ** 2
        )
        adjacent = geo_sq_dist <= r * r
        cnt_k = np.bincount(pair_i[adjacent], minlength=len(Ui))
        cnt_w = np.bincount(pair_j[adjacent], minlength=len(Uj))

    return bf_core(in_p.sum(), in_q.sum(), (in_p & in_q).sum(), cnt_k, cnt_w)
