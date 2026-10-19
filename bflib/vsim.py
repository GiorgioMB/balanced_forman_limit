"""Local (Palm) geometric graphs around a planted vertex x on several geometries.

For a planted vertex x (index 0) we sample a Poisson process of intensity n (w.r.t. Riemannian
volume, lambda = n*omega_d*r^d) on the geodesic ball B(x, 2r).  The geometric graph on these
points contains every vertex, edge, triangle and 4-cycle needed to compute the exact BF
curvature of all edges {x, j}.  Also returns tangent coordinates of each point at x.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.special import gamma
from scipy.integrate import quad


def omega(d):
    return np.pi ** (d / 2) / gamma(d / 2 + 1)


def _unit(rng, N, k):
    vecs = rng.standard_normal((N, k))
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _cap_alpha(rng, k, amax, N):
    """polar angle on S^k with density prop. to sin^{k-1} on [0, amax] (rejection)."""
    angles = np.empty(0)
    while len(angles) < N:
        n_draw = int(1.3 * (N - len(angles))) + 50
        cand = amax * rng.random(n_draw) ** (1 / k)
        accept = rng.random(n_draw) < (
            np.sin(cand) / np.where(cand > 0, cand, 1)
        ) ** (k - 1)
        angles = np.concatenate([angles, cand[accept]])
    return angles[:N]


_cache = {}


def _capvol(k, amax):
    key = (k, round(amax, 12))
    if key not in _cache:
        _cache[key] = (
            2
            * np.pi ** (k / 2)
            / gamma(k / 2)
            * quad(lambda a: np.sin(a) ** (k - 1), 0, amax)[0]
        )
    return _cache[key]


def sample_local(rng, geo, lam, r, R=None):
    """Return dict with: emb (embedding for kd-tree superset query), rad (query radius),
    exact (function(I,J)->bool adjacency on candidate pairs), tan (tangent coords at x), d.
    """
    R = 2 * r if R is None else R

    if geo.startswith("flat"):
        d = int(geo[4:])
        n = lam / (omega(d) * r**d)
        n_points = rng.poisson(n * omega(d) * R**d)
        points = (
            _unit(rng, n_points, d) * (R * rng.random(n_points) ** (1 / d))[:, None]
        )
        points = np.vstack([np.zeros(d), points])
        return dict(d=d, emb=points, rad=r, exact=None, tan=points)

    if geo in ("S2", "S3"):
        k = int(geo[1])
        n = lam / (omega(k) * r**k)
        n_points = rng.poisson(n * _capvol(k, R))
        radial = _cap_alpha(rng, k, R, n_points)
        dirs = _unit(rng, n_points, k)

        radial = np.concatenate([[0.0], radial])
        dirs = np.vstack([np.eye(k)[0], dirs])

        sphere_pts = np.column_stack([np.cos(radial), np.sin(radial)[:, None] * dirs])
        return dict(
            d=k,
            emb=sphere_pts,
            rad=2 * np.sin(r / 2),
            exact=None,
            tan=radial[:, None] * dirs,
        )

    if geo == "H2":
        n = lam / (omega(2) * r**2)
        n_points = rng.poisson(n * 2 * np.pi * (np.cosh(R) - 1))
        radial = np.arccosh(1 + rng.random(n_points) * (np.cosh(R) - 1))
        azimuth = 2 * np.pi * rng.random(n_points)

        radial = np.concatenate([[0.0], radial])
        azimuth = np.concatenate([[0.0], azimuth])

        hyperboloid = np.column_stack(
            [
                np.cosh(radial),
                np.sinh(radial) * np.cos(azimuth),
                np.sinh(radial) * np.sin(azimuth),
            ]
        )
        tangent = np.column_stack(
            [radial * np.cos(azimuth), radial * np.sin(azimuth)]
        )  # normal coords: 1-Lipschitz superset

        minkowski = np.array([1.0, -1.0, -1.0])
        cosh_r = np.cosh(r)
        exact_adj = lambda rows, cols: (
            (hyperboloid[rows] * minkowski) * hyperboloid[cols]
        ).sum(1) <= cosh_r

        return dict(d=2, emb=tangent, rad=r, exact=exact_adj, tan=tangent)

    if geo == "S2xR":
        n = lam / (omega(3) * r**3)
        n_points = rng.poisson(n * 2 * np.pi * (1 - np.cos(R)) * 2 * R)
        radial = np.arccos(1 - rng.random(n_points) * (1 - np.cos(R)))
        azimuth = 2 * np.pi * rng.random(n_points)
        s = rng.uniform(-R, R, n_points)

        keep = radial**2 + s**2 <= R**2
        radial, azimuth, s = radial[keep], azimuth[keep], s[keep]

        radial = np.concatenate([[0.0], radial])
        azimuth = np.concatenate([[0.0], azimuth])
        s = np.concatenate([[0.0], s])

        sphere_pts = np.column_stack(
            [
                np.cos(radial),
                np.sin(radial) * np.cos(azimuth),
                np.sin(radial) * np.sin(azimuth),
            ]
        )
        embedding = np.column_stack([sphere_pts, s])

        exact_adj = lambda rows, cols: (
            np.arccos(np.clip((sphere_pts[rows] * sphere_pts[cols]).sum(1), -1, 1)) ** 2
            + (s[rows] - s[cols]) ** 2
            <= r * r
        )

        tangent = np.column_stack(
            [radial * np.cos(azimuth), radial * np.sin(azimuth), s]
        )
        return dict(d=3, emb=embedding, rad=r, exact=exact_adj, tan=tangent)

    raise ValueError(geo)


def adjacency(loc):
    pairs = cKDTree(loc["emb"]).query_pairs(loc["rad"], output_type="ndarray")
    if loc["exact"] is not None and len(pairs):
        pairs = pairs[loc["exact"](pairs[:, 0], pairs[:, 1])]

    n_points = len(loc["emb"])
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])

    return csr_matrix(
        (np.ones(len(rows), np.int32), (rows, cols)), shape=(n_points, n_points)
    )


def bf_vertex_edges(A, x=0, js=None):
    """Exact BF curvature of edges {x,j} for j in js (default: all neighbours of x).
    Returns arrays BF, S (4-cycle term), tri."""
    indptr, indices = A.indptr, A.indices
    Nx = indices[indptr[x] : indptr[x + 1]]
    if js is None:
        js = Nx

    n_vertices = A.shape[0]
    in_Nx = np.zeros(n_vertices, bool)
    in_Nx[Nx] = True

    results = np.zeros((len(js), 3))
    dx = len(Nx)

    for edge_idx, j in enumerate(js):
        Nj = indices[indptr[j] : indptr[j + 1]]
        in_Nj = np.zeros(n_vertices, bool)
        in_Nj[Nj] = True
        tri = int(in_Nx[Nj].sum())
        dj = len(Nj)

        Ux = Nx[~in_Nj[Nx]]
        Ux = Ux[Ux != j]
        Uj = Nj[~in_Nx[Nj]]
        Uj = Uj[Uj != x]

        dmax, dmin = max(dx, dj), min(dx, dj)
        if dmin <= 1:
            continue

        sq = 0.0
        if len(Ux) and len(Uj):
            ind_Uj = np.zeros(n_vertices)
            ind_Uj[Uj] = 1.0
            ind_Ux = np.zeros(n_vertices)
            ind_Ux[Ux] = 1.0

            cnt_k = A[Ux] @ ind_Uj
            cnt_w = A[Uj] @ ind_Ux
            gam = max(cnt_k.max(), cnt_w.max())
            if gam > 0:
                sq = (np.count_nonzero(cnt_k) + np.count_nonzero(cnt_w)) / (gam * dmax)

        bf = 2 / dx + 2 / dj - 2 + 2 * tri / dmax + tri / dmin + sq
        results[edge_idx] = (bf, sq, tri)

    return results
