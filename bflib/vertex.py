"""Local (Palm) geometric graphs around a planted vertex x (Section S5).

For a planted vertex x (index 0) we sample a Poisson process of intensity n with respect to the
Riemannian volume, lambda = n omega_d r^d, on the geodesic ball B(x, 2r). The geometric graph on
these points contains every vertex, edge, triangle and 4-cycle needed for the exact BF curvature
of all edges {x, j}. Tangent coordinates at x give the latent lengths t_xj and directions v_xj.
"""

import numpy as np
from scipy.integrate import quad
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.special import gamma

from .planted import _cap_polar_angles, _unit_vectors
from .theory import omega

_cache = {}


def _cap_volume(k, amax):
    key = (k, round(amax, 12))
    if key not in _cache:
        _cache[key] = 2 * np.pi ** (k / 2) / gamma(k / 2) * quad(lambda a: np.sin(a) ** (k - 1), 0, amax)[0]
    return _cache[key]


def sample_local(rng, geometry, lam, r):
    """Poisson points on B(x, 2r) around x = point 0.

    Returns a dict with: emb (embedding for a superset k-d tree query), rad (query radius),
    exact (exact adjacency test on candidate pairs, or None if the query is exact), tan (tangent
    coordinates at x) and d.
    """
    R = 2 * r
    if geometry.startswith("flat"):
        d = int(geometry[4:])
        n = lam / (omega(d) * r**d)
        count = rng.poisson(n * omega(d) * R**d)
        points = _unit_vectors(rng, count, d) * (R * rng.random(count) ** (1 / d))[:, None]
        points = np.vstack([np.zeros(d), points])
        return dict(d=d, emb=points, rad=r, exact=None, tan=points)

    if geometry in ("S2", "S3"):
        k = int(geometry[1])
        n = lam / (omega(k) * r**k)
        count = rng.poisson(n * _cap_volume(k, R))
        radial = np.concatenate([[0.0], _cap_polar_angles(rng, k, R, count)])
        dirs = np.vstack([np.eye(k)[0], _unit_vectors(rng, count, k)])
        sphere = np.column_stack([np.cos(radial), np.sin(radial)[:, None] * dirs])
        return dict(d=k, emb=sphere, rad=2 * np.sin(r / 2), exact=None, tan=radial[:, None] * dirs)

    if geometry == "H2":
        n = lam / (omega(2) * r**2)
        count = rng.poisson(n * 2 * np.pi * (np.cosh(R) - 1))
        radial = np.concatenate([[0.0], np.arccosh(1 + rng.random(count) * (np.cosh(R) - 1))])
        azimuth = np.concatenate([[0.0], 2 * np.pi * rng.random(count)])
        hyperboloid = np.column_stack(
            [np.cosh(radial), np.sinh(radial) * np.cos(azimuth), np.sinh(radial) * np.sin(azimuth)]
        )
        tangent = np.column_stack([radial * np.cos(azimuth), radial * np.sin(azimuth)])
        minkowski = np.array([1.0, -1.0, -1.0])
        cosh_r = np.cosh(r)
        exact = lambda rows, cols: ((hyperboloid[rows] * minkowski) * hyperboloid[cols]).sum(1) <= cosh_r
        return dict(d=2, emb=tangent, rad=r, exact=exact, tan=tangent)

    if geometry == "S2xR":
        n = lam / (omega(3) * r**3)
        count = rng.poisson(n * 2 * np.pi * (1 - np.cos(R)) * 2 * R)
        radial = np.arccos(1 - rng.random(count) * (1 - np.cos(R)))
        azimuth = 2 * np.pi * rng.random(count)
        s = rng.uniform(-R, R, count)
        keep = radial**2 + s**2 <= R**2
        radial = np.concatenate([[0.0], radial[keep]])
        azimuth = np.concatenate([[0.0], azimuth[keep]])
        s = np.concatenate([[0.0], s[keep]])
        sphere = np.column_stack([np.cos(radial), np.sin(radial) * np.cos(azimuth), np.sin(radial) * np.sin(azimuth)])
        exact = lambda rows, cols: (
            np.arccos(np.clip((sphere[rows] * sphere[cols]).sum(1), -1, 1)) ** 2 + (s[rows] - s[cols]) ** 2 <= r * r
        )
        tangent = np.column_stack([radial * np.cos(azimuth), radial * np.sin(azimuth), s])
        return dict(d=3, emb=np.column_stack([sphere, s]), rad=r, exact=exact, tan=tangent)

    raise ValueError(f"unknown geometry {geometry!r}")


def adjacency(local):
    pairs = cKDTree(local["emb"]).query_pairs(local["rad"], output_type="ndarray")
    if local["exact"] is not None and len(pairs):
        pairs = pairs[local["exact"](pairs[:, 0], pairs[:, 1])]
    n = len(local["emb"])
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    return csr_matrix((np.ones(len(rows), np.int32), (rows, cols)), shape=(n, n))


def bf_vertex_edges(A, x, js):
    """Exact BF curvature of the edges {x, j}, j in js. Returns an array of rows (BF, S, triangles)."""
    indptr, indices = A.indptr, A.indices
    N_x = indices[indptr[x] : indptr[x + 1]]
    n = A.shape[0]
    in_Nx = np.zeros(n, bool)
    in_Nx[N_x] = True
    out = np.zeros((len(js), 3))
    d_x = len(N_x)
    for e, j in enumerate(js):
        N_j = indices[indptr[j] : indptr[j + 1]]
        in_Nj = np.zeros(n, bool)
        in_Nj[N_j] = True
        tri = int(in_Nx[N_j].sum())
        d_j = len(N_j)
        U_x = N_x[~in_Nj[N_x]]
        U_x = U_x[U_x != j]
        U_j = N_j[~in_Nx[N_j]]
        U_j = U_j[U_j != x]
        d_max, d_min = max(d_x, d_j), min(d_x, d_j)
        if d_min <= 1:
            continue
        four_cycle = 0.0
        if len(U_x) and len(U_j):
            ind_Uj = np.zeros(n)
            ind_Uj[U_j] = 1.0
            ind_Ux = np.zeros(n)
            ind_Ux[U_x] = 1.0
            cnt_k = A[U_x] @ ind_Uj
            cnt_w = A[U_j] @ ind_Ux
            gamma_max = max(cnt_k.max(), cnt_w.max())
            if gamma_max > 0:
                four_cycle = (np.count_nonzero(cnt_k) + np.count_nonzero(cnt_w)) / (gamma_max * d_max)
        out[e] = (2 / d_x + 2 / d_j - 2 + 2 * tri / d_max + tri / d_min + four_cycle, four_cycle, tri)
    return out
