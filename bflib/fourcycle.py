"""Constants of the 4-cycle term: u_d(t), vbar_d(t) and c_d(t) = 2 u_d / vbar_d of eq. (3.1).

At unit scale U_t = B(0,1) minus B(t e_1,1), U'_t = B(t e_1,1) minus B(0,1), and v_t(z) is the
normalised volume of B(z,1) n U'_t. For d = 2, Lemma 3.4(b) gives u_2(t) = 2 v_t(h) at the rim point
h; v_t is computed here by adaptive quadrature in polar coordinates around z, and its maximum over
the right boundary of U_t is located by a scan (Remark 3.5). For every d the constants are also
estimated by Monte Carlo integration over the lunes.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from .theory import f_lens, omega


# ------------------------------------------------------------------------------- d = 2 by quadrature
def _ray_interval(z, c, u):
    """Parameters s >= 0 with |z + s u - c| <= 1, as an interval (or None)."""
    b = np.dot(z - c, u)
    disc = b * b - (np.dot(z - c, z - c) - 1)
    if disc < 0:
        return None
    root = np.sqrt(disc)
    return -b - root, -b + root


def visibility_2d(t, z):
    """v_t(z) = Area(B(z,1) n U'_t) / pi for d = 2, by quadrature over directions around z."""
    c1, c0 = np.array([t, 0.0]), np.zeros(2)

    def radial_mass(theta):
        u = np.array([np.cos(theta), np.sin(theta)])
        inside_q = _ray_interval(z, c1, u)
        if inside_q is None:
            return 0.0
        lo, hi = max(inside_q[0], 0.0), min(inside_q[1], 1.0)
        if hi <= lo:
            return 0.0
        segments = [(lo, hi)]
        inside_p = _ray_interval(z, c0, u)
        if inside_p is not None:
            a, b = inside_p
            segments = []
            if lo < min(a, hi):
                segments.append((lo, min(a, hi)))
            if max(b, lo) < hi:
                segments.append((max(b, lo), hi))
        return sum(0.5 * (q * q - p * p) for p, q in segments)

    return quad(radial_mass, 0, 2 * np.pi, limit=400, epsabs=1e-11)[0] / np.pi


def constants_2d(t, n_scan=200):
    """u_2(t) = 2 v_t(h), v_t(h), and the maximum of v_t over a scan of the right boundary of U_t."""
    ell = np.sqrt(1 - t * t / 4)
    h = np.array([t / 2, ell])
    v_h = visibility_2d(t, h)
    best = v_h
    # arc of the circle around t e_1 inside B(0,1), beyond the rim point h
    angle_h = np.arctan2(ell, t / 2 - t)
    for a in np.linspace(angle_h, np.pi, n_scan)[1:]:
        z = np.array([t, 0.0]) + (1 - 1e-9) * np.array([np.cos(a), np.sin(a)])
        if z @ z <= 1:
            best = max(best, visibility_2d(t, z))
    # arc of the unit circle beyond the rim point (x_1 < t/2)
    for a in np.linspace(np.arctan2(ell, t / 2), np.pi, n_scan)[1:]:
        z = (1 - 1e-9) * np.array([np.cos(a), np.sin(a)])
        if (z - [t, 0]) @ (z - [t, 0]) > 1:
            best = max(best, visibility_2d(t, z))
    return dict(u=2 * v_h, v_rim=v_h, v_boundary_max=best, c=2 * 2 * v_h / best)


# ------------------------------------------------------------------------------- Monte Carlo, any d
def _sample_lune(rng, d, t, side, count):
    centre_i = np.zeros(d)
    centre_i[0] = -t / 2
    centre_j = -centre_i
    centre, other = (centre_j, centre_i) if side == "j" else (centre_i, centre_j)
    batches = []
    while sum(len(b) for b in batches) < count:
        pts = centre + rng.uniform(-1, 1, (2 * count, d))
        batches.append(pts[(((pts - centre) ** 2).sum(1) <= 1) & (((pts - other) ** 2).sum(1) > 1)])
    return np.concatenate(batches)[:count]


def constants_monte_carlo(rng, d, t, n_lune_j, n_lune_i=40_000):
    """Monte Carlo estimates of u_d(t), vbar_d(t), c_d(t), with standard errors.

    v_t is estimated by the fraction of n_lune_j uniform points of U'_t within distance 1; its supremum
    by the best of n_lune_i uniform points of U_t, refined by Nelder-Mead. The standard errors are
    binomial (for u, and for v at the maximiser) combined by the delta method for c; they do not
    include the bias of maximising a noisy function, which is of the same order.
    """
    f = float(f_lens(d, t))
    vol_U = omega(d) * (1 - f)
    lune_j = _sample_lune(rng, d, t, "j", n_lune_j)
    tree = cKDTree(lune_j)
    lune_i = _sample_lune(rng, d, t, "i", n_lune_i)
    n_nbrs = np.asarray(tree.query_ball_point(lune_i, 1.0, return_length=True))
    reach = vol_U * n_nbrs / len(lune_j)
    seen = np.mean(n_nbrs > 0)
    u = (1 - f) * seen
    centre_i = np.zeros(d)
    centre_i[0] = -t / 2
    centre_j = -centre_i

    def neg_v(point):
        if ((point - centre_i) ** 2).sum() > 1 or ((point - centre_j) ** 2).sum() <= 1:
            return 0.0
        return -vol_U * tree.query_ball_point(point, 1.0, return_length=True) / len(lune_j)

    starts = lune_i[np.argsort(-reach)[:5]]
    g_max = max(reach.max(), max(-minimize(neg_v, s, method="Nelder-Mead",
                                           options={"xatol": 1e-4, "fatol": 1e-6}).fun for s in starts))
    v_bar = g_max / omega(d)
    q = g_max / vol_U  # fraction of the lune seen from the maximiser
    se_u = (1 - f) * np.sqrt(seen * (1 - seen) / len(lune_i))
    se_v = (1 - f) * np.sqrt(q * (1 - q) / len(lune_j))
    c = 2 * u / v_bar
    se_c = c * np.sqrt((se_u / u) ** 2 + (se_v / v_bar) ** 2)
    return dict(u=u, u_se=se_u, v=v_bar, v_se=se_v, c=c, c_se=se_c)
