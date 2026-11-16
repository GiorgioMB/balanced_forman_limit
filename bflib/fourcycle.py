"""Constants of the 4-cycle term: u_d(t), vbar_d(t) and c_d(t) = 2 u_d / vbar_d of eq. (3.1).

At unit scale U_t = B(0,1) minus B(t e_1,1), U'_t = B(t e_1,1) minus B(0,1), and v_t(z) is the
normalised volume of B(z,1) n U'_t. For d = 2, Lemma 3.4(b) gives u_2(t) = 2 v_t(h) at the rim point
h; v_t is computed here by adaptive quadrature in polar coordinates around z, and its profile along the
right boundary of U_t is computed on a scan of both arcs (Remark 3.5). For every d the constants are
also estimated by Monte Carlo integration over the lunes.
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


def _unit_circle_crossings(z, c):
    """Points where the unit circle about z meets the unit circle about c."""
    gap = c - z
    dist = float(np.hypot(*gap))
    if dist == 0 or dist > 2:
        return []
    along, across = dist / 2, np.sqrt(max(1 - dist * dist / 4, 0.0))
    base, normal = z + along * gap / dist, np.array([-gap[1], gap[0]]) / dist
    return [base + across * normal, base - across * normal]


def _kink_angles(t, z):
    """Directions from z at which the radial mass of visibility_2d is not smooth: towards the rim points h+-
    (where the order of the two circles along the ray changes), towards the points where a ray leaves B(z,1)
    on one of the circles (where the clipping at 1 starts or stops) and, when z lies outside a disc, along
    its tangents (where the ray starts or stops meeting it)."""
    c0, c1 = np.zeros(2), np.array([t, 0.0])
    ell = np.sqrt(1 - t * t / 4)
    targets = [np.array([t / 2, ell]), np.array([t / 2, -ell])]
    for c in (c0, c1):
        targets += _unit_circle_crossings(z, c)
    angles = [np.arctan2(p[1] - z[1], p[0] - z[0]) for p in targets]
    for c in (c0, c1):
        dist = float(np.hypot(*(c - z)))
        if dist > 1:
            towards, half = np.arctan2(c[1] - z[1], c[0] - z[0]), np.arcsin(1 / dist)
            angles += [towards - half, towards + half]
    angles = np.mod(angles, 2 * np.pi)
    return sorted({float(a) for a in angles if 1e-12 < a < 2 * np.pi - 1e-12})


def visibility_2d(t, z):
    """v_t(z) = Area(B(z,1) n U'_t) / pi for d = 2, by quadrature over directions around z.

    The radial mass vanishes outside a few angular windows, which are narrow for small t; the angles at
    which it is not smooth are therefore passed to quad as breakpoints (without them, quad can sample only
    zeros and return 0, as it did at t = 0.1 for points of the right boundary of U_t with v_t > 0)."""
    z = np.asarray(z, float)
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

    return quad(radial_mass, 0, 2 * np.pi, points=_kink_angles(t, z), limit=400, epsabs=1e-13,
                epsrel=1e-12)[0] / np.pi


MONOTONE_TOL = 1e-10  # far above the quadrature error (about 1e-12), far below the steps of the scans


def _profile(values):
    """Largest increase along a scan, and whether the scan decreases strictly while v_t > 0."""
    values = np.asarray(values)
    steps = np.diff(values)
    positive = values[1:] > MONOTONE_TOL
    return float(steps.max()), bool(np.all(steps <= MONOTONE_TOL) and np.all(steps[positive] < 0))


def constants_2d(t, n_scan=200):
    """u_2(t) = 2 v_t(h) (Lemma 3.4(b)), and the profile of v_t along the right boundary of U_t (Remark 3.5).

    The right boundary consists of the arc of the circle about t e_1 inside B(0,1) and the arcs of the unit
    circle beyond the rim points, which meet at h+-. By the symmetry x_2 -> -x_2 it suffices to scan, from h+,
    the first arc down to (t - 1, 0) and the second around to (-1, 0), with n_scan points each; every scanned
    point lies in U_t, at distance 1e-9 from the arc. Returns u, v_rim = v_t(h), the maximum of v_t over both
    scans, c = 2 u / max, and for each arc the largest increase of v_t away from h and whether v_t decreases
    strictly while it is positive."""
    ell = np.sqrt(1 - t * t / 4)
    h = np.array([t / 2, ell])
    v_h = visibility_2d(t, h)
    angle_h = np.arctan2(ell, t / 2 - t)  # h seen from t e_1
    circle_q = [np.array([t, 0.0]) + (1 + 1e-9) * np.array([np.cos(a), np.sin(a)])
                for a in np.linspace(angle_h, np.pi, n_scan)[1:]]
    circle_p = [(1 - 1e-9) * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(np.arctan2(ell, t / 2), np.pi, n_scan)[1:]]
    assert all(z @ z <= 1 and (z - [t, 0]) @ (z - [t, 0]) > 1 for z in circle_q + circle_p)  # all in U_t
    out = dict(t=t, u=2 * v_h, v_rim=v_h, n_scan=n_scan)
    best = v_h
    for name, arc in (("arc_q", circle_q), ("arc_p", circle_p)):
        values = [v_h] + [visibility_2d(t, z) for z in arc]
        best = max(best, max(values))
        out[name + "_max_increase"], out[name + "_decreasing"] = _profile(values)
        out[name + "_end"] = float(values[-1])
    out.update(v_boundary_max=best, c=2 * 2 * v_h / best)
    return out


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
