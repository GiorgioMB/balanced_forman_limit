"""Exact lens ratios Vol(B(p,r) n B(q,r)) / Vol(B(p,r)) and clustering probabilities C(p,r) by
adaptive quadrature, from spherical and hyperbolic trigonometry (Sections S1 and S4)."""

import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.special import gamma

from .theory import f_lens

# The tolerances requested below are at the level of rounding error; quad then reports that it cannot
# certify them, although the results agree with the closed forms and with each other to ~1e-12.
warnings.filterwarnings("ignore", category=IntegrationWarning, module=__name__)


def _sphere_inner(alpha, R, dl, k):
    """Measure of the directions at angle alpha from p whose point lies within R of q (d(p,q) = dl)."""
    if alpha <= 0:
        return 0.0
    c = (np.cos(R) - np.cos(alpha) * np.cos(dl)) / (np.sin(alpha) * np.sin(dl))
    c = min(max(c, -1.0), 1.0)
    if k == 2:
        return 2 * np.arccos(c)
    if k == 3:
        return 2 * np.pi * (1 - c)
    area = 2 * np.pi ** ((k - 1) / 2) / gamma((k - 1) / 2)
    return area * quad(lambda phi: np.sin(phi) ** (k - 2), 0, np.arccos(c))[0]


def sphere_lens(k, R, dl):
    """Volume of B(p,R) n B(q,R) on the unit sphere S^k, d(p,q) = dl."""
    points = [R - dl] if R > dl else None
    return quad(lambda a: np.sin(a) ** (k - 1) * _sphere_inner(a, R, dl, k), 0, R,
                points=points, limit=400, epsabs=1e-15, epsrel=1e-12)[0]


def sphere_ball(k, R):
    area = 2 * np.pi ** (k / 2) / gamma(k / 2)
    return area * quad(lambda a: np.sin(a) ** (k - 1), 0, R, epsabs=1e-15, epsrel=1e-13)[0]


def hyperbolic_lens(R, dl):
    """Area of B(p,R) n B(q,R) in H^2, d(p,q) = dl."""
    def inner(a):
        if a <= 0:
            return 0.0
        c = (np.cosh(a) * np.cosh(dl) - np.cosh(R)) / (np.sinh(a) * np.sinh(dl))
        return 2 * np.arccos(min(max(c, -1.0), 1.0))

    points = [R - dl] if R > dl else None
    return quad(lambda a: np.sinh(a) * inner(a), 0, R, points=points, limit=400, epsabs=1e-15, epsrel=1e-12)[0]


def hyperbolic_ball(R):
    return 2 * np.pi * (np.cosh(R) - 1)


def _cap(R):
    return 2 * np.pi * (1 - np.cos(R)) if R > 0 else 0.0


def s2xr_lens_ratio_R(r, dl):
    """S^2 x R, edge along the line: slices at height s are caps of radius sqrt(r^2 - max(s^2,(s-dl)^2))."""
    lens = quad(lambda s: _cap(np.sqrt(max(r * r - max(s * s, (s - dl) ** 2), 0))), dl - r, r,
                points=[dl / 2], limit=400, epsabs=1e-15, epsrel=1e-12)[0]
    ball = quad(lambda s: _cap(np.sqrt(max(r * r - s * s, 0))), -r, r, limit=400, epsabs=1e-15, epsrel=1e-12)[0]
    return lens / ball


def s2xr_lens_ratio_S(r, dl):
    """S^2 x R, edge along the sphere: slices at height s are spherical lenses of radius sqrt(r^2 - s^2)."""
    s_max = np.sqrt(r * r - dl * dl / 4)
    lens = 2 * quad(lambda s: sphere_lens(2, np.sqrt(r * r - s * s), dl), 0, s_max,
                    limit=200, epsabs=1e-14, epsrel=1e-10)[0]
    ball = 2 * quad(lambda s: _cap(np.sqrt(r * r - s * s)), 0, r, limit=400, epsabs=1e-15, epsrel=1e-12)[0]
    return lens / ball


LENS_RATIO = {
    "S2": lambda r, dl: sphere_lens(2, r, dl) / sphere_ball(2, r),
    "H2": lambda r, dl: hyperbolic_lens(r, dl) / hyperbolic_ball(r),
    "S3": lambda r, dl: sphere_lens(3, r, dl) / sphere_ball(3, r),
    "S2xR_R": s2xr_lens_ratio_R,
    "S2xR_S": s2xr_lens_ratio_S,
}


def lens_ratio(geometry, r, t):
    return LENS_RATIO[geometry](r, t * r)


def clustering_probability(surface, r):
    """C(p,r): probability that two independent uniform points of B(p,r) are within distance r
    (Corollary 5.2), on the unit sphere S^2, S^3 or the hyperbolic plane H^2."""
    if surface == "S2":
        lens, ball, shell = (lambda x: sphere_lens(2, r, x)), sphere_ball(2, r), (lambda x: 2 * np.pi * np.sin(x))
    elif surface == "H2":
        lens, ball, shell = (lambda x: hyperbolic_lens(r, x)), hyperbolic_ball(r), (lambda x: 2 * np.pi * np.sinh(x))
    elif surface == "S3":
        lens, ball, shell = (lambda x: sphere_lens(3, r, x)), sphere_ball(3, r), (lambda x: 4 * np.pi * np.sin(x) ** 2)
    else:
        raise ValueError(surface)
    return quad(lambda x: shell(x) * lens(x), 0, r, limit=200, epsabs=1e-16, epsrel=1e-11)[0] / ball**2


def flat_clustering(d):
    """The flat value 3 f_d(1) / 2 (Lemma 3.7)."""
    return 1.5 * float(f_lens(d, 1.0))
