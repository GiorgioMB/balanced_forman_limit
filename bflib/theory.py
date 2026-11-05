"""Closed-form quantities from the paper.

Every function here is a formula stated in the manuscript; the comments give the place where it
appears. Nothing in this module is random.
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import betainc, gamma
from scipy.stats import norm


def omega(d):
    """Volume of the unit ball in R^d (Section 2.1)."""
    return np.pi ** (d / 2) / gamma(d / 2 + 1)


def f_lens(d, t):
    """Normalised lens volume f_d(t) = I_{1-t^2/4}((d+1)/2, 1/2) (Lemma 2.2(a))."""
    return betainc((d + 1) / 2, 0.5, 1 - np.asarray(t, float) ** 2 / 4)


def Psi(d, t):
    """Flat limit of BF curvature, Psi_d(t) = -2 + 3 f_d(t) (Theorem 3.1)."""
    return -2 + 3 * f_lens(d, t)


def kappa(d, t):
    """Curvature profile kappa_d(t) of the lens expansion (Proposition 5.1, eq. (5.1))."""
    t = np.asarray(t, float)
    return omega(d - 1) / (6 * (d + 1) * omega(d)) * t * (1 - t**2 / 4) ** ((d + 1) / 2)


def W_mean(f):
    """E W_t = -f sqrt((1-f)/pi) (Theorem 3.2)."""
    return -f * np.sqrt((1 - f) / np.pi)


def W_var(f):
    """Var W_t = f(1-f)(9-4f) - f^2(1-f)/pi (Theorem 3.2)."""
    return f * (1 - f) * (9 - 4 * f) - f**2 * (1 - f) / np.pi


def skew_normal_pdf(w, f):
    """Density of W_t: (2/s) phi(w/s) N(alpha w/s), s^2 = f(1-f)(9-4f), alpha = -sqrt(f/(2-f))/3."""
    s = np.sqrt(f * (1 - f) * (9 - 4 * f))
    alpha = -np.sqrt(f / (2 - f)) / 3
    return 2 / s * norm.pdf(w / s) * norm.cdf(alpha * w / s)


def bias_expansion(d, t, lam, c_dt):
    """E BF = Psi - f sqrt((1-f)/pi) lam^{-1/2} + (4 - 3f + c_d(t)) / lam (Proposition 3.3)."""
    f = float(f_lens(d, t))
    return float(Psi(d, t)) + W_mean(f) / np.sqrt(lam) + (4 - 3 * f + c_dt) / lam


def mean_over_T(d, g):
    """E g(T) for T with density d t^{d-1} on [0,1], the law of the normalised edge length."""
    return quad(lambda t: d * t ** (d - 1) * g(t), 0, 1, limit=200)[0]


def a_d(d):
    """a_d = E[f_d(T) sqrt((1 - f_d(T))/pi)] (Theorem 3.6)."""
    return mean_over_T(d, lambda t: float(f_lens(d, t)) * np.sqrt((1 - float(f_lens(d, t))) / np.pi))


def vartheta_d(d):
    """vartheta_d = 6(d+1)/(d(d+2)) E kappa_d(T) (eq. (7.1))."""
    return 6 * (d + 1) / (d * (d + 2)) * mean_over_T(d, lambda t: float(kappa(d, t)))


def eta_d(d):
    """eta_d = -Cov(Psi_d(T), T^2) / (6d) (eq. (7.2))."""
    e_t2 = mean_over_T(d, lambda t: t * t)
    e_psi = mean_over_T(d, lambda t: float(Psi(d, t)))
    e_psi_t2 = mean_over_T(d, lambda t: float(Psi(d, t)) * t * t)
    return -(e_psi_t2 - e_psi * e_t2) / (6 * d)


def rho_quadratic_form(geometry):
    """rho(v) = Ric(v,v) + Scal/(d+2) for the five test geometries (Section 8.3)."""
    return {
        "S2": (2, 1.0 + 2.0 / 4),
        "H2": (2, -1.0 - 2.0 / 4),
        "S3": (3, 2.0 + 6.0 / 5),
        "S2xR_S": (3, 1.0 + 2.0 / 5),
        "S2xR_R": (3, 0.0 + 2.0 / 5),
    }[geometry]


def folded_mean_excess(mu, sigma):
    """E|mu + sigma G| - E|sigma G| for standard Gaussian G."""
    if sigma <= 0:
        return abs(mu)
    a = abs(mu) / sigma
    folded = sigma * (2 * norm.pdf(a) + a * (2 * norm.cdf(a) - 1))
    return folded - sigma * np.sqrt(2 / np.pi)


def density_correction(d, t, lam_local, mu):
    """Correction -(f/2)(E|mu + sigma G| - E|sigma G|), sigma = sqrt(2(1-f)/lam_local) (Prop. 5.8(b))."""
    f = float(f_lens(d, t))
    sigma = np.sqrt(2 * (1 - f) / lam_local)
    return -0.5 * f * folded_mean_excess(mu, sigma)


# ---------------------------------------------------------------- two edges sharing a vertex (Thm B.3)
def _disc_intersection_area(centres):
    """Area of the intersection of unit discs with the given centres, by quadrature in x."""
    lo = max(c[0] for c in centres) - 1.0
    hi = min(c[0] for c in centres) + 1.0
    if hi <= lo:
        return 0.0

    def width(x):
        bottom, top = -np.inf, np.inf
        for cx, cy in centres:
            half_chord = np.sqrt(max(0.0, 1.0 - (x - cx) ** 2))
            bottom, top = max(bottom, cy - half_chord), min(top, cy + half_chord)
        return max(0.0, top - bottom)

    kinks = {c[0] + s for c in centres for s in (-1.0, 1.0)}
    for i, (x1, y1) in enumerate(centres):
        for x2, y2 in centres[i + 1:]:
            dist = np.hypot(x2 - x1, y2 - y1)
            if 0 < dist < 2:
                half_chord = np.sqrt(1 - dist * dist / 4)
                kinks |= {(x1 + x2) / 2 + s * half_chord * (y1 - y2) / dist for s in (-1.0, 1.0)}
    breakpoints = sorted(k for k in kinks if lo < k < hi)
    return quad(width, lo, hi, points=breakpoints or None, limit=500, epsabs=1e-13, epsrel=1e-12)[0]


def pair_limit_correlation(theta_deg, tj=0.75, tk=0.75):
    """Exact limit correlation of BF on edges {x,j}, {x,k} at angle theta (d = 2; Theorem B.3).

    Each limit is a linear form in (G(L1), G(Bx), G(Bj), G(L2), G(Bk)) minus (f/2)|G(Bx) - G(B.)|;
    a linear form is uncorrelated with |any linear form| of a centred Gaussian vector, and
    Cov(|U|,|V|) = (2/pi) sU sV Upsilon(rho) (Section S7).
    """
    theta = np.deg2rad(theta_deg)
    zj, zk = (tj, 0.0), (tk * np.cos(theta), tk * np.sin(theta))
    fj, fk = float(f_lens(2, tj)), float(f_lens(2, tk))
    triple = _disc_intersection_area([(0.0, 0.0), zj, zk]) / np.pi
    fjk = float(f_lens(2, np.hypot(zj[0] - zk[0], zj[1] - zk[1])))
    cov = np.array(
        [
            [fj, fj, fj, triple, triple],
            [fj, 1.0, fj, fk, fk],
            [fj, fj, 1.0, triple, fjk],
            [triple, fk, triple, fk, fk],
            [triple, fk, fjk, fk, 1.0],
        ]
    )
    l1 = np.array([3, -1.5 * fj, -1.5 * fj, 0, 0])
    l2 = np.array([0, -1.5 * fk, 0, 3, -1.5 * fk])
    u1 = np.array([0, 1.0, -1.0, 0, 0])
    u2 = np.array([0, 1.0, 0, 0, -1.0])
    s1, s2 = np.sqrt(u1 @ cov @ u1), np.sqrt(u2 @ cov @ u2)
    rho = np.clip(u1 @ cov @ u2 / (s1 * s2), -1.0, 1.0)
    cov_abs = 2 / np.pi * s1 * s2 * upsilon(rho)
    covariance = l1 @ cov @ l2 + 0.25 * fj * fk * cov_abs
    var1 = l1 @ cov @ l1 + 0.25 * fj**2 * s1**2 * (1 - 2 / np.pi)
    var2 = l2 @ cov @ l2 + 0.25 * fk**2 * s2**2 * (1 - 2 / np.pi)
    return float(covariance / np.sqrt(var1 * var2))


def upsilon(s):
    """Upsilon(s) = sqrt(1 - s^2) + s arcsin(s) - 1 (Proposition 7.5)."""
    s = np.clip(s, -1.0, 1.0)
    return np.sqrt(1 - s * s) + s * np.arcsin(s) - 1


# ---------------------------------------------------------------- the variance constant (Prop. 7.5)
def sigma2_qmc(d, m=20, seed=0, second_chaos=False):
    """sigma_d^2 of Proposition 7.5 by scrambled Sobol quasi-Monte Carlo with 2^m points.

    Integrates f(|u|) f(|u'|) sqrt((1-f(|u|))(1-f(|u'|))) Upsilon(varsigma) over |u|,|u'| <= 1 and
    |w| <= 4 (the integrand vanishes beyond), and multiplies by 1/(4 pi omega_d^3). With
    second_chaos=True, Upsilon is replaced by its leading term varsigma^2/2 (Remark 7.10).
    Returns (estimate, standard error over 8 independent scramblings).
    """
    from scipy.stats import qmc

    def ball(points, radius):
        # map the unit cube to the ball of the given radius (radial CDF r^d, uniform direction)
        rad = radius * points[:, 0] ** (1 / d)
        gauss = norm.ppf(np.clip(points[:, 1:d + 1], 1e-12, 1 - 1e-12))
        direction = gauss / np.linalg.norm(gauss, axis=1, keepdims=True)
        return rad[:, None] * direction

    estimates = []
    for rep in range(8):
        sample = qmc.Sobol(3 * (d + 1), scramble=True, seed=seed + rep).random_base2(m)
        u = ball(sample[:, : d + 1], 1.0)
        up = ball(sample[:, d + 1 : 2 * (d + 1)], 1.0)
        w = ball(sample[:, 2 * (d + 1) :], 4.0)

        def f_of(x):
            dist = np.linalg.norm(x, axis=1)
            return np.where(dist < 2, f_lens(d, np.minimum(dist, 2)), 0.0)

        fu, fup = f_of(u), f_of(up)
        scale = np.sqrt(np.maximum((1 - fu) * (1 - fup), 1e-300))
        num = f_of(w) - f_of(w + up) - f_of(u - w) + f_of(u - w - up)
        vs = np.clip(num / (2 * scale), -1, 1)
        kernel = vs * vs / 2 if second_chaos else upsilon(vs)
        integrand = fu * fup * scale * kernel
        volume = omega(d) * omega(d) * omega(d) * 4.0**d
        estimates.append(integrand.mean() * volume / (4 * np.pi * omega(d) ** 3))
    estimates = np.array(estimates)
    return float(estimates.mean()), float(estimates.std(ddof=1) / np.sqrt(len(estimates)))


def b_constant(d, n_samples=4_000_000, seed=0):
    """b_d of the second-order kernel heuristic (Section S8), by Monte Carlo.

    b_d = omega_d^{-2} int int_{|u|<=1, |u-w|<=1<|w|} f(|u-w|) / sqrt(4 pi (1 - f(|u-w|))) du dw;
    with v = w - u both u and v range over the unit ball. Returns (estimate, standard error).
    """
    rng = np.random.default_rng(seed)

    def ball(count):
        gauss = rng.standard_normal((count, d))
        return gauss / np.linalg.norm(gauss, axis=1, keepdims=True) * rng.random(count)[:, None] ** (1 / d)

    u, v = ball(n_samples), ball(n_samples)
    f = f_lens(d, np.linalg.norm(v, axis=1))
    values = np.where(np.linalg.norm(u + v, axis=1) > 1, f / np.sqrt(4 * np.pi * (1 - f)), 0.0)
    return float(values.mean()), float(values.std() / np.sqrt(n_samples))
