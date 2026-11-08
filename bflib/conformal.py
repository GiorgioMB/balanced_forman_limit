"""Geodesic balls and lenses of a conformal metric g = exp(2 phi) (dx_1^2 + dx_2^2) near the origin of R^2
(Section S15).

Geodesics are shot with the classical Runge-Kutta method, together with their derivative in the initial
angle (a Jacobi field), so that the circle of radius r about c is the curve theta -> exp_c(r u(theta)) with a
known tangent. Areas use Green's formula with a radial potential: for any centre O,

    Vol_g(D) = oint_{dD} Phi(x) (x - O) x dx,    Phi(x) = int_0^1 exp(2 phi(O + s (x - O))) s ds,

since div(Phi(x) (x - O)) = exp(2 phi(x)). The boundary of a lens consists of two arcs of circles, whose end
points (the corners of the lens) are found by Newton's method; the integrals over the arcs and over s use
Gauss-Legendre quadrature, and those over full circles the trapezoidal rule. For phi = 0 and for the round
sphere phi = log(2 / (1 + |x|^2)) the results agree with the closed forms to about 1e-14 (bflib.selftest).
"""

import numpy as np
from numpy.polynomial.legendre import leggauss


class ConformalMetric:
    """phi with its gradient and Hessian; the Gaussian curvature is K = -exp(-2 phi) Laplacian(phi)."""

    def __init__(self, phi, grad, hess, name=""):
        self.phi, self.grad, self.hess, self.name = phi, grad, hess, name

    def curvature(self, x):
        x = np.asarray(x, float)
        return -np.exp(-2 * self.phi(x)) * np.trace(self.hess(x), axis1=-2, axis2=-1)

    def curvature_gradient(self, x, h=1e-4):
        """Gradient of K by central differences of order 4 (for the predictions only)."""
        x = np.asarray(x, float)
        out = np.zeros(2)
        for k in range(2):
            e = np.zeros(2)
            e[k] = h
            out[k] = (8 * (self.curvature(x + e) - self.curvature(x - e))
                      - (self.curvature(x + 2 * e) - self.curvature(x - 2 * e))) / (12 * h)
        return out


def _diag(values):
    """Stack of 2 x 2 diagonal matrices from an array (..., 2) of diagonal entries."""
    out = np.zeros(values.shape[:-1] + (2, 2))
    out[..., 0, 0], out[..., 1, 1] = values[..., 0], values[..., 1]
    return out


FLAT = ConformalMetric(lambda x: np.zeros(np.shape(x)[:-1]), lambda x: np.zeros(np.shape(x)),
                       lambda x: np.zeros(np.shape(x)[:-1] + (2, 2)), "flat")


def _sphere_phi(x):
    return np.log(2.0) - np.log1p((np.asarray(x) ** 2).sum(-1))


def _sphere_grad(x):
    x = np.asarray(x, float)
    return -2 * x / (1 + (x**2).sum(-1))[..., None]


def _sphere_hess(x):
    x = np.asarray(x, float)
    q = 1 + (x**2).sum(-1)
    return -2 * np.eye(2) / q[..., None, None] + 4 * x[..., :, None] * x[..., None, :] / (q**2)[..., None, None]


ROUND_SPHERE = ConformalMetric(_sphere_phi, _sphere_grad, _sphere_hess, "unit sphere (stereographic)")


def _test_phi(x):
    x = np.asarray(x, float)
    return -(x**2).sum(-1) / 4 + x[..., 0] ** 3 / 6


def _test_grad(x):
    x = np.asarray(x, float)
    return np.stack([-x[..., 0] / 2 + x[..., 0] ** 2 / 2, -x[..., 1] / 2], -1)


def _test_hess(x):
    x = np.asarray(x, float)
    return _diag(np.stack([-0.5 + x[..., 0], np.full(x.shape[:-1], -0.5)], -1))


# K = exp(-2 phi) (1 - x_1): K(0) = 1, grad K(0) = (-1, 0); symmetric under x_2 -> -x_2, not locally symmetric
ASYMMETRIC = ConformalMetric(_test_phi, _test_grad, _test_hess, "phi = -|x|^2/4 + x_1^3/6")


# ------------------------------------------------------------------------------- geodesics
def _rhs(metric, state):
    """Geodesic equations x' = v, v' = -2 (grad phi . v) v + |v|^2 grad phi, and their linearisation
    (J, W) = (dx, dv) in the initial angle. state has shape (m, 8): x, v, J, W."""
    x, v, J, W = state[:, 0:2], state[:, 2:4], state[:, 4:6], state[:, 6:8]
    g = metric.grad(x)
    H = metric.hess(x)
    gv, vv = (g * v).sum(1), (v * v).sum(1)
    acc = -2 * gv[:, None] * v + vv[:, None] * g
    HJ = np.einsum("mij,mj->mi", H, J)
    gW, vW = (g * W).sum(1), (v * W).sum(1)
    dacc = (-2 * (HJ * v).sum(1)[:, None] * v + vv[:, None] * HJ
            - 2 * gW[:, None] * v - 2 * gv[:, None] * W + 2 * vW[:, None] * g)
    return np.concatenate([v, acc, W, dacc], axis=1)


def shoot(metric, centre, angles, length, steps=400):
    """End points exp_c(length u(theta)) and their derivatives in theta, for the unit initial directions of
    coordinate angle theta (angles are preserved by a conformal metric). Returns arrays (m, 2), (m, 2)."""
    centre = np.asarray(centre, float)
    angles = np.atleast_1d(np.asarray(angles, float))
    speed = np.exp(-metric.phi(centre))
    u = np.column_stack([np.cos(angles), np.sin(angles)])
    du = np.column_stack([-np.sin(angles), np.cos(angles)])
    state = np.concatenate([np.tile(centre, (len(angles), 1)), speed * u, np.zeros_like(u), speed * du], axis=1)
    h = length / steps
    for _ in range(steps):
        k1 = _rhs(metric, state)
        k2 = _rhs(metric, state + 0.5 * h * k1)
        k3 = _rhs(metric, state + 0.5 * h * k2)
        k4 = _rhs(metric, state + h * k3)
        state = state + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return state[:, 0:2], state[:, 4:6]


def exp_map(metric, centre, angle, length, steps=400):
    return shoot(metric, centre, [angle], length, steps)[0][0]


# ------------------------------------------------------------------------------- areas
_SIGMA, _SIGMA_W = (np.asarray(a) for a in leggauss(40))
_SIGMA, _SIGMA_W = (_SIGMA + 1) / 2, _SIGMA_W / 2


def _green(metric, origin, points, tangents):
    """The integrand Phi(x) (x - O) x x' of Green's formula at boundary points x with tangents x'."""
    rel = points - origin
    seg = origin + _SIGMA[None, :, None] * rel[:, None, :]  # (m, s, 2)
    Phi = (np.exp(2 * metric.phi(seg)) * _SIGMA[None, :] * _SIGMA_W[None, :]).sum(1)
    return Phi * (rel[:, 0] * tangents[:, 1] - rel[:, 1] * tangents[:, 0])


def ball_area(metric, centre, r, nodes=256, steps=400):
    """Vol_g(B(c, r)) by the trapezoidal rule over the (periodic) circle parameter."""
    angles = 2 * np.pi * np.arange(nodes) / nodes
    points, tangents = shoot(metric, centre, angles, r, steps)
    return float(_green(metric, np.asarray(centre, float), points, tangents).mean() * 2 * np.pi)


def _arc_integral(metric, origin, centre, r, lo, hi, nodes, steps):
    x, w = leggauss(nodes)
    angles = lo + (hi - lo) * (x + 1) / 2
    points, tangents = shoot(metric, centre, angles, r, steps)
    return float((_green(metric, origin, points, tangents) * w).sum() * (hi - lo) / 2)


def lens_corner(metric, x, y, r, theta_x, theta_y, steps=400):
    """Solve exp_x(r u(theta_x)) = exp_y(r u(theta_y)) by Newton's method from the given angles; stops when
    the step has reached rounding level."""
    for _ in range(40):
        px, tx = shoot(metric, x, [theta_x], r, steps)
        py, ty = shoot(metric, y, [theta_y], r, steps)
        residual = px[0] - py[0]
        step = np.linalg.solve(np.column_stack([tx[0], -ty[0]]), residual)
        theta_x, theta_y = theta_x - step[0], theta_y - step[1]
        if np.abs(step).max() < 1e-14:
            break
    if np.abs(residual).max() > 1e-12 * r:
        raise RuntimeError("lens corner: Newton's method did not converge")
    return theta_x, theta_y


def lens_area(metric, x, y, r, nodes=96, steps=400):
    """Vol_g(B(x, r) n B(y, r)) for d_g(x, y) < 2r, small r."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    beta = np.arctan2(*(y - x)[::-1])
    half = np.arccos(min(1.0, np.linalg.norm(y - x) * np.exp(_mean_phi(metric, x, y)) / (2 * r)))
    upper = lens_corner(metric, x, y, r, beta + half, beta + np.pi - half, steps)
    lower = lens_corner(metric, x, y, r, beta - half, beta + np.pi + half, steps)
    origin = (x + y) / 2
    # counterclockwise: along the circle about x from the lower to the upper corner, then along the circle about y
    total = _arc_integral(metric, origin, x, r, lower[0], upper[0], nodes, steps)
    total += _arc_integral(metric, origin, y, r, upper[1], lower[1], nodes, steps)
    return total


def _mean_phi(metric, x, y):
    """phi at the coordinate midpoint, to convert coordinate distances for the starting values only."""
    return float(metric.phi((x + y) / 2))


def lens_ratio(metric, x, angle, t, r, **kw):
    """Vol_g(B(x, r) n B(y, r)) / Vol_g(B(x, r)) for y = exp_x(t r u(angle)); also returns y and the areas."""
    y = exp_map(metric, x, angle, t * r, kw.get("steps", 400))
    lens = lens_area(metric, x, y, r, **kw)
    ball = ball_area(metric, x, r, steps=kw.get("steps", 400))
    return lens / ball, dict(y=y, lens=lens, ball=ball)
