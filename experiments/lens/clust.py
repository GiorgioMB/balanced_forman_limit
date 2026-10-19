"""Exact clustering C(p,r) on S^2, H^2, S^3 by quadrature (Cor. clustering check)."""

import json
import sys

import numpy as np
from scipy.integrate import quad

from bflib.bfsim import f_lens, kappa
from bflib.paths import RES
from exact import sphere_lens, sphere_ball, hyp_lens, hyp_ball

r = 0.05

results = {}
for name, d, Scal, lens, ball, density in [
    (
        "S2",
        2,
        2.0,
        lambda dist: sphere_lens(2, r, dist),
        sphere_ball(2, r),
        lambda dist: 2 * np.pi * np.sin(dist),
    ),
    (
        "H2",
        2,
        -2.0,
        lambda dist: hyp_lens(r, dist),
        hyp_ball(r),
        lambda dist: 2 * np.pi * np.sinh(dist),
    ),
    (
        "S3",
        3,
        6.0,
        lambda dist: sphere_lens(3, r, dist),
        sphere_ball(3, r),
        lambda dist: 4 * np.pi * np.sin(dist) ** 2,
    ),
]:
    C = (
        quad(
            lambda dist: density(dist) * lens(dist), 0, r, limit=200, epsabs=1e-16, epsrel=1e-11
        )[0]
        / ball**2
    )

    dev = (C - 1.5 * f_lens(d, 1.0)) / r**2
    pred = 3 * kappa(d, 1.0) / (d + 2) * Scal

    results[name] = dict(C=C, dev=float(dev), pred=float(pred))
    print(name, f"{dev:.6f}", f"{pred:.6f}")

json.dump(results, open(f"{RES}/clust_exact.json", "w"), indent=1)
