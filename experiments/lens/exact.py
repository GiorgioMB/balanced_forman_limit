import json
import sys

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma

from bflib.bfsim import omega, f_lens, kappa
from bflib.paths import RES


def _inner_sph(alpha, R, dl, k):
    if alpha <= 0:
        return 0.0

    cos_bound = (np.cos(R) - np.cos(alpha) * np.cos(dl)) / (np.sin(alpha) * np.sin(dl))
    cos_bound = min(max(cos_bound, -1.0), 1.0)

    if k == 2:
        return 2 * np.arccos(cos_bound)
    if k == 3:
        return 2 * np.pi * (1 - cos_bound)

    area_Sk2 = 2 * np.pi ** ((k - 1) / 2) / gamma((k - 1) / 2)
    return (
        area_Sk2 * quad(lambda phi: np.sin(phi) ** (k - 2), 0, np.arccos(cos_bound))[0]
    )


def sphere_lens(k, R, dl):
    breakpoints = [R - dl] if R > dl else None
    return quad(
        lambda a: np.sin(a) ** (k - 1) * _inner_sph(a, R, dl, k),
        0,
        R,
        points=breakpoints,
        limit=400,
        epsabs=1e-15,
        epsrel=1e-12,
    )[0]


def sphere_ball(k, R):
    area_Sk1 = 2 * np.pi ** (k / 2) / gamma(k / 2)
    return (
        area_Sk1
        * quad(lambda a: np.sin(a) ** (k - 1), 0, R, epsabs=1e-15, epsrel=1e-13)[0]
    )


def hyp_lens(R, dl):
    def inner(a):
        if a <= 0:
            return 0.0

        cos_bound = (np.cosh(a) * np.cosh(dl) - np.cosh(R)) / (np.sinh(a) * np.sinh(dl))
        cos_bound = min(max(cos_bound, -1.0), 1.0)
        return 2 * np.arccos(cos_bound)

    breakpoints = [R - dl] if R > dl else None
    return quad(
        lambda a: np.sinh(a) * inner(a),
        0,
        R,
        points=breakpoints,
        limit=400,
        epsabs=1e-15,
        epsrel=1e-12,
    )[0]


def hyp_ball(R):
    return 2 * np.pi * (np.cosh(R) - 1)


def s2xr_R(r, dl):
    cap_area = lambda R: 2 * np.pi * (1 - np.cos(R)) if R > 0 else 0.0

    lens_vol = quad(
        lambda s: cap_area(np.sqrt(max(r * r - max(s * s, (s - dl) ** 2), 0))),
        dl - r,
        r,
        points=[dl / 2],
        limit=400,
        epsabs=1e-15,
        epsrel=1e-12,
    )[0]

    ball_vol = quad(
        lambda s: cap_area(np.sqrt(max(r * r - s * s, 0))),
        -r,
        r,
        limit=400,
        epsabs=1e-15,
        epsrel=1e-12,
    )[0]

    return lens_vol / ball_vol


def s2xr_S(r, dl):
    s_max = np.sqrt(r * r - dl * dl / 4)

    lens_vol = (
        2
        * quad(
            lambda s: sphere_lens(2, np.sqrt(r * r - s * s), dl),
            0,
            s_max,
            limit=200,
            epsabs=1e-14,
            epsrel=1e-10,
        )[0]
    )

    ball_vol = (
        2
        * quad(
            lambda s: 2 * np.pi * (1 - np.cos(np.sqrt(r * r - s * s))),
            0,
            r,
            limit=400,
            epsabs=1e-15,
            epsrel=1e-12,
        )[0]
    )

    return lens_vol / ball_vol


geos = {
    "S2": (2, 1.5, lambda r, dl: sphere_lens(2, r, dl) / sphere_ball(2, r)),
    "H2": (2, -1.5, lambda r, dl: hyp_lens(r, dl) / hyp_ball(r)),
    "S3": (3, 3.2, lambda r, dl: sphere_lens(3, r, dl) / sphere_ball(3, r)),
    "S2xR_R": (3, 0.4, s2xr_R),
    "S2xR_S": (3, 1.4, s2xr_S),
}

if __name__ == "__main__":
    import time

    start_time = time.time()
    results = {}

    for name, (d, rho, lens_ratio) in [
        (k, v) for k, v in geos.items() if k == sys.argv[1]
    ]:
        for t in [0.25, 0.5, 0.75, 1.0]:
            row = [
                (lens_ratio(r, t * r) - f_lens(d, t)) / r**2 for r in [0.2, 0.1, 0.05]
            ]
            pred = kappa(d, t) * rho

            results[f"tab_{name}_{t}"] = [float(x) for x in row] + [float(pred)]
            print(
                f"{name:7s} t={t:4.2f}: r=.2 {row[0]:+.6f}  r=.1 {row[1]:+.6f}  r=.05 {row[2]:+.6f} | pred {pred:+.6f}",
                flush=True,
            )

        for t in [0.5, 0.75, 1.0]:
            results[f"r05_{name}_{t}"] = float(lens_ratio(0.5, 0.5 * t))

    json.dump(results, open(f"{RES}/exact_" + sys.argv[1] + ".json", "w"), indent=1)
    print("time", time.time() - start_time)
