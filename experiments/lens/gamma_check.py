import sys

import numpy as np
from scipy.integrate import quad

from bflib.bfsim import omega, f_lens, kappa
from exact import sphere_lens, sphere_ball, hyp_lens, hyp_ball


def gamma_d(d):
    mean_kappa = quad(lambda t: d * t ** (d - 1) * kappa(d, t), 0, 1)[0]
    mean_f_term = quad(
        lambda t: d * t ** (d - 1) * f_lens(d, t) * (t * t / d - 1 / (d + 2)), 0, 1
    )[0]

    return (
        2 * (d + 1) / (d * (d + 2)) * mean_kappa - mean_f_term / 6,
        mean_kappa,
        mean_f_term,
    )


for d in [2, 3]:
    gamma_val, mean_kappa, mean_f_term = gamma_d(d)
    print(
        f"d={d}: gamma_d={gamma_val:.6f}  E[kappa]={mean_kappa:.6f}  E[f(T)(T^2/d-1/(d+2))]={mean_f_term:.6f}"
    )

gamma2 = gamma_d(2)[0]
gamma3 = gamma_d(3)[0]

C2 = 1.5 * f_lens(2, 1.0)
C3 = 1.5 * f_lens(3, 1.0)

for r in [0.2, 0.1, 0.05]:
    area_S2 = sphere_ball(2, r)
    C_S2 = (
        quad(
            lambda dist: sphere_lens(2, r, dist) * 2 * np.pi * np.sin(dist),
            0,
            r,
            limit=200,
            epsabs=1e-15,
            epsrel=1e-11,
        )[0]
        / area_S2**2
    )

    area_H2 = hyp_ball(r)
    C_H2 = (
        quad(
            lambda dist: hyp_lens(r, dist) * 2 * np.pi * np.sinh(dist),
            0,
            r,
            limit=200,
            epsabs=1e-15,
            epsrel=1e-11,
        )[0]
        / area_H2**2
    )

    vol_S3 = sphere_ball(3, r)
    C_S3 = (
        quad(
            lambda dist: sphere_lens(3, r, dist) * 4 * np.pi * np.sin(dist) ** 2,
            0,
            r,
            limit=200,
            epsabs=1e-15,
            epsrel=1e-11,
        )[0]
        / vol_S3**2
    )

    print(
        f"r={r}: S2 (C-C2)/r^2={(C_S2-C2)/r**2:+.6f} pred {gamma2*2:+.6f} | H2 {(C_H2-C2)/r**2:+.6f} pred {-gamma2*2:+.6f} | S3 {(C_S3-C3)/r**2:+.6f} pred {gamma3*6:+.6f}"
    )
