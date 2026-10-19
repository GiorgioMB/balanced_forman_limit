"""Planted edges on curved spaces and flat references (r=0.5, lambda=1500). usage: run_curved.py geo t N chunk dummy"""

import sys

import numpy as np

from bflib.bfsim import flat_trial, sphere_trial, hyp_trial, s2xr_trial
from bflib.paths import RES

geo = sys.argv[1]
t = float(sys.argv[2])
n_trials = int(sys.argv[3])
chunk = int(sys.argv[4])

lam = 1500
r = 0.5

trial = {
    "S2": lambda gen: sphere_trial(gen, 2, lam, t, r),
    "S3": lambda gen: sphere_trial(gen, 3, lam, t, r),
    "H2": lambda gen: hyp_trial(gen, lam, t, r),
    "S2xR_S": lambda gen: s2xr_trial(gen, lam, t, r, "S"),
    "S2xR_R": lambda gen: s2xr_trial(gen, lam, t, r, "R"),
    "flat2": lambda gen: flat_trial(gen, 2, lam, t),
    "flat3": lambda gen: flat_trial(gen, 3, lam, t),
}[geo]

rng = np.random.default_rng(
    [
        abs(hash(geo)) % (2**31) if False else sum(map(ord, geo)),
        int(round(t * 100)),
        chunk,
        20261010,
    ]
)

samples = np.array([trial(rng)[:2] for _ in range(n_trials)])
np.save(f"{RES}/curved/c_{geo}_{t}_{n_trials}_{chunk}.npy", samples)
