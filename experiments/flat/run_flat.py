"""Planted edges in the flat Palm model (torus, locally). usage: run_flat.py d lam t N chunk"""

import sys

import numpy as np

from bflib.bfsim import flat_trial
from bflib.paths import RES

d = int(sys.argv[1])
lam = int(sys.argv[2])
t = float(sys.argv[3])
n_trials = int(sys.argv[4])
chunk = int(sys.argv[5])

rng = np.random.default_rng([d, lam, int(round(t * 100)), chunk, 20261009])
samples = np.array([flat_trial(rng, d, lam, t)[:2] for _ in range(n_trials)])
np.save(f"{RES}/flat/f_{d}_{lam}_{t}_{chunk}.npy", samples)
