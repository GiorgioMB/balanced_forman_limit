import glob
import os
import sys
import time
import zlib

import numpy as np

from oq2orc import trial
from bflib.paths import RESULTS

cfg = sys.argv[1]
worker = int(os.environ["W"])
time_limit = float(os.environ.get("TL", "4.5"))

geo, lam, r, target = {
    "flat2": ("flat2", 300, 1.0, None),
    "S2": ("S2", 300, 0.6, None),
    "H2": ("H2", 300, 0.6, None),
    "flat3": ("flat3", 300, 1.0, None),
    "S2xR_R": ("S2xR", 300, 0.6, np.array([0, 0, 1.0])),
    "S2xR_S": ("S2xR", 300, 0.6, np.array([1.0, 0, 0])),
}[cfg]

batch = len(glob.glob(f"{RESULTS}/oq2/{cfg}_w{worker}_b*"))
rng = np.random.default_rng(zlib.crc32(cfg.encode()) + worker * 10**6 + batch * 1000)

start_time = time.time()
samples = []
while time.time() - start_time < time_limit:
    samples.append(trial(rng, geo, lam, r, target))

np.save(f"{RESULTS}/oq2/{cfg}_w{worker}_b{batch}.npy", np.array(samples))
