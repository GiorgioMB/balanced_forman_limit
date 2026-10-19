import glob
import os
import sys
import time
import zlib

import numpy as np

from e3sim import trial
from bflib.paths import RESULTS

lam = float(sys.argv[1])
r = float(sys.argv[2])
t = float(sys.argv[3])
density_amp = float(sys.argv[4])
direc = sys.argv[5]
mode = sys.argv[6]

worker = int(os.environ["W"])
time_limit = float(os.environ.get("TL", "5.5"))

tag = f"l{int(lam)}_r{r}_t{t}_a{density_amp}_{direc}_{mode}"
batch = len(glob.glob(f"{RESULTS}/e3/{tag}_w{worker}_b*.npy"))
rng = np.random.default_rng(zlib.crc32(tag.encode()) + worker * 10**6 + batch * 1000)

start_time = time.time()
results = []
while time.time() - start_time < time_limit:
    results.append(trial(rng, lam, r, t, density_amp, direc, mode))

np.save(f"{RESULTS}/e3/{tag}_w{worker}_b{batch}.npy", np.array(results))
