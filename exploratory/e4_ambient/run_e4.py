import glob
import os
import sys
import time
import zlib

import numpy as np

from e4sim import trial
from bflib.paths import RESULTS

lam = float(sys.argv[1])
r = float(sys.argv[2])
t = float(sys.argv[3])
R = float(sys.argv[4])
direction = sys.argv[5]
ambient = sys.argv[6] == "amb"
worker = int(os.environ["W"])
time_limit = float(os.environ.get("TL", "4.5"))

tag = f"l{int(lam)}_r{r}_t{t}_R{R}_{direction}_{sys.argv[6]}"
batch = len(glob.glob(f"{RESULTS}/e4/{tag}_w{worker}_b*.npy"))
rng = np.random.default_rng(zlib.crc32(tag.encode()) + worker * 10**6 + batch * 1000)

start_time = time.time()
samples = []
while time.time() - start_time < time_limit:
    samples.append(trial(rng, lam, r, t, R, direction, ambient))

np.save(f"{RESULTS}/e4/{tag}_w{worker}_b{batch}.npy", np.array(samples))
