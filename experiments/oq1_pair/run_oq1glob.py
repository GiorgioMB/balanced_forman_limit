import glob
import os
import sys
import time
import zlib

import numpy as np

from oq1glob import rep
from bflib.paths import RESULTS

geo = sys.argv[1]
lam = float(sys.argv[2])
r = float(sys.argv[3])
worker = int(os.environ["W"])
time_limit = float(os.environ.get("TL", "1.5"))

tag = f"glob_{geo}_l{int(lam)}_r{r}"
batch = len(glob.glob(f"{RESULTS}/oq1/{tag}_w{worker}_b*"))

start_time = time.time()
results = []
rep_idx = 0
while time.time() - start_time < time_limit:
    seed = zlib.crc32(tag.encode()) + worker * 10**6 + batch * 100 + rep_idx
    results.append(rep(geo, lam, r, seed))
    rep_idx += 1

np.save(f"{RESULTS}/oq1/{tag}_w{worker}_b{batch}.npy", np.array(results))
