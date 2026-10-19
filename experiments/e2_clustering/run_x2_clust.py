import glob
import os
import sys
import time
import zlib

import numpy as np

import e2lib as E
from bflib.paths import RESULTS

geo = sys.argv[1]
lam = float(sys.argv[2])
r = float(sys.argv[3])
worker = int(os.environ["W"])
time_limit = float(os.environ.get("TL", "5.5"))

tag = f"{geo}_l{int(lam)}_r{r:.4f}"
batch = len(glob.glob(f"{RESULTS}/e2/{tag}_w{worker}_b*.npy"))

start_time = time.time()
results = []
rep_idx = 0
while time.time() - start_time < time_limit:
    results.append(
        E.rep(
            geo,
            lam,
            r,
            zlib.crc32(tag.encode()) + worker * 10**7 + batch * 10**4 + rep_idx,
        )
    )
    rep_idx += 1

np.save(f"{RESULTS}/e2/{tag}_w{worker}_b{batch}.npy", np.array(results))
