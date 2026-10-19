import os
import time

import numpy as np

from bflib.bfsim import flat_trial
from bflib.paths import RESULTS

worker = int(os.environ["W"])

samples_by_lam = {}
rng = np.random.default_rng(777 + worker)
start_time = time.time()

for lam in [2, 5, 10, 20, 50, 200]:
    samples_by_lam[str(lam)] = np.array(
        [
            flat_trial(rng, 2, lam, 0.75)[0]
            for _ in range(6000 if lam <= 50 else 2000)
        ]
    )

np.savez(f"{RESULTS}/e5/fixedlam_w{worker}.npz", **samples_by_lam)
