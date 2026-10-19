import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from bflib.bfsim import *
from bflib.paths import RES


def job(args):
    d, lam, t, n_trials, seed = args
    rng = np.random.default_rng(seed)
    samples = np.array([flat_trial(rng, d, lam, t) for _ in range(n_trials)])
    return (d, lam, t, samples)


if __name__ == "__main__":
    dim = int(sys.argv[1])
    lams = [int(lam_str) for lam_str in sys.argv[2].split(",")]

    tasks = []
    seed = 1000 * dim
    for t in [0.25, 0.5, 0.75, 1.0]:
        for lam in lams:
            n_trials = 4000 if lam <= 3200 else (2000 if lam == 6400 else 1000)
            for batch in range(4):
                seed += 1
                tasks.append((dim, lam, t, n_trials // 4, seed + lam))

    start_time = time.time()
    results = {}
    with Pool(4) as pool:
        for d, lam, t, samples in pool.imap_unordered(job, tasks, chunksize=1):
            results.setdefault(f"{d}_{lam}_{t}", []).append(samples)

    path = f'{RES}/e1_d{dim}_{sys.argv[2].replace(",","-")}.npz'
    np.savez_compressed(
        path, **{key: np.concatenate(val) for key, val in results.items()}
    )
    print("saved", path, time.time() - start_time)
