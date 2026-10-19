import glob
import os
import time

import numpy as np

from oq1pair import trial
from bflib.paths import RESULTS

worker = int(os.environ["W"])
time_limit = float(os.environ.get("TL", "4.5"))
data_dir = f"{RESULTS}/oq1"
thetas = [10, 30, 60, 90, 120, 150, 180]

n_done = {
    theta_deg: sum(len(np.load(path)) for path in glob.glob(f"{data_dir}/pair_th{theta_deg}_w*"))
    for theta_deg in thetas
}
todo = [theta_deg for theta_deg in thetas if n_done[theta_deg] < 8000]

if todo:
    theta_deg = todo[worker % len(todo)]
    batch = len(glob.glob(f"{data_dir}/pair_th{theta_deg}_w{worker}_b*"))
    rng = np.random.default_rng(10**6 * theta_deg + 1000 * worker + batch)

    start_time = time.time()
    results = []
    while time.time() - start_time < time_limit:
        results.append(trial(rng, 1000, 0.75, np.deg2rad(theta_deg)))

    np.save(f"{data_dir}/pair_th{theta_deg}_w{worker}_b{batch}.npy", np.array(results))
