import collections
import glob

import numpy as np

from bflib.paths import WORK_RES

runs = collections.defaultdict(list)
for path in glob.glob(f"{WORK_RES}/c_*.txt"):
    parts = path.split("/")[-1].split("_")
    cfg = (int(parts[1]), parts[2])
    try:
        data = np.loadtxt(path, ndmin=2)
    except Exception:
        continue
    if data.size:
        runs[cfg].append(data[:, 1:])

for cfg in sorted(runs, key=lambda cfg: (cfg[1], cfg[0])):
    data = np.vstack(runs[cfg])
    mean = data.mean(0)
    se = data.std(0) / np.sqrt(len(data))
    rest = data[:, 0] - data[:, 1] - data[:, 2]

    print(
        "lam=%4d eps=%s n=%6d  tot %.3f±%.3f  S %.3f±%.3f  J %.3f±%.3f (J/sqrt=%.4f)  rest %.3f±%.3f"
        % (
            cfg[0],
            cfg[1],
            len(data),
            mean[0],
            se[0],
            mean[1],
            se[1],
            mean[2],
            se[2],
            mean[2] / np.sqrt(cfg[0]),
            rest.mean(),
            rest.std() / np.sqrt(len(data)),
        )
    )
