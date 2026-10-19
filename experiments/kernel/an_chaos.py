import collections
import glob

import numpy as np

from bflib.paths import WORK_RES

runs = collections.defaultdict(list)
for path in glob.glob(f"{WORK_RES}/c_*.txt"):
    parts = path.split("/")[-1].split("_")
    cfg = (int(parts[1]), float(parts[2]))
    data = np.loadtxt(path, ndmin=2)
    if data.size:
        runs[cfg].append(data[:, 1:])

for lam in sorted({cfg[0] for cfg in runs}):
    eps_values = sorted(eps for (cfg_lam, eps) in runs if cfg_lam == lam)
    if len(eps_values) < 8:
        continue

    eps_grid = np.array([0.0] + eps_values + [3.0])
    means = []
    for eps in eps_values:
        means.append(np.vstack(runs[(lam, eps)]).mean(0))
    means = np.array(means)
    means = np.vstack([means[0], means, np.zeros(3)])

    for col, name in [(0, "full"), (2, "J")]:
        integrand = means[:, col] ** 2 * eps_grid
        print(lam, name, "Var2/(n lam)=%.4f" % np.trapz(integrand, eps_grid))
