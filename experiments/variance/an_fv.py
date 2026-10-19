import collections
import glob
import json

import numpy as np

from bflib.paths import WORK_RES

runs = collections.defaultdict(list)
for path in glob.glob(f"{WORK_RES}/fv_*.txt"):
    parts = path.split("/")[-1].split("_")
    cfg = (int(parts[1]), int(parts[2]))
    try:
        data = np.loadtxt(path, ndmin=2)
    except Exception:
        continue
    if data.size:
        runs[cfg].append(data)

rows = []
for cfg in sorted(runs):
    data = np.vstack(runs[cfg])
    data = data[np.unique(data[:, 0], return_index=True)[1]]

    bin_sums = data[:, 1::2]
    bin_counts = data[:, 2::2]
    bin_means = bin_sums.sum(0) / np.maximum(bin_counts.sum(0), 1)
    F = (bin_sums - bin_counts * bin_means).sum(1)

    n_graphs = len(F)
    lam, n0 = cfg
    var_F = F.var(ddof=1)
    moment4 = ((F - F.mean()) ** 4).mean()
    se = np.sqrt(max(moment4 - var_F * var_F, 0) / n_graphs)

    # bootstrap s.e.
    rng = np.random.default_rng(0)
    boot_vars = [F[rng.integers(0, n_graphs, n_graphs)].var(ddof=1) for _ in range(400)]

    r = np.sqrt(lam / (np.pi * n0))
    print(
        "lam=%4d n0=%5d r=%.3f graphs=%5d  Var/(n0 lam)=%.3f ± %.3f (boot %.3f)  meanE=%.0f"
        % (
            lam,
            n0,
            r,
            n_graphs,
            var_F / (n0 * lam),
            se / (n0 * lam),
            np.std(boot_vars) / (n0 * lam),
            bin_counts.sum(1).mean(),
        )
    )
    rows.append(
        dict(
            lam=lam,
            n0=n0,
            r=r,
            graphs=n_graphs,
            ratio=var_F / (n0 * lam),
            se=np.std(boot_vars) / (n0 * lam),
        )
    )

json.dump(rows, open(f"{WORK_RES}/var_table.json", "w"), indent=1)
