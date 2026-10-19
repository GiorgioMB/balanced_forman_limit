import glob
import json

import numpy as np

from bflib.paths import RESULTS

rows = []
for lam, n0 in [(10, 2000), (20, 2000), (40, 2000), (40, 1000), (80, 1000)]:
    files = glob.glob(f"{RESULTS}/var/var_l{lam}_n{n0}_w*")
    data = np.concatenate([np.load(path) for path in files])

    bin_sums = data[:, 2:22]
    bin_counts = data[:, 22:42]
    # pooled bin means = m_lam(t)-Psi(t)
    bin_means = bin_sums.sum(0) / np.maximum(bin_counts.sum(0), 1)
    F = (bin_sums - bin_counts * bin_means).sum(1)

    n_reps = len(F)
    var_F = F.var(ddof=1)
    se = var_F * np.sqrt(2 / (n_reps - 1))

    rows.append(
        dict(
            lam=lam,
            n0=n0,
            reps=n_reps,
            varF=float(var_F),
            varF_se=float(se),
            varF_over_n=float(var_F / n0),
            varF_over_nlam=float(var_F / (n0 * lam)),
        )
    )
    print(
        f"lam={lam:3d} n0={n0}: reps {n_reps}  Var F/n = {var_F/n0:.3f} ± {se/n0:.3f}"
        f"   Var F/(n lam) = {var_F/(n0*lam):.4f}"
    )

lams = np.array([row["lam"] for row in rows if row["n0"] == 2000] + [80.0])
var_over_n = np.array(
    [row["varF_over_n"] for row in rows if row["n0"] == 2000]
    + [rows[-1]["varF_over_n"]]
)
print(
    "log-log slope of Var F/n vs lambda:",
    np.polyfit(np.log(lams), np.log(var_over_n), 1)[0],
)

json.dump(rows, open(f"{RESULTS}/var_summary.json", "w"), indent=1)
