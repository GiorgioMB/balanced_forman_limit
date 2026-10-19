import glob
import json

import numpy as np

from bflib.bfsim import Psi, f_lens, W_mean, W_var
from bflib.paths import RESULTS

summary = {"regime": [], "fixed": [], "knn": []}

for n in [1000, 3000, 10000]:
    for scaling, lam_val in [("2logn", 2 * np.log(n)), ("log2n", np.log(n) ** 2)]:
        tag = f"rgg_n{n}_p{round(lam_val,4):g}"
        files = glob.glob(f"{RESULTS}/e5/{tag}_w*")
        data = [np.load(path) for path in files]

        bf = np.concatenate([chunk["bf"] for chunk in data])
        t = np.concatenate([chunk["t"] for chunk in data])
        dev = np.abs(bf - Psi(2, t))
        rate = np.sqrt(np.log(n) / lam_val)

        record = dict(
            n=n,
            scaling=scaling,
            lam=float(lam_val),
            E=len(bf),
            maxdev=float(dev.max()),
            meandev=float(dev.mean()),
            rate=float(rate),
            ratio=float(dev.max() / rate),
        )
        summary["regime"].append(record)
        print(record)

data = [np.load(path) for path in glob.glob(f"{RESULTS}/e5/fixedlam_w*")]
for lam in ["2", "5", "10", "20", "50", "200"]:
    samples = np.concatenate([chunk[lam] for chunk in data])
    f = float(f_lens(2, 0.75))

    record = dict(
        lam=int(lam),
        N=len(samples),
        mean=float(samples.mean()),
        sd=float(samples.std()),
        psi=float(Psi(2, 0.75)),
        zero_frac=float(np.mean(samples == 0)),
        clt_mean=float(Psi(2, 0.75) + W_mean(f) / np.sqrt(int(lam))),
        clt_sd=float(np.sqrt(W_var(f) / int(lam))),
    )
    summary["fixed"].append(record)
    print(record)

bins = np.linspace(0, 1.6, 17)
for tag in ["knn_n20000_p20", "knn_n20000_p50", "rgg_n20000_p50"]:
    data = [np.load(path) for path in glob.glob(f"{RESULTS}/e5/{tag}_w*")]
    bf = np.concatenate([chunk["bf"] for chunk in data])
    t = np.concatenate([chunk["t"] for chunk in data])

    bin_idx = np.digitize(t, bins) - 1
    mean_by_bin = [
        float(bf[bin_idx == bin_no].mean()) if (bin_idx == bin_no).sum() > 30 else None
        for bin_no in range(16)
    ]
    frac_long = float(np.mean(t > 1))

    record = dict(
        tag=tag,
        E=len(bf),
        bins=bins.tolist(),
        mean_by_bin=mean_by_bin,
        frac_t_gt_1=frac_long,
        mean=float(bf.mean()),
    )
    summary["knn"].append(record)

    print(
        tag,
        len(bf),
        "frac t>1",
        round(frac_long, 3),
        "mean BF",
        round(bf.mean(), 4),
        [None if val is None else round(val, 3) for val in mean_by_bin],
    )

print(
    "Psi at bin centers",
    [
        round(float(Psi(2, centre)), 3) if centre <= 1 else None
        for centre in (bins[:-1] + bins[1:]) / 2
    ],
)

json.dump(summary, open(f"{RESULTS}/e5_summary.json", "w"), indent=1)
