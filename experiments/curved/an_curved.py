import glob
import json
import sys

import numpy as np

from bflib.bfsim import Psi, f_lens, kappa
from bflib.paths import RES

exact_ratios = json.load(open(f"{RES}/exact.json"))

r = 0.5
GEOMETRIES = {
    "S2": (2, 1.5),
    "H2": (2, -1.5),
    "S3": (3, 3.2),
    "S2xR_S": (3, 1.4),
    "S2xR_R": (3, 0.4),
}


def load(geo, t):
    files = sorted(glob.glob(f"{RES}/curved/c_{geo}_{t}_2000_*.npy"))
    samples = np.concatenate([np.load(path) for path in files])
    assert len(files) == 20 and samples.shape == (40000, 2)
    return samples[:, 0]


rows = []
flat = {}
for d in [2, 3]:
    for t in [0.5, 0.75, 1.0]:
        bf = load(f"flat{d}", t)
        flat[(d, t)] = (bf.mean(), bf.std(ddof=1) / np.sqrt(len(bf)))

        rows.append(
            dict(
                geo=f"flat{d}",
                t=t,
                mean=float(bf.mean()),
                se=float(flat[(d, t)][1]),
                offset=float(bf.mean() - Psi(d, t)),
                offset_z=float((bf.mean() - Psi(d, t)) / flat[(d, t)][1]),
            )
        )

for geo, (d, rho) in GEOMETRIES.items():
    for t in [0.5, 0.75, 1.0]:
        bf = load(geo, t)
        mean, se = bf.mean(), bf.std(ddof=1) / np.sqrt(len(bf))

        flat_mean, flat_se = flat[(d, t)]
        diff = mean - flat_mean
        se_diff = np.sqrt(se**2 + flat_se**2)

        exact = 3 * (exact_ratios[f"r05_{geo}_{t}"] - float(f_lens(d, t)))
        lead = 3 * float(kappa(d, t)) * rho * r * r

        rows.append(
            dict(
                geo=geo,
                t=t,
                diff=float(diff),
                se=float(se_diff),
                lead=lead,
                exact=float(exact),
                z=float((diff - exact) / se_diff),
                relerr_lead=float((diff - lead) / lead),
                zflat=float(diff / se_diff),
            )
        )

json.dump(rows, open(f"{RES}/curved_summary.json", "w"), indent=1)

for row in rows:
    print({key: (round(val, 5) if isinstance(val, float) else val) for key, val in row.items()})

curved_rows = [row for row in rows if "z" in row]
z_values = [row["z"] for row in curved_rows]
rel_errs = [abs(row["relerr_lead"]) for row in curved_rows]
print(
    "z range", min(z_values), max(z_values), "max rel", max(rel_errs), "mean rel", np.mean(rel_errs)
)
print(
    "S2xR_R zflat", [round(row["zflat"], 1) for row in curved_rows if row["geo"] == "S2xR_R"]
)
