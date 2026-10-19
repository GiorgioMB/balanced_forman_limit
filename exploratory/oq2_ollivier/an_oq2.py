import glob
import json

import numpy as np

from bflib.bfsim import kappa
from bflib.paths import RESULTS


def load(geo):
    return np.concatenate([np.load(path) for path in glob.glob(f"{RESULTS}/oq2/{geo}_w*")])


rows = []
for geo, ref, d, ric, scal in [
    ("S2", "flat2", 2, 1, 2),
    ("H2", "flat2", 2, -1, -2),
    ("S2xR_S", "flat3", 3, 1, 2),
    ("S2xR_R", "flat3", 3, 0, 2),
]:
    curved = load(geo)
    flat = load(ref)
    r = 0.6

    rec = dict(geo=geo, n=len(curved), ric_vv=ric, scal=scal)
    for col, name in [(1, "BF"), (2, "ORC")]:
        shift = curved[:, col].mean() - flat[:, col].mean()
        se = np.sqrt(
            curved[:, col].var() / len(curved) + flat[:, col].var() / len(flat)
        )
        sd = flat[:, col].std()
        rec[name] = dict(
            shift=float(shift),
            se=float(se),
            sd_edge=float(sd),
            snr_edge=float(shift / sd),
        )
    rec["BF_pred"] = float(3 * kappa(d, 0.75) * (ric + scal / (d + 2)) * r * r)
    rec["ORC_ollivier_ref"] = float(r * r * ric / (2 * (d + 2)))
    rows.append(rec)

    print(
        f"{geo:7s} BF shift {rec['BF']['shift']:+.4f}±{rec['BF']['se']:.4f}"
        f" (pred {rec['BF_pred']:+.4f}) SNR/edge {rec['BF']['snr_edge']:+.2f}"
        f" | ORC shift {rec['ORC']['shift']:+.4f}±{rec['ORC']['se']:.4f}"
        f" (eps^2Ric/2(d+2)={rec['ORC_ollivier_ref']:+.4f})"
        f" SNR/edge {rec['ORC']['snr_edge']:+.2f}"
    )

for ref in ["flat2", "flat3"]:
    flat = load(ref)
    print(
        ref,
        "mean BF-Psi %.4f  mean ORC %.4f  sd BF %.4f sd ORC %.4f corr %.3f"
        % (
            flat[:, 1].mean(),
            flat[:, 2].mean(),
            flat[:, 1].std(),
            flat[:, 2].std(),
            np.corrcoef(flat[:, 1], flat[:, 2])[0, 1],
        ),
    )

json.dump(rows, open(f"{RESULTS}/oq2_summary.json", "w"), indent=1)
