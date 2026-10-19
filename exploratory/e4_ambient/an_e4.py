import glob
import json

import numpy as np

from e4sim import lens_ratio
from bflib.bfsim import f_lens
from bflib.paths import RESULTS


def load(tag):
    return np.concatenate([np.load(path) for path in glob.glob(f"{RESULTS}/e4/{tag}_w*")])


rows = []
for t in [0.75, 1.0]:
    flat = load(f"l2000_r0.5_t{t}_R1.0_axial_geo")

    for r in [0.5, 0.8]:
        for direction in ["circ", "axial"]:
            files = glob.glob(f"{RESULTS}/e4/l2000_r{r}_t{t}_R1.0_{direction}_amb_w*")
            if not files:
                continue

            ambient = load(f"l2000_r{r}_t{t}_R1.0_{direction}_amb")
            diff = ambient.mean() - flat.mean()
            se = np.sqrt(ambient.var() / len(ambient) + flat.var() / len(flat))
            pred = 3 * (lens_ratio(r, t, 1.0, direction, M=2500) - f_lens(2, t))

            rows.append(
                dict(
                    t=t,
                    r=r,
                    dir=direction,
                    n=len(ambient),
                    diff=float(diff),
                    se=float(se),
                    pred=float(pred),
                )
            )
            print(
                f"t={t} r={r} {direction:5s}: ambient-graph BF minus flat {diff:+.5f}±{se:.5f}"
                f"   exact ambient-lens prediction {pred:+.5f}  z {(diff-pred)/se:+.2f}"
            )

    print(
        "  flat check: geodesic graph on cylinder mean BF - Psi =",
        flat.mean() - (-2 + 3 * f_lens(2, t)),
    )

json.dump(rows, open(f"{RESULTS}/e4_summary.json", "w"), indent=1)
