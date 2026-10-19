import glob
import json

import numpy as np

from bflib.bfsim import f_lens
from bflib.paths import RESULTS


def load(tag):
    samples = np.concatenate([np.load(path) for path in glob.glob(f"{RESULTS}/e3/{tag}_w*")])
    return samples


rows = []
for t in [0.75, 1.0]:
    baseline = load(f"l2000_r0.02_t{t}_a0.0_par_fixed")

    for mode in ["fixed", "adapt"]:
        for direc in ["par", "perp"]:
            for r in [0.01, 0.02, 0.04]:
                files = glob.glob(
                    f"{RESULTS}/e3/l2000_r{r}_t{t}_a0.5_{direc}_{mode}_w*"
                )
                if not files:
                    continue

                samples = load(f"l2000_r{r}_t{t}_a0.5_{direc}_{mode}")
                diff = samples.mean() - baseline.mean()
                se = np.sqrt(
                    samples.var() / len(samples) + baseline.var() / len(baseline)
                )

                pred = (
                    -0.5 * f_lens(2, t) * t * r * np.pi
                    if (direc == "par" and mode == "fixed")
                    else 0.0
                )

                rows.append(
                    dict(
                        t=t,
                        mode=mode,
                        dir=direc,
                        r=r,
                        n=len(samples),
                        diff=float(diff),
                        se=float(se),
                        pred=float(pred),
                    )
                )

                print(
                    f"t={t} {mode:5s} {direc:4s} r={r}: diff {diff:+.5f}±{se:.5f}  first-order pred {pred:+.5f}  ratio {diff/pred if pred else float('nan'):.3f}"
                )

json.dump(rows, open(f"{RESULTS}/e3_summary.json", "w"), indent=1)
