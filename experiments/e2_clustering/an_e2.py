import glob
import json

import numpy as np

from bflib.paths import RESULTS

exact_table = json.load(open(f"{RESULTS}/e2_exact.json"))
C2 = exact_table["C2"]
g2 = 0.017229153  # gamma_2 = 3 kappa_2(1)/4

rows = []
for geo, r, chi, area, exact_key in [
    ("bolza", 0.6, -2, 4 * np.pi, "H2_0.6"),
    ("S2", 0.6, 2, 4 * np.pi, "S2_0.6"),
    ("T2", 0.16925687506432687, 0, 1.0, None),
    ("S2", 0.45, 2, 4 * np.pi, "S2_0.45"),
    ("S2", 0.3, 2, 4 * np.pi, "S2_0.3"),
]:
    tag = f"{geo}_l60_r{r:.4f}"
    samples = np.concatenate([np.load(path) for path in glob.glob(f"{RESULTS}/e2/{tag}_w*")])
    clust = samples[:, 0]
    clust_mean, se = clust.mean(), clust.std() / np.sqrt(len(clust))

    lead = C2 + g2 * (4 * np.pi * chi / area) * r * r
    exact = exact_table[exact_key] if exact_key else C2

    scale = area / (4 * np.pi * g2 * r * r)
    chihat = (clust_mean - C2) * scale
    chise = se * scale

    rows.append(
        dict(
            geo=geo,
            r=r,
            chi=chi,
            A=area,
            reps=len(clust),
            mean=clust_mean,
            se=se,
            lead=lead,
            exact=exact,
            chihat=chihat,
            chise=chise,
            meandeg=float(samples[:, 2].mean()),
        )
    )
    print(
        f"{tag:16s} reps {len(clust)} "
        f"mean C {clust_mean:.5f}±{se:.5f}  "
        f"leading {lead:.5f}  "
        f"exact {exact:.5f}  "
        f"chi_hat {chihat:+.3f}±{chise:.3f} (chi={chi})  "
        f"r^2/A={r*r/area:.4f}"
    )

json.dump(rows, open(f"{RESULTS}/e2_summary.json", "w"), indent=1)
