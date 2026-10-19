import glob
import json

import numpy as np

from oq1pair import theory
from bflib.paths import RESULTS

rows = []
for theta_deg in [10, 30, 60, 90, 120, 150, 180]:
    pair_bf = np.concatenate(
        [np.load(path) for path in glob.glob(f"{RESULTS}/oq1/pair_th{theta_deg}_w*")]
    )

    corr = np.corrcoef(pair_bf.T)[0, 1]
    n_samples = len(pair_bf)
    se = (1 - corr * corr) / np.sqrt(n_samples - 3)
    corr_theory, _ = theory(0.75, np.deg2rad(theta_deg), M=300000, G=300000)

    rows.append(
        dict(theta=theta_deg, n=n_samples, corr=float(corr), se=float(se), theory=corr_theory)
    )
    print(theta_deg, n_samples, "sim %.3f ± %.3f   theory %.3f" % (corr, se, corr_theory))

json.dump(rows, open(f"{RESULTS}/oq1_pair.json", "w"), indent=1)
