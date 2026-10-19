import sys

import numpy as np

from bflib.bfsim import *
from bflib.paths import RES

for dim in [2, 3]:
    data = np.load(f"{RES}/e1_d{dim}_100-200-400-800-1600-3200-6400-12800.npz")
    print(f"===== d={dim} =====")

    for t in [0.25, 0.5, 0.75, 1.0]:
        f = float(f_lens(dim, t))
        psi = float(Psi(dim, t))
        print(f" t={t}  Psi={psi:.4f}  E[W]={W_mean(f):.4f}  Var[W]={W_var(f):.4f}")

        for lam in [100, 200, 400, 800, 1600, 3200, 6400, 12800]:
            samples = data[f"{dim}_{lam}_{t}"]
            bf = samples[:, 0]
            sq = samples[:, 1]

            bias = bf.mean() - psi
            se = bf.std() / np.sqrt(len(bf))

            print(
                f"   lam={lam:6d} N={len(bf):5d} mean(BF-Psi)={bias:+.5f}±{se:.5f}  "
                f"sqrt(lam)*mean={np.sqrt(lam)*bias:+.4f}  "
                f"sqrt(lam)*(mean-4/lam-Sq)={np.sqrt(lam)*(bias-4/lam-sq.mean()):+.4f}  "
                f"lam*Var={lam*bf.var():.4f}  "
                f"lam*E[Sq]={lam*sq.mean():.3f}  "
                f"rms={np.sqrt(((bf-psi)**2).mean()):.4f}"
            )
