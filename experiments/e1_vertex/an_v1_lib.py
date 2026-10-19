import glob
import json

import numpy as np

from bflib.bfsim import Psi, kappa
from bflib.paths import RESULTS

data_dir = f"{RESULTS}/e1"


def load(pref):
    files = sorted(glob.glob(f"{data_dir}/{pref}_c*.npz"))
    chunks = [np.load(f) for f in files]
    return {
        key: np.concatenate([chunk[key] for chunk in chunks])
        for key in ["t", "bf", "sq", "v", "vid"]
    }


def bias_fit(Z, d, mask=None):
    sel = np.ones(len(Z["t"]), bool) if mask is None else mask
    t_sel = Z["t"][sel]
    return np.polyfit(t_sel, Z["bf"][sel] - Psi(d, t_sel), 3)


def scal_hat(Z, d, r, coef):
    resid = Z["bf"] - Psi(d, Z["t"]) - np.polyval(coef, Z["t"])
    kap = kappa(d, Z["t"])
    scal_coef = 2 * (d + 1) / (d * (d + 2))

    vertex_ids = np.unique(Z["vid"])
    vertex_index = np.searchsorted(vertex_ids, Z["vid"])
    num = np.bincount(vertex_index, weights=resid)
    den = np.bincount(vertex_index, weights=kap)
    return num / (3 * r * r * scal_coef * den)
