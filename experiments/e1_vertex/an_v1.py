import json

import numpy as np

from bflib.bfsim import Psi, kappa
from bflib.paths import RESULTS
from an_v1_lib import load, bias_fit, scal_hat

summary = {}
configs = [
    ("flat2_l2000_r1.0", 2, 1.0, "flat2_l2000_r1.0", 0),
    ("S2_l2000_r0.5", 2, 0.5, "flat2_l2000_r1.0", 2),
    ("H2_l2000_r0.5", 2, 0.5, "flat2_l2000_r1.0", -2),
    ("flat3_l600_r1.0", 3, 1.0, "flat3_l600_r1.0", 0),
    ("S3_l600_r0.6", 3, 0.6, "flat3_l600_r1.0", 6),
    ("S2xR_l600_r0.6", 3, 0.6, "flat3_l600_r1.0", 2),
    ("flat2_l500_r1.0", 2, 1.0, "flat2_l500_r1.0", 0),
    ("S2_l500_r0.3", 2, 0.3, "flat2_l500_r1.0", 2),
    ("S2_l500_r0.5", 2, 0.5, "flat2_l500_r1.0", 2),
    ("S2_l500_r0.7", 2, 0.7, "flat2_l500_r1.0", 2),
]
fits = {}

for name, d, r, ref, scal_true in configs:
    data = load(name)
    lam = float(name.split("_l")[1].split("_")[0])

    # cross-fit: fit on even vertices, evaluate on odd (r is arbitrary for flat; use r=0.5 scale)
    if name == ref:
        even = data["vid"] % 2 == 0
        coef_even = bias_fit(data, d, even)
        fits[ref] = bias_fit(data, d)
        data_odd = {key: val[~even] for key, val in data.items()}
        scal_est = scal_hat(data_odd, d, 0.5, coef_even)
    else:
        scal_est = scal_hat(data, d, r, fits[ref])

    record = dict(
        d=d,
        lam=lam,
        r=r,
        true=scal_true,
        nv=len(scal_est),
        mean=float(scal_est.mean()),
        se=float(scal_est.std() / np.sqrt(len(scal_est))),
        sd=float(scal_est.std()),
        rmse=float(np.sqrt(np.mean((scal_est - scal_true) ** 2))),
        lr4=lam * r**4,
    )
    summary[name] = record

    print(
        f"{name:18s} true {scal_true:+d}  "
        f"mean Scal_hat {record['mean']:+.3f} ± {record['se']:.3f}  "
        f"per-vertex sd {record['sd']:.2f}  "
        f"rmse {record['rmse']:.2f}  "
        f"lam r^4 {record['lr4']:.1f}  "
        f"nv {len(scal_est)}"
    )

# Ricci tensor recovery on S2xR and S3 (pooled weighted LS)
for name, ref, d, r in [
    ("S2xR_l600_r0.6", "flat3_l600_r1.0", 3, 0.6),
    ("S3_l600_r0.6", "flat3_l600_r1.0", 3, 0.6),
    ("S2_l2000_r0.5", "flat2_l2000_r1.0", 2, 0.5),
    ("H2_l2000_r0.5", "flat2_l2000_r1.0", 2, 0.5),
]:
    data = load(name)
    kap = kappa(d, data["t"])
    response = (
        data["bf"] - Psi(d, data["t"]) - np.polyval(fits[ref], data["t"])
    ) / (3 * kap * r * r)

    directions = data["v"]
    pairs = [(row, col) for row in range(d) for col in range(row, d)]
    design = np.column_stack(
        [(1 if row == col else 2) * directions[:, row] * directions[:, col] for row, col in pairs]
    )
    weights = kap**2
    beta = np.linalg.lstsq(
        design * np.sqrt(weights)[:, None],
        response * np.sqrt(weights),
        rcond=None,
    )[0]

    # vertex-cluster bootstrap for SE
    vertex_ids = np.unique(data["vid"])
    rng = np.random.default_rng(0)
    boot_betas = []
    rows_of_vertex = {vertex: np.where(data["vid"] == vertex)[0] for vertex in vertex_ids}
    for _ in range(200):
        sample = np.concatenate(
            [rows_of_vertex[vertex] for vertex in rng.choice(vertex_ids, len(vertex_ids))]
        )
        boot_betas.append(
            np.linalg.lstsq(
                design[sample] * np.sqrt(weights[sample])[:, None],
                response[sample] * np.sqrt(weights[sample]),
                rcond=None,
            )[0]
        )
    boot_betas = np.array(boot_betas)

    def to_matrix(coefs):
        matrix = np.zeros((d, d))
        for (row, col), val in zip(pairs, coefs):
            matrix[row, col] = matrix[col, row] = val
        return matrix

    A_hat = to_matrix(beta)
    trace = np.trace(A_hat)
    Ric = A_hat - trace / (2 * (d + 1)) * np.eye(d)
    scal = (d + 2) / (2 * (d + 1)) * trace

    A_boot = np.array([to_matrix(coefs) for coefs in boot_betas])
    Ric_boot = np.array(
        [matrix - np.trace(matrix) / (2 * (d + 1)) * np.eye(d) for matrix in A_boot]
    )
    scal_boot = (d + 2) / (2 * (d + 1)) * np.trace(A_boot, axis1=1, axis2=2)

    summary["ricci_" + name] = dict(
        A=A_hat.tolist(),
        A_se=A_boot.std(0).tolist(),
        Ric=Ric.tolist(),
        Ric_se=Ric_boot.std(0).tolist(),
        Scal=float(scal),
        Scal_se=float(scal_boot.std()),
    )

    np.set_printoptions(precision=3, suppress=True)
    print(
        name,
        "\n A_hat=\n", A_hat,
        "\n se=\n", A_boot.std(0),
        "\n Ric_hat=\n", Ric,
        "\n se\n", Ric_boot.std(0),
        "\n Scal_hat=%.3f ± %.3f" % (scal, scal_boot.std()),
    )

json.dump(summary, open(f"{RESULTS}/e1_summary.json", "w"), indent=1)
