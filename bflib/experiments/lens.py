"""Exact lens ratios and clustering probabilities by quadrature (Table S3; the check of Corollary 5.2
in Section S4; the exact-lens column of Table 2; the exact column of Table 3)."""

from ..lens import clustering_probability, flat_clustering, lens_ratio
from ..pipeline import Task, run_tasks
from ..theory import f_lens, kappa, rho_quadratic_form

NAME = "lens"
TITLE = "Exact lens volumes and clustering probabilities"
ITEMS = ["Table S3", "Section S4 (Corollary 5.2)", "Table 2 (exact-lens column)", "Table 3 (exact column)"]
REQUIRES = []

GEOMETRIES = ["S2", "H2", "S3", "S2xR_R", "S2xR_S"]
T_TABLE = [0.25, 0.5, 0.75, 1.0]
R_TABLE = [0.2, 0.1, 0.05]


def geometry_values(geometry):
    d, rho = rho_quadratic_form(geometry)
    out = {}
    for t in T_TABLE:
        out[f"table_{t}"] = [(lens_ratio(geometry, r, t) - float(f_lens(d, t))) / r**2 for r in R_TABLE]
        out[f"pred_{t}"] = float(kappa(d, t)) * rho
    for t in (0.5, 0.75, 1.0):  # radius of the planted-edge runs of Table 2
        out[f"r0.5_{t}"] = lens_ratio(geometry, 0.5, t)
    return out


def clustering_values():
    out = {f"corollary_{s}_0.05": clustering_probability(s, 0.05) for s in ("S2", "H2", "S3")}
    # Table 3: the Bolza surface is locally isometric to H^2, and r = 0.6 is below its injectivity radius
    for surface, r in [("H2", 0.6), ("S2", 0.6), ("S2", 0.45), ("S2", 0.3)]:
        out[f"table3_{surface}_{r}"] = clustering_probability(surface, r)
    return out


def run(ctx):
    tasks = [Task(ctx.raw(NAME, f"{g}.json"), geometry_values, (g,)) for g in GEOMETRIES]
    tasks.append(Task(ctx.raw(NAME, "clustering.json"), clustering_values, ()))
    run_tasks(ctx, NAME, tasks)


def analyse(ctx):
    import json

    values = {g: json.load(open(ctx.raw(NAME, f"{g}.json"))) for g in GEOMETRIES}
    clustering = json.load(open(ctx.raw(NAME, "clustering.json")))
    ctx.save_summary(NAME, dict(geometries=values, clustering=clustering))

    labels = {"S2": ("$S^2$", r"\tfrac32"), "H2": ("$H^2$", r"-\tfrac32"), "S3": ("$S^3$", r"\tfrac{16}5"),
              "S2xR_R": (r"$S^2\!\times\!\R$, $v\in T\R$", r"\tfrac25"),
              "S2xR_S": (r"$S^2\!\times\!\R$, $v\in TS^2$", r"\tfrac75")}
    rows = []
    for g in GEOMETRIES:
        for i, t in enumerate(T_TABLE):
            name, rho = labels[g] if i == 0 else ("", "")
            cells = " & ".join(f"${x:+.6f}$" for x in values[g][f"table_{t}"] + [values[g][f"pred_{t}"]])
            rows.append(f"{name} & {'$' + rho + '$' if rho else ''} & {t:.2f} & {cells} \\\\")
        rows.append("\\addlinespace[2pt]")
    ctx.write_table("tableS3", "Table S3 body", rows[:-1])

    worst = max(abs(values[g][f"table_{t}"][-1] - values[g][f"pred_{t}"]) / abs(values[g][f"pred_{t}"])
                for g in GEOMETRIES for t in T_TABLE)
    ctx.note(f"- Lens expansion: at r = 0.05 the normalised deviation matches kappa_d(t) rho to relative error "
             f"{worst:.1e} in all 20 cases.")
    for surface, d, scal in [("S2", 2, 2.0), ("H2", 2, -2.0), ("S3", 3, 6.0)]:
        dev = (clustering[f"corollary_{surface}_0.05"] - flat_clustering(d)) / 0.05**2
        pred = 3 * float(kappa(d, 1.0)) / (d + 2) * scal
        ctx.note(f"- Corollary 5.2 on {surface}: (C(p,r) - 3f_d(1)/2)/r^2 = {dev:.6f} at r = 0.05, prediction {pred:.6f}.")
