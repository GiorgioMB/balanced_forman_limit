"""Mean local clustering on closed surfaces and the plug-in Euler characteristic (Section 8.4, Table 3,
Figure S4(a))."""

import zlib

import numpy as np

from ..graphs import AREA, EULER, mean_local_clustering, sample_surface, surface_edges
from ..lens import flat_clustering
from ..pipeline import Task, run_tasks
from ..theory import kappa

NAME = "clustering"
TITLE = "Mean local clustering and the Euler characteristic"
ITEMS = ["Section 8.4 (mean clustering)", "Table 3", "Figure S4(a)"]
REQUIRES = ["lens"]

LAM = 60
CHUNK = 100


def design(ctx):
    """(surface, r, number of graphs). r^2/|M| = 0.0286 for the first three rows."""
    r_torus = float(np.sqrt(0.36 / (4 * np.pi)))  # same r^2/|M| as r = 0.6 on surfaces of area 4 pi
    rows = [("bolza", 0.6, 2483), ("S2", 0.6, 3229), ("T2", r_torus, 3168), ("S2", 0.45, 1803), ("S2", 0.3, 770)]
    return rows if ctx.paper else [(s, r, 30) for s, r, _ in rows]


def _tag(surface, r):
    return f"{surface}_l{LAM}_r{r:.4f}"


def chunk(surface, r, first, count):
    """Graphs first, ..., first + count - 1; graph i uses the seed crc32(tag) + i."""
    base = zlib.crc32(_tag(surface, r).encode())
    out = []
    for i in range(first, first + count):
        rng = np.random.default_rng(base + i)
        n = rng.poisson(LAM * AREA[surface] / (np.pi * r * r))
        points = sample_surface(surface, rng, n)
        out.append(mean_local_clustering(surface_edges(surface, points, r), n))
    return np.array(out)


def _chunks(total):
    return [(first, min(CHUNK, total - first)) for first in range(0, total, CHUNK)]


def run(ctx):
    run_tasks(ctx, NAME, [Task(ctx.raw(NAME, f"{_tag(s, r)}_{first}.npy"), chunk, (s, r, first, count))
                          for s, r, total in design(ctx) for first, count in _chunks(total)])


def analyse(ctx):
    exact_values = ctx.load_summary("lens")["clustering"]
    flat_value = flat_clustering(2)
    gamma2 = 3 * float(kappa(2, 1.0)) / 4  # coefficient of Scal r^2 in Corollary 5.2, d = 2
    rows = []
    for surface, r, total in design(ctx):
        data = np.concatenate([np.load(ctx.raw(NAME, f"{_tag(surface, r)}_{first}.npy")) for first, _ in _chunks(total)])
        clust = data[:, 0]
        mean, se = clust.mean(), clust.std() / np.sqrt(len(clust))
        area, chi = AREA[surface], EULER[surface]
        lead = flat_value + gamma2 * (4 * np.pi * chi / area) * r * r
        key = {"bolza": f"table3_H2_{r}", "S2": f"table3_S2_{r}"}.get(surface)
        exact = exact_values[key] if key else flat_value
        scale = area / (4 * np.pi * gamma2 * r * r)
        rows.append(dict(surface=surface, chi=chi, r=r, r2_over_area=r * r / area, graphs=len(clust), mean=mean, se=se,
                         leading=lead, exact=exact, chi_hat=(mean - flat_value) * scale, chi_se=se * scale,
                         mean_degree=float(data[:, 2].mean())))
    ctx.save_summary(NAME, rows)

    first = [row for row in rows if abs(row["r2_over_area"] - 0.0286) < 1e-3]
    ctx.note("- Plug-in Euler characteristic at r^2/|M| = 0.0286: " + ", ".join(
        f"{row['surface']}: {row['chi_hat']:+.3f} +- {row['chi_se']:.3f} (chi = {row['chi']})" for row in first) + ".")
    worst = max(abs(row["mean"] - row["exact"]) / row["se"] for row in rows)
    ctx.note(f"- Sampling means within {worst:.1f} standard errors of the exact C(p,r) (paper: 1.8).")
    names = {"bolza": "Bolza ($g=2$)", "S2": "$S^2$", "T2": "$T^2$"}
    ctx.write_table("table3", "Table 3 body", [
        f"{names[row['surface']]} & ${row['chi']:+d}$ & {row['r']:.3f} & {row['r2_over_area']:.4f} & {row['graphs']} & "
        f"{row['mean']:.5f}\\,$\\pm$\\,{row['se']:.5f} & {row['leading']:.5f} & {row['exact']:.5f} & "
        f"${row['chi_hat']:+.3f}\\pm{row['chi_se']:.3f}$ \\\\".replace("$+0$", "$0$") for row in rows])
