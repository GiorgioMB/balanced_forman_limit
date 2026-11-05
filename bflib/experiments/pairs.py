"""Correlation of BF curvature on two edges sharing a vertex (Theorem B.3; Section S7, Table S4,
Figure S4(c))."""

import numpy as np

from ..pipeline import Task, run_tasks
from ..planted import pair_trial
from ..theory import pair_limit_correlation

NAME = "pairs"
TITLE = "Two edges sharing a vertex"
ITEMS = ["Section S7", "Table S4", "Figure S4(c)"]
REQUIRES = []

LAM, T = 1000, 0.75
ANGLES = [10, 30, 60, 90, 120, 150, 180]
PAIRS = [8506, 11691, 11536, 8705, 11122, 14681, 12476]
CHUNK = 500


def design(ctx):
    return dict(zip(ANGLES, PAIRS)) if ctx.paper else {a: 300 for a in ANGLES}


def chunk(theta_deg, chunk_id, count):
    rng = np.random.default_rng([theta_deg, chunk_id, 20261011])
    return np.array([pair_trial(rng, LAM, T, np.deg2rad(theta_deg)) for _ in range(count)])


def _chunks(total):
    return [(c, min(CHUNK, total - c * CHUNK)) for c in range((total + CHUNK - 1) // CHUNK)]


def run(ctx):
    run_tasks(ctx, NAME, [Task(ctx.raw(NAME, f"pair_{a}_{c}.npy"), chunk, (a, c, n))
                          for a, total in design(ctx).items() for c, n in _chunks(total)])


def analyse(ctx):
    rows = []
    for angle, total in design(ctx).items():
        data = np.vstack([np.load(ctx.raw(NAME, f"pair_{angle}_{c}.npy")) for c, _ in _chunks(total)])
        corr = float(np.corrcoef(data.T)[0, 1])
        se = (1 - corr * corr) / np.sqrt(len(data) - 3)
        rows.append(dict(theta=angle, pairs=len(data), corr=corr, se=se, limit=pair_limit_correlation(angle, T, T)))
    ctx.save_summary(NAME, rows)
    z = [(r["corr"] - r["limit"]) / r["se"] for r in rows]
    ctx.note("- Correlations " + ", ".join(f"{r['corr']:+.3f}" for r in rows) + f" for angles {ANGLES} degrees; "
             f"standardised differences from the exact limit in [{min(z):.1f}, {max(z):.1f}].")
    ctx.write_table("tableS4", "Table S4 body", [
        f"${r['theta']}^\\circ$ & {r['pairs']:,} & ${r['corr']:+.3f}\\pm{r['se']:.3f}$ & ${r['limit']:+.3f}$ \\\\"
        .replace(",", "{,}") for r in rows])
