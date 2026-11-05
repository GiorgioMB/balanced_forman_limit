"""Variance of the centred sum F_n on the flat torus, and of its max/min part F_n^K and of the
symmetrised sum F_n^sym (Section S8, Table S5; Remark 7.10)."""

import zlib

import numpy as np

from ..pipeline import Task, bfgraph_binned, run_tasks

NAME = "variance"
TITLE = "Variance of the centred sum of BF curvatures"
ITEMS = ["Section S8 (variance)", "Table S5", "Remark 7.10"]
REQUIRES = []

N_BINS = 100


def design(ctx):
    """(lambda, n0, graphs): Poisson(n0) points on T^2, radius sqrt(lambda / (pi n0))."""
    if ctx.paper:
        return [(10, 2000, 1840), (20, 2000, 1440), (40, 1000, 1600), (40, 2000, 800), (80, 4000, 360), (160, 4000, 108)]
    return [(10, 1000, 30), (20, 1000, 20)]


def _per_task(lam):
    return 40 if lam <= 40 else 10 if lam <= 80 else 4


def _tasks_of(ctx, lam, n0, graphs):
    base = zlib.crc32(f"variance_l{lam}_n{n0}".encode()) % 10**8
    per = _per_task(lam)
    return [Task(ctx.raw(NAME, f"var_l{lam}_n{n0}_{first}.npy"), bfgraph_binned,
                 ("torus2", n0, lam, "nominal", base + first, min(per, graphs - first), N_BINS))
            for first in range(0, graphs, per)]


def run(ctx):
    run_tasks(ctx, NAME, [t for cfg in design(ctx) for t in _tasks_of(ctx, *cfg)])


def analyse(ctx):
    rng = np.random.default_rng(0)
    rows = []
    for lam, n0, graphs in design(ctx):
        data = np.vstack([np.load(t.out) for t in _tasks_of(ctx, lam, n0, graphs)])
        body = data[:, 3:].reshape(len(data), N_BINS, 4)
        count, s_bf, s_kink = body[:, :, 0], body[:, :, 1], body[:, :, 2]
        # m_lambda(t) and its max/min analogue, estimated by pooling all graphs of the row by bins of t
        m_bf = s_bf.sum(0) / np.maximum(count.sum(0), 1)
        m_kink = s_kink.sum(0) / np.maximum(count.sum(0), 1)
        F = (s_bf - count * m_bf).sum(1)
        FK = (s_kink - count * m_kink).sum(1)
        Fsym = F + FK  # BF^sym = BF + kink
        row = dict(lam=lam, n0=n0, r=float(np.sqrt(lam / (np.pi * n0))), graphs=len(F),
                   mean_edges=float(data[:, 2].mean()))
        for key, values, norm in [("F", F, n0 * lam), ("FK", FK, n0 * lam**2), ("Fsym", Fsym, n0 * lam)]:
            boot = [values[rng.integers(0, len(values), len(values))].var(ddof=1) for _ in range(400)]
            row[key] = float(values.var(ddof=1) / norm)
            row[key + "_se"] = float(np.std(boot) / norm)
        rows.append(row)
    ctx.save_summary(NAME, rows)
    ctx.note("- Var F_n / (n lambda): " + ", ".join(f"{r['F']:.3f} (lambda={r['lam']}, n0={r['n0']})" for r in rows)
             + " (Remark 7.10: between 0.28 and 0.35).")
    ctx.note("- Var F_n^K / (n lambda^2): " + ", ".join(f"{r['FK']:.2e}" for r in rows)
             + " (Proposition 7.5: tends to sigma_2^2, about 6.7e-4).")
    ctx.note("- Var F_n^sym / (n lambda): " + ", ".join(f"{r['Fsym']:.3f}" for r in rows) + ".")
    ctx.note(f"- Largest radius {max(r['r'] for r in rows):.3f} (must be < 1/8).")
    ctx.write_table("tableS5", "Table S5 body (F^K column in units of 1e-4)", [
        f"{r['lam']} & {r['n0']} & {r['r']:.3f} & {r['graphs']} & ${r['F']:.3f}\\pm{r['F_se']:.3f}$ & "
        f"${1e4 * r['FK']:.2f}\\pm{1e4 * r['FK_se']:.2f}$ & ${r['Fsym']:.3f}\\pm{r['Fsym_se']:.3f}$\\\\" for r in rows])
