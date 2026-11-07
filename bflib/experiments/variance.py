"""Variance of the centred sum F_n on the flat torus, and of its max/min part F_n^K and of the
symmetrised sum F_n^sym (Section 8.4, 'The variance mechanism'; Section S8, Table S5; Remark 7.10), and the
shape of the law of F_n across graphs (Theorem 7.6)."""

import zlib

import numpy as np
from scipy import stats

from ..pipeline import Task, bfgraph_binned, run_tasks

NAME = "variance"
TITLE = "Variance of the centred sum of BF curvatures"
ITEMS = ["Section 8.4 (variance mechanism)", "Section S8 (variance)", "Table S5", "Remark 7.10",
         "Theorem 7.6 (law of F_n)"]
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
        # Var F_n = Var F^sym + Var F^K - 2 Cov(F^sym, F^K): the covariance is what lowers Var F_n
        row["cov_sym_K"] = float(np.cov(Fsym, FK, ddof=1)[0, 1] / (n0 * lam))
        row["corr_sym_K"] = float(np.corrcoef(Fsym, FK)[0, 1])
        # the law of F_n across graphs, against the normal approximation of Theorem 7.6
        row["lam_over_sqrt_n0"] = float(lam / np.sqrt(n0))
        row["skewness"] = float(stats.skew(F, bias=False))
        row["excess_kurtosis"] = float(stats.kurtosis(F, bias=False))
        row["shapiro_p"] = float(stats.shapiro(F).pvalue)
        rows.append(row)
    ctx.save_summary(NAME, rows)
    ctx.note("- Var F_n / (n lambda): " + ", ".join(f"{r['F']:.3f} (lambda={r['lam']}, n0={r['n0']})" for r in rows)
             + " (Sections 8.4 and S8: decreases from 0.36 to 0.24).")
    ctx.note("- Var F_n^K / (n lambda^2): " + ", ".join(f"{r['FK']:.2e}" for r in rows)
             + " (Section S8: between 6.9e-4 and 7.7e-4; Proposition 7.5: tends to sigma_2^2, about 6.75e-4).")
    ctx.note("- Var F_n^sym / (n lambda): " + ", ".join(f"{r['Fsym']:.3f}" for r in rows)
             + " (Section S8: between 0.41 and 0.44).")
    ctx.note("- Cov(F^sym, F^K) / (n lambda): " + ", ".join(f"{r['cov_sym_K']:.3f}" for r in rows) + "; correlation "
             + ", ".join(f"{r['corr_sym_K']:.2f}" for r in rows)
             + " (Section S8: from about 0.03 to 0.16, correlation 0.55 to 0.70).")
    ctx.note("- Law of F_n across graphs (Theorem 7.6): skewness " + ", ".join(f"{r['skewness']:+.2f}" for r in rows)
             + "; excess kurtosis " + ", ".join(f"{r['excess_kurtosis']:+.2f}" for r in rows)
             + "; Shapiro-Wilk p " + ", ".join(f"{r['shapiro_p']:.2f}" for r in rows) + ".")
    dense = [r for r in rows if r["lam_over_sqrt_n0"] > 1]
    if dense:
        ctx.note("- Rows with lambda > sqrt(n0), where for d = 2 also the average is consistent: " + ", ".join(
            f"lambda={r['lam']}, n0={r['n0']} (lambda/sqrt(n0) = {r['lam_over_sqrt_n0']:.2f}, p = {r['shapiro_p']:.2f})"
            for r in dense) + ".")
    ctx.note(f"- Largest radius {max(r['r'] for r in rows):.3f} (must be < 1/8).")
    ctx.write_table("tableS5", "Table S5 body (F^K column in units of 1e-4)", [
        f"{r['lam']} & {r['n0']} & {r['r']:.3f} & {r['graphs']} & ${r['F']:.3f}\\pm{r['F_se']:.3f}$ & "
        f"${1e4 * r['FK']:.2f}\\pm{1e4 * r['FK_se']:.2f}$ & ${r['Fsym']:.3f}\\pm{r['Fsym_se']:.3f}$\\\\" for r in rows])
