"""Second-order kernel E D^2_{x,y} F at dist(x,y) = 0.05 r, and the second-chaos part of Var F_n from the
kernel on a grid of distances in [0, 3r] (Section S8, Table S7)."""

import zlib

import numpy as np

from ..pipeline import Task, kernel_samples, run_tasks

NAME = "kernel"
TITLE = "Second-order kernel of the sum of BF curvatures"
ITEMS = ["Section S8 (kernel)", "Table S7"]
REQUIRES = ["variance"]  # the second-chaos parts are compared with the variances of Table S5

EPS = 0.05
GRID = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]  # with EPS and the end point 3, where the kernel vanishes


def design(ctx):
    """Samples at EPS for each lambda, and per-distance samples of the grid for the two chaos lambdas."""
    if ctx.paper:
        return {20: 16000, 40: 8000, 80: 1610, 160: 389, 320: 21}, {20: 4000, 80: 1000}
    return {20: 40, 40: 20}, {20: 20}


def _per_task(lam):
    return {20: 1000, 40: 500, 80: 100, 160: 20, 320: 3}[lam]


def _tasks_of(ctx, lam, eps, samples):
    base = zlib.crc32(f"kernel_l{lam}_e{eps}".encode()) % 10**8
    per = min(_per_task(lam), samples)
    return [Task(ctx.raw(NAME, f"k_l{lam}_e{eps}_{first}.npy"), kernel_samples, (lam, eps, base + first,
                                                                                 min(per, samples - first)))
            for first in range(0, samples, per)]


def _all_tasks(ctx):
    main, chaos = design(ctx)
    tasks = {(lam, EPS): _tasks_of(ctx, lam, EPS, n) for lam, n in main.items()}
    for lam, n in chaos.items():
        for eps in GRID:
            tasks[(lam, eps)] = _tasks_of(ctx, lam, eps, n)
    return tasks


def run(ctx):
    run_tasks(ctx, NAME, [t for group in _all_tasks(ctx).values() for t in group])


def analyse(ctx):
    tasks = _all_tasks(ctx)
    data = {key: np.vstack([np.load(t.out) for t in group])[:, 1:] for key, group in tasks.items()}
    main, chaos = design(ctx)
    rows = []
    for lam in main:
        tot, S, J = data[(lam, EPS)].T
        rest = tot - S - J
        se = lambda x: float(x.std() / np.sqrt(len(x)))
        rows.append(dict(lam=lam, samples=len(tot), J=float(J.mean()), J_se=se(J), J_over_sqrt=float(J.mean() / np.sqrt(lam)),
                         S=float(S.mean()), S_se=se(S), rest=float(rest.mean()), rest_se=se(rest),
                         total=float(tot.mean()), total_se=se(tot)))
    variance = {(r["lam"], r["n0"]): r for r in ctx.load_summary("variance")}
    chaos_rows = []
    for lam in chaos:
        grid = np.array([0.0, EPS] + GRID + [3.0])
        means = np.array([data[(lam, EPS)].mean(0)] + [data[(lam, e)].mean(0) for e in [EPS] + GRID] + [np.zeros(3)])
        # the second-chaos part of Var F_n is n lambda int_0^3 k(eps)^2 eps d eps (d = 2), with k(0) taken as k(EPS)
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        part_total = float(trapezoid(means[:, 0] ** 2 * grid, grid))
        part_J = float(trapezoid(means[:, 2] ** 2 * grid, grid))
        totals = [v["F"] for (l, _), v in variance.items() if l == lam]
        chaos_rows.append(dict(lam=lam, total=part_total, J=part_J, J_over_lam=part_J / lam,
                               share=(part_total / np.mean(totals)) if totals else None))
    ctx.save_summary(NAME, dict(kernel=rows, chaos=chaos_rows))
    ratios = [r["J_over_sqrt"] for r in rows]
    ctx.note("- Max/min part / sqrt(lambda): " + ", ".join(f"{x:.4f}" for x in ratios)
             + " (paper: -0.076 to -0.093, heuristic limit about -0.0993).")
    for c in chaos_rows:
        share = f", {100 * c['share']:.0f}% of Var F_n" if c["share"] is not None else ""
        ctx.note(f"- lambda = {c['lam']}: second-chaos part {c['total']:.3f} n lambda{share}; max/min part alone "
                 f"{c['J']:.3f} n lambda = {1e4 * c['J_over_lam']:.1f}e-4 n lambda^2.")
    ctx.write_table("tableS7", "Table S7 body", [
        f"{r['lam']} & {r['samples']:,} & ${r['J']:+.3f}\\pm{r['J_se']:.3f}$ & ${r['J_over_sqrt']:+.4f}$ & "
        f"${r['S']:+.3f}\\pm{r['S_se']:.3f}$ & ${r['rest']:+.3f}\\pm{r['rest_se']:.3f}$ & "
        f"${r['total']:+.3f}\\pm{r['total_se']:.3f}$ \\\\".replace(",", "{,}") for r in rows])
