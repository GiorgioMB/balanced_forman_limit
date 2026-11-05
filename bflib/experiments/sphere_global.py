"""Global averages of BF curvature on the unit sphere (Section 8.4, 'Global BF averages'; Table 4).

On the unit sphere Scal = 2, so Theorem 7.2 predicts that the edgewise-centred average is
2 vartheta_2 r^2 = r^2/12, and Corollary 7.3 that the plain average minus the flat mean at the mean
degree is 2 (vartheta_2 + eta_2) r^2 = 3 sqrt(3) r^2 / (16 pi).

Low degree (lambda = 30): every edge is centred at Psi_2(t_e) on the sphere and on a torus with the
same number of points; the difference of the two averages estimates the edgewise-centred average of
Theorem 7.2, because the torus average estimates the mean of m_lambda - Psi_2 over the flat law of T.

Larger degrees (Table 4): the edgewise centring uses m_lambda(t) estimated from flat tori with the same
expected number of points and the same lambda, pooled in 100 bins of t; the plain average is centred by
the plain average over a torus with the same expected number of points at the mean degree of each
sphere graph.
"""

import zlib

import numpy as np

from ..pipeline import Task, bfgraph_binned, run_tasks
from ..theory import eta_d, vartheta_d

NAME = "sphere_global"
TITLE = "Global BF averages on the unit sphere"
ITEMS = ["Section 8.4 (global BF averages)", "Table 4"]
REQUIRES = []

N_BINS = 100
EDGEWISE = 2 * 1 / 24  # 2 vartheta_2: average scalar curvature of the unit sphere is 2


def design(ctx):
    low = dict(lam=30, radii=[0.3, 0.2], graphs=40 if ctx.paper else 3)
    high = dict(lam=[100, 200, 400] if ctx.paper else [100], r=0.3, graphs=40 if ctx.paper else 2)
    return low, high


def _base(label):
    return zlib.crc32(label.encode()) % 10**8


def _graphs_per_task(n0):
    return 4 if n0 < 5000 else 1


def _task_list(ctx, label, space, n0, param, rmode, graphs):
    tasks, per = [], _graphs_per_task(n0)
    for first in range(0, graphs, per):
        count = min(per, graphs - first)
        tasks.append(Task(ctx.raw(NAME, f"{label}_{first}.npy"), bfgraph_binned,
                          (space, n0, param, rmode, _base(label) + first, count, N_BINS)))
    return tasks


def _load(ctx, label, n0, graphs):
    per = _graphs_per_task(n0)
    return np.vstack([np.load(ctx.raw(NAME, f"{label}_{first}.npy")) for first in range(0, graphs, per)])


def _labels(lam, r):
    return f"sphere_l{lam}_r{r}", f"torus_l{lam}_r{r}"


def run(ctx):
    low, high = design(ctx)
    tasks = []
    for r in low["radii"]:
        n0 = 4 * low["lam"] / r**2
        s, t = _labels(low["lam"], r)
        tasks += _task_list(ctx, s, "sphere2", n0, r, "nominal", low["graphs"])
        tasks += _task_list(ctx, t, "torus2", n0, low["lam"], "realised", low["graphs"])
    for lam in high["lam"]:
        n0 = 4 * lam / high["r"] ** 2
        s, t = _labels(lam, high["r"])
        tasks += _task_list(ctx, s, "sphere2", n0, high["r"], "nominal", high["graphs"])
        tasks += _task_list(ctx, t, "torus2", n0, lam, "nominal", high["graphs"])
    run_tasks(ctx, NAME + " (sphere graphs and flat references)", tasks)

    # second stage: one torus at the mean degree of each sphere graph
    tasks = []
    for lam in high["lam"]:
        n0 = 4 * lam / high["r"] ** 2
        s, _ = _labels(lam, high["r"])
        spheres = _load(ctx, s, n0, high["graphs"])
        for g, row in enumerate(spheres):
            mean_degree = 2 * row[2] / row[1]
            label = f"torus_at_mean_degree_l{lam}_g{g}"
            tasks.append(Task(ctx.raw(NAME, f"{label}.npy"), bfgraph_binned,
                              ("torus2", n0, float(mean_degree), "nominal", _base(label), 1, N_BINS)))
    run_tasks(ctx, NAME + " (tori at the mean degree)", tasks)


def _bins(data):
    """Per-graph arrays (E, count, sum BF, sum kink, sum Psi) from the binned output."""
    body = data[:, 3:].reshape(len(data), -1, 4)
    return data[:, 2], body[:, :, 0], body[:, :, 1], body[:, :, 2], body[:, :, 3]


def analyse(ctx):
    low, high = design(ctx)
    plain = 2 * (vartheta_d(2) + eta_d(2))
    out = dict(low=[], high=[])
    for r in low["radii"]:
        n0 = 4 * low["lam"] / r**2
        s_label, t_label = _labels(low["lam"], r)
        averages = []
        for label in (s_label, t_label):
            E, _, sum_bf, _, sum_psi = _bins(_load(ctx, label, n0, low["graphs"]))
            averages.append((sum_bf.sum(1) - sum_psi.sum(1)) / E)
        sphere, torus = averages
        excess = sphere.mean() - torus.mean()
        se = np.sqrt(sphere.var(ddof=1) / len(sphere) + torus.var(ddof=1) / len(torus))
        out["low"].append(dict(r=r, n0=n0, lam=low["lam"], graphs=len(sphere), excess=excess, se=se,
                               prediction=EDGEWISE * r * r, lam_r4=low["lam"] * r**4))
        ctx.note(f"- lambda = {low['lam']}, r = {r}: edgewise excess {excess:.4f} +- {se:.4f} ({excess / se:.1f} sigma), "
                 f"prediction r^2/12 = {EDGEWISE * r * r:.4f}, lambda r^4 = {low['lam'] * r**4:.2f}.")

    rng = np.random.default_rng(0)
    for lam in high["lam"]:
        r, n0, graphs = high["r"], 4 * lam / high["r"] ** 2, high["graphs"]
        s_label, t_label = _labels(lam, r)
        E_s, c_s, bf_s, _, _ = _bins(_load(ctx, s_label, n0, graphs))
        E_t, c_t, bf_t, _, _ = _bins(_load(ctx, t_label, n0, graphs))

        def edgewise(sphere_idx, torus_idx):
            m = bf_t[torus_idx].sum(0) / np.maximum(c_t[torus_idx].sum(0), 1)  # flat m_lambda(t) by bins
            return ((bf_s[sphere_idx] - c_s[sphere_idx] * m).sum(1) / E_s[sphere_idx]).mean()

        all_s, all_t = np.arange(graphs), np.arange(graphs)
        value = edgewise(all_s, all_t)
        boot = [edgewise(rng.integers(0, graphs, graphs), rng.integers(0, graphs, graphs)) for _ in range(400)]
        paired = []
        for g in range(graphs):
            E_h, _, bf_h, _, _ = _bins(np.load(ctx.raw(NAME, f"torus_at_mean_degree_l{lam}_g{g}.npy")))
            paired.append(bf_s[g].sum() / E_s[g] - bf_h.sum() / E_h[0])
        paired = np.array(paired)
        row = dict(lam=lam, n0=n0, r=r, graphs=graphs, edges_per_graph=float(E_s.mean()),
                   edgewise=value / r**2, edgewise_se=float(np.std(boot)) / r**2,
                   plain=paired.mean() / r**2, plain_se=paired.std(ddof=1) / np.sqrt(graphs) / r**2,
                   mean_degree=float((2 * E_s / _load(ctx, s_label, n0, graphs)[:, 1]).mean()))
        out["high"].append(row)
        ctx.note(f"- lambda = {lam}: r^-2 edgewise {row['edgewise']:.4f} +- {row['edgewise_se']:.4f} (limit {EDGEWISE:.4f}), "
                 f"r^-2 plain {row['plain']:.4f} +- {row['plain_se']:.4f} (limit {plain:.4f}), {graphs} graphs of "
                 f"~{row['edges_per_graph']:,.0f} edges.")
    ctx.save_summary(NAME, out)
    rows = [f"{h['lam']} & {round(h['n0'])} & ${h['edges_per_graph']:.1e}$ & {h['graphs']} & "
            f"${h['edgewise']:.4f}\\pm{h['edgewise_se']:.4f}$ & ${h['plain']:.4f}\\pm{h['plain_se']:.4f}$ \\\\"
            .replace("e+0", "\\times10^").replace("e+", "\\times10^") for h in out["high"]]
    ctx.write_table("table4", "Table 4 body (rows above the limit row)", rows)
