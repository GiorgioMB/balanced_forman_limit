"""Global averages of BF curvature on the unit sphere (Section 8.4, 'Global BF averages'; Table 4).

On the unit sphere Scal = 2, so Theorem 7.2 predicts that the edgewise-centred average is
2 vartheta_2 r^2 = r^2/12, and Corollary 7.3 that the plain average minus the flat mean at the mean
degree is 2 (vartheta_2 + eta_2) r^2 = 3 sqrt(3) r^2 / (16 pi).

Low degree (lambda = 30): every edge is centred at Psi_2(t_e) on the sphere and on a torus with the
same number of points; the difference of the two averages estimates the edgewise-centred average of
Theorem 7.2, because the torus average estimates the mean of m_lambda - Psi_2 over the flat law of T.
The expected degree on the sphere is lambda' = 2 lambda (1 - cos r) / r^2, about lambda (1 - r^2/12), so
the sphere is compared with a torus of slightly larger degree; through the terms of order lambda^{-1/2}
and 1/lambda of m_lambda (Proposition 3.3, with c_2 = 4) this shifts the prediction r^2/12 by
E[m_lambda'(T) - m_lambda(T)] (Section 8.4). The per-edge signal-to-noise ratio compares this excess with
the noise of a single edge: the standard deviation of BF(e) - Psi_2(t_e) about its mean at the same length
(pooled within the bins of t), from the edge lists of the same sphere graphs (bfgraph, mode edgelist, with
the seeds of the binned runs).

Larger degrees (Table 4): the edgewise centring uses m_lambda(t) estimated from flat tori with the same
expected number of points and the same lambda, pooled in 100 bins of t; the plain average is centred by
the plain average over a torus with the same expected number of points at the mean degree of each
sphere graph.
"""

import zlib

import numpy as np

from ..pipeline import Task, bfgraph_binned, bfgraph_edgelist, run_tasks
from ..theory import Psi, eta_d, mean_flat_offset, vartheta_d

NAME = "sphere_global"
TITLE = "Global BF averages on the unit sphere"
ITEMS = ["Section 8.4 (global BF averages)", "Table 4"]
REQUIRES = []

N_BINS = 100
EDGEWISE = 2 * 1 / 24  # 2 vartheta_2: average scalar curvature of the unit sphere is 2
C2 = 4.0  # c_2(t) = 4 for every t (Remark 3.5, Section S2)


def sphere_degree(lam, r):
    """Expected degree on the unit sphere with geodesic radius r and n0 = 4 lambda / r^2 points."""
    return 2 * lam * (1 - np.cos(r)) / r**2


def degree_shift(lam, r):
    """E[m_lambda'(T) - m_lambda(T)] for the sphere's degree lambda', from Proposition 3.3 (Section 8.4)."""
    return mean_flat_offset(2, sphere_degree(lam, r), C2) - mean_flat_offset(2, lam, C2)


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


def edge_moments(space, n0, param, rmode, seed0, count, n_bins):
    """Per graph and bin of t: the number of edges and the sums of x and x^2, x = BF(e) - Psi_2(t_e), for the
    graphs seed0, ..., seed0 + count - 1 of bfgraph_binned (the edge-list mode rebuilds the same graphs)."""
    moments = np.zeros((count, 3, n_bins))
    for k in range(count):
        g = bfgraph_edgelist(space, n0, param, rmode, seed0 + k)
        t = g["t"].astype(np.float64)
        x = g["bf"] - Psi(2, t)
        b = np.minimum((t * n_bins).astype(np.int64), n_bins - 1)
        moments[k] = [np.bincount(b, minlength=n_bins), np.bincount(b, x, minlength=n_bins),
                      np.bincount(b, x * x, minlength=n_bins)]
    return dict(moments=moments)


def _moment_tasks(ctx, label, space, n0, param, rmode, graphs):
    tasks, per = [], _graphs_per_task(n0)
    for first in range(0, graphs, per):
        count = min(per, graphs - first)
        tasks.append(Task(ctx.raw(NAME, f"{label}_moments_{first}.npz"), edge_moments,
                          (space, n0, param, rmode, _base(label) + first, count, N_BINS)))
    return tasks


def _load_moments(ctx, label, n0, graphs):
    per = _graphs_per_task(n0)
    return np.concatenate([np.load(ctx.raw(NAME, f"{label}_moments_{first}.npz"))["moments"]
                           for first in range(0, graphs, per)])


def edge_noise(moments):
    """Pooled within-bin standard deviation of BF(e) - Psi_2(t_e) over all edges of all graphs."""
    n, s1, s2 = moments.sum(0)
    keep = n > 1
    within = (s2[keep] - s1[keep] ** 2 / n[keep]).sum() / (n[keep].sum() - keep.sum())
    return float(np.sqrt(within))


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
        tasks += _moment_tasks(ctx, s, "sphere2", n0, r, "nominal", low["graphs"])
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
        averages, degrees = [], []
        for label in (s_label, t_label):
            data = _load(ctx, label, n0, low["graphs"])
            E, _, sum_bf, _, sum_psi = _bins(data)
            averages.append((sum_bf.sum(1) - sum_psi.sum(1)) / E)
            degrees.append(float((2 * E / data[:, 1]).mean()))
        sphere, torus = averages
        excess = sphere.mean() - torus.mean()
        se = np.sqrt(sphere.var(ddof=1) / len(sphere) + torus.var(ddof=1) / len(torus))
        shifted = EDGEWISE * r * r + degree_shift(low["lam"], r)
        moments = _load_moments(ctx, s_label, n0, low["graphs"])
        E_sphere = _bins(_load(ctx, s_label, n0, low["graphs"]))[0]
        assert np.array_equal(moments[:, 0].sum(1), E_sphere), "edge lists and binned runs differ"
        noise = edge_noise(moments)
        out["low"].append(dict(r=r, n0=n0, lam=low["lam"], graphs=len(sphere), excess=excess, se=se,
                               prediction=EDGEWISE * r * r, prediction_degree_shift=shifted,
                               degree_ratio=degrees[0] / degrees[1], degree_ratio_expected=sphere_degree(1, r),
                               edge_noise=noise, snr_per_edge=excess / noise, snr_per_edge_prediction=shifted / noise,
                               lam_r4=low["lam"] * r**4))
        ctx.note(f"- lambda = {low['lam']}, r = {r}: edgewise excess {excess:.4f} +- {se:.4f} ({excess / se:.1f} sigma), "
                 f"prediction r^2/12 = {EDGEWISE * r * r:.4f}, lambda r^4 = {low['lam'] * r**4:.2f}.")
        ctx.note(f"  Mean degree sphere / torus {degrees[0] / degrees[1]:.4f} (expected 2(1 - cos r)/r^2 = "
                 f"{sphere_degree(1, r):.4f}, about 1 - r^2/12 = {1 - r * r / 12:.4f}); prediction shifted by it through "
                 f"Proposition 3.3: {shifted:.4f}, {(excess - shifted) / se:+.1f} standard errors from the excess "
                 "(Section 8.4: 0.0089 at r = 0.3 and 0.0040 at r = 0.2, within one standard error).")
        ctx.note(f"  Per edge: noise sd(BF(e) - Psi_2(t_e) | t_e) = {noise:.4f} over {int(E_sphere.sum()):,} edges; signal-to-noise "
                 f"ratio excess / noise = {excess / noise:.3f} (prediction / noise = {shifted / noise:.3f}), against "
                 f"{excess / se:.1f} for the average over all edges (Section 8.4: per-edge ratio 0.03-0.05).")

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
        row["degree_shift_over_r2"] = degree_shift(lam, r) / r**2
        out["high"].append(row)
        ctx.note(f"- lambda = {lam}: r^-2 edgewise {row['edgewise']:.4f} +- {row['edgewise_se']:.4f} (limit {EDGEWISE:.4f}), "
                 f"r^-2 plain {row['plain']:.4f} +- {row['plain_se']:.4f} (limit {plain:.4f}), {graphs} graphs of "
                 f"~{row['edges_per_graph']:,.0f} edges.")
        ctx.note(f"  Mean-degree shift of the edgewise average, centred at the nominal lambda: "
                 f"{row['degree_shift_over_r2']:.4f} r^2 (Section 8.4: about 0.004, 0.002 and 0.001 r^2 at lambda = 100, "
                 f"200, 400); plain minus edgewise {row['plain'] - row['edgewise']:.4f} against 2 eta_2 = {2 * eta_d(2):.4f} "
                 "(Section 8.4: about 0.019 at lambda = 400).")
    ctx.save_summary(NAME, out)
    rows = [f"{h['lam']} & {round(h['n0'])} & ${h['edges_per_graph']:.1e}$ & {h['graphs']} & "
            f"${h['edgewise']:.4f}\\pm{h['edgewise_se']:.4f}$ & ${h['plain']:.4f}\\pm{h['plain_se']:.4f}$ \\\\"
            .replace("e+0", "\\times10^").replace("e+", "\\times10^") for h in out["high"]]
    ctx.write_table("table4", "Table 4 body (rows above the limit row)", rows)
