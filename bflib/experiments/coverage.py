"""Coverage of the confidence intervals of Theorem 7.14 (Section S16, Table S9).

Poisson RGGs with Poisson(n0) points on the flat unit torus T^2, where the target is 0, and on the unit
sphere S^2 with geodesic radius r and nominal expected degree lambda = n0 r^2 / 4, where the target is
2 vartheta_2 r^2 = r^2/12 (the unit sphere does not have unit area, so the average scalar curvature 2
replaces int_M Scal). For each graph, F_n = sum_e (BF(e) - m_lambda(t_e)) and Fbar_n = F_n / |E| as in
(7.1). The centring m_lambda(t) is estimated by pooling flat tori at the same lambda in 100 bins of t;
these reference tori are independent of the graphs tested (their seeds are disjoint).

The cells of (7.3). On T^2 they are the squares of a G x G grid with r/2 < 1/G <= r/sqrt(2), so that the
centres are a maximal r/2-separated set, and N(a) holds the cells at distance at most 4r from Q_a. On S^2
they are the Voronoi cells of a Fibonacci set whose minimal distance exceeds r/2 and whose covering radius
is at most r/2, and N(a) holds the cells whose centres are within 5r of that of Q_a: since every cell lies
within r/2 of its centre, this contains every cell at distance at most 4r.

For each graph the cell sums X_a of (7.3) give sigmahat_n^2 = sum_a X_a sum_{b in N(a)} X_b, and the
cell-centred variant uses X_a - Fbar_n E_a instead, where E_a is half the number of edge ends in Q_a, so
that the centred sums add up to zero. Intervals: Fbar_n +- z sd for the oracle (sd the standard deviation
of Fbar_n over the graphs of a row), and Fbar_n +- z sigmahat_n / |E| for the studentised forms.
"""

import zlib

import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.spatial import SphericalVoronoi, cKDTree
from scipy.stats import norm

from ..pipeline import Task, bfgraph_binned, bfgraph_edgelist, run_tasks

NAME = "coverage"
TITLE = "Coverage of the confidence intervals"
ITEMS = ["Section S16", "Table S9", "Theorem 7.14 (coverage)"]
REQUIRES = ["constants"]  # sigma_2^2, for the scale of the limit law

N_BINS = 100
FIB = 4 * 3.0922 * 2.7281  # Fibonacci sets: minimal distance 3.0922 / sqrt(N), covering radius 2.7281 / sqrt(N)
LEVELS = (0.90, 0.95)


def design(ctx):
    """Rows (space, lambda, n0, graphs) and, for each lambda, the flat reference (n0, number of tori)."""
    if ctx.paper:
        rows = [("torus2", 50, 12500, 400), ("torus2", 50, 50000, 400), ("sphere2", 50, 50000, 200),
                ("sphere2", 200, 16000, 240)]
        refs = {50: (50000, 80), 200: (16000, 100)}
    else:
        rows = [("torus2", 20, 2000, 24), ("sphere2", 20, 3000, 16)]
        refs = {20: (3000, 6)}
    return rows, refs


def radius(space, lam, n0):
    return float(np.sqrt(lam / (np.pi * n0))) if space == "torus2" else float(np.sqrt(4 * lam / n0))


def param(space, lam, n0):
    """The parameter of bfgraph: lambda on T^2 and the geodesic radius on S^2."""
    return float(lam) if space == "torus2" else radius(space, lam, n0)


def _base(label):
    return zlib.crc32(label.encode()) % 10**8


def _row_label(space, lam, n0):
    return f"{space}_l{lam}_n{n0}"


def _ref_label(lam, n0):
    return f"ref_l{lam}_n{n0}"


def _per_task(space, n0):
    return 20 if n0 < 20000 and space == "torus2" else 5 if n0 > 20000 else 3


def _seed_ranges(ctx):
    rows, refs = design(ctx)
    ranges = [(("torus2", n0, float(lam)), _base(_ref_label(lam, n0)), count) for lam, (n0, count) in refs.items()]
    ranges += [((space, n0, param(space, lam, n0)), _base(_row_label(space, lam, n0)), graphs)
               for space, lam, n0, graphs in rows]
    return ranges


def check_seeds(ctx):
    """Graphs with the same space, n0 and parameter and the same seed coincide: their seeds must differ."""
    ranges = _seed_ranges(ctx)
    for a, (key_a, lo_a, n_a) in enumerate(ranges):
        for key_b, lo_b, n_b in ranges[a + 1:]:
            assert key_a != key_b or lo_a + n_a <= lo_b or lo_b + n_b <= lo_a, (key_a, key_b)


# ------------------------------------------------------------------------------- cells
def fibonacci_sphere(count):
    k = np.arange(count)
    z = 1 - (2 * k + 1) / count
    phi = np.pi * (3 - np.sqrt(5)) * k
    rho = np.sqrt(1 - z * z)
    return np.column_stack([rho * np.cos(phi), rho * np.sin(phi), z])


def _geodesic(chord):
    return 2 * np.arcsin(np.minimum(1.0, chord / 2))


def cells(space, r):
    """The cells of (7.3) for radius r: a dict with the number of cells, the function assigning points to
    cells, the 0/1 matrix of 'b in N(a)' (with a in N(a)) and the separation and covering radius of the
    centres, in units of r."""
    if space == "torus2":
        G = int(round(2 ** 0.75 / r))
        h = 1.0 / G
        reach = int(np.ceil(4 * r / h)) + 1
        assert r / 2 < h <= r / np.sqrt(2) and 2 * reach + 1 < G
        offsets = [(i, j) for i in range(-reach, reach + 1) for j in range(-reach, reach + 1)
                   if h * np.hypot(max(abs(i) - 1, 0), max(abs(j) - 1, 0)) <= 4 * r]
        a = np.arange(G * G)
        ax, ay = a // G, a % G
        rows = np.repeat(a, len(offsets))
        cols = np.concatenate([((ax + i) % G) * G + (ay + j) % G for i, j in offsets]).reshape(len(offsets), -1).T.ravel()
        adjacency = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(G * G, G * G))

        def assign(pos):
            c = np.minimum((pos * G).astype(np.int64), G - 1)
            return c[:, 0] * G + c[:, 1]

        return dict(count=G * G, assign=assign, adjacency=adjacency, separation=h / r, covering=h / np.sqrt(2) / r,
                    rule="squares, dist(Q_a, Q_b) <= 4r")
    count = int(round(FIB / r**2))
    sites = fibonacci_sphere(count)
    tree = cKDTree(sites)
    separation = float(_geodesic(tree.query(sites, k=2)[0][:, 1].min()))
    covering = float(_geodesic(tree.query(SphericalVoronoi(sites).vertices)[0].max()))
    assert separation > r / 2 and covering <= r / 2
    pairs = tree.query_pairs(2 * np.sin(5 * r / 2), output_type="ndarray")
    adjacency = csr_matrix((np.ones(2 * len(pairs)), (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
                           shape=(count, count)) + identity(count, format="csr")

    def assign(pos):
        return tree.query(pos)[1]

    return dict(count=count, assign=assign, adjacency=adjacency, separation=separation / r, covering=covering / r,
                rule="Fibonacci Voronoi cells, centres within 5r")


# ------------------------------------------------------------------------------- tasks
def _ref_tasks(ctx, lam):
    n0, tori = design(ctx)[1][lam]
    base, per = _base(_ref_label(lam, n0)), 4 if n0 > 20000 else 3
    return [Task(ctx.raw(NAME, f"{_ref_label(lam, n0)}_{first}.npy"), bfgraph_binned,
                 ("torus2", n0, lam, "nominal", base + first, min(per, tori - first), N_BINS))
            for first in range(0, tori, per)]


def flat_mean(ctx, lam):
    """m_lambda(t) in 100 bins of t, pooled over the reference tori."""
    data = np.vstack([np.load(t.out) for t in _ref_tasks(ctx, lam)])
    body = data[:, 3:].reshape(len(data), N_BINS, 4)
    count = body[:, :, 0].sum(0)
    if ctx.paper:
        assert count.min() > 0
    return body[:, :, 1].sum(0) / np.maximum(count, 1), int(count.sum()), len(data)


def reference_error(ctx, lam):
    """Standard error of the pooled centring, as it enters Fbar_n: the standard deviation over the reference
    tori of their own centred averages, divided by the square root of their number."""
    data = np.vstack([np.load(t.out) for t in _ref_tasks(ctx, lam)])
    body = data[:, 3:].reshape(len(data), N_BINS, 4)
    m = flat_mean(ctx, lam)[0]
    own = (body[:, :, 1] - body[:, :, 0] * m).sum(1) / data[:, 2]
    return float(own.std(ddof=1) / np.sqrt(len(own)))


def graph_chunk(space, lam, n0, seed0, count, m_table):
    """Cell sums X_a, edge ends 2 E_a and (seed, points, edges, F_n, sum of BF, r) for graphs seed0, ..."""
    geometry = cells(space, radius(space, lam, n0))
    X, ends, scalars = [], [], []
    for seed in range(seed0, seed0 + count):
        g = bfgraph_edgelist(space, n0, param(space, lam, n0), "nominal", seed)
        b = np.minimum((g["t"].astype(np.float64) * N_BINS).astype(np.int64), N_BINS - 1)  # as in bfgraph binned
        w = g["bf"] - m_table[b]
        cell = geometry["assign"](g["pos"])
        ci, cj = cell[g["i"]], cell[g["j"]]
        K = geometry["count"]
        X.append(0.5 * (np.bincount(ci, w, K) + np.bincount(cj, w, K)))
        ends.append((np.bincount(ci, minlength=K) + np.bincount(cj, minlength=K)).astype(np.int32))
        scalars.append((seed, len(g["pos"]), len(w), w.sum(), g["bf"].sum(), g["r"]))
    return dict(X=np.array(X), ends=np.array(ends), scalars=np.array(scalars, dtype=np.float64))


def _graph_tasks(ctx, space, lam, n0, graphs, m_table):
    base, per = _base(_row_label(space, lam, n0)), _per_task(space, n0)
    return [Task(ctx.raw(NAME, f"{_row_label(space, lam, n0)}_{first}.npz"), graph_chunk,
                 (space, lam, n0, base + first, min(per, graphs - first), m_table))
            for first in range(0, graphs, per)]


def run(ctx):
    check_seeds(ctx)
    rows, refs = design(ctx)
    run_tasks(ctx, NAME + " (flat references)", [t for lam in refs for t in _ref_tasks(ctx, lam)])
    tasks = []
    for space, lam, n0, graphs in rows:
        tasks += _graph_tasks(ctx, space, lam, n0, graphs, flat_mean(ctx, lam)[0])
    run_tasks(ctx, NAME + " (graphs)", tasks)


# ------------------------------------------------------------------------------- analysis
def _coverage(stat, half_width):
    """Fraction of graphs whose interval stat +- half_width covers 0 (an undefined width never covers)."""
    ok = np.isfinite(half_width) & (np.abs(stat) <= half_width)
    return float(ok.mean())


def analyse(ctx):
    rows, refs = design(ctx)
    sigma2 = ctx.load_summary("constants")["sigma2_d2"]["value"]
    out = dict(rows=[], references={})
    for lam in refs:
        _, edges, tori = flat_mean(ctx, lam)
        out["references"][lam] = dict(n0=refs[lam][0], tori=tori, edges=edges, error=reference_error(ctx, lam))
    for space, lam, n0, graphs in rows:
        m_table = flat_mean(ctx, lam)[0]
        parts = [np.load(t.out) for t in _graph_tasks(ctx, space, lam, n0, graphs, m_table)]
        X = np.vstack([p["X"] for p in parts])
        ends = np.vstack([p["ends"] for p in parts])
        seed, points, E, F, sum_bf, r_used = np.vstack([p["scalars"] for p in parts]).T
        r = radius(space, lam, n0)
        assert np.allclose(r_used, r)
        geometry = cells(space, r)
        A = geometry["adjacency"]
        target = 0.0 if space == "torus2" else r * r / 12
        bar = F / E
        sd = bar.std(ddof=1)
        dev = bar - target
        sig2 = np.einsum("ga,ga->g", X, (A @ X.T).T)
        Xc = X - bar[:, None] * (ends / 2)
        sig2c = np.einsum("ga,ga->g", Xc, (A @ Xc.T).T)
        varF = F.var(ddof=1)
        row = dict(space=space, lam=lam, n0=n0, graphs=len(bar), r=r, lam_over_sqrt_n0=lam / np.sqrt(n0),
                   lam_r4=lam * r**4, target=target, mean_points=float(points.mean()), mean_edges=float(E.mean()),
                   cells=geometry["count"], cell_rule=geometry["rule"], mean_neighbourhood=float(A.sum() / A.shape[0]),
                   separation_over_r=geometry["separation"], covering_over_r=geometry["covering"],
                   mean=float(bar.mean()), sd=float(sd), bias_over_sd=float(dev.mean() / sd),
                   bias_over_sd_se=float(1 / np.sqrt(len(bar))),
                   reference_error_over_sd=float(out["references"][lam]["error"] / sd),
                   var_F_over_n_lam=float(varF / (n0 * lam)), sqrt_n_sd_over_limit=float(np.sqrt(n0) * sd / (2 * np.sqrt(sigma2))),
                   sigma_hat2_over_varF=float(sig2.mean() / varF), sigma_hat2_rel_sd=float(sig2.std(ddof=1) / sig2.mean()),
                   negative=int((sig2 <= 0).sum()),
                   centred_sigma_hat2_over_varF=float(sig2c.mean() / varF),
                   centred_sigma_hat2_rel_sd=float(sig2c.std(ddof=1) / sig2c.mean()),
                   centred_negative=int((sig2c <= 0).sum()),
                   # the uncentred estimator contains the large-scale part of F_n^2, the centred one does not
                   corr_F2_sigma_hat2=float(np.corrcoef(F**2, sig2)[0, 1]),
                   corr_F2_centred_sigma_hat2=float(np.corrcoef(F**2, sig2c)[0, 1]))
        row["blocks_over_neighbourhood"] = row["cells"] / row["mean_neighbourhood"]
        for level in LEVELS:
            z = norm.ppf(0.5 + level / 2)
            key = f"{round(100 * level)}"
            row[f"oracle_{key}"] = _coverage(dev, np.full(len(dev), z * sd))
            row[f"student_{key}"] = _coverage(dev, z * np.sqrt(np.where(sig2 > 0, sig2, np.nan)) / E)
            row[f"centred_{key}"] = _coverage(dev, z * np.sqrt(np.where(sig2c > 0, sig2c, np.nan)) / E)
        row["coverage_se_95"] = float(np.sqrt(0.95 * 0.05 / len(bar)))
        out["rows"].append(row)
    ctx.save_summary(NAME, out)

    for lam, ref in out["references"].items():
        ctx.note(f"- Flat reference for lambda = {lam}: {ref['tori']} tori with n0 = {ref['n0']:,}, {ref['edges']:,} edges.")
    for row in out["rows"]:
        name = "T^2" if row["space"] == "torus2" else "S^2"
        ctx.note(f"- {name}, lambda = {row['lam']}, n0 = {row['n0']:,} (r = {row['r']:.4f}, lambda/sqrt(n0) = "
                 f"{row['lam_over_sqrt_n0']:.2f}, lambda r^4 = {row['lam_r4']:.2g}), {row['graphs']} graphs of about "
                 f"{row['mean_edges']:,.0f} edges; {row['cells']} cells ({row['cell_rule']}; separation "
                 f"{row['separation_over_r']:.3f} r, covering radius {row['covering_over_r']:.3f} r), mean neighbourhood "
                 f"{row['mean_neighbourhood']:.1f}, N/Dbar = {row['blocks_over_neighbourhood']:.1f}.")
        ctx.note(f"  target {row['target']:.3e}; mean Fbar {row['mean']:.3e}, sd {row['sd']:.3e}, bias/sd "
                 f"{row['bias_over_sd']:+.3f} +- {row['bias_over_sd_se']:.3f} (sampling) and +- "
                 f"{row['reference_error_over_sd']:.3f} (centring); Var F_n/(n0 lambda) {row['var_F_over_n_lam']:.3f}; "
                 f"sqrt(n0) sd / (2 sigma_2) = {row['sqrt_n_sd_over_limit']:.2f}.")
        ctx.note(f"  coverage at 90% / 95%: oracle {row['oracle_90']:.3f} / {row['oracle_95']:.3f}; studentised "
                 f"{row['student_90']:.3f} / {row['student_95']:.3f} (E sigmahat^2 / Var F_n = {row['sigma_hat2_over_varF']:.2f}, "
                 f"relative sd {row['sigma_hat2_rel_sd']:.2f}, {row['negative']} non-positive); cell-centred "
                 f"{row['centred_90']:.3f} / {row['centred_95']:.3f} (ratio {row['centred_sigma_hat2_over_varF']:.2f}, relative sd "
                 f"{row['centred_sigma_hat2_rel_sd']:.2f}, {row['centred_negative']} non-positive); Monte Carlo se of a "
                 f"95% coverage {row['coverage_se_95']:.3f}.")
        ctx.note(f"  correlation of F_n^2 with sigmahat^2 {row['corr_F2_sigma_hat2']:+.2f}, with the centred estimator "
                 f"{row['corr_F2_centred_sigma_hat2']:+.2f}.")
    names = {"torus2": "$\\T^2$", "sphere2": "$S^2$"}
    ctx.write_table("tableS9", "Table S9 body (90% coverages are in the report)", [
        f"{names[row['space']]} & {row['lam']} & {row['n0']:,} & {row['lam_over_sqrt_n0']:.2f} & "
        f"{row['blocks_over_neighbourhood']:.1f} & {row['graphs']} & ${row['bias_over_sd']:+.2f}$ & {row['oracle_95']:.3f} & "
        f"{row['sigma_hat2_over_varF']:.2f} & {row['student_95']:.3f} & {row['centred_sigma_hat2_over_varF']:.2f} & "
        f"{row['centred_95']:.3f} \\\\".replace(",", "{,}") for row in out["rows"]])
