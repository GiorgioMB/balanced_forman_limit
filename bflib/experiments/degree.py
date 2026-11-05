"""Logarithmic and fixed expected degree (Section S6; Figure S4(b))."""

import zlib

import numpy as np
from scipy.spatial import cKDTree

from ..graphs import bf_edges, csr_from_pairs
from ..pipeline import Task, run_tasks
from ..planted import flat_trial
from ..theory import Psi, W_mean, W_var, bias_expansion, f_lens

NAME = "degree"
TITLE = "Logarithmic and fixed expected degree"
ITEMS = ["Section S6", "Figure S4(b)"]
REQUIRES = []

FIXED_LAMBDAS = [2, 5, 10, 20, 50, 200]
EDGE_SLICE = 50_000


def log_configs(ctx):
    """(n, scaling, lambda rounded to 4 decimals as used for the radius and the seed)."""
    sizes = [1000, 3000, 10000] if ctx.paper else [1000]
    return [(n, scaling, round(value, 4)) for n in sizes
            for scaling, value in [("2logn", 2 * np.log(n)), ("log2n", np.log(n) ** 2)]]


def _graph(n, lam):
    rng = np.random.default_rng(zlib.crc32(f"rgg_n{n}_p{lam:g}".encode()))
    points = rng.random((n, 2))
    r = np.sqrt(lam / (n * np.pi))
    edges = cKDTree(points, boxsize=1.0).query_pairs(r, output_type="ndarray")
    return points, edges, r


def log_slice(n, lam, first):
    """(t, BF) for the edges first, ..., first + EDGE_SLICE - 1 of the binomial RGG on T^2."""
    points, edges, r = _graph(n, lam)
    part = edges[first : first + EDGE_SLICE]
    disp = points[part[:, 0]] - points[part[:, 1]]
    disp -= np.round(disp)
    return np.column_stack([np.linalg.norm(disp, axis=1) / r, bf_edges(csr_from_pairs(edges, n), part)[:, 0]])


def fixed_worker(worker, n_small, n_large):
    """Planted edges with t = 0.75 at fixed lambda; four independent streams 777 + worker."""
    rng = np.random.default_rng(777 + worker)
    return {str(lam): np.array([flat_trial(rng, 2, lam, 0.75)[0] for _ in range(n_small if lam <= 50 else n_large)])
            for lam in FIXED_LAMBDAS}


def run(ctx):
    tasks = []
    for n, _, lam in log_configs(ctx):
        n_edges = len(_graph(n, lam)[1])
        tasks += [Task(ctx.raw(NAME, f"rgg_n{n}_p{lam:g}_{first}.npy"), log_slice, (n, lam, first))
                  for first in range(0, n_edges, EDGE_SLICE)]
    sizes = (6000, 2000) if ctx.paper else (400, 100)
    tasks += [Task(ctx.raw(NAME, f"fixed_w{w}.npz"), fixed_worker, (w, *sizes)) for w in range(4)]
    run_tasks(ctx, NAME, tasks)


def analyse(ctx):
    summary = dict(log=[], fixed=[])
    for n, scaling, lam in log_configs(ctx):
        n_edges = len(_graph(n, lam)[1])
        t, bf = np.vstack([np.load(ctx.raw(NAME, f"rgg_n{n}_p{lam:g}_{first}.npy"))
                           for first in range(0, n_edges, EDGE_SLICE)]).T
        dev = np.abs(bf - Psi(2, t))
        exact_lam = 2 * np.log(n) if scaling == "2logn" else np.log(n) ** 2
        rate = np.sqrt(np.log(n) / exact_lam)
        summary["log"].append(dict(n=n, scaling=scaling, lam=exact_lam, E=len(bf), maxdev=float(dev.max()),
                                   meandev=float(dev.mean()), ratio=float(dev.max() / rate)))
    workers = [np.load(ctx.raw(NAME, f"fixed_w{w}.npz")) for w in range(4)]
    f = float(f_lens(2, 0.75))
    for lam in FIXED_LAMBDAS:
        bf = np.concatenate([w[str(lam)] for w in workers])
        summary["fixed"].append(dict(lam=lam, N=len(bf), mean=float(bf.mean()), sd=float(bf.std()),
                                     zero_fraction=float(np.mean(bf == 0)), psi=float(Psi(2, 0.75)),
                                     expansion=bias_expansion(2, 0.75, lam, 4.0),
                                     clt_mean=float(Psi(2, 0.75) + W_mean(f) / np.sqrt(lam)),
                                     clt_sd=float(np.sqrt(W_var(f) / lam))))
    ctx.save_summary(NAME, summary)

    two = [r for r in summary["log"] if r["scaling"] == "2logn"]
    sq = [r for r in summary["log"] if r["scaling"] == "log2n"]
    ctx.note("- lambda = 2 log n: max deviation " + ", ".join(f"{r['maxdev']:.2f}" for r in two) + "; mean deviation "
             + ", ".join(f"{r['meandev']:.2f}" for r in two) + " (paper: max above 1.17, mean from 0.27 to 0.23).")
    ctx.note("- lambda = (log n)^2: max deviation " + ", ".join(f"{r['maxdev']:.2f}" for r in sq) + ", i.e. "
             + ", ".join(f"{r['ratio']:.1f}" for r in sq) + " times sqrt(log n / lambda) (paper: 0.63, 0.58, 0.83; 1.6 to 2.5).")
    ctx.note("- Fixed lambda, t = 0.75: means " + ", ".join(f"{r['mean']:+.2f} (lambda={r['lam']})" for r in summary["fixed"])
             + f"; Psi_2(0.75) = {Psi(2, 0.75):+.2f}.")
    at = {r["lam"]: r for r in summary["fixed"]}
    ctx.note(f"- Proposition 3.3 with c_2(0.75) = 4: {at[50]['expansion']:.3f} at lambda = 50 (simulated {at[50]['mean']:.3f}), "
             f"{at[200]['expansion']:.3f} at lambda = 200 (simulated {at[200]['mean']:.3f}).")
    p_one = 2 * np.exp(-2) - np.exp(-2 * (2 - f))
    ctx.note(f"- At lambda = 2 an endpoint has degree one with probability 2e^-2 - e^(-2(2-f)) = {p_one:.2f} "
             "(Section S6: 0.22), and then BF = 0 by convention.")
