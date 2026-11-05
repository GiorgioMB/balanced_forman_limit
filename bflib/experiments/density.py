"""Planted edges under the density varpi(x) = 1 + a cos(2 pi x_1) on T^2 (Proposition 5.8; Section S9,
Table S7).

The edge is centred at x_1 = 1/4, where |grad log varpi| is largest, and points along the gradient. Its
mean curvature is centred by the flat Palm mean at the local expected degree lambda_varpi = lambda
varpi(p), estimated by planted edges in the flat model at that degree, and compared with
-(f/2)(E|mu + sigma G| - E|sigma G|), mu = t r |d_v log varpi(p)|, of Proposition 5.8(b). The same
centring is applied to BF^sym, whose centred mean should be O(r^2 + 1/lambda).
"""

import numpy as np

from ..pipeline import Task, run_tasks
from ..planted import cosine_density, density_endpoints, density_trial, flat_trial
from ..theory import density_correction

NAME = "density"
TITLE = "Non-uniform sampling"
ITEMS = ["Section S9", "Table S7", "Proposition 5.8 (numerical observation)"]
REQUIRES = []

LAM, AMPLITUDE = 2000, 0.5
RADII = [0.01, 0.02, 0.04]
T_VALUES = [0.75, 1.0]
CHUNK = 1000


def design(ctx):
    return 20_000 if ctx.paper else 300


def local_quantities(r, t):
    """lambda_varpi = lambda varpi(p) and mu = t r |d_v log varpi(p)| at the endpoint p."""
    p, _ = density_endpoints(r, t)
    x = p[0]
    varpi = float(cosine_density(p[None], AMPLITUDE)[0])
    grad_log = 2 * np.pi * AMPLITUDE * np.sin(2 * np.pi * x) / varpi
    return LAM * varpi, t * r * abs(grad_log)


def chunk(kind, r, t, chunk_id, count):
    rng = np.random.default_rng([int(round(r * 1000)), int(round(t * 100)), chunk_id, kind, 20261012])
    if kind == 0:
        trials = [density_trial(rng, LAM, r, t, AMPLITUDE) for _ in range(count)]
    else:
        lam_local, _ = local_quantities(r, t)
        trials = [flat_trial(rng, 2, lam_local, t) for _ in range(count)]
    return np.array([(bf, bf + kink) for bf, _, kink in trials])


def _chunks(total):
    return [(c, min(CHUNK, total - c * CHUNK)) for c in range((total + CHUNK - 1) // CHUNK)]


def _path(ctx, kind, r, t, c):
    return ctx.raw(NAME, f"{'density' if kind == 0 else 'flat'}_r{r}_t{t}_{c}.npy")


def run(ctx):
    run_tasks(ctx, NAME, [Task(_path(ctx, kind, r, t, c), chunk, (kind, r, t, c, n))
                          for kind in (0, 1) for r in RADII for t in T_VALUES for c, n in _chunks(design(ctx))])


def analyse(ctx):
    rows = []
    for r in RADII:
        for t in T_VALUES:
            sim = np.vstack([np.load(_path(ctx, 0, r, t, c)) for c, _ in _chunks(design(ctx))])
            ref = np.vstack([np.load(_path(ctx, 1, r, t, c)) for c, _ in _chunks(design(ctx))])
            diff = sim.mean(0) - ref.mean(0)
            se = np.sqrt(sim.var(0, ddof=1) / len(sim) + ref.var(0, ddof=1) / len(ref))
            lam_local, mu = local_quantities(r, t)
            pred = density_correction(2, t, lam_local, mu)
            rows.append(dict(r=r, t=t, lam=LAM, edges=len(sim), lam_local=lam_local, mu=mu, bf=float(diff[0]),
                             bf_se=float(se[0]), prediction=pred, relative_deviation=float(diff[0] / pred - 1),
                             bf_sym=float(diff[1]), bf_sym_se=float(se[1])))
    ctx.save_summary(NAME, rows)
    ctx.note(f"- Setting: planted edges in R^2 with intensity n (1 + {AMPLITUDE} cos 2 pi x_1), lambda = {LAM}, "
             f"edge along the gradient at x_1 = 1/4, {design(ctx):,} edges per cell and as many flat references.")
    for row in rows:
        ctx.note(f"- r = {row['r']}, t = {row['t']}: centred mean {row['bf']:+.5f} +- {row['bf_se']:.5f}, prediction "
                 f"{row['prediction']:+.5f} (relative deviation {100 * row['relative_deviation']:+.0f}%); BF^sym "
                 f"{row['bf_sym']:+.5f} +- {row['bf_sym_se']:.5f}.")
    ctx.write_table("tableS7", "Table S7 body", [
        f"{row['r']:.2f} & {row['lam']} & {row['t']:.2f} & ${row['bf']:+.5f}\\pm{row['bf_se']:.5f}$ & "
        f"${row['prediction']:+.5f}$ & ${row['bf_sym']:+.5f}\\pm{row['bf_sym_se']:.5f}$\\\\" for row in rows])
