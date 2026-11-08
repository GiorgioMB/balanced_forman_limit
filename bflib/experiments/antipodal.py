"""Antipodal cancellation of the curvature terms of order r^3 (Lemma 7.23; Section S15, Table S8).

The surface is R^2 near the origin with g = exp(2 phi)(dx_1^2 + dx_2^2), phi = -|x|^2/4 + x_1^3/6: its Gaussian
curvature K = exp(-2 phi)(1 - x_1) has K(0) = 1 and grad K(0) = (-1, 0), so it is not locally symmetric, and g is
symmetric under x_2 -> -x_2. For x = 0, y_v = exp_x(t r v) and the lens ratio R(v) = Vol(B(x,r) n B(y_v,r)) / Vol(B(x,r)),
computed exactly by bflib.conformal, Proposition 5.1 gives R(v) = f_2(t) + kappa_2(t) rho r^2 + O(r^3) with
rho = 3K(0)/2, and Lemma 7.23 says that the antipodal average (R(v) + R(-v))/2 equals f_2(t) + kappa_2(t) rho r^2 up to
O(r^4 + r^5/t), and that Vol(B(y_v,r)) + Vol(B(y_-v,r)) - 2 Vol(B(x,r)) = O(r^6). The terms of each order in r are
alternately even and odd in v, so all four normalised quantities below converge at rate r^2.

The heuristic for the odd part evaluates the lens volume at the curvature of the midpoint of the edge: with
Vol(L) = pi r^2 (f + (3 kappa/2 - f/12) K r^2) on a surface of constant curvature K and
Vol(B(x,r)) = pi r^2 (1 - K r^2/12), K(midpoint) - K(x) = (t r/2) d_v K gives the odd part
(3 kappa_2(t)/2 - f_2(t)/12) (t/2) d_v K r^3. It is not proved.
"""

import numpy as np

from .. import conformal
from ..pipeline import Task, run_tasks
from ..theory import f_lens, kappa

NAME = "antipodal"
TITLE = "Antipodal cancellation of the curvature terms of order r^3"
ITEMS = ["Section S15", "Table S8", "Lemma 7.23 (numerical check)"]
REQUIRES = []

DIRECTIONS = {"+e1": 0.0, "-e1": np.pi, "+e2": np.pi / 2, "-e2": -np.pi / 2}


def design(ctx):
    """Values of t, radii (halving) and discretisation: Gauss-Legendre nodes per arc and Runge-Kutta steps."""
    if ctx.paper:
        return [0.25, 0.5, 0.75, 1.0], [0.4, 0.2, 0.1, 0.05], dict(nodes=96, steps=400)
    return [0.5, 1.0], [0.4, 0.2], dict(nodes=48, steps=100)


def configuration(r, t, nodes, steps):
    """Volumes of B(x,r) and, for each direction v, of B(x,r) n B(y_v,r) and B(y_v,r), on the test surface."""
    metric, x = conformal.ASYMMETRIC, np.zeros(2)
    out = dict(r=r, t=t, ball_x=conformal.ball_area(metric, x, r, steps=steps))
    for name, angle in DIRECTIONS.items():
        y = conformal.exp_map(metric, x, angle, t * r, steps)
        out[name] = dict(y=y.tolist(), lens=conformal.lens_area(metric, x, y, r, nodes=nodes, steps=steps),
                         ball=conformal.ball_area(metric, y, r, steps=steps))
    return out


def symmetric_check(r, t):
    """On the round sphere (constant curvature, so locally symmetric) the two lens volumes coincide."""
    metric, x = conformal.ROUND_SPHERE, np.array([0.3, -0.2])
    out = {}
    for angle in (0.4, 1.7):
        lenses = [conformal.lens_area(metric, x, conformal.exp_map(metric, x, a, t * r), r) for a in (angle, angle + np.pi)]
        out[str(angle)] = lenses
    return out


def _tasks(ctx):
    t_values, radii, acc = design(ctx)
    tasks = [Task(ctx.raw(NAME, f"conf_r{r}_t{t}.json"), configuration, (r, t, acc["nodes"], acc["steps"]))
             for r in radii for t in t_values]
    # the same configuration at a finer discretisation, as a check of the quadrature and of the shooting
    tasks.append(Task(ctx.raw(NAME, "accuracy_r0.4_t1.0.json"), configuration,
                      (0.4, 1.0, int(1.5 * acc["nodes"]), 2 * acc["steps"])))
    tasks.append(Task(ctx.raw(NAME, "sphere_r0.2_t0.75.json"), symmetric_check, (0.2, 0.75)))
    return tasks


def run(ctx):
    run_tasks(ctx, NAME, _tasks(ctx))


def _load(ctx, name):
    import json

    with open(ctx.raw(NAME, name)) as fh:
        return json.load(fh)


def _richardson(values):
    """Limit of values at RADII (halving) with an error of order r^2, from the last two radii."""
    return (4 * values[-1] - values[-2]) / 3


def analyse(ctx):
    T_VALUES, RADII, _ = design(ctx)
    metric = conformal.ASYMMETRIC
    K0 = float(metric.curvature(np.zeros(2)))
    gradK = metric.curvature_gradient(np.zeros(2))
    rho = 1.5 * K0  # Ric(v,v) + Scal/(d+2) = K + 2K/4 on a surface
    rows = []
    for t in T_VALUES:
        f, k = float(f_lens(2, t)), float(kappa(2, t))
        entry = dict(t=t, f=f, kappa=k)
        for axis, (plus, minus, dK) in {"e1": ("+e1", "-e1", gradK[0]), "e2": ("+e2", "-e2", gradK[1])}.items():
            odd, even, ball_odd, ball_even = [], [], [], []
            for r in RADII:
                c = _load(ctx, f"conf_r{r}_t{t}.json")
                Rp, Rm = c[plus]["lens"] / c["ball_x"], c[minus]["lens"] / c["ball_x"]
                odd.append((Rp - Rm) / (2 * r**3))
                even.append(((Rp + Rm) / 2 - f - k * rho * r * r) / r**4)
                ball_odd.append((c[plus]["ball"] - c[minus]["ball"]) / (2 * np.pi * r**5))
                ball_even.append((c[plus]["ball"] + c[minus]["ball"] - 2 * c["ball_x"]) / (np.pi * r**6))
            entry[axis] = dict(d_v_K=float(dK), odd=odd, odd_limit=_richardson(odd),
                               odd_heuristic=(1.5 * k - f / 12) * (t / 2) * dK, even=even, even_limit=_richardson(even),
                               ball_odd=ball_odd, ball_odd_limit=_richardson(ball_odd), ball_odd_prediction=-t * dK / 12,
                               ball_even=ball_even, ball_even_limit=_richardson(ball_even))
        rows.append(entry)
    fine = _load(ctx, "accuracy_r0.4_t1.0.json")
    base = _load(ctx, "conf_r0.4_t1.0.json")
    accuracy = max(max(abs(fine[v]["lens"] / base[v]["lens"] - 1), abs(fine[v]["ball"] / base[v]["ball"] - 1))
                   for v in DIRECTIONS)
    sphere = _load(ctx, "sphere_r0.2_t0.75.json")
    sphere_odd = max(abs(a - b) / (a + b) for a, b in sphere.values())
    ctx.save_summary(NAME, dict(K0=K0, grad_K0=gradK.tolist(), rho=rho, radii=RADII, rows=rows,
                                discretisation_change=accuracy, sphere_relative_odd_part=sphere_odd))

    ctx.note(f"- Test surface phi = -|x|^2/4 + x_1^3/6: K(0) = {K0:.6f}, grad K(0) = ({gradK[0]:+.6f}, {gradK[1]:+.6f}), "
             f"rho = {rho}. Radii {RADII}; normalised quantities converge at rate r^2, limits by Richardson extrapolation.")
    for row in rows:
        e1, e2 = row["e1"], row["e2"]
        ctx.note(f"- t = {row['t']}: odd part (R(e_1) - R(-e_1))/(2 r^3) = " + ", ".join(f"{x:+.4e}" for x in e1["odd"])
                 + f", limit {e1['odd_limit']:+.4e}, midpoint heuristic {e1['odd_heuristic']:+.4e} "
                 f"({100 * (e1['odd_limit'] / e1['odd_heuristic'] - 1):+.2f}%); antipodal remainder / r^4 = "
                 + ", ".join(f"{x:+.4e}" for x in e1["even"]) + f", limit {e1['even_limit']:+.4e}.")
        ctx.note(f"  v = e_2: odd part at most {max(abs(x) for x in e2['odd']):.1e} (symmetry x_2 -> -x_2); "
                 f"remainder / r^4 limit {e2['even_limit']:+.4e}. Balls along e_1: odd part / (pi r^5) limit "
                 f"{e1['ball_odd_limit']:+.5f} (prediction -t d_v K/12 = {e1['ball_odd_prediction']:+.5f}); "
                 f"(Vol B(y_+) + Vol B(y_-) - 2 Vol B(x)) / (pi r^6) = " + ", ".join(f"{x:+.4f}" for x in e1["ball_even"]) + ".")
    acc = design(ctx)[2]
    ctx.note(f"- Finer discretisation ({int(1.5 * acc['nodes'])} nodes, {2 * acc['steps']} steps) at r = 0.4, t = 1 "
             f"changes the volumes by at most "
             f"{accuracy:.1e} relative. On the round sphere the two lens volumes agree to {sphere_odd:.1e} relative.")

    table = []  # in units of 1e-3
    for i, row in enumerate(rows):
        e1 = row["e1"]
        cells = " & ".join(f"${1e3 * x:+.3f}$" for x in e1["odd"])
        table.append(f"{row['t']:.2f} & odd part$/r^3$ & {cells} & ${1e3 * e1['odd_limit']:+.3f}$ & "
                     f"${1e3 * e1['odd_heuristic']:+.3f}$ \\\\")
        cells = " & ".join(f"${1e3 * x:+.3f}$" for x in e1["even"])
        table.append(f" & remainder$/r^4$ & {cells} & ${1e3 * e1['even_limit']:+.3f}$ & \\\\")
        if i + 1 < len(rows):
            table.append("\\addlinespace[2pt]")
    ctx.write_table("tableS8", "Table S8 body (in units of 1e-3)", table)
