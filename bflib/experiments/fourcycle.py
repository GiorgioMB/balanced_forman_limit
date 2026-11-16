"""Constants u_d, vbar_d, c_d of the 4-cycle term (Table S1, Remark 3.5, Section S2).

For d = 2 the constants come from quadrature: u_2(t) = 2 v_t(h) by Lemma 3.4(b), and the profile of v_t
along both arcs of the right boundary of U_t, for the values of t of Remark 3.5, shows that v_t decreases
away from h+-, so that vbar_2(t) = v_t(h) and c_2(t) = 4. For d = 3 they come from Monte Carlo integration.
"""

import numpy as np

from ..fourcycle import constants_2d, constants_monte_carlo
from ..pipeline import Task, run_tasks

NAME = "fourcycle"
TITLE = "Constants of the 4-cycle term"
ITEMS = ["Table S1 (constants)", "Remark 3.5 (profile of v_t for t in {0.1, ..., 1})", "Section S2"]
REQUIRES = []

T_VALUES = [0.25, 0.5, 0.75, 1.0]  # Table S1
T_PROFILE = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]  # Remark 3.5, numerical observation


def quadrature(t):
    return constants_2d(t)


def monte_carlo(seed, n_j2, n_j3, n_i):
    """All eight configurations in one random stream, in the order d = 2, 3 and t increasing.
    Seed 7 gives the values reported in the paper; other seeds are independent replicates."""
    rng = np.random.default_rng(seed)
    return {f"{d}_{t}": constants_monte_carlo(rng, d, t, n_j2 if d == 2 else n_j3, n_i)
            for d in (2, 3) for t in T_VALUES}


def run(ctx):
    sizes = (400_000, 600_000, 40_000) if ctx.paper else (20_000, 30_000, 2_000)
    # (the file name differs from that of earlier versions, whose quadrature could miss narrow windows)
    tasks = [Task(ctx.raw(NAME, f"profile_{t}.json"), quadrature, (t,)) for t in T_PROFILE]
    tasks += [Task(ctx.raw(NAME, f"monte_carlo_seed{seed}.json"), monte_carlo, (seed, *sizes))
              for seed in [7] + replicate_seeds(ctx)]
    run_tasks(ctx, NAME, tasks)


def replicate_seeds(ctx):
    """Independent replicates; their spread is the Monte Carlo standard error of one run."""
    return list(range(8, 16)) if ctx.paper else [8, 9]


def analyse(ctx):
    import json

    quad = {t: json.load(open(ctx.raw(NAME, f"profile_{t}.json"))) for t in T_PROFILE}
    mc = json.load(open(ctx.raw(NAME, "monte_carlo_seed7.json")))
    reps = [json.load(open(ctx.raw(NAME, f"monte_carlo_seed{s}.json"))) for s in replicate_seeds(ctx)]
    for key in mc:
        for q in ("u", "v", "c"):
            # standard error of a single run: the spread of independent replicates (this includes the
            # bias of maximising a noisy function, which the binomial errors do not)
            mc[key][q + "_se"] = float(np.std([rep[key][q] for rep in reps], ddof=1))
    # d = 2: u_2 = 2 v_t(h) (Lemma 3.4(b)) and the maximum of v_t over the right boundary of U_t;
    # d = 3: Monte Carlo over the lunes
    constants = {}
    for t in T_VALUES:
        q = quad[t]
        constants[f"2_{t}"] = dict(u=q["u"], v=q["v_boundary_max"], c=q["c"], source="quadrature")
        constants[f"3_{t}"] = dict(mc[f"3_{t}"], source="monte carlo")
    ctx.save_summary(NAME, dict(constants=constants, monte_carlo=mc, quadrature=quad))

    ctx.note("- d = 2 by quadrature: " + ", ".join(
        f"t={t}: u={quad[t]['u']:.5f}, vbar={quad[t]['v_boundary_max']:.5f}, c={quad[t]['c']:.4f}" for t in T_VALUES))
    decreasing = {t: quad[t]["arc_q_decreasing"] and quad[t]["arc_p_decreasing"] for t in T_PROFILE}
    ctx.note(f"- Remark 3.5: v_t decreases monotonically along both arcs away from h+- for t in {T_PROFILE}: "
             + ", ".join(f"{t}: {ok}" for t, ok in decreasing.items()) + f" (all: {all(decreasing.values())}; "
             f"{quad[T_PROFILE[0]]['n_scan']} points per arc). Largest increase along an arc: "
             f"{max(max(q['arc_q_max_increase'], q['arc_p_max_increase']) for q in quad.values()):.1e}.")
    rim_is_max = all(abs(quad[t]["v_boundary_max"] - quad[t]["v_rim"]) < 1e-9 for t in T_PROFILE)
    ctx.note(f"- Hence v_t is maximal at h+- on the right boundary, vbar_2 = v_t(h) and c_2 = 4, for every t: {rim_is_max}. "
             f"At t = 1, v_1(h) - 1/6 = {quad[1.0]['v_rim'] - 1 / 6:.1e} (Remark 3.5: v_1(h) = 1/6).")
    c2 = [mc[f"2_{t}"] for t in T_VALUES]
    ctx.note("- Monte Carlo c_2(t): " + ", ".join(f"{m['c']:.2f} +- {m['c_se']:.2f}" for m in c2)
             + " (Remark 3.5: 4.00 +- 0.03).")
    dev = max(abs(mc[f"2_{t}"]["u"] / quad[t]["u"] - 1) for t in T_VALUES)
    ctx.note(f"- Largest relative deviation of the Monte Carlo u_2 from quadrature: {100 * dev:.1f}% (Section S2: 0.8%).")
    ctx.note("- d = 3: " + ", ".join(f"t={t}: c={mc[f'3_{t}']['c']:.2f} +- {mc[f'3_{t}']['c_se']:.2f}" for t in T_VALUES)
             + " (Remark 3.5: 7.67, 7.39, 6.99, 6.42).")
