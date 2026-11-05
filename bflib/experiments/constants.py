"""Constants of the paper: the table of Remark 3.7(c), a_d (Theorem 3.6), vartheta_d and eta_d (Section 7.1),
the coefficient of Corollary 5.2, the exponents of Table 1, sigma_d^2 of Proposition 7.5 with its
second-chaos part (Remark 7.10), and b_2 of Section S8."""

from fractions import Fraction

import numpy as np

from ..pipeline import Task, run_tasks
from ..theory import a_d, b_constant, eta_d, f_lens, kappa, mean_over_T, sigma2_qmc, vartheta_d

NAME = "constants"
TITLE = "Constants and thresholds"
ITEMS = ["Table 1", "Remark 3.7(c)", "Theorem 3.6 (a_d)", "Section 7.1 (vartheta_d, eta_d)",
         "Remark 7.10 (sigma_d^2)", "Section S8 (b_2)", "Corollary 5.2"]
REQUIRES = []


def sigma2(d, m, second_chaos):
    value, se = sigma2_qmc(d, m=m, second_chaos=second_chaos)
    return dict(value=value, se=se)


def b2(n_samples):
    value, se = b_constant(2, n_samples)
    return dict(value=value, se=se)


def run(ctx):
    m = 20 if ctx.paper else 12
    tasks = [Task(ctx.raw(NAME, f"sigma2_d{d}.json"), sigma2, (d, m, False)) for d in (2, 3)]
    tasks.append(Task(ctx.raw(NAME, "sigma2_second_chaos_d2.json"), sigma2, (2, m, True)))
    tasks.append(Task(ctx.raw(NAME, "b2.json"), b2, (4_000_000 if ctx.paper else 200_000,)))
    run_tasks(ctx, NAME, tasks)


def analyse(ctx):
    import json

    out = {}
    table = []
    for d in (2, 3, 4, 5, 10, 20):
        f1 = float(f_lens(d, 1.0))
        mean = mean_over_T(d, lambda t: -2 + 3 * float(f_lens(d, t)))
        out[f"remark37_d{d}"] = dict(f1=f1, psi1=-2 + 3 * f1, mean=mean, mean_closed_form=-2 + 4.5 * f1)
        table.append(f"{d} & {f1:.4f} & ${-2 + 3 * f1:.3f}$ & ${mean:.3f}$ \\\\")
    ctx.write_table("remark37", "Remark 3.7(c): d, f_d(1), Psi_d(1), mean of mu_infinity", table)
    ctx.note("- Remark 3.7(c): " + "; ".join(
        f"d={d}: f={v['f1']:.4f}, Psi(1)={v['psi1']:.3f}, mean={v['mean']:.3f}"
        for d, v in ((k[10:], v) for k, v in out.items())) + ".")

    for d in (2, 3):
        out[f"a_{d}"], out[f"vartheta_{d}"], out[f"eta_{d}"] = a_d(d), vartheta_d(d), eta_d(d)
    ctx.note(f"- a_2 = {out['a_2']:.4f}, a_3 = {out['a_3']:.4f} (Theorem 3.6: 0.199, 0.184).")
    ctx.note(f"- vartheta_2 = {out['vartheta_2']:.6f} (1/24 = {1 / 24:.6f}), vartheta_3 = {out['vartheta_3']:.6f} "
             f"(67/2560 = {67 / 2560:.6f}).")
    ctx.note(f"- eta_2 = {out['eta_2']:.6f} (3 sqrt3/(32 pi) - 1/24 = {3 * np.sqrt(3) / (32 * np.pi) - 1 / 24:.6f}), "
             f"eta_3 = {out['eta_3']:.6f} (7/1280 = {7 / 1280:.6f}).")
    ctx.note(f"- On surfaces: 4 pi vartheta_2 = {4 * np.pi * out['vartheta_2']:.6f} (pi/6 = {np.pi / 6:.6f}); "
             f"4 pi (vartheta_2 + eta_2) = {4 * np.pi * (out['vartheta_2'] + out['eta_2']):.6f} "
             f"(3 sqrt3/8 = {3 * np.sqrt(3) / 8:.6f}).")
    out["clustering_coefficient_d2"] = 3 * float(kappa(2, 1.0)) / 4
    ctx.note(f"- Corollary 5.2, d = 2: 3 kappa_2(1)/4 = {out['clustering_coefficient_d2']:.5f} (paper: 0.01723).")

    thresholds = {}
    for d in (2, 3):
        thresholds[d] = dict(
            single_edge=Fraction(4, d + 4), bf_average=max(Fraction(0), 1 - Fraction(d, 4)),
            smooth_average=max(Fraction(0), Fraction(4 - d, d + 4)), normal_approximation=Fraction(1, 2),
            variance_regime=Fraction(6, d + 6))
        ctx.note(f"- Table 1, d = {d}: alpha > {thresholds[d]['single_edge']} (one edge), > {thresholds[d]['bf_average']} "
                 f"(BF average), > {thresholds[d]['smooth_average']} (smooth average), < 1/2 (normal approximation); "
                 f"lambda r^6 -> 0 iff alpha < {thresholds[d]['variance_regime']}.")
    out["table1"] = {d: {k: str(v) for k, v in th.items()} for d, th in thresholds.items()}

    s2, s3 = (json.load(open(ctx.raw(NAME, f"sigma2_d{d}.json"))) for d in (2, 3))
    chaos = json.load(open(ctx.raw(NAME, "sigma2_second_chaos_d2.json")))
    b = json.load(open(ctx.raw(NAME, "b2.json")))
    out.update(sigma2_d2=s2, sigma2_d3=s3, sigma2_second_chaos_d2=chaos, b2=b)
    ctx.note(f"- sigma_2^2 = {s2['value']:.3e} +- {s2['se']:.1e}, sigma_3^2 = {s3['value']:.3e} +- {s3['se']:.1e} "
             f"(Remark 7.10: 6.74e-4, 4.23e-4); varsigma^2/2 term for d = 2: {chaos['value']:.3e} (6.67e-4).")
    ctx.note(f"- The kink dominates Var F_n only for lambda above about 0.3 / sigma_2^2 = {0.3 / s2['value']:.0f} (paper: 450).")
    ctx.note(f"- b_2 = {b['value']:.4f} +- {b['se']:.4f} (Section S8: 0.0993).")
    ctx.save_summary(NAME, out)
