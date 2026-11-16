"""Constants of the paper: the table of Remark 3.7(c), a_d (Theorem 3.6), vartheta_d and eta_d (Section 7,
eqs. (7.1) and (7.2)), the coefficient of Corollary 5.2, the exponents of Table 1 and of Theorem 7.14,
sigma_d^2 of Proposition 7.5 with its second-chaos part (Remark 7.10), and b_2 of Section S8."""

from fractions import Fraction

import numpy as np

from ..pipeline import Task, run_tasks
from ..theory import a_d, b_constant, eta_d, f_lens, kappa, mean_over_T, sigma2_qmc, vartheta_d

NAME = "constants"
TITLE = "Constants and thresholds"
ITEMS = ["Table 1", "Remark 3.7(c)", "Theorem 3.6 (a_d)", "Section 7 (vartheta_d, eta_d)",
         "Theorem 7.14 (windows)", "Remark 7.10 (sigma_d^2)", "Section S8 (b_2)", "Corollary 5.2"]
REQUIRES = []


def crossing(d, n=0, lam=0, r=0):
    """For lambda_n = n^alpha and r_n = (lambda_n / (omega_d n))^(1/d), the product of the powers n**n,
    lambda_n**lam and r_n**r is a constant times n**e(alpha), with e affine in alpha. Returns the zero of e
    and its slope."""
    n, lam, r = Fraction(n), Fraction(lam), Fraction(r)
    slope = lam + r / d
    return -(n - r / d) / slope, slope


def table1_thresholds(d):
    """The exponents of Table 1 and of Theorem 7.14 for lambda_n = n^alpha, derived from the conditions."""
    def above(**powers):  # the quantity tends to infinity exactly for alpha above the value
        value, slope = crossing(d, **powers)
        assert slope > 0
        return max(Fraction(0), value)

    def below(**powers):  # the quantity tends to zero exactly for alpha below the value
        value, slope = crossing(d, **powers)
        assert slope > 0
        return value

    normal = below(lam=1, r=6)                               # lambda r^6 -> 0 (on a curved M)
    ci_upper = min(below(n=Fraction(1, 2), lam=Fraction(1, 2), r=6),  # sqrt(n lambda) r^6 -> 0
                   below(n=Fraction(1, 2), r=4),             # sqrt(n) r^4 -> 0
                   normal)
    symmetric_upper = below(n=Fraction(1, 2), r=4)            # locally symmetric M: no sqrt(lambda) r^6 term
    return dict(
        flat_limit=Fraction(0),                               # lambda >> log n
        single_edge=above(lam=1, r=4),                        # lambda r^4 -> infinity
        bf_average=above(n=Fraction(1, 2), r=2),              # sqrt(n) r^2 -> infinity
        normal_approximation=normal,
        ci_lower=above(n=-Fraction(1, 2), lam=1),             # lambda / sqrt(n) -> infinity
        ci_upper=ci_upper,
        ci_upper_locally_symmetric=symmetric_upper,
        ci_upper_studentised=min(ci_upper, below(lam=1, r=4)),  # and lambda r^4 -> 0 (unless M = T^d)
        smooth_average=above(n=1, lam=1, r=4))                # n lambda r^4 -> infinity


TABLE1_ROWS = [  # the first two columns of Table 1, and the cells as functions of the thresholds
    (r"Uniform flat limit (\cref{thm:A}(i)) & $\lambda_n\gg\log n$", lambda th: f"$\\alpha>{th['flat_limit']}$"),
    (r"Curvature on one edge / all edges (\cref{cor:B}) & $\lambda_nr_n^4\to\infty$ / $\gg\log n$",
     lambda th: f"$\\alpha>{th['single_edge']}$"),
    (r"BF average consistent (\cref{thm:C,cor:D}) & $\sqrt n\,r_n^2\to\infty$",
     lambda th: f"$\\alpha>{th['bf_average']}$"),
    (r"Normal approximation (\cref{thm:D}) & $\lambda_nr_n^6\to0$", lambda th: f"$\\alpha<{th['normal_approximation']}$"),
    (r"Confidence intervals (\cref{thm:E}) & $\lambda_n\gg\sqrt n$, $\sqrt{n\lambda_n}\,r_n^6\to0$",
     lambda th: f"${th['ci_lower']}<\\alpha<{th['ci_upper']}$"),
    (r"Smooth statistic, average (\cref{thm:smooth-global}) & $n\lambda_nr_n^4\to\infty$",
     lambda th: f"$\\alpha>{th['smooth_average']}$"),
]


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
    dims = (2, 3, 4, 5, 10, 20)
    for d in dims:
        f1 = float(f_lens(d, 1.0))
        mean = mean_over_T(d, lambda t: -2 + 3 * float(f_lens(d, t)))
        out[f"remark37_d{d}"] = dict(f1=f1, psi1=-2 + 3 * f1, mean=mean, mean_closed_form=-2 + 4.5 * f1)
    # the layout of Remark 3.7(c): one column per dimension, below the header row "$d$ & 2 & 3 & ..."
    values = [out[f"remark37_d{d}"] for d in dims]
    ctx.write_table("remark37", "Remark 3.7(c), rows below the header $d$ & " + " & ".join(map(str, dims)), [
        "$f_d(1)$ & " + " & ".join(f"{v['f1']:.4f}" for v in values) + "\\\\",
        "$\\Psi_d(1)$ & " + " & ".join(f"${v['psi1']:.3f}$" for v in values) + "\\\\",
        "mean of $\\mu_\\infty$ & " + " & ".join(f"${v['mean']:.3f}$" for v in values) + "\\\\"])
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
    ctx.note(f"- Corollary 5.2, d = 2: 3 kappa_2(1)/4 = {out['clustering_coefficient_d2']:.5f} "
             "(paper, after Corollary 5.2: 0.01723).")

    thresholds = {d: table1_thresholds(d) for d in (2, 3)}
    for d, th in thresholds.items():
        ctx.note(f"- Table 1, d = {d}: alpha > {th['flat_limit']} (uniform flat limit), > {th['single_edge']} (one edge), "
                 f"> {th['bf_average']} (BF average), < {th['normal_approximation']} (normal approximation on a curved M), "
                 f"{th['ci_lower']} < alpha < {th['ci_upper']} (confidence intervals), > {th['smooth_average']} "
                 "(smooth average).")
    ctx.note("- Theorem 7.14 and the caption of Table 1: upper limits " + " and ".join(
        str(th["ci_upper"]) for th in thresholds.values()) + " (paper: beta_2 = 5/7, beta_3 = 3/5), on locally symmetric M "
        + " and ".join(str(th["ci_upper_locally_symmetric"]) for th in thresholds.values()) + " (paper: 3/4, 5/8); "
        "studentised " + " and ".join(str(th["ci_upper_studentised"]) for th in thresholds.values())
        + " (paper: 2/3, 4/7).")
    ctx.write_table("table1", "Table 1 body (columns d = 2 and d = 3)", [
        f"{label} & {cell(thresholds[2])} & {cell(thresholds[3])}\\\\" for label, cell in TABLE1_ROWS])
    out["table1"] = {d: {k: str(v) for k, v in th.items()} for d, th in thresholds.items()}

    s2, s3 = (json.load(open(ctx.raw(NAME, f"sigma2_d{d}.json"))) for d in (2, 3))
    chaos = json.load(open(ctx.raw(NAME, "sigma2_second_chaos_d2.json")))
    b = json.load(open(ctx.raw(NAME, "b2.json")))
    out.update(sigma2_d2=s2, sigma2_d3=s3, sigma2_second_chaos_d2=chaos, b2=b)
    ctx.note(f"- sigma_2^2 = {s2['value']:.3e} +- {s2['se']:.1e}, sigma_3^2 = {s3['value']:.3e} +- {s3['se']:.1e} "
             f"(Remark 7.10: 6.75e-4, 4.23e-4); varsigma^2/2 term for d = 2: {chaos['value']:.3e} (6.67e-4).")
    ctx.note(f"- The kink dominates Var F_n only for lambda above about 0.4 / sigma_2^2 = {0.4 / s2['value']:.0f}, where "
             "0.4 n lambda is the rest of the variance (Sections 8.4 and S8) (Remark 7.10: 600).")
    ctx.note(f"- b_2 = {b['value']:.4f} +- {b['se']:.4f} (Section S8: 0.0994).")
    ctx.save_summary(NAME, out)
