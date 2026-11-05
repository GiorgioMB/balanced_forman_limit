"""Planted edges on S^2, H^2, S^3 and S^2 x R against flat references (Section 8.3; Table 2, Figure 3)."""

import numpy as np

from ..pipeline import Task, figure_style, run_tasks
from ..planted import flat_trial, hyperbolic_trial, s2xr_trial, sphere_trial
from ..theory import Psi, f_lens, kappa, rho_quadratic_form

NAME = "curved"
TITLE = "Planted edges on curved spaces"
ITEMS = ["Section 8.3", "Table 2", "Figure 3"]
REQUIRES = ["lens"]

LAM, R = 1500, 0.5
T_VALUES = [0.5, 0.75, 1.0]
CURVED = ["S2", "H2", "S3", "S2xR_S", "S2xR_R"]
FLAT = {"S2": "flat2", "H2": "flat2", "S3": "flat3", "S2xR_S": "flat3", "S2xR_R": "flat3"}


def design(ctx):
    """(chunks, trials per chunk): 40,000 edges per cell at paper scale, 840,000 in total."""
    return (20, 2000) if ctx.paper else (1, 150)


def trial(rng, geometry, t):
    if geometry == "S2":
        return sphere_trial(rng, 2, LAM, t, R)
    if geometry == "S3":
        return sphere_trial(rng, 3, LAM, t, R)
    if geometry == "H2":
        return hyperbolic_trial(rng, LAM, t, R)
    if geometry.startswith("S2xR"):
        return s2xr_trial(rng, LAM, t, R, geometry[-1])
    return flat_trial(rng, int(geometry[-1]), LAM, t)


def chunk(geometry, t, chunk_id, n_trials):
    rng = np.random.default_rng([sum(map(ord, geometry)), int(round(t * 100)), chunk_id, 20261010])
    return np.array([trial(rng, geometry, t)[:2] for _ in range(n_trials)])


def _path(ctx, geometry, t, c):
    return ctx.raw(NAME, f"c_{geometry}_{t}_{c}.npy")


def run(ctx):
    n_chunks, per = design(ctx)
    run_tasks(ctx, NAME, [Task(_path(ctx, g, t, c), chunk, (g, t, c, per))
                          for g in CURVED + ["flat2", "flat3"] for t in T_VALUES for c in range(n_chunks)])


def _load(ctx, geometry, t):
    n_chunks, _ = design(ctx)
    return np.concatenate([np.load(_path(ctx, geometry, t, c)) for c in range(n_chunks)])[:, 0]


def analyse(ctx):
    exact = ctx.load_summary("lens")["geometries"]
    flat = {}
    rows = []
    for d in (2, 3):
        for t in T_VALUES:
            bf = _load(ctx, f"flat{d}", t)
            flat[(d, t)] = (bf.mean(), bf.std(ddof=1) / np.sqrt(len(bf)))
            rows.append(dict(geometry=f"flat{d}", t=t, n=len(bf), mean=float(bf.mean()), se=float(flat[(d, t)][1]),
                             offset=float(bf.mean() - Psi(d, t))))
    for g in CURVED:
        d, rho = rho_quadratic_form(g)
        for t in T_VALUES:
            bf = _load(ctx, g, t)
            diff = bf.mean() - flat[(d, t)][0]
            se = np.sqrt(bf.var(ddof=1) / len(bf) + flat[(d, t)][1] ** 2)
            exact_diff = 3 * (exact[g][f"r0.5_{t}"] - float(f_lens(d, t)))
            lead = 3 * float(kappa(d, t)) * rho * R * R
            rows.append(dict(geometry=g, t=t, n=len(bf), diff=float(diff), se=float(se), lead=lead,
                             exact=float(exact_diff), z=float((diff - exact_diff) / se),
                             relerr_lead=float((diff - lead) / lead), z_flat=float(diff / se)))
    ctx.save_summary(NAME, rows)

    curved = [r for r in rows if "z" in r]
    z = np.array([r["z"] for r in curved])
    rel = np.abs([r["relerr_lead"] for r in curved])
    ctx.note(f"- {sum(r['n'] for r in rows):,} planted edges at r = {R}, lambda = {LAM}.")
    ctx.note(f"- z-scores against the exact lens in [{z.min():.2f}, {z.max():.2f}], {int((np.abs(z) <= 2).sum())} of "
             f"{len(z)} within two standard errors, sum z^2 = {np.sum(z**2):.1f} on {len(z)} degrees of freedom "
             "(paper: [-2.40, 1.40], 14 of 15, 12.2).")
    ctx.note(f"- Agreement with 3 kappa_d rho r^2 within {100 * rel.max():.1f}%, mean absolute relative error "
             f"{100 * rel.mean():.1f}% (paper: 4.7%, 1.6%).")
    flat_line = [r["z_flat"] for r in curved if r["geometry"] == "S2xR_R"]
    ctx.note(f"- S^2 x R along the line: {min(flat_line):.1f} to {max(flat_line):.1f} standard errors from the flat "
             "reference (paper: 19.6 to 22.4).")
    offsets = [abs(r["offset"]) for r in rows if r["geometry"].startswith("flat")]
    ctx.note(f"- Flat references differ from Psi_d(t) by up to {max(offsets):.1e} (paper: 2.2e-3).")

    labels = {"S2": "$S^2$", "H2": "$H^2$", "S3": "$S^3$", "S2xR_S": r"$S^2\!\times\!\R$, $v\in TS^2$",
              "S2xR_R": r"$S^2\!\times\!\R$, $v\in T\R$"}
    table = []
    for g in CURVED:
        for i, r in enumerate(x for x in curved if x["geometry"] == g):
            table.append(f"{labels[g] if i == 0 else ''} & {r['t']:.2f} & ${r['diff']:+.5f}\\pm{r['se']:.5f}$ & "
                         f"${r['lead']:+.5f}$ & ${r['exact']:+.5f}$ & ${r['z']:+.2f}$ \\\\")
        table.append("\\addlinespace[2pt]")
    ctx.write_table("table2", "Table 2 body", table[:-1])
    _figure(ctx, curved)


def _figure(ctx, curved):
    plt = figure_style()
    styles = {"S2": (r"$S^2$ ($\rho=3/2$)", "C3", "o"), "H2": (r"$H^2$ ($\rho=-3/2$)", "C0", "v"),
              "S3": (r"$S^3$ ($\rho=16/5$)", "C1", "s"),
              "S2xR_S": (r"$S^2\!\times\!\mathbb{R}$, $v\in TS^2$ ($\rho=7/5$)", "C2", "D"),
              "S2xR_R": (r"$S^2\!\times\!\mathbb{R}$, $v\in T\mathbb{R}$ ($\rho=2/5$)", "C4", "^")}
    fig, ax = plt.subplots(1, 1, figsize=(4.6, 2.9))
    grid = np.linspace(0, 1, 200)
    for g in CURVED:
        d, rho = rho_quadratic_form(g)
        label, colour, marker = styles[g]
        sub = [r for r in curved if r["geometry"] == g]
        ax.plot(grid, 3 * kappa(d, grid) * rho * R**2, color=colour, lw=0.8)
        ax.errorbar([r["t"] for r in sub], [r["diff"] for r in sub], yerr=[2 * r["se"] for r in sub], fmt=marker,
                    color=colour, ms=3.5, capsize=1.5, elinewidth=0.7, label=label)
        ax.plot([r["t"] for r in sub], [r["exact"] for r in sub], "x", color="k", ms=3.5, mew=0.7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel=r"$t$", ylabel=r"$\mathrm{E\,BF}_{M}-\mathrm{E\,BF}_{\mathrm{flat}}$")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.42 * (hi - lo))  # a band above the curves for the legend
    ax.legend(frameon=False, fontsize=6.6, loc="upper left", ncol=2, columnspacing=1.0)
    ax.set_title(r"$r=0.5$, $\lambda=1500$; lines: $3\kappa_d(t)\rho\,r^2$; $\times$: exact lens", fontsize=8)
    fig.tight_layout(pad=0.3)
    fig.savefig(ctx.figure("fig_curved.pdf"))
    plt.close(fig)
