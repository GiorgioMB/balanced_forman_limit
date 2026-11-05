"""Single planted edges on the flat torus (Section 8.2; Figures 1, 2 and S1; Table S1, last columns)."""

import numpy as np

from ..pipeline import Task, figure_style, run_tasks
from ..planted import flat_trial
from ..theory import Psi, W_mean, W_var, f_lens, skew_normal_pdf

NAME = "flat"
TITLE = "Planted edges on the flat torus"
ITEMS = ["Section 8.2", "Figure 1", "Figure 2", "Figure S1", "Table S1 (columns 1600 ES, 12800 ES)"]
REQUIRES = ["fourcycle"]  # Figure S1 draws the limits c_d(t)

T_VALUES = [0.25, 0.5, 0.75, 1.0]


def design(ctx):
    """lambda values and, per lambda, (number of chunks, trials per chunk). 216,000 edges at paper scale."""
    if ctx.paper:
        lams = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
        return {lam: ((8 if lam <= 3200 else 4 if lam == 6400 else 2), 500) for lam in lams}
    return {lam: (1, 60) for lam in [100, 200, 400, 800]}


def chunk(d, lam, t, chunk_id, n_trials):
    rng = np.random.default_rng([d, lam, int(round(t * 100)), chunk_id, 20261009])
    return np.array([flat_trial(rng, d, lam, t)[:2] for _ in range(n_trials)])


def _path(ctx, d, lam, t, c):
    return ctx.raw(NAME, f"f_{d}_{lam}_{t}_{c}.npy")


def run(ctx):
    tasks = [Task(_path(ctx, d, lam, t, c), chunk, (d, lam, t, c, per))
             for d in (2, 3) for t in T_VALUES for lam, (n_chunks, per) in design(ctx).items()
             for c in range(n_chunks)]
    run_tasks(ctx, NAME, tasks)


def load(ctx, d, lam, t):
    n_chunks, _ = design(ctx)[lam]
    return np.concatenate([np.load(_path(ctx, d, lam, t, c)) for c in range(n_chunks)])


def analyse(ctx):
    lams = list(design(ctx))
    summary = {}
    for d in (2, 3):
        for t in T_VALUES:
            f, psi = float(f_lens(d, t)), float(Psi(d, t))
            rows = []
            for lam in lams:
                bf, sq = load(ctx, d, lam, t).T
                centred = np.sqrt(lam) * (bf.mean() - psi - (4 - 3 * f) / lam - sq.mean())
                se = np.sqrt(lam) * np.std(bf - sq, ddof=1) / np.sqrt(len(bf))
                rows.append(dict(lam=lam, n=len(bf), rms=float(np.sqrt(((bf - psi) ** 2).mean())),
                                 var_ratio=float(lam * bf.var(ddof=1) / W_var(f)),
                                 bias_scaled=float(centred), bias_se=float(se),
                                 z=float((centred - W_mean(f)) / se), lam_ES=float(lam * sq.mean())))
            large = np.array(lams) >= 400
            slope = np.polyfit(np.log(np.array(lams)[large]), np.log([r["rms"] for r in rows])[large], 1)[0]
            summary[f"{d}_{t}"] = dict(d=d, t=t, f=f, slope=float(slope), rows=rows)
    ctx.save_summary(NAME, summary)

    slopes = [v["slope"] for v in summary.values()]
    ratios = [r["var_ratio"] for v in summary.values() for r in v["rows"]]
    z = [r["z"] for v in summary.values() for r in v["rows"] if r["lam"] >= 400]
    n_edges = sum(r["n"] for v in summary.values() for r in v["rows"])
    ctx.note(f"- {n_edges:,} planted edges.")
    ctx.note(f"- Rate: least-squares slopes over lambda >= 400 in [{min(slopes):.3f}, {max(slopes):.3f}], "
             f"mean {np.mean(slopes):.3f} (paper: [-0.512, -0.487], mean -0.498).")
    ctx.note(f"- Variance: lambda Var(BF) / Var W_t in [{min(ratios):.2f}, {max(ratios):.2f}] over "
             f"{len(ratios)} configurations, mean {np.mean(ratios):.2f} (paper: [0.92, 1.11], 64, mean 0.99).")
    ctx.note(f"- Bias: {len(z)} standardised discrepancies for lambda >= 400, mean {np.mean(z):.2f}, "
             f"sd {np.std(z, ddof=1):.2f} (paper: 48, mean 0.12, sd 1.01).")
    _figures(ctx, summary, lams)
    write_table_s1(ctx)


def _no_minor_labels(axes):
    from matplotlib.ticker import NullFormatter

    for a in axes:
        a.xaxis.set_minor_formatter(NullFormatter())


def _figures(ctx, summary, lams):
    plt = figure_style()
    from matplotlib.lines import Line2D

    colours = {0.25: "#1b9e77", 0.5: "#d95f02", 0.75: "#7570b3", 1.0: "#e7298a"}
    lam_arr = np.array(lams)

    # Figure 1: rate, variance, bias
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.35))
    for d, marker, ls in [(2, "o", "-"), (3, "s", "--")]:
        for t in T_VALUES:
            entry = summary[f"{d}_{t}"]
            f, rows = entry["f"], entry["rows"]
            style = dict(color=colours[t], mfc="none" if d == 3 else colours[t], mew=0.7, ms=2.8)
            ax[0].loglog(lam_arr, [r["rms"] for r in rows], marker, ls="none", **style)
            ax[0].loglog(lam_arr, np.sqrt((W_var(f) + W_mean(f) ** 2) / lam_arr), ls, color=colours[t], lw=0.7)
            ax[1].semilogx(lam_arr, [r["var_ratio"] for r in rows], marker + ls, lw=0.6, **style)
            ax[2].errorbar(lam_arr * (1.04 if d == 3 else 0.96), np.array([r["bias_scaled"] for r in rows]) / W_mean(f),
                           yerr=2 * np.array([r["bias_se"] for r in rows]) / abs(W_mean(f)), fmt=marker,
                           elinewidth=0.5, capsize=0, **style)
    ax[0].set(xlabel=r"$\lambda$", ylabel=r"RMS of $\mathrm{BF}-\Psi_d(t)$")
    ax[0].set_title(r"(a) rate $\lambda^{-1/2}$", fontsize=9)
    ax[1].axhline(1, color="k", lw=0.6)
    ax[1].set(ylim=(0.8, 1.2), xlabel=r"$\lambda$", ylabel=r"$\lambda\,\mathrm{Var}(\mathrm{BF})\,/\,\mathrm{Var}(W)$")
    ax[1].set_title("(b) variance", fontsize=9)
    ax[2].set_xscale("log")
    ax[2].axhline(1, color="k", lw=0.6)
    ax[2].set(ylim=(0, 2), xlabel=r"$\lambda$",
              ylabel=r"$\sqrt{\lambda}\,(\mathrm{bias}-\frac{4-3f}{\lambda}-\mathrm{E}\,\mathcal{S})\,/\,\mathrm{E}W$")
    ax[2].set_title(r"(c) $\lambda^{-1/2}$ bias", fontsize=9)
    handles = [Line2D([], [], color=colours[t], marker="o", ls="", ms=3, label=f"$t={t}$") for t in T_VALUES]
    handles += [Line2D([], [], color="gray", marker="o", ls="-", ms=3, label="$d=2$"),
                Line2D([], [], color="gray", marker="s", mfc="none", ls="--", ms=3, label="$d=3$")]
    ax[0].legend(handles=handles, loc="upper right", frameon=False, ncol=2, handlelength=1.2, columnspacing=0.6)
    _no_minor_labels(ax)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(ctx.figure("fig_rate.pdf"))
    plt.close(fig)

    # Figure 2: empirical law against the skew-normal limit, pooled over the three largest lambda < max
    pool = [lam for lam in (1600, 3200, 6400) if lam in lams] or lams[-3:]
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.3))
    for panel, (d, t) in enumerate([(2, 0.5), (3, 1.0)]):
        f = float(f_lens(d, t))
        w = np.concatenate([np.sqrt(lam) * (load(ctx, d, lam, t)[:, 0] - Psi(d, t) - (4 - 3 * f) / lam
                                            - load(ctx, d, lam, t)[:, 1]) for lam in pool])
        ax[panel].hist(w, bins=45, density=True, color="0.82", edgecolor="0.55", lw=0.35,
                       label=f"simulation ($N={len(w)}$)")
        grid = np.linspace(-5, 5, 400)
        ax[panel].plot(grid, skew_normal_pdf(grid, f), "k", lw=1.1, label="skew-normal limit $W$")
        ax[panel].plot(grid, np.exp(-grid**2 / (2 * W_var(f))) / np.sqrt(2 * np.pi * W_var(f)), "k:", lw=0.8,
                       label=r"$N(0,\mathrm{Var}\,W)$")
        ax[panel].axvline(W_mean(f), color="k", lw=0.5, ls="--")
        ax[panel].set(xlim=(-5, 5), ylim=(0, 0.48))
        ax[panel].set_title(f"$d={d}$, $t={t}$, $\\lambda\\in\\{{{','.join(map(str, pool))}\\}}$", fontsize=8.5)
        ax[panel].set_xlabel(r"$\sqrt{\lambda}\,(\mathrm{BF}-\Psi_d(t)-\mathcal{S}-(4-3f_d)/\lambda)$")
    ax[0].legend(frameon=False, loc="upper left", fontsize=6.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(ctx.figure("fig_clt.pdf"))
    plt.close(fig)

    # Figure S1: lambda E S against the limit c_d(t) of Proposition 3.3
    constants = ctx.load_summary("fourcycle")["constants"]
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.2))
    for panel, d in enumerate((2, 3)):
        for t in T_VALUES:
            ax[panel].semilogx(lam_arr, [r["lam_ES"] for r in summary[f"{d}_{t}"]["rows"]], "o-", ms=2.8,
                               color=colours[t], lw=0.7, label=f"$t={t}$")
            ax[panel].axhline(constants[f"{d}_{t}"]["c"], color=colours[t], ls=":", lw=0.9)
        ax[panel].set(xlabel=r"$\lambda$", ylabel=r"$\lambda\,\mathrm{E}\,\mathcal{S}$")
        ax[panel].set_title(f"$d={d}$", fontsize=9)
    ax[0].legend(frameon=False, loc="lower right", ncol=2)
    _no_minor_labels(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(ctx.figure("fig_sq.pdf"))
    plt.close(fig)


def write_table_s1(ctx):
    """Table S1: constants of the 4-cycle term against lambda E S at the two largest lambda."""
    summary, constants = ctx.load_summary(NAME), ctx.load_summary("fourcycle")["constants"]
    lams = sorted({r["lam"] for v in summary.values() for r in v["rows"]})
    lo, hi = (1600, 12800) if {1600, 12800} <= set(lams) else (lams[-2], lams[-1])
    rows = []
    for d in (2, 3):
        for t in T_VALUES:
            c = constants[f"{d}_{t}"]
            es = {r["lam"]: r["lam_ES"] for r in summary[f"{d}_{t}"]["rows"]}
            if d == 2:
                u, v = ("$1/3$", "$1/6$") if t == 1.0 else (f"{c['u']:.5f}", f"{c['v']:.5f}")
                rows.append(f"{d} & {t:.2f} & {u} & {v} & 4 & {es[lo]:.2f} & {es[hi]:.2f} \\\\")
            else:
                rows.append(f"{d} & {t:.2f} & {c['u']:.4f}\\,$\\pm$\\,{c['u_se']:.4f} & {c['v']:.4f}\\,$\\pm$\\,"
                            f"{c['v_se']:.4f} & {c['c']:.2f}\\,$\\pm$\\,{c['c_se']:.2f} & {es[lo]:.2f} & {es[hi]:.2f} \\\\")
        if d == 2:
            rows.append("\\addlinespace[2pt]")
    ctx.write_table("tableS1", f"Table S1 body; simulated columns at lambda = {lo} and {hi}", rows)
