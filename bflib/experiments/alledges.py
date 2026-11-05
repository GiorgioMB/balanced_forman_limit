"""BF curvature on all edges of random geometric graphs on the torus (Section S3; Table S2, Figure S2;
the maximal-deviation statement after Theorem 3.1)."""

import numpy as np
from scipy.optimize import brentq

from ..pipeline import Task, bfgraph_edges, figure_style, run_tasks
from ..theory import Psi, f_lens, omega

NAME = "alledges"
TITLE = "All edges of random geometric graphs on the torus"
ITEMS = ["Table S2", "Figure S2", "Section 3.1 (maximal deviation)", "Section 8.2 (all edges)"]
REQUIRES = []


def design(ctx):
    """(d, n, lambda, seed). The seeds are those of the runs reported in Table S2."""
    if ctx.paper:
        return [(2, 10000, 25, 1), (2, 10000, 50, 2), (2, 10000, 100, 3), (2, 10000, 200, 4), (2, 10000, 400, 5),
                (3, 25000, 25, 6), (3, 25000, 50, 7), (3, 25000, 100, 8), (3, 25000, 200, 9)]
    return [(2, 2000, 25, 1), (2, 2000, 50, 2), (3, 4000, 25, 6)]


def _path(ctx, d, n, lam, seed):
    return ctx.raw(NAME, f"g_{d}_{n}_{lam}_seed{seed}.npy")


def run(ctx):
    run_tasks(ctx, NAME, [Task(_path(ctx, *cfg), bfgraph_edges, cfg) for cfg in design(ctx)])


def analyse(ctx):
    rows = []
    for d, n, lam, seed in design(ctx):
        t, bf, sq = np.load(_path(ctx, d, n, lam, seed)).astype(float).T
        dev = np.abs(bf - Psi(d, t))
        levels = (np.arange(len(bf)) + 0.5) / len(bf)
        # W1 between the empirical law and the law of Psi_d(T): quantile coupling
        w1 = float(np.mean(np.abs(np.sort(bf) - np.sort(Psi(d, (1 - levels) ** (1 / d))))))
        rows.append(dict(d=d, n=n, lam=lam, seed=seed, E=len(bf), W1=w1, meandev=float(dev.mean()),
                         maxdev=float(dev.max()), ratio=float(dev.max() / np.sqrt(np.log(n) / lam)),
                         meanbf=float(bf.mean()), meanlim=float(-2 + 4.5 * f_lens(d, 1.0)),
                         lam_max_S=float(lam * sq[t >= 0.25].max()),
                         r=float((lam / (n * omega(d))) ** (1 / d))))
    ctx.save_summary(NAME, rows)
    ratios = [r["ratio"] for r in rows]
    ctx.note(f"- {sum(r['E'] for r in rows):,} edges; max_e |BF - Psi_d(t_e)| / sqrt(log n / lambda) in "
             f"[{min(ratios):.2f}, {max(ratios):.2f}] (paper: [1.77, 2.15]); largest r = {max(r['r'] for r in rows):.4f} "
             f"(must be < 1/8).")
    ctx.write_table("tableS2", "Table S2 body", [
        f"{r['d']} & {r['n']:,} & {r['lam']} & {r['E']:,} & {r['W1']:.4f} & {r['meandev']:.4f} & {r['maxdev']:.4f} & "
        f"{r['ratio']:.2f} & ${r['meanbf']:+.4f}$ & ${r['meanlim']:+.4f}$ & {r['lam_max_S']:.2f} \\\\".replace(",", "{,}")
        + ("\n\\addlinespace[2pt]" if (r["d"] == 2 and i + 1 < len(rows) and rows[i + 1]["d"] == 3) else "")
        for i, r in enumerate(rows)])
    _figure(ctx, rows)


def _figure(ctx, rows):
    plt = figure_style()
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3))
    for panel, d in enumerate((2, 3)):
        largest = max((r for r in rows if r["d"] == d), key=lambda r: r["lam"])
        bf = np.load(_path(ctx, d, largest["n"], largest["lam"], largest["seed"]))[:, 1].astype(float)
        ax[panel].hist(bf, bins=60, density=True, color="0.8", edgecolor="0.55", lw=0.3,
                       label=f"all $|E|={len(bf):,}$ edges")
        y = np.linspace(float(Psi(d, 1.0)) + 1e-6, 1 - 1e-6, 300)
        tau = np.array([brentq(lambda s: float(Psi(d, s)) - v, 0, 1) for v in y])
        density = d * omega(d) / omega(d - 1) / 3 * tau ** (d - 1) * (1 - tau**2 / 4) ** (-(d - 1) / 2)
        ax[panel].plot(y, density, "k", lw=1.1, label=r"$\varphi_\infty$")
        ax[panel].set_title(f"$d={d}$, $n={largest['n']}$, $\\lambda={largest['lam']}$", fontsize=9)
        ax[panel].set_xlabel("BF curvature")
        ax[panel].legend(frameon=False, loc="upper right", fontsize=7)
    for d, marker in [(2, "o"), (3, "s")]:
        sub = [r for r in rows if r["d"] == d]
        lams = [r["lam"] for r in sub]
        for key, colour in [("maxdev", "C3"), ("meandev", "C0"), ("W1", "C2")]:
            ax[2].loglog(lams, [r[key] for r in sub], marker + "-", color=colour, ms=3, lw=0.7,
                         mfc="none" if d == 3 else colour)
    lam_line = np.array([25, 400])
    n_ref = max(r["n"] for r in rows if r["d"] == 2)
    ax[2].loglog(lam_line, 1.9 * np.sqrt(np.log(n_ref) / lam_line), "k--", lw=0.7)
    ax[2].loglog(lam_line, 1.0 / np.sqrt(lam_line), "k:", lw=0.7)
    ax[2].text(140, 0.75, r"$\max_e|\mathrm{BF}-\Psi|$", color="C3", fontsize=7)
    ax[2].text(26, 0.062, r"mean$_e|\mathrm{BF}-\Psi|$", color="C0", fontsize=7)
    ax[2].text(26, 0.0075, r"$W_1(\mu_n,\mu_\infty)$", color="C2", fontsize=7)
    ax[2].set_xlabel(r"$\lambda$")
    ax[2].set_title(r"dashed: $1.9\sqrt{\log n/\lambda}$; dotted: $\lambda^{-1/2}$", fontsize=7.5)
    fig.tight_layout(pad=0.3, w_pad=0.5)
    fig.savefig(ctx.figure("fig_global.pdf"))
    plt.close(fig)
