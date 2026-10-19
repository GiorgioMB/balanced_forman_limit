"""Figure 7(c): correlation of BF curvature on two edges sharing a vertex (d = 2).

Simulated correlations (Table 6) are plotted against the exact Gaussian limit of the
joint-fluctuation theorem (\\cref{thm:joint}).  Each limit variable is
    W_e = 3 G(L_e) - (3/2) f_e (G(B_x) + G(B_e)) - (1/2) f_e |G(B_x) - G(B_e)|,
where G is the Gaussian random measure with Cov(G(A), G(B)) = |A n B| / pi.  A linear
form of a centred Gaussian vector is uncorrelated with |any linear form|, and
Cov(|U|,|V|) = (2/pi) sU sV (sqrt(1 - rho^2) + rho arcsin(rho) - 1), so the correlation
is explicit in the lens areas and the area of B_x n B_j n B_k.

Standalone:   python fig_more_c.py      -> fig_more_c.pdf, and prints the Table 6 limit column
In a script:  from fig_more_c import plot_panel_c;  plot_panel_c(axes[2])
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import betainc

# Table 6: angle (deg), edges, simulated correlation, standard error (lambda = 1000, t = 0.75).
# Replace with your unrounded simulation output if you have it.
SIM = np.array(
    [
        (10, 8506, 0.897, 0.002),
        (30, 11691, 0.704, 0.005),
        (60, 11536, 0.423, 0.008),
        (90, 8705, 0.160, 0.010),
        (120, 11122, -0.077, 0.009),
        (150, 14681, -0.267, 0.008),
        (180, 12476, -0.345, 0.008),
    ]
)


def lens(t):
    """f_2(t): area of the intersection of two unit discs at distance t, divided by pi."""
    return betainc(1.5, 0.5, 1.0 - t * t / 4.0)


def disc_intersection_area(centres):
    """Area of the intersection of unit discs with the given centres, by quadrature in x."""
    lo = max(c[0] for c in centres) - 1.0
    hi = min(c[0] for c in centres) + 1.0
    if hi <= lo:
        return 0.0

    def width(x):
        bot, top = -np.inf, np.inf
        for cx, cy in centres:
            half_chord = np.sqrt(max(0.0, 1.0 - (x - cx) ** 2))
            bot, top = max(bot, cy - half_chord), min(top, cy + half_chord)
        return max(0.0, top - bot)

    # breakpoints: disc edges and circle-circle intersections (where the integrand has kinks)
    kinks = {c[0] + s for c in centres for s in (-1.0, 1.0)}
    for i, (x1, y1) in enumerate(centres):
        for x2, y2 in centres[i + 1 :]:
            dist = np.hypot(x2 - x1, y2 - y1)
            if 0 < dist < 2:
                half_chord = np.sqrt(1 - dist * dist / 4)
                kinks |= {
                    (x1 + x2) / 2 + s * half_chord * (y1 - y2) / dist
                    for s in (-1.0, 1.0)
                }

    breakpoints = sorted(k for k in kinks if lo < k < hi)
    return quad(
        width,
        lo,
        hi,
        points=breakpoints or None,
        limit=500,
        epsabs=1e-13,
        epsrel=1e-12,
    )[0]


def limit_corr(theta_deg, tj=0.75, tk=0.75):
    """Exact limit correlation of BF curvature on edges {x,j}, {x,k} at angle theta (d = 2)."""
    theta = np.deg2rad(theta_deg)
    zj, zk = (tj, 0.0), (tk * np.cos(theta), tk * np.sin(theta))
    fj, fk = lens(tj), lens(tk)
    triple_area = disc_intersection_area([(0.0, 0.0), zj, zk]) / np.pi
    fjk = lens(np.hypot(zj[0] - zk[0], zj[1] - zk[1]))

    # covariance of (G(L1), G(Bx), G(Bj), G(L2), G(Bk)), with L1 = Bx n Bj and L2 = Bx n Bk
    cov_matrix = np.array(
        [
            [fj, fj, fj, triple_area, triple_area],
            [fj, 1.0, fj, fk, fk],
            [fj, fj, 1.0, triple_area, fjk],
            [triple_area, fk, triple_area, fk, fk],
            [triple_area, fk, fjk, fk, 1.0],
        ]
    )

    l1 = np.array([3, -1.5 * fj, -1.5 * fj, 0, 0])  # linear part, edge {x,j}
    l2 = np.array([0, -1.5 * fk, 0, 3, -1.5 * fk])  # linear part, edge {x,k}
    u1 = np.array([0, 1.0, -1.0, 0, 0])  # U1 = G(Bx) - G(Bj)
    u2 = np.array([0, 1.0, 0, 0, -1.0])  # U2 = G(Bx) - G(Bk)

    s1, s2 = np.sqrt(u1 @ cov_matrix @ u1), np.sqrt(u2 @ cov_matrix @ u2)
    rho = np.clip(u1 @ cov_matrix @ u2 / (s1 * s2), -1.0, 1.0)

    cov_abs = 2 / np.pi * s1 * s2 * (np.sqrt(1 - rho**2) + rho * np.arcsin(rho) - 1)
    cov = l1 @ cov_matrix @ l2 + 0.25 * fj * fk * cov_abs
    var1 = l1 @ cov_matrix @ l1 + 0.25 * fj**2 * s1**2 * (1 - 2 / np.pi)
    var2 = l2 @ cov_matrix @ l2 + 0.25 * fk**2 * s2**2 * (1 - 2 / np.pi)

    return cov / np.sqrt(var1 * var2)


def plot_panel_c(ax, t=0.75, nse=2, label="(c)"):
    """Draw panel (c): simulation (+- nse standard errors) against the exact limit (line)."""
    grid = np.linspace(0, 180, 361)
    ax.plot(
        grid,
        [limit_corr(th, t, t) for th in grid],
        "-",
        color="k",
        lw=1.2,
        label="limit",
    )

    ax.errorbar(
        SIM[:, 0],
        SIM[:, 2],
        yerr=nse * SIM[:, 3],
        fmt="o",
        ms=4,
        capsize=2,
        color="C0",
        label=rf"simulation ($\pm{nse}$ s.e.)",
    )

    ax.axhline(0, ls=":", lw=0.8, color="0.5")
    ax.set_xlim(-5, 185)
    ax.set_xticks(range(0, 181, 30))
    ax.set_xlabel(r"angle $\theta$ (deg)")
    ax.set_ylabel("correlation")
    if label:
        ax.set_title(label, loc="left")
    ax.legend(frameon=False, fontsize="small")


if __name__ == "__main__":
    print("angle   limit    simulated        z")
    for th, _, sim_corr, se in SIM:
        lim_corr = limit_corr(th)
        print(
            f"{th:5.0f}  {lim_corr:+.3f}   {sim_corr:+.3f}+-{se:.3f}   {(sim_corr - lim_corr) / se:+.2f}"
        )

    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    plot_panel_c(ax)
    fig.tight_layout()
    fig.savefig("fig_more_c.pdf")
    print("wrote fig_more_c.pdf")
