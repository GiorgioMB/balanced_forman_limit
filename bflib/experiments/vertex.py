"""Vertex-level estimators of the scalar curvature and the Ricci tensor (Corollary 5.4(c),(d);
Section S5, Figure S3). These are oracle estimators: they use the latent lengths and directions."""

import zlib

import numpy as np

from ..pipeline import Task, figure_style, run_tasks
from ..theory import Psi, kappa
from ..vertex import adjacency, bf_vertex_edges, sample_local

NAME = "vertex"
TITLE = "Vertex-level curvature and the Ricci tensor"
ITEMS = ["Section S5", "Figure S3"]
REQUIRES = []

T_MIN, MAX_EDGES = 0.5, 200
# (geometry, lambda, r, true Scal, flat reference)
CONFIGS = [
    ("flat2", 2000, 1.0, 0, "flat2_l2000_r1.0"), ("S2", 2000, 0.5, 2, "flat2_l2000_r1.0"),
    ("H2", 2000, 0.5, -2, "flat2_l2000_r1.0"),
    ("flat3", 600, 1.0, 0, "flat3_l600_r1.0"), ("S3", 600, 0.6, 6, "flat3_l600_r1.0"),
    ("S2xR", 600, 0.6, 2, "flat3_l600_r1.0"),
    ("flat2", 500, 1.0, 0, "flat2_l500_r1.0"), ("S2", 500, 0.3, 2, "flat2_l500_r1.0"),
    ("S2", 500, 0.5, 2, "flat2_l500_r1.0"), ("S2", 500, 0.7, 2, "flat2_l500_r1.0"),
]


def design(ctx, geometry, lam):
    """(chunks, vertices per chunk)."""
    if lam == 2000:
        return (16, 25) if ctx.paper else (2, 2)
    if lam == 600:
        return (160, 1) if ctx.paper else (4, 1)
    return ((300 if geometry == "flat2" else 200), 1) if ctx.paper else (4, 1)


def name_of(geometry, lam, r):
    return f"{geometry}_l{lam}_r{r}"


def chunk(geometry, lam, r, chunk_id, n_vertices):
    rng = np.random.default_rng(zlib.crc32(f"{geometry}{float(lam)}{float(r)}{chunk_id}".encode()))
    parts = {key: [] for key in ("t", "bf", "sq", "v", "vid")}
    degrees = []
    for vertex in range(n_vertices):
        local = sample_local(rng, geometry, lam, r)
        A = adjacency(local)
        nbrs = A.indices[A.indptr[0] : A.indptr[1]]
        tangents = local["tan"][nbrs]
        t_norm = np.linalg.norm(tangents, axis=1) / r
        candidates = np.where(t_norm >= T_MIN)[0]
        pick = rng.choice(candidates, min(MAX_EDGES, len(candidates)), replace=False)
        values = bf_vertex_edges(A, 0, nbrs[pick])
        parts["t"].append(t_norm[pick])
        parts["bf"].append(values[:, 0])
        parts["sq"].append(values[:, 1])
        parts["v"].append(tangents[pick] / np.linalg.norm(tangents[pick], axis=1, keepdims=True))
        parts["vid"].append(np.full(len(pick), chunk_id * n_vertices + vertex))
        degrees.append(len(nbrs))
    out = {key: np.concatenate(val) for key, val in parts.items()}
    out["deg"] = np.array(degrees)
    return out


def run(ctx):
    tasks = []
    for geometry, lam, r, _, _ in CONFIGS:
        n_chunks, per = design(ctx, geometry, lam)
        tasks += [Task(ctx.raw(NAME, f"{name_of(geometry, lam, r)}_c{c}.npz"), chunk, (geometry, lam, r, c, per))
                  for c in range(n_chunks)]
    run_tasks(ctx, NAME, tasks)


def load(ctx, geometry, lam, r):
    n_chunks, _ = design(ctx, geometry, lam)
    files = [np.load(ctx.raw(NAME, f"{name_of(geometry, lam, r)}_c{c}.npz")) for c in range(n_chunks)]
    return {key: np.concatenate([f[key] for f in files]) for key in ("t", "bf", "sq", "v", "vid")}


def bias_fit(data, d, mask=None):
    """Cubic fit in t of BF - Psi_d on a flat reference: the flat expectation at the same lambda."""
    sel = np.ones(len(data["t"]), bool) if mask is None else mask
    return np.polyfit(data["t"][sel], data["bf"][sel] - Psi(d, data["t"][sel]), 3)


def scal_hat(data, d, r, coef):
    """Per-vertex Scal estimator of Corollary 5.4(c), with the realised sum of kappa_d in the denominator."""
    resid = data["bf"] - Psi(d, data["t"]) - np.polyval(coef, data["t"])
    ids = np.unique(data["vid"])
    index = np.searchsorted(ids, data["vid"])
    num = np.bincount(index, weights=resid)
    den = np.bincount(index, weights=kappa(d, data["t"]))
    return num / (3 * r * r * 2 * (d + 1) / (d * (d + 2)) * den)


def ricci_fit(data, d, r, coef):
    """Weighted least squares for the bilinear form rho_x of Corollary 5.4(d), pooled over vertices,
    with a vertex-cluster bootstrap; returns Ric and Scal with standard errors."""
    kap = kappa(d, data["t"])
    response = (data["bf"] - Psi(d, data["t"]) - np.polyval(coef, data["t"])) / (3 * kap * r * r)
    pairs = [(i, j) for i in range(d) for j in range(i, d)]
    design_matrix = np.column_stack([(1 if i == j else 2) * data["v"][:, i] * data["v"][:, j] for i, j in pairs])
    w = np.sqrt(kap**2)

    def solve(rows):
        beta = np.linalg.lstsq(design_matrix[rows] * w[rows, None], response[rows] * w[rows], rcond=None)[0]
        A = np.zeros((d, d))
        for (i, j), b in zip(pairs, beta):
            A[i, j] = A[j, i] = b
        return A - np.trace(A) / (2 * (d + 1)) * np.eye(d), (d + 2) / (2 * (d + 1)) * np.trace(A)

    ric, scal = solve(np.arange(len(response)))
    ids = np.unique(data["vid"])
    rows_of = {v: np.where(data["vid"] == v)[0] for v in ids}
    rng = np.random.default_rng(0)
    boot = [solve(np.concatenate([rows_of[v] for v in rng.choice(ids, len(ids))])) for _ in range(200)]
    return dict(Ric=ric, Ric_se=np.std([b[0] for b in boot], axis=0), Scal=scal, Scal_se=float(np.std([b[1] for b in boot])))


def analyse(ctx):
    data = {name_of(g, lam, r): load(ctx, g, lam, r) for g, lam, r, _, _ in CONFIGS}
    fits, summary = {}, {}
    for geometry, lam, r, scal, ref in CONFIGS:
        d = 3 if geometry in ("flat3", "S3", "S2xR") else 2
        name = name_of(geometry, lam, r)
        if name == ref:  # flat reference: cross-fit the bias curve on even vertices, evaluate on odd ones
            even = data[name]["vid"] % 2 == 0
            fits[ref] = bias_fit(data[name], d)
            estimates = scal_hat({k: v[~even] for k, v in data[name].items()}, d, 0.5, bias_fit(data[name], d, even))
        else:
            estimates = scal_hat(data[name], d, r, fits[ref])
        summary[name] = dict(d=d, lam=lam, r=r, true=scal, vertices=len(estimates), mean=float(estimates.mean()),
                             se=float(estimates.std() / np.sqrt(len(estimates))), sd=float(estimates.std()),
                             rmse=float(np.sqrt(np.mean((estimates - scal) ** 2))), lam_r4=lam * r**4,
                             estimates=estimates)
    for geometry, lam, r in [("S2xR", 600, 0.6), ("S3", 600, 0.6), ("S2", 2000, 0.5), ("H2", 2000, 0.5)]:
        d = 3 if geometry in ("S3", "S2xR") else 2
        ref = "flat3_l600_r1.0" if d == 3 else "flat2_l2000_r1.0"
        summary["ricci_" + name_of(geometry, lam, r)] = ricci_fit(data[name_of(geometry, lam, r)], d, r, fits[ref])
    ctx.save_summary(NAME, summary)

    for key in ("S2_l2000_r0.5", "H2_l2000_r0.5", "S3_l600_r0.6", "S2xR_l600_r0.6", "flat2_l2000_r1.0", "flat3_l600_r1.0"):
        s = summary[key]
        ctx.note(f"- {key}: mean Scal_hat {s['mean']:+.2f} +- {s['se']:.2f} (true {s['true']}), per-vertex sd {s['sd']:.2f}, "
                 f"{s['vertices']} vertices, lambda r^4 = {s['lam_r4']:.0f}.")
    rmse = sorted((s["lam_r4"], s["rmse"]) for k, s in summary.items() if k.startswith("S2_l"))
    ctx.note("- RMSE on S^2: " + ", ".join(f"{e:.1f} at lambda r^4 = {l:.0f}" for l, e in rmse) + ".")
    for key in ("ricci_S2xR_l600_r0.6", "ricci_S3_l600_r0.6", "ricci_S2_l2000_r0.5", "ricci_H2_l2000_r0.5"):
        fit = summary[key]
        ctx.note(f"- {key[6:]}: diag(Ric_hat) = ({', '.join(f'{x:.2f}' for x in np.diag(fit['Ric']))}) +- "
                 f"({', '.join(f'{x:.2f}' for x in np.diag(fit['Ric_se']))}), Scal_hat {fit['Scal']:.2f} +- {fit['Scal_se']:.2f}.")
    _figure(ctx, summary)


def _figure(ctx, summary):
    plt = figure_style()
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3))
    for key, colour, label in [("S2_l2000_r0.5", "C3", "$S^2$ (Scal $=2$)"), ("H2_l2000_r0.5", "C0", "$H^2$ (Scal $=-2$)"),
                               ("S3_l600_r0.6", "C1", "$S^3$ (Scal $=6$)")]:
        ax[0].hist(summary[key]["estimates"], bins=30, histtype="step", color=colour, density=True, label=label)
    ax[0].set_xlabel(r"$\widehat{\mathrm{Scal}}_i$")
    ax[0].legend(frameon=False, fontsize=6.5)
    ax[0].set_title("(a) per-vertex estimates", fontsize=8.5)
    circles = np.array(sorted((s["lam_r4"], s["rmse"]) for k, s in summary.items() if k.startswith("S2_l")))
    ax[1].loglog(circles[:, 0], circles[:, 1], "o", color="C3", ms=4)
    for key, colour in [("S3_l600_r0.6", "C1"), ("S2xR_l600_r0.6", "C2"), ("H2_l2000_r0.5", "C0")]:
        ax[1].loglog(summary[key]["lam_r4"], summary[key]["rmse"], "s", color=colour, ms=4)
    x = np.array([3, 150])
    ax[1].loglog(x, 11 * x**-0.5, "k--", lw=0.7)
    ax[1].set(xlabel=r"$\lambda r^4$", ylabel="RMSE")
    ax[1].set_title(r"(b) error $\propto(\lambda r^4)^{-1/2}$", fontsize=8.5)
    fit = summary["ricci_S2xR_l600_r0.6"]
    entries = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    ax[2].errorbar(range(6), [fit["Ric"][e] for e in entries], yerr=[2 * fit["Ric_se"][e] for e in entries], fmt="o",
                   color="C2", ms=4, capsize=2)
    ax[2].plot(range(6), [1, 1, 0, 0, 0, 0], "kx", ms=6)
    ax[2].set_xticks(range(6))
    ax[2].set_xticklabels(["$R_{11}$", "$R_{22}$", "$R_{33}$", "$R_{12}$", "$R_{13}$", "$R_{23}$"], fontsize=7)
    ax[2].axhline(0, color="k", lw=0.4)
    ax[2].set_title(r"(c) Ricci tensor, $S^2\times\mathbb{R}$", fontsize=8.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(ctx.figure("fig_e1.pdf"))
    plt.close(fig)
