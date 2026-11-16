"""How much is computed: planted edges, and edges of whole graphs, whose BF curvature is computed exactly
(Section 1: "about 2 x 10^6 planted edges and 2 x 10^9 edges of full graphs"; Sections 8.1 and S1: "up to
about 3.6 x 10^6 edges per graph").

The counts are expected values computed from the designs of the experiments at the chosen scale; nothing
is simulated. A torus graph with n points and expected degree lambda has (n - 1) lambda / 2 edges in
expectation (binomial points, or Poisson points with the realised radius of bfgraph's mode 'realised'), a
torus graph with Poisson(n0) points at the nominal radius n0 lambda / 2, and a graph on the unit sphere with
Poisson(n0) points and geodesic radius r has n0^2 (1 - cos r) / 4. A planted vertex contributes its 200
sampled incident edges: at these degrees every vertex has more than 200 neighbours with t >= 0.5. Graphs
on which only local clustering is computed (Table 3), and the local graphs of the second-order kernel
(Table S6), are not counted.
"""

import numpy as np

from . import alledges, coverage, curved, degree, density, flat, pairs, sphere_global, variance, vertex

NAME = "census"
TITLE = "Scale of the computations"
ITEMS = ["Section 1 (numbers of planted edges and edges of full graphs)", "Sections 8.1 and S1 (edges per graph)"]
REQUIRES = []


def _torus(n0, lam):
    return n0 * lam / 2


def _sphere(n0, r):
    return n0 * n0 * (1 - np.cos(r)) / 4


def planted_edges(ctx):
    per_config = sum(n_chunks * per for n_chunks, per in flat.design(ctx).values())
    out = {"flat (Section 8.2)": 2 * len(flat.T_VALUES) * per_config}  # d = 2, 3
    n_chunks, per = curved.design(ctx)
    out["curved and flat references (Section 8.3)"] = (len(curved.CURVED) + 2) * len(curved.T_VALUES) * n_chunks * per
    out["planted vertices (Section S5)"] = sum(np.prod(vertex.design(ctx, g, lam)) * vertex.MAX_EDGES
                                               for g, lam, _, _, _ in vertex.CONFIGS)
    small, large = degree.fixed_sizes(ctx)
    out["fixed lambda (Section S6)"] = 4 * sum(small if lam <= 50 else large for lam in degree.FIXED_LAMBDAS)
    out["two edges at a vertex (Section S7)"] = 2 * sum(pairs.design(ctx).values())
    out["density and flat references (Section S9)"] = 2 * len(density.RADII) * len(density.T_VALUES) * density.design(ctx)
    return {k: int(v) for k, v in out.items()}


def graph_edges(ctx):
    """Expected number of edges per experiment, and the largest expected graph."""
    totals, largest = {}, (0.0, "")

    def add(label, edges_per_graph, graphs, what):
        nonlocal largest
        totals[label] = totals.get(label, 0.0) + edges_per_graph * graphs
        largest = max(largest, (edges_per_graph, what))

    for d, n, lam, _ in alledges.design(ctx):
        add("all edges on the torus (Section S3)", (n - 1) * lam / 2, 1, f"T^{d}, n = {n}, lambda = {lam}")
    for n, _, lam in degree.log_configs(ctx):
        add("logarithmic degree (Section S6)", (n - 1) * lam / 2, 1, f"T^2, n = {n}, lambda = {lam}")
    low, high = sphere_global.design(ctx)
    for r in low["radii"]:
        n0 = 4 * low["lam"] / r**2
        add("global averages on S^2 (Section 8.4)", _sphere(n0, r), low["graphs"], f"S^2, lambda = {low['lam']}, r = {r}")
        add("global averages on S^2 (Section 8.4)", (n0 - 1) * low["lam"] / 2, low["graphs"], f"T^2, lambda = {low['lam']}")
    for lam in high["lam"]:
        n0, r = 4 * lam / high["r"] ** 2, high["r"]
        add("global averages on S^2 (Section 8.4)", _sphere(n0, r), high["graphs"], f"S^2, lambda = {lam}, r = {r}")
        add("global averages on S^2 (Section 8.4)", _torus(n0, lam), high["graphs"], f"T^2, lambda = {lam}")
        add("global averages on S^2 (Section 8.4)", _torus(n0, sphere_global.sphere_degree(lam, r)), high["graphs"],
            f"T^2 at the mean degree, lambda = {lam}")
    for lam, n0, graphs in variance.design(ctx):
        add("variance (Section S8)", _torus(n0, lam), graphs, f"T^2, lambda = {lam}, n0 = {n0}")
    rows, refs = coverage.design(ctx)
    for lam, (n0, tori) in refs.items():
        add("coverage (Section S16)", _torus(n0, lam), tori, f"T^2, lambda = {lam}, n0 = {n0}")
    for space, lam, n0, graphs in rows:
        edges = _torus(n0, lam) if space == "torus2" else _sphere(n0, coverage.radius(space, lam, n0))
        add("coverage (Section S16)", edges, graphs, f"{'T^2' if space == 'torus2' else 'S^2'}, lambda = {lam}, n0 = {n0}")
    return {k: float(v) for k, v in totals.items()}, largest


def run(ctx):
    pass


def analyse(ctx):
    planted = planted_edges(ctx)
    graphs, (largest, where) = graph_edges(ctx)
    total_planted, total_graphs = sum(planted.values()), sum(graphs.values())
    edge_level = planted["flat (Section 8.2)"] + planted["curved and flat references (Section 8.3)"]
    without_s16 = total_graphs - graphs.get("coverage (Section S16)", 0.0)
    ctx.save_summary(NAME, dict(planted=planted, graphs=graphs, largest_graph=dict(edges=largest, graph=where),
                                total_planted=total_planted, total_graph_edges=total_graphs))
    ctx.note("- Planted edges: " + "; ".join(f"{k} {v:,}" for k, v in planted.items()) + f"; total {total_planted:,} "
             f"(of which {edge_level:,} in Sections 8.2 and 8.3).")
    ctx.note("- Edges of whole graphs (expected): " + "; ".join(f"{k} {v:.2e}" for k, v in graphs.items())
             + f"; total {total_graphs:.2e}, of which {without_s16:.2e} outside Section S16.")
    ctx.note(f"- Largest graph: {where}, about {largest:.2e} edges.")
    ctx.note("  (Section 1: about 2e6 planted edges and 2e9 edges of full graphs; Sections 8.1 and S1: up to about 3.6e6 "
             "edges per graph.)")
