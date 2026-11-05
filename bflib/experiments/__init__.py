"""Registry of experiments, in an order compatible with their dependencies."""

from . import (alledges, clustering, constants, curved, degree, density, figure_s4, flat, fourcycle, kernel, lens,
               pairs, sphere_global, variance, vertex)

EXPERIMENTS = {m.NAME: m for m in [constants, fourcycle, flat, alledges, lens, curved, clustering, sphere_global,
                                   vertex, degree, pairs, figure_s4, variance, kernel, density]}

# manuscript items -> experiments that produce them
ALIASES = {
    "table1": ["constants"], "fig1": ["flat"], "fig2": ["flat"], "table2": ["curved"], "fig3": ["curved"],
    "table3": ["clustering"], "table4": ["sphere_global"],
    "figS1": ["flat"], "figS2": ["alledges"], "figS3": ["vertex"], "figS4": ["figure_s4"],
    "tableS1": ["flat"], "tableS2": ["alledges"], "tableS3": ["lens"], "tableS4": ["pairs"],
    "tableS5": ["variance"], "tableS6": ["kernel"], "tableS7": ["density"],
    "sec8.2": ["flat", "alledges"], "sec8.3": ["curved"], "sec8.4": ["clustering", "sphere_global"],
    "S1": ["lens", "clustering"], "S2": ["fourcycle", "flat"], "S3": ["alledges"], "S4": ["lens"], "S5": ["vertex"],
    "S6": ["degree"], "S7": ["pairs"], "S8": ["variance", "kernel"], "S9": ["density"],
}

# approximate single-core cost at paper scale, in CPU-hours
# (the work parallelises over --jobs)
PAPER_COST = {"constants": 0.05, "fourcycle": 0.2, "flat": 1, "alledges": 0.02, "lens": 0.01, "curved": 2,
              "clustering": 0.3, "sphere_global": 1, "vertex": 1.5, "degree": 0.1, "pairs": 0.1, "figure_s4": 0,
              "variance": 0.2, "kernel": 0.5, "density": 0.3}


def resolve(names):
    """Experiments for the requested names (experiments, manuscript items or 'all'), with dependencies."""
    wanted = []
    for name in names:
        if name == "all":
            wanted += list(EXPERIMENTS)
        elif name in EXPERIMENTS:
            wanted.append(name)
        elif name in ALIASES:
            wanted += ALIASES[name]
        else:
            raise SystemExit(f"unknown target {name!r}; see python reproduce.py --list")
    closed = set()

    def add(name):
        for dep in EXPERIMENTS[name].REQUIRES:
            add(dep)
        closed.add(name)

    for name in wanted:
        add(name)
    return [name for name in EXPERIMENTS if name in closed]
