#!/usr/bin/env python3
"""Single entry point: reproduces every number, table and figure of the paper and its supplement.

    python reproduce.py --list                         what can be reproduced, and how long it takes
    python reproduce.py selftest                       check the implementations against Definition 2.1
    python reproduce.py all --scale smoke              every experiment at reduced size (minutes)
    python reproduce.py all --scale paper --jobs 16    every experiment at the size reported in the paper
    python reproduce.py fig1 table4 tableS5            selected manuscript items (dependencies included)

Results go to results/<scale>/: figures/ (the PDF files included by the manuscript), tables/ (LaTeX table
bodies), summaries/ (JSON) and REPORT.md (every reported number next to its place in the paper).
Simulations are split into cached tasks, so an interrupted run resumes where it stopped.
"""

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")  # one thread per worker; parallelism comes from --jobs

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from bflib.experiments import ALIASES, EXPERIMENTS, PAPER_COST, resolve  # noqa: E402
from bflib.pipeline import Context  # noqa: E402


def list_targets():
    print("Experiments (run with: python reproduce.py <name> ...):\n")
    for name, module in EXPERIMENTS.items():
        cost = f"~{PAPER_COST[name]:g} CPU-h" if PAPER_COST[name] else "-"
        print(f"  {name:14s} {cost:>11s}   {module.TITLE}")
        print(f"  {'':14s} {'':11s}   produces: {'; '.join(module.ITEMS)}")
    print("\nManuscript items can be used as names too:")
    print("  " + ", ".join(ALIASES))
    total = sum(PAPER_COST.values())
    print(f"\nPaper scale in total: roughly {round(total)} CPU-hours; --scale smoke runs everything in a few minutes.")


def write_report(ctx):
    import numpy, scipy  # noqa: E401
    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True).stdout.strip() or "unknown"
    except OSError:
        commit = "unknown"
    lines = [f"# Reproduction report ({ctx.scale} scale)", "",
             f"Commit {commit}; Python {platform.python_version()}, NumPy {numpy.__version__}, SciPy {scipy.__version__}; "
             f"updated {time.strftime('%Y-%m-%d %H:%M')}.", ""]
    if ctx.scale != "paper":
        lines += ["Smoke scale uses reduced sample sizes: the numbers check that the pipeline runs, not the paper's values.", ""]
    for name, module in EXPERIMENTS.items():
        notes = ctx.root / "summaries" / f"{name}.report.md"
        if notes.exists():
            lines += [f"## {module.TITLE}", "", "Manuscript: " + "; ".join(module.ITEMS), "", notes.read_text().rstrip(), ""]
    produced = sorted(p.relative_to(ctx.root).as_posix() for sub in ("figures", "tables") for p in (ctx.root / sub).iterdir())
    lines += ["## Files", "", *[f"- {p}" for p in produced], ""]
    (ctx.root / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*", help="experiments, manuscript items, 'all' or 'selftest'")
    parser.add_argument("--scale", choices=["smoke", "paper"], default="smoke")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="worker processes (default: all cores)")
    parser.add_argument("--results", default=str(REPO / "results"), help="results directory (default: ./results)")
    parser.add_argument("--list", action="store_true", help="list experiments and manuscript items")
    args = parser.parse_args()

    if args.list or not args.targets:
        list_targets()
        return 0
    if args.targets == ["selftest"]:
        from bflib import selftest
        return 1 if selftest.run() else 0

    names = resolve([t for t in args.targets if t != "selftest"])
    ctx = Context(scale=args.scale, root=Path(args.results) / args.scale, jobs=max(1, args.jobs))
    print(f"scale {args.scale}, {ctx.jobs} worker(s), results in {ctx.root}")
    print("experiments: " + ", ".join(names))
    for name in names:
        module = EXPERIMENTS[name]
        start = time.time()
        print(f"\n== {name}: {module.TITLE}", flush=True)
        ctx.report = []
        module.run(ctx)
        module.analyse(ctx)
        (ctx.root / "summaries" / f"{name}.report.md").write_text("\n".join(ctx.report) + "\n")
        for line in ctx.report:
            print("  " + line)
        print(f"  ({time.time() - start:.0f} s)", flush=True)
    write_report(ctx)
    print(f"\nreport: {ctx.root / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
