# Code for "Balanced Forman curvature on random geometric graphs"

## Layout

```
bflib/         shared code: bfsim (planted-edge BF), allbf (BF of all edges), vsim (planted vertex), paths
c/             C programs (d2c, fullgraph, fullvar) and a Makefile
experiments/   one folder per paper item: simulation, analysis and driver scripts together
exploratory/   experiments not reported in the paper
figures/       figure scripts that combine several experiments
tools/         par.sh, go.sh, runjobs.sh (generic runners) and env.sh (sourced by the drivers)
```

Helpers used by only one experiment (`e2lib.py`, `bolza.py`, `e3sim.py`, `oq1pair.py`, `exact.py`, ...)
live in that experiment's folder and are imported from there.

## Setup

```
pip install -e .            # makes `bflib` importable; add .[figs] for matplotlib, .[ollivier] for POT
make -C c                   # gcc -O3 -march=native, -lm
```

Scripts can be run from any directory. The shell drivers (`next_*.sh`, `runjobs*.sh`, `rund2.sh`,
`runfv.sh`, ...) also work without the install step, since `tools/env.sh` puts the repository on
`PYTHONPATH`. Seeds are fixed by configuration.

Data locations are set in `bflib/paths.py` (and mirrored in `tools/env.sh`); the defaults are the
paths the scripts have always used:

| Variable | Default | Used by |
|---|---|---|
| `BF_ROOT` | `/tmp/bf` (`res/`, `fig/` below it) | flat, lens, full-graph and curved experiments, `figures/plots.py` |
| `BF_RESULTS` | `../results` next to this repository | all other experiments |
| `BF_FIGS` | `../figs` next to this repository | `fig_e1.py`, `fig_more.py`, `fig_new.py` |
| `BF_WORK` | `/tmp/w` (`res/` below it) | raw text output of `d2c` and `fullvar` |

## Paper items

| Paper item | Producer | Summary file in `results/` |
|---|---|---|
| Section 8.2 (single edge on the torus), Fig. 1 (rate), Fig. 2 (CLT), Fig. 3 and Table 1 (4-cycle term, last two columns) | `experiments/flat/`: `run_flat.py` (via `jobs.txt`, `tools/runjobs.sh`), `an_flat.py`; `figures/plots.py` | `flat_summary.json`, `flat_edges_d{2,3}.npz` |
| Table 1, constants u, g, c | `experiments/flat/sq_const.py` | `sq_const.json` |
| Table 2 and Fig. 4 (all edges of a graph) | `c/fullgraph.c`, `experiments/global/an_global.py`, `figures/plots.py` | `global_summary.json`, `global_d*_n*_lam*.npz` |
| Table 3 (lens expansion), exact-lens column of Table 4 | `experiments/lens/exact.py` | `exact.json` |
| Clustering check after Table 3 | `experiments/lens/clust.py` (`gamma_check.py`: the same constants by direct quadrature) | `clust_exact.json` |
| Table 4 and Fig. 5 (curved spaces) | `experiments/curved/`: `run_curved.py` (via `jobs_c.txt`, `runjobs_c.sh`), `an_curved.py`; `figures/plots.py` | `curved_summary.json`, `curved_raw.tgz` |
| Section 8.6 and Fig. 6 (vertex-level curvature, Ricci tensor) | `experiments/e1_vertex/`: `run_v1.py` (via `next_e1.sh` / `drive_e1.sh`, uses `bflib/vsim.py`), `an_v1.py` (+ `an_v1_lib.py`), `fig_e1.py` | `e1_summary.json` |
| Table 5 (clustering, Euler characteristic) | `experiments/e2_clustering/`: `run_x2_clust.py` (via `next_e2.sh`), `an_e2.py`, `e2lib.py`, `bolza.py` | `e2_summary.json`, `e2_exact.json` |
| Section 8.8 (logarithmic and fixed degree), Fig. 7(b) | `experiments/e5_degree/`: `run_e5.py` (via `next_e5.sh`), `run_e5b.py`, `an_e5.py`; `figures/fig_more.py` | `e5_summary.json` |
| Table 6 (two edges sharing a vertex), Fig. 7(c) | `experiments/oq1_pair/`: `oq1pair.py`, `run_oq1pair.py`, `an_oq1pair.py`; `figures/fig_more.py`, `figures/fig_7c_rerun.py` | `oq1_pair.json` |
| Global averages on the sphere | `experiments/oq1_pair/`: `oq1glob.py`, `run_oq1glob.py` | raw arrays in `raw_results.tgz` (`results/oq1/glob_*`) |
| Table 8 (variance of F_n) | `c/fullvar.c`, `experiments/variance/`: `runfv.sh`, `an_fv.py` | `var_table.json`, `var_table.txt`, `fullvar_raw.tgz` |
| Table 7 (second-order kernel) | `c/d2c.c`, `experiments/kernel/`: `rund2.sh`, `an_c.py`, `an_chaos.py` | `kernel_summary.txt`, `kernel_raw.tgz` |
| Remark on non-uniform sampling | `experiments/e3_density/`: `e3sim.py`, `run_e3.py` (via `next_e3.sh`), `an_e3.py` | `e3_summary.json` |

Other scripts kept next to the experiment they relate to:
`experiments/e1_vertex/run_e1.py` + `an_e1.py` (planted-edge rate runs in d = 2, 3),
`experiments/e2_clustering/run_e2.py` (all-edge BF on the torus in Python),
`experiments/variance/e7var.py` + `an_var.py` + `next_var.sh` (Python variance of F, `var_summary.json`).

Exploratory experiments that are not reported in the paper, in `exploratory/`: ambient distances
(`e4_ambient/`, `e4_summary.json`), curvature-based rewiring (`e6_rewiring/`, `e6_*.json`),
BF versus Ollivier curvature (`oq2_ollivier/`, `oq2_summary.json`), kNN graphs (`run_e5.py knn ...`)
and `coeffs.py`. `figures/fig_new.py` draws earlier versions of figures from these.
