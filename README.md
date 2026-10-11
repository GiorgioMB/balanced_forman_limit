# Code for "Balanced Forman curvature on random geometric graphs"

Scripts use absolute working directories (`/tmp/bf` for the planted-edge,
lens and full-graph experiments, `/tmp/w` for the others). Copy `code/` there,
or edit the paths at the top of each script. C programs compile with
`gcc -O3 -march=native -o NAME NAME.c -lm`. Seeds are fixed by configuration.

| Paper item | Producer | Summary file in `results/` |
|---|---|---|
| Section 8.2 (single edge on the torus), Fig. 1 (rate), Fig. 2 (CLT), Fig. 3 and Table 1 (4-cycle term, last two columns) | `run_flat.py` (via `jobs.txt`, `runjobs.sh`), `an_flat.py`, `plots.py` | `flat_summary.json`, `flat_edges_d{2,3}.npz` |
| Table 1, constants u, g, c | `sq_const.py` | `sq_const.json` |
| Table 2 and Fig. 4 (all edges of a graph) | `fullgraph.c`, `an_global.py`, `plots.py` | `global_summary.json`, `global_d*_n*_lam*.npz` |
| Table 3 (lens expansion), exact-lens column of Table 4 | `exact.py` | `exact.json` |
| Clustering check after Table 3 | `clust.py` | `clust_exact.json` |
| Table 4 and Fig. 5 (curved spaces) | `run_curved.py` (via `jobs_c.txt`, `runjobs_c.sh`), `an_curved.py`, `plots.py` | `curved_summary.json`, `curved_raw.tgz` |
| Section 8.6 and Fig. 6 (vertex-level curvature, Ricci tensor) | `run_e1.py`, `an_e1.py`, `fig_e1.py`, `vsim.py` | `e1_summary.json` |
| Table 5 (clustering, Euler characteristic) | `run_e2.py`, `an_e2.py`, `e2lib.py`, `bolza.py` | `e2_summary.json`, `e2_exact.json` |
| Section 8.8 (logarithmic and fixed degree), Fig. 7(b) | `run_e5.py`, `run_e5b.py`, `an_e5.py`, `fig_more.py` | `e5_summary.json` |
| Table 6 (two edges sharing a vertex), Fig. 7(c) | `oq1pair.py`, `run_oq1pair.py`, `an_oq1pair.py` | `oq1_pair.json` |
| Global averages on the sphere | `oq1glob.py`, `run_oq1glob.py` | raw arrays in `raw_results.tgz` (`results/oq1/glob_*`) |
| Table 8 (variance of F_n) | `fullvar.c`, `runfv.sh`, `an_fv.py` | `var_table.json`, `var_table.txt`, `fullvar_raw.tgz` |
| Table 7 (second-order kernel) | `d2c.c`, `rund2.sh`, `an_c.py`, `an_chaos.py` | `kernel_summary.txt`, `kernel_raw.tgz` |
| Remark on non-uniform sampling | `e3sim.py`, `run_e3.py`, `an_e3.py` | `e3_summary.json` |

Exploratory experiments that are not reported in the paper: ambient distances
(`e4sim.py`, `e4_summary.json`), curvature-based rewiring (`e6lib.py`, `e6_*.json`),
BF versus Ollivier curvature (`oq2orc.py`, `oq2_summary.json`) and kNN graphs.
