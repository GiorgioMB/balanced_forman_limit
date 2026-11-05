# Balanced Forman curvature on random geometric graphs: reproduction code

This repository reproduces every number, table and figure of the paper *Balanced Forman Curvature and
Neighbourhood-Overlap Statistics on Random Geometric Graphs* and of its supplementary material. 

## Requirements

Python 3.9 or later with NumPy, SciPy and Matplotlib, and a C compiler (`cc`, `gcc` or `clang`; set `CC` to
choose one). The C programs are compiled automatically on first use.

```
python -m pip install -r requirements.txt
```

## One entry point

Everything runs through `reproduce.py`:

```
python reproduce.py --list                         # what can be reproduced, and the cost of each part
python reproduce.py selftest                       # all implementations against Definition 2.1 (seconds)
python reproduce.py all --scale smoke              # every experiment at reduced size (minutes)
python reproduce.py all --scale paper --jobs 16    # every experiment at the size reported in the paper
python reproduce.py table2 fig3                    # selected items; dependencies are run automatically
```

`--scale smoke` (the default) checks that the whole pipeline works; its numbers are noisy. `--scale paper`
uses the sample sizes of the paper. The work is split into small cached tasks: an interrupted run resumes
where it stopped, and rerunning a finished experiment only redoes the (fast) analysis.

Results are written to `results/<scale>/`:

| Directory | Contents |
|---|---|
| `figures/` | the figure files included by the manuscript (`fig_rate.pdf`, ..., `fig_more.pdf`) |
| `tables/` | the body rows of every table, in the manuscript's LaTeX format |
| `summaries/` | every computed quantity, as JSON |
| `REPORT.md` | each reported number, next to the place in the paper where it appears |
| `raw/` | the cached simulation output |

## Manuscript items

| Item | Command | Output |
|---|---|---|
| Table 1 (thresholds), Remark 3.7(c), constants of Sections 3.4 and 7.1 | `python reproduce.py table1` | `REPORT.md`, `tables/remark37.tex` |
| Section 8.2, Figures 1 and 2 | `python reproduce.py fig1` | `figures/fig_rate.pdf`, `figures/fig_clt.pdf` |
| Section 8.3, Table 2, Figure 3 | `python reproduce.py table2` | `tables/table2.tex`, `figures/fig_curved.pdf` |
| Section 8.4, Table 3 (mean clustering) | `python reproduce.py table3` | `tables/table3.tex` |
| Section 8.4, global BF averages and Table 4 | `python reproduce.py table4` | `tables/table4.tex` |
| Section S2, Table S1, Figure S1, Remark 3.5 | `python reproduce.py tableS1` | `tables/tableS1.tex`, `figures/fig_sq.pdf` |
| Section S3, Table S2, Figure S2 | `python reproduce.py tableS2` | `tables/tableS2.tex`, `figures/fig_global.pdf` |
| Section S4, Table S3, check of Corollary 5.2 | `python reproduce.py tableS3` | `tables/tableS3.tex` |
| Section S5, Figure S3 | `python reproduce.py figS3` | `figures/fig_e1.pdf` |
| Section S6, Figure S4(b) | `python reproduce.py S6` | `REPORT.md` |
| Section S7, Table S4, Figure S4(c) | `python reproduce.py tableS4` | `tables/tableS4.tex` |
| Figure S4 | `python reproduce.py figS4` | `figures/fig_more.pdf` |
| Section S8, Tables S5 and S6, Remark 7.10 | `python reproduce.py S8` | `tables/tableS5.tex`, `tables/tableS6.tex` |
| Section S9, Table S7, Proposition 5.8 | `python reproduce.py tableS7` | `tables/tableS7.tex` |

## Cost

At paper scale the experiments take roughly 7 CPU-hours in total, most of it for the planted edges on
curved spaces and the planted vertices (`--list` gives the cost of each part). With `--jobs N` the work is
spread over N processes; memory use stays below 1 GB per process.

## Layout

```
reproduce.py            the entry point
bflib/theory.py         closed-form quantities: f_d, Psi_d, kappa_d, the law of W_t, vartheta_d, eta_d, sigma_d^2, ...
bflib/planted.py        exact BF curvature of planted edges (flat, S^k, H^2, S^2 x R, a density gradient, two edges)
bflib/vertex.py         local graphs around a planted vertex (Section S5)
bflib/graphs.py         BF curvature on whole graphs, clustering, graphs on T^2, S^2 and the Bolza surface
bflib/bolza.py          the Bolza surface (Section S1)
bflib/lens.py           exact lens volumes and clustering probabilities by quadrature
bflib/fourcycle.py      the constants u_d, vbar_d, c_d of the 4-cycle term
bflib/experiments/      one module per experiment; each lists the manuscript items it produces
bflib/selftest.py       the implementations against a literal implementation of Definition 2.1
bflib/pipeline.py       cached parallel tasks, the C programs, output helpers
csrc/bfgraph.c          BF curvature on all edges of graphs on T^2, T^3 and S^2
csrc/kernel.c           the second-order kernel of Section S8
```

## Reproducibility

Every task has a fixed seed, so a run regenerates the same numbers. The planted-edge samplers draw their
random numbers in a fixed order; the C programs use the xoshiro256** generator seeded by splitmix64 and an
exact Poisson sampler. `python reproduce.py selftest` replays the C generator in Python and checks the C
programs edge by edge against Definition 2.1.
