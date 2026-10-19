"""Where scripts read and write data. Defaults are the locations the scripts used before the
reorganisation; set the environment variables to move them."""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# planted-edge, lens and full-graph experiments (formerly /tmp/bf/res and /tmp/bf/fig)
BF_ROOT = Path(os.environ.get("BF_ROOT", "/tmp/bf"))
RES = BF_ROOT / "res"
FIG = BF_ROOT / "fig"

# the other experiments (formerly ../results and ../figs, i.e. siblings of this repository)
RESULTS = Path(os.environ.get("BF_RESULTS", REPO.parent / "results"))
FIGS = Path(os.environ.get("BF_FIGS", REPO.parent / "figs"))

# raw text output of the C programs d2c and fullvar (formerly /tmp/w/res)
WORK_RES = Path(os.environ.get("BF_WORK", "/tmp/w")) / "res"
