# Sourced by the driver scripts. Same defaults as bflib/paths.py; override by exporting them.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TOOLS=$REPO/tools
export BF_ROOT=${BF_ROOT:-/tmp/bf}
export BF_RESULTS=${BF_RESULTS:-$(dirname "$REPO")/results}
export BF_WORK=${BF_WORK:-/tmp/w}
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
