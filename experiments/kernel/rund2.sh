#!/bin/bash
. "$(dirname "$0")/../../tools/env.sh"
# rund2.sh lam nsamp tag : 4 workers, distinct seeds, 6.5 s budget
for w in 0 1 2 3; do
  timeout -k 1 6.5 "$REPO/c/d2c" $1 ${EPS:-0.05} $(( $3*1000000 + w*100000 + 1 )) $2 0 1 \
    >> $BF_WORK/res/c_$1_${EPS:-0.05}_$3_$w.txt 2>/dev/null &
done
wait
