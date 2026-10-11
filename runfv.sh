#!/bin/bash
# runfv.sh lam n0 ngraphs tag : 4 workers, distinct seeds, each whole graphs
for w in 0 1 2 3; do timeout -k 1 6.5 ./fullvar $1 $2 $(( $4*1000000 + w*100000 + 1 )) $3 0 1 100 >> /tmp/w/res/fv_$1_$2_$4_$w.txt 2>/dev/null & done; wait
