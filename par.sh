#!/bin/bash
# par.sh script args... : 4 parallel workers, single-threaded BLAS, ~95s budget
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NW=4
for w in 0 1 2 3; do W=$w timeout -k 1 7 "$@" > /tmp/par_$w.log 2>&1 & done
wait
tail -n 2 /tmp/par_*.log | grep -v "^$" | grep -v "==>" | head -8
