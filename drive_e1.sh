#!/bin/bash
# runs configs in order until ~160s used; idempotent
START=$(date +%s)
while read geo lam r nch; do
  NOW=$(date +%s); EL=$((NOW-START)); if [ $EL -gt 25 ]; then break; fi
  if [[ $geo == *3* || $geo == S2xR ]]; then export VPC=5; else export VPC=25; fi; python3 run_v1.py $geo $lam $r $nch | grep -v "remaining chunks 0 "
done << 'CFG'
flat2 2000 1.0 16
S2 2000 0.5 16
H2 2000 0.5 16
flat3 2000 1.0 40
S3 2000 0.5 40
S2xR 2000 0.5 40
flat2 500 1.0 16
flat2 1000 1.0 16
S2 500 0.3 8
S2 500 0.5 8
S2 500 0.7 8
S2 1000 0.3 8
S2 1000 0.5 8
S2 1000 0.7 8
S2 2000 0.3 8
S2 2000 0.7 8
CFG
