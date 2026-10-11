#!/bin/bash
export VPC=1 TL=5.5
while read geo lam r nch; do
  have=$(ls ../results/e1/ 2>/dev/null | grep -c "^${geo}_l${lam}_r${r}\(\.0\)\?_c")
  if [ "$have" -lt "$nch" ]; then
    echo "$geo $lam $r have $have/$nch"; bash par.sh python3 run_v1.py $geo $lam $r $nch; exit 0
  fi
done << 'CFG'
flat3 600 1.0 160
S3 600 0.6 160
S2xR 600 0.6 160
flat2 500 1.0 300
S2 500 0.3 200
S2 500 0.5 200
S2 500 0.7 200
CFG
echo ALLDONE
