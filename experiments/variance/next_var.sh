cd "$(dirname "$0")" && . ../../tools/env.sh
while read lam n0 tl N; do
  have=$(python3 -c "import numpy as np,glob; print(sum(len(np.load(f)) for f in glob.glob('$BF_RESULTS/var/var_l${lam}_n${n0}_w*')))")
  if [ "$have" -lt "$N" ]; then echo "$lam $n0 $have/$N"; TL=$tl bash "$TOOLS/par.sh" python3 e7var.py $lam $n0; exit 0; fi
done << 'CFG'
10 2000 4.0 48
20 2000 3.5 48
40 2000 2.3 32
80 1000 2.3 24
40 1000 3.5 32
CFG
echo ALLDONE
