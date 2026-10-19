cd "$(dirname "$0")" && . ../../tools/env.sh
export TL=4.5
while read kind n p N; do
  tag=$(python3 -c "print(f'${kind}_n${n}_p{float($p):g}')")
  st=$(python3 -c "
import numpy as np,glob
fs=glob.glob('$BF_RESULTS/e5/${tag}_w*_s*.npz'); done=sum(len(np.load(f)['bf']) for f in fs)
E=int(np.load(fs[0])['E']) if fs else 10**12
N='$N'; tgt=E if N=='ALL' else min(E,int(N))
print(1 if done>=tgt-4 else 0, done, tgt)")
  set -- $st
  if [ "$1" = "0" ]; then echo "$tag $2/$3"; bash "$TOOLS/par.sh" python3 run_e5.py $kind $n $p; exit 0; fi
done << 'CFG'
rgg 1000 13.8155 ALL
rgg 1000 47.7171 ALL
rgg 3000 16.0127 ALL
rgg 3000 64.1019 ALL
rgg 10000 18.4207 ALL
knn 20000 20 40000
knn 20000 50 40000
rgg 20000 50 40000
rgg 10000 84.8304 ALL
CFG
echo ALLDONE
