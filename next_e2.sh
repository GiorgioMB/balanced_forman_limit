export TL=5.5
while read geo lam r N; do
  tag=$(python3 -c "print(f'${geo}_l${lam}_r{float($r):.4f}')")
  have=$(python3 -c "import numpy as np,glob; print(sum(len(np.load(f)) for f in glob.glob('../results/e2/${tag}_w*_b*.npy')))")
  if [ "$have" -lt "$N" ]; then echo "$tag $have/$N"; bash par.sh python3 run_x2_clust.py $geo $lam $r; exit 0; fi
done << 'CFG'
bolza 60 0.6 2400
S2 60 0.6 1000
T2 60 0.16925687506432687 1000
S2 60 0.45 800
S2 60 0.3 500
CFG
echo ALLDONE
