export TL=4.5
while read lam r t R d a N; do
  tag="l${lam}_r${r}_t${t}_R${R}_${d}_${a}"
  have=$(python3 -c "import numpy as np,glob; print(sum(len(np.load(f)) for f in glob.glob('../results/e4/${tag}_w*_b*.npy')))")
  if [ "$have" -lt "$N" ]; then echo "$tag $have/$N"; bash par.sh python3 run_e4.py $lam $r $t $R $d $a; exit 0; fi
done << 'CFG'
2000 0.5 0.75 1.0 axial geo 24000
2000 0.5 1.0 1.0 axial geo 24000
2000 0.8 0.75 1.0 circ amb 24000
2000 0.8 0.75 1.0 axial amb 24000
2000 0.8 1.0 1.0 circ amb 24000
2000 0.8 1.0 1.0 axial amb 24000
2000 0.5 0.75 1.0 circ amb 24000
2000 0.5 0.75 1.0 axial amb 24000
CFG
echo ALLDONE
