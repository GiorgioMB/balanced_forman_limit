export TL=4.5
while read lam r t a d mode N; do
  tag="l${lam}_r${r}_t${t}_a${a}_${d}_${mode}"
  have=$(python3 -c "import numpy as np,glob; print(sum(len(np.load(f)) for f in glob.glob('../results/e3/${tag}_w*_b*.npy')))")
  if [ "$have" -lt "$N" ]; then echo "$tag $have/$N"; bash par.sh python3 run_e3.py $lam $r $t $a $d $mode; exit 0; fi
done << 'CFG'
2000 0.02 0.75 0.0 par fixed 6000
2000 0.02 1.0 0.0 par fixed 6000
2000 0.01 0.75 0.5 par fixed 6000
2000 0.02 0.75 0.5 par fixed 6000
2000 0.04 0.75 0.5 par fixed 6000
2000 0.01 1.0 0.5 par fixed 6000
2000 0.02 1.0 0.5 par fixed 6000
2000 0.04 1.0 0.5 par fixed 6000
2000 0.02 0.75 0.5 perp fixed 6000
2000 0.04 0.75 0.5 perp fixed 6000
2000 0.02 1.0 0.5 perp fixed 6000
2000 0.04 1.0 0.5 perp fixed 6000
2000 0.02 0.75 0.5 par adapt 6000
2000 0.04 0.75 0.5 par adapt 6000
2000 0.02 1.0 0.5 par adapt 6000
2000 0.04 1.0 0.5 par adapt 6000
CFG
echo ALLDONE
