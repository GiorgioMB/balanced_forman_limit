export TL=4.5
while read c N; do
  have=$(python3 -c "import numpy as np,glob; print(sum(len(np.load(f)) for f in glob.glob('../results/oq2/${c}_w*')))")
  if [ "$have" -lt "$N" ]; then echo "$c $have/$N"; bash par.sh python3 run_oq2.py $c; exit 0; fi
done << 'CFG'
flat2 2500
S2 2500
H2 2500
flat3 2000
S2xR_R 2000
S2xR_S 2000
CFG
echo ALLDONE
