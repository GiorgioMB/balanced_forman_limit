import numpy as np, glob, json
from e4sim import lens_ratio
from bfsim import f_lens
def L(tag): return np.concatenate([np.load(f) for f in glob.glob(f'../results/e4/{tag}_w*')])
rows=[]
for t in [0.75,1.0]:
    F=L(f'l2000_r0.5_t{t}_R1.0_axial_geo')
    for r in [0.5,0.8]:
        for d in ['circ','axial']:
            fs=glob.glob(f'../results/e4/l2000_r{r}_t{t}_R1.0_{d}_amb_w*')
            if not fs: continue
            X=L(f'l2000_r{r}_t{t}_R1.0_{d}_amb'); diff=X.mean()-F.mean(); se=np.sqrt(X.var()/len(X)+F.var()/len(F))
            pred=3*(lens_ratio(r,t,1.0,d,M=2500)-f_lens(2,t))
            rows.append(dict(t=t,r=r,dir=d,n=len(X),diff=float(diff),se=float(se),pred=float(pred)))
            print(f"t={t} r={r} {d:5s}: ambient-graph BF minus flat {diff:+.5f}±{se:.5f}   exact ambient-lens prediction {pred:+.5f}  z {(diff-pred)/se:+.2f}")
    print('  flat check: geodesic graph on cylinder mean BF - Psi =', F.mean()-(-2+3*f_lens(2,t)))
json.dump(rows,open('../results/e4_summary.json','w'),indent=1)
