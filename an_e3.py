import numpy as np, glob, json
from bfsim import f_lens
def L(tag):
    x=np.concatenate([np.load(f) for f in glob.glob(f'../results/e3/{tag}_w*')]); return x
rows=[]
for t in [0.75,1.0]:
    F=L(f'l2000_r0.02_t{t}_a0.0_par_fixed')
    for mode in ['fixed','adapt']:
        for d in ['par','perp']:
            for r in [0.01,0.02,0.04]:
                fs=glob.glob(f'../results/e3/l2000_r{r}_t{t}_a0.5_{d}_{mode}_w*')
                if not fs: continue
                X=L(f'l2000_r{r}_t{t}_a0.5_{d}_{mode}')
                diff=X.mean()-F.mean(); se=np.sqrt(X.var()/len(X)+F.var()/len(F))
                pred=-0.5*f_lens(2,t)*t*r*np.pi if (d=='par' and mode=='fixed') else 0.0
                rows.append(dict(t=t,mode=mode,dir=d,r=r,n=len(X),diff=float(diff),se=float(se),pred=float(pred)))
                print(f"t={t} {mode:5s} {d:4s} r={r}: diff {diff:+.5f}±{se:.5f}  first-order pred {pred:+.5f}  ratio {diff/pred if pred else float('nan'):.3f}")
json.dump(rows,open('../results/e3_summary.json','w'),indent=1)
