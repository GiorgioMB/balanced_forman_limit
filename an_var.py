import numpy as np, glob, json
rows=[]
for lam,n0 in [(10,2000),(20,2000),(40,2000),(40,1000),(80,1000)]:
    fs=glob.glob(f'../results/var/var_l{lam}_n{n0}_w*'); X=np.concatenate([np.load(f) for f in fs])
    S=X[:,2:22]; C=X[:,22:42]; b=S.sum(0)/np.maximum(C.sum(0),1)    # pooled bin means = m_lam(t)-Psi(t)
    F=(S-C*b).sum(1); R=len(F)
    v=F.var(ddof=1); se=v*np.sqrt(2/(R-1))
    rows.append(dict(lam=lam,n0=n0,reps=R,varF=float(v),varF_se=float(se),varF_over_n=float(v/n0),varF_over_nlam=float(v/(n0*lam))))
    print(f"lam={lam:3d} n0={n0}: reps {R}  Var F/n = {v/n0:.3f} ± {se/n0:.3f}   Var F/(n lam) = {v/(n0*lam):.4f}")
L=np.array([r['lam'] for r in rows if r['n0']==2000]+[80.]); V=np.array([r['varF_over_n'] for r in rows if r['n0']==2000]+[rows[-1]['varF_over_n']])
print('log-log slope of Var F/n vs lambda:', np.polyfit(np.log(L),np.log(V),1)[0])
json.dump(rows,open('../results/var_summary.json','w'),indent=1)
