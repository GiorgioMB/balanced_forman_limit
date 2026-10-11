"""Summaries for all-edge RGG experiments on T^d (fullgraph.c output)."""
import numpy as np, glob, json, sys
sys.path.insert(0,'/tmp/bf'); from bfsim import Psi, f_lens
out=[]
for d,n,lams in [(2,10000,[25,50,100,200,400]),(3,25000,[25,50,100,200])]:
    for lam in lams:
        A=np.concatenate([np.fromfile(f,dtype=np.float32).reshape(-1,3) for f in sorted(glob.glob(f'/tmp/bf/res/g_{d}_{n}_{lam}_*.bin'))]).astype(float)
        t,bf,sq=A[:,0],A[:,1],A[:,2]; dev=np.abs(bf-Psi(d,t)); E=len(bf)
        # W1 between empirical law of BF and law of Psi_d(T), T density d t^{d-1}: quantile coupling
        u=(np.arange(E)+0.5)/E; q_inf=np.sort(Psi(d,(1-u)**(1/d)))
        W1=float(np.mean(np.abs(np.sort(bf)-q_inf)))
        rec=dict(d=d,n=n,lam=lam,E=E,W1=W1,meandev=float(dev.mean()),maxdev=float(dev.max()),
                 ratio=float(dev.max()/np.sqrt(np.log(n)/lam)),meanbf=float(bf.mean()),
                 meanlim=float(-2+4.5*f_lens(d,1.0)),lam_max_sq=float(lam*sq[t>=0.25].max()))
        out.append(rec); print(rec)
        if (d,lam) in [(2,400),(3,200)]:
            np.savez_compressed(f'/tmp/bf/res/global_d{d}_n{n}_lam{lam}.npz',bf=bf.astype(np.float32),n=n,lam=lam)
json.dump(out,open('/tmp/bf/res/global_summary.json','w'),indent=1)
