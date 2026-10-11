import numpy as np, sys, json
sys.path.insert(0,'/tmp/bf'); from bfsim import omega, f_lens
from scipy.spatial import cKDTree
from scipy.optimize import minimize
rng=np.random.default_rng(7)
def sample_lune(d,t,side,M):
    p=np.zeros(d); p[0]=-t/2; q=-p
    c = q if side=='j' else p; o = p if side=='j' else q
    out=[]
    while sum(len(x) for x in out)<M:
        X=c+rng.uniform(-1,1,(2*M,d)); X=X[(((X-c)**2).sum(1)<=1)&(((X-o)**2).sum(1)>1)]; out.append(X)
    return np.concatenate(out)[:M]
res={}
for d in [2,3]:
    for t in [0.25,0.5,0.75,1.0]:
        f=float(f_lens(d,t)); VU=omega(d)*(1-f)
        Uj=sample_lune(d,t,'j',400000 if d==2 else 600000); Tj=cKDTree(Uj)
        Ui=sample_lune(d,t,'i',40000)
        cnt=np.asarray(Tj.query_ball_point(Ui,1.0,return_length=True))
        v=VU*cnt/len(Uj)
        u=(1-f)*np.mean(cnt>0)
        # refine the max of v by local search from the best candidates (restricted to U_i)
        p=np.zeros(d); p[0]=-t/2; q=-p
        def negv(z):
            if ((z-p)**2).sum()>1 or ((z-q)**2).sum()<=1: return 0.0
            return -VU*Tj.query_ball_point(z,1.0,return_length=True)/len(Uj)
        best=Ui[np.argsort(-v)[:5]]
        g=max(v.max(), max(-minimize(negv,z0,method='Nelder-Mead',options={'xatol':1e-4,'fatol':1e-6}).fun for z0 in best))
        g/=omega(d)
        res[f"{d}_{t}"]=dict(u=u,g=g,lim=2*u/g)
        print(f"d={d} t={t}: u={u:.4f} (1-f={1-f:.4f})  g={g:.4f}  2u/g={2*u/g:.3f}",flush=True)
json.dump(res,open('/tmp/bf/res/sq_const.json','w'),indent=1)
