import numpy as np, sys, time, os
from multiprocessing import Pool
sys.path.insert(0,'/tmp/bf'); from bfsim import *
def job(args):
    d, lam, t, ntr, seed = args
    rng = np.random.default_rng(seed)
    return (d, lam, t, np.array([flat_trial(rng, d, lam, t) for _ in range(ntr)]))
if __name__ == '__main__':
    D = int(sys.argv[1]); lams=[int(x) for x in sys.argv[2].split(',')]
    tasks=[]; seed=1000*D
    for t in [0.25,0.5,0.75,1.0]:
        for lam in lams:
            ntr = 4000 if lam<=3200 else (2000 if lam==6400 else 1000)
            for b in range(4):
                seed+=1; tasks.append((D,lam,t,ntr//4,seed+lam))
    t0=time.time(); res={}
    with Pool(4) as P:
        for d,lam,t,out in P.imap_unordered(job, tasks, chunksize=1):
            res.setdefault(f"{d}_{lam}_{t}", []).append(out)
    fn=f'/tmp/bf/res/e1_d{D}_{sys.argv[2].replace(",","-")}.npz'
    np.savez_compressed(fn, **{k: np.concatenate(v) for k,v in res.items()})
    print('saved',fn, time.time()-t0)
