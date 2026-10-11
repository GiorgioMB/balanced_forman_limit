import numpy as np, os, time
from bfsim import flat_trial
W=int(os.environ['W']); res={}
rng=np.random.default_rng(777+W); t0=time.time()
for lam in [2,5,10,20,50,200]:
    res[str(lam)]=np.array([flat_trial(rng,2,lam,0.75)[0] for _ in range(6000 if lam<=50 else 2000)])
np.savez(f'../results/e5/fixedlam_w{W}.npz',**res)
