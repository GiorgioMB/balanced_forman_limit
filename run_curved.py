"""Planted edges on curved spaces and flat references (r=0.5, lambda=1500). usage: run_curved.py geo t N chunk dummy"""
import numpy as np, sys
sys.path.insert(0,'/tmp/bf'); from bfsim import flat_trial, sphere_trial, hyp_trial, s2xr_trial
geo=sys.argv[1]; t=float(sys.argv[2]); N=int(sys.argv[3]); ch=int(sys.argv[4]); lam=1500; r=0.5
F={'S2':lambda g:sphere_trial(g,2,lam,t,r),'S3':lambda g:sphere_trial(g,3,lam,t,r),'H2':lambda g:hyp_trial(g,lam,t,r),
   'S2xR_S':lambda g:s2xr_trial(g,lam,t,r,'S'),'S2xR_R':lambda g:s2xr_trial(g,lam,t,r,'R'),
   'flat2':lambda g:flat_trial(g,2,lam,t),'flat3':lambda g:flat_trial(g,3,lam,t)}[geo]
rng=np.random.default_rng([abs(hash(geo))%(2**31) if False else sum(map(ord,geo)),int(round(t*100)),ch,20261010])
A=np.array([F(rng)[:2] for _ in range(N)])
np.save(f'/tmp/bf/res/curved/c_{geo}_{t}_{N}_{ch}.npy',A)
