"""Planted edges in the flat Palm model (torus, locally). usage: run_flat.py d lam t N chunk"""
import numpy as np, sys
sys.path.insert(0,'/tmp/bf'); from bfsim import flat_trial
d=int(sys.argv[1]); lam=int(sys.argv[2]); t=float(sys.argv[3]); N=int(sys.argv[4]); ch=int(sys.argv[5])
rng=np.random.default_rng([d,lam,int(round(t*100)),ch,20261009])
A=np.array([flat_trial(rng,d,lam,t)[:2] for _ in range(N)])
np.save(f'/tmp/bf/res/flat/f_{d}_{lam}_{t}_{ch}.npy',A)
