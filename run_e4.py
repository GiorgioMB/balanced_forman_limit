import numpy as np, sys, os, time, zlib, glob
from e4sim import trial
lam=float(sys.argv[1]); r=float(sys.argv[2]); t=float(sys.argv[3]); R=float(sys.argv[4]); d=sys.argv[5]; amb=sys.argv[6]=='amb'
W=int(os.environ['W']); TL=float(os.environ.get('TL','4.5'))
tag=f'l{int(lam)}_r{r}_t{t}_R{R}_{d}_{sys.argv[6]}'; b=len(glob.glob(f'../results/e4/{tag}_w{W}_b*.npy'))
rng=np.random.default_rng(zlib.crc32(tag.encode())+W*10**6+b*1000); t0=time.time(); out=[]
while time.time()-t0<TL: out.append(trial(rng,lam,r,t,R,d,amb))
np.save(f'../results/e4/{tag}_w{W}_b{b}.npy',np.array(out))
