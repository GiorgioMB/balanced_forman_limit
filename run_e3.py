import numpy as np, sys, os, time, zlib, glob
from e3sim import trial
lam=float(sys.argv[1]); r=float(sys.argv[2]); t=float(sys.argv[3]); a=float(sys.argv[4]); direc=sys.argv[5]; mode=sys.argv[6]
W=int(os.environ['W']); TL=float(os.environ.get('TL','5.5'))
tag=f'l{int(lam)}_r{r}_t{t}_a{a}_{direc}_{mode}'; b=len(glob.glob(f'../results/e3/{tag}_w{W}_b*.npy'))
rng=np.random.default_rng(zlib.crc32(tag.encode())+W*10**6+b*1000)
t0=time.time(); out=[]
while time.time()-t0<TL: out.append(trial(rng,lam,r,t,a,direc,mode))
np.save(f'../results/e3/{tag}_w{W}_b{b}.npy',np.array(out))
