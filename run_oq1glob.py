import numpy as np, os, time, glob, sys, zlib
from oq1glob import rep
geo=sys.argv[1]; lam=float(sys.argv[2]); r=float(sys.argv[3]); W=int(os.environ['W']); TL=float(os.environ.get('TL','1.5'))
tag=f'glob_{geo}_l{int(lam)}_r{r}'; b=len(glob.glob(f'../results/oq1/{tag}_w{W}_b*'))
t0=time.time(); out=[]; k=0
while time.time()-t0<TL:
    out.append(rep(geo,lam,r,zlib.crc32(tag.encode())+W*10**6+b*100+k)); k+=1
np.save(f'../results/oq1/{tag}_w{W}_b{b}.npy',np.array(out))
