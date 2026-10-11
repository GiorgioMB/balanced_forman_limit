import numpy as np, sys, os, time, zlib, glob, e2lib as E
geo=sys.argv[1]; lam=float(sys.argv[2]); r=float(sys.argv[3]); W=int(os.environ['W']); TL=float(os.environ.get('TL','5.5'))
tag=f'{geo}_l{int(lam)}_r{r:.4f}'; b=len(glob.glob(f'../results/e2/{tag}_w{W}_b*.npy'))
t=time.time(); res=[]; k=0
while time.time()-t<TL:
    res.append(E.rep(geo,lam,r,zlib.crc32(tag.encode())+W*10**7+b*10**4+k)); k+=1
np.save(f'../results/e2/{tag}_w{W}_b{b}.npy',np.array(res))
