import numpy as np, os, time, glob, sys, zlib
from oq2orc import trial
cfg=sys.argv[1]; W=int(os.environ['W']); TL=float(os.environ.get('TL','4.5'))
geo,lam,r,tg={'flat2':('flat2',300,1.0,None),'S2':('S2',300,0.6,None),'H2':('H2',300,0.6,None),'flat3':('flat3',300,1.0,None),
 'S2xR_R':('S2xR',300,0.6,np.array([0,0,1.])),'S2xR_S':('S2xR',300,0.6,np.array([1.,0,0]))}[cfg]
b=len(glob.glob(f'../results/oq2/{cfg}_w{W}_b*')); rng=np.random.default_rng(zlib.crc32(cfg.encode())+W*10**6+b*1000)
t0=time.time(); out=[]
while time.time()-t0<TL: out.append(trial(rng,geo,lam,r,tg))
np.save(f'../results/oq2/{cfg}_w{W}_b{b}.npy',np.array(out))
