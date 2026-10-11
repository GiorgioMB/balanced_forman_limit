import numpy as np, os, time, glob
from oq1pair import trial
W=int(os.environ['W']); TL=float(os.environ.get('TL','4.5')); D='../results/oq1'
ths=[10,30,60,90,120,150,180]
cnt={th:sum(len(np.load(f)) for f in glob.glob(f'{D}/pair_th{th}_w*')) for th in ths}
todo=[th for th in ths if cnt[th]<8000]
if todo:
    th=todo[W%len(todo)]; b=len(glob.glob(f'{D}/pair_th{th}_w{W}_b*'))
    rng=np.random.default_rng(10**6*th+1000*W+b); t0=time.time(); out=[]
    while time.time()-t0<TL: out.append(trial(rng,1000,0.75,np.deg2rad(th)))
    np.save(f'{D}/pair_th{th}_w{W}_b{b}.npy',np.array(out))
