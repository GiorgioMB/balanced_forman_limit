import numpy as np, os, time, glob
from e6lib import build
from allbf import bf_edges
W=int(os.environ['W']); TL=float(os.environ.get('TL','4.5')); D='../results/e6'
X,P,A=build(2026); mine=np.arange(W,len(P),4)
done=sorted(glob.glob(f'{D}/bf_w{W}_s*.npz')); start=sum(len(np.load(f)['idx']) for f in done)
t0=time.time(); res=[]; idx=[]; k=start
while k<len(mine) and time.time()-t0<TL:
    sl=mine[k:k+300]; res.append(bf_edges(A,P[sl])); idx.append(sl); k+=len(sl)
if res: R=np.concatenate(res); np.savez(f'{D}/bf_w{W}_s{len(done)}.npz',idx=np.concatenate(idx),bf=R[:,0],sq=R[:,1],E=len(P))
