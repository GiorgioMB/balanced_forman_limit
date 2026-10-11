# E1: Palm-vertex simulations. usage: python run_v1.py geo lam r nchunks [t0] [m]
import numpy as np, sys, os, time, zlib
from multiprocessing import Pool
from vsim import sample_local, adjacency, bf_vertex_edges
geo=sys.argv[1]; lam=float(sys.argv[2]); r=float(sys.argv[3]); nch=int(sys.argv[4])
t0=float(sys.argv[5]) if len(sys.argv)>5 else 0.5; m=int(sys.argv[6]) if len(sys.argv)>6 else 200
VPC=int(os.environ.get('VPC','25'))
D='../results/e1'
def chunk(c):
    fn=f'{D}/{geo}_l{int(lam)}_r{r}_c{c}.npz'
    if os.path.exists(fn): return fn
    rng=np.random.default_rng(zlib.crc32(f'{geo}{lam}{r}{c}'.encode()))
    T=[];B=[];S=[];V=[];VID=[];DEG=[]
    for v in range(VPC):
        loc=sample_local(rng,geo,lam,r); A=adjacency(loc)
        nb=A.indices[A.indptr[0]:A.indptr[1]]
        tn=loc['tan'][nb]; tt=np.linalg.norm(tn,axis=1)/r
        cand=np.where(tt>=t0)[0]; pick=rng.choice(cand,min(m,len(cand)),replace=False)
        out=bf_vertex_edges(A,0,nb[pick])
        T.append(tt[pick]); B.append(out[:,0]); S.append(out[:,1]); V.append(tn[pick]/np.linalg.norm(tn[pick],axis=1,keepdims=True))
        VID.append(np.full(len(pick),c*VPC+v)); DEG.append(len(nb))
    np.savez_compressed(fn,t=np.concatenate(T),bf=np.concatenate(B),sq=np.concatenate(S),v=np.concatenate(V),vid=np.concatenate(VID),deg=np.array(DEG))
    return fn
if __name__=='__main__':
    t=time.time(); W=int(os.environ.get('W','0')); NW=int(os.environ.get('NW','1')); TL=float(os.environ.get('TL','95'))
    for c in range(W,nch,NW):
        if time.time()-t>TL: break
        chunk(c)
