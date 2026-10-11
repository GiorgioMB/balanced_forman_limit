import numpy as np, sys, time, os
from multiprocessing import Pool
from scipy.spatial import cKDTree
sys.path.insert(0,'/tmp/bf'); from bfsim import *
G={}
def init(X,indptr,indices,r):
    G['X']=X; G['ip']=indptr; G['ix']=indices; G['r']=r
def work(edges):
    X=G['X']; ip=G['ip']; ix=G['ix']; r2=G['r']**2
    out=np.empty((len(edges),3),np.float32)
    for e,(i,j) in enumerate(edges):
        Ni=ix[ip[i]:ip[i+1]]; Nj=ix[ip[j]:ip[j+1]]
        tri=np.intersect1d(Ni,Nj,assume_unique=True).size
        Ui=np.setdiff1d(Ni,Nj,assume_unique=True); Ui=Ui[Ui!=j]
        Uj=np.setdiff1d(Nj,Ni,assume_unique=True); Uj=Uj[Uj!=i]
        if Ui.size and Uj.size:
            D=X[Ui][:,None,:]-X[Uj][None,:,:]; D-=np.round(D)
            A=(D*D).sum(2)<=r2; ck=A.sum(1); cw=A.sum(0)
        else:
            ck=np.zeros(Ui.size,int); cw=np.zeros(Uj.size,int)
        # bf_core expects counts excluding the planted pair: d_i = 1 + n_p  ->  n_p = |N_i| - 1
        b,sq,_=bf_core(Ni.size-1, Nj.size-1, tri, ck, cw)
        dd=X[i]-X[j]; dd-=np.round(dd)
        out[e]=(np.sqrt((dd*dd).sum())/G['r'], b, sq)
    return out
if __name__=='__main__':
    d=int(sys.argv[1]); n=int(sys.argv[2]); lam=float(sys.argv[3])
    fn=f'/tmp/bf/res/e2_d{d}_n{n}_lam{int(lam)}.npz'
    if os.path.exists(fn): sys.exit(0)
    rng=np.random.default_rng(int(1e6*d+lam))
    r=(lam/(n*omega(d)))**(1/d); assert r<0.25
    X=rng.random((n,d)); T=cKDTree(X,boxsize=1.0)
    nb=T.query_ball_point(X,r)
    indptr=np.zeros(n+1,np.int64); lists=[]
    for i,l in enumerate(nb):
        a=np.array(sorted(l),np.int64); a=a[a!=i]; lists.append(a); indptr[i+1]=indptr[i]+a.size
    indices=np.concatenate(lists)
    E=np.array([(i,j) for i in range(n) for j in lists[i] if j>i],np.int64)
    t0=time.time()
    chunks=np.array_split(E,64)
    with Pool(4,initializer=init,initargs=(X,indptr,indices,r)) as P:
        res=np.concatenate(P.map(work,chunks))
    np.savez_compressed(fn,t=res[:,0],bf=res[:,1],sq=res[:,2],r=r,n=n,lam=lam,deg=np.diff(indptr))
    print(fn,'edges',len(E),'time',round(time.time()-t0,1),flush=True)
