"""Two rooms on a torus [0,2(1+L)) x [0,1): room A x in [0,1), room B x in [1+L, 2+L); two walls of
thickness L with a corridor of width w centred at y=1/2. Uniform points, Euclidean (periodic) RGG."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
Lw=0.3; w=0.1; r=0.05; lam=30.0
PX=2*(1+Lw)
def in_domain(X):
    x=X[:,0]%PX; y=X[:,1]
    wall=((x>=1)&(x<1+Lw))|((x>=2+Lw)&(x<PX))
    return ~wall | (np.abs(y-0.5)<=w/2)
def build(seed):
    rng=np.random.default_rng(seed); dens=lam/(np.pi*r*r)
    N=rng.poisson(dens*PX); X=np.column_stack([rng.random(N)*PX,rng.random(N)])
    X=X[in_domain(X)]
    T=cKDTree(X,boxsize=[PX,1.0]); P=T.query_pairs(r,output_type='ndarray'); n=len(X)
    A=csr_matrix((np.ones(2*len(P)),(np.r_[P[:,0],P[:,1]],np.r_[P[:,1],P[:,0]])),shape=(n,n)); A.sort_indices()
    return X,P,A
def region(X):
    x=X[:,0]%PX
    return np.where(x<1,0,np.where(x<1+Lw,2,np.where(x<2+Lw,1,2)))   # 0=A,1=B,2=corridor
def edge_len(X,P):
    d=X[P[:,0]]-X[P[:,1]]; d[:,0]-=PX*np.round(d[:,0]/PX); d[:,1]-=np.round(d[:,1]); return np.linalg.norm(d,axis=1)
