"""Global mean of centred BF on full RGGs: unit sphere S^2 (geodesic) vs flat torus with the same lambda."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from allbf import bf_edges
from bfsim import Psi
def graph(geo,lam,r,seed):
    rng=np.random.default_rng(seed)
    if geo=='S2':
        n=rng.poisson(lam*4/(r*r)); u=rng.standard_normal((n,3)); X=u/np.linalg.norm(u,axis=1,keepdims=True)
        P=cKDTree(X).query_pairs(2*np.sin(r/2),output_type='ndarray')
        t=np.arccos(np.clip((X[P[:,0]]*X[P[:,1]]).sum(1),-1,1))/r
    else:  # torus with the same n: side 1, radius rT so that n*pi*rT^2 = lam
        n=rng.poisson(lam*4/(r*r)); rT=np.sqrt(lam/(n*np.pi)); X=rng.random((n,2))
        P=cKDTree(X,boxsize=1.0).query_pairs(rT,output_type='ndarray'); dv=X[P[:,0]]-X[P[:,1]]; dv-=np.round(dv); t=np.linalg.norm(dv,axis=1)/rT
    A=csr_matrix((np.ones(2*len(P)),(np.r_[P[:,0],P[:,1]],np.r_[P[:,1],P[:,0]])),shape=(n,n)); A.sort_indices()
    return n,P,A,t
def rep(geo,lam,r,seed):
    n,P,A,t=graph(geo,lam,r,seed); R=bf_edges(A,P)
    return n,len(P),float(np.mean(R[:,0]-Psi(2,t)))
