"""Variance of F = sum_e (BF(e) - m_lam(t_e)) on the flat torus (Poisson model), as a function of lambda.
Per graph we store binned sums of BF-Psi and counts (20 bins in t) so that m_lam can be estimated by pooling."""
import numpy as np, os, sys, time, glob, zlib
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from allbf import bf_edges
from bfsim import Psi
lam=float(sys.argv[1]); n0=int(sys.argv[2]); W=int(os.environ['W']); TL=float(os.environ.get('TL','2.5'))
tag=f'var_l{int(lam)}_n{n0}'; b=len(glob.glob(f'../results/var/{tag}_w{W}_b*'))
bins=np.linspace(0,1,21); t0=time.time(); out=[]; k=0
while time.time()-t0<TL:
    rng=np.random.default_rng(zlib.crc32(tag.encode())+W*10**6+b*100+k); k+=1
    n=rng.poisson(n0); r=np.sqrt(lam/(n0*np.pi)); X=rng.random((n,2))
    P=cKDTree(X,boxsize=1.0).query_pairs(r,output_type='ndarray'); dv=X[P[:,0]]-X[P[:,1]]; dv-=np.round(dv); t=np.linalg.norm(dv,axis=1)/r
    A=csr_matrix((np.ones(2*len(P)),(np.r_[P[:,0],P[:,1]],np.r_[P[:,1],P[:,0]])),shape=(n,n)); A.sort_indices()
    y=bf_edges(A,P)[:,0]-Psi(2,t); j=np.minimum(np.digitize(t,bins)-1,19)
    out.append(np.r_[n,len(P),np.bincount(j,weights=y,minlength=20),np.bincount(j,minlength=20)])
np.save(f'../results/var/{tag}_w{W}_b{b}.npy',np.array(out))
