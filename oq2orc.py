"""Per-edge comparison of BF and data-cloud Ollivier curvature (uniform measures on 1-hop neighbourhoods,
geodesic ground cost; kappa = 1 - W1/d(x,j)) on a selected edge of a planted vertex."""
import numpy as np, ot
from vsim import sample_local, adjacency, bf_vertex_edges
from bfsim import Psi
def geod(geo,loc,I,J,r):
    E=loc['emb']
    if geo.startswith('flat'): return np.linalg.norm(E[I][:,None]-E[J][None],axis=2)
    if geo in ('S2','S3'): return np.arccos(np.clip(E[I]@E[J].T,-1,1))
    if geo=='H2':
        T=loc['tan']; a=np.linalg.norm(T,axis=1); u=T/np.where(a>0,a,1)[:,None]
        H=np.column_stack([np.cosh(a),np.sinh(a)[:,None]*u])
        G=H[I][:,None,0]*H[J][None,:,0]-(H[I][:,None,1:]*H[J][None,:,1:]).sum(2); return np.arccosh(np.maximum(G,1))
    if geo=='S2xR':
        Y=E[:,:3]; s=E[:,3]; return np.sqrt(np.arccos(np.clip(Y[I]@Y[J].T,-1,1))**2+(s[I][:,None]-s[J][None])**2)
def trial(rng,geo,lam,r,target):
    loc=sample_local(rng,geo,lam,r); A=adjacency(loc)
    nb=A.indices[A.indptr[0]:A.indptr[1]]; T=loc['tan'][nb]; tt=np.linalg.norm(T,axis=1)/r
    c=np.where((tt>=0.7)&(tt<=0.8))[0]
    if target is None: j=c[rng.integers(len(c))]
    else: j=c[np.argmax(np.abs(T[c]@target)/np.linalg.norm(T[c],axis=1))]
    jj=nb[j]; bf=bf_vertex_edges(A,0,np.array([jj]))[0,0]
    Nx=A.indices[A.indptr[0]:A.indptr[1]]; Nj=A.indices[A.indptr[jj]:A.indptr[jj+1]]
    M=geod(geo,loc,Nx,Nj,r); d=geod(geo,loc,np.array([0]),np.array([jj]),r)[0,0]
    W=ot.emd2(np.full(len(Nx),1/len(Nx)),np.full(len(Nj),1/len(Nj)),M)
    return tt[j], bf-Psi(loc['d'],tt[j]), 1-W/d
