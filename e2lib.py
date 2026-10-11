import numpy as np, os
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
import bolza as B
GFILE='../results/bolza_group_r0.6.npy'
def group():
    if not os.path.exists(GFILE): np.save(GFILE,np.array(B.group_elements(0.6,4)))
    return np.load(GFILE)
def edges(geo,X,r):
    n=len(X)
    if geo=='T2':
        P=cKDTree(X,boxsize=1.0).query_pairs(r,output_type='ndarray'); return P
    if geo=='S2':
        return cKDTree(X).query_pairs(2*np.sin(r/2),output_type='ndarray')
    if geo=='bolza':
        G=group(); imgs=[];own=[]
        for g in G:
            imgs.append((g@X.T).T); own.append(np.arange(n))
        Y=np.concatenate(imgs); own=np.concatenate(own)
        TQ=cKDTree(B.poincare(Y)); rad=np.sinh(r/2)
        L=TQ.query_ball_point(B.poincare(X),rad*1.0000001)
        I=np.repeat(np.arange(n),[len(l) for l in L]); Jy=np.concatenate([np.array(l,int) for l in L])
        ch=(X[I]*np.array([1.,-1.,-1.])*Y[Jy]).sum(1)          # = cosh d
        ok=(ch<=np.cosh(r))&(own[Jy]!=I)
        I,Jo=I[ok],own[Jy[ok]]
        P=np.unique(np.sort(np.column_stack([I,Jo]),axis=1),axis=0); return P
def sample(geo,rng,n):
    if geo=='T2': return rng.random((n,2))
    if geo=='S2':
        u=rng.standard_normal((n,3)); return u/np.linalg.norm(u,axis=1,keepdims=True)
    if geo=='bolza': return B.sample(rng,n)
AREA={'T2':1.0,'S2':4*np.pi,'bolza':4*np.pi}
def clustering(P,n):
    A=csr_matrix((np.ones(2*len(P)),(np.r_[P[:,0],P[:,1]],np.r_[P[:,1],P[:,0]])),shape=(n,n))
    deg=np.asarray(A.sum(1)).ravel()
    tri=np.asarray((A@A).multiply(A).sum(1)).ravel()/2
    ok=deg>=2; c=np.zeros(n); c[ok]=tri[ok]/(deg[ok]*(deg[ok]-1)/2)
    return c[ok].mean(), ok.mean(), deg.mean()
def rep(geo,lam,r,seed):
    rng=np.random.default_rng(seed); n=rng.poisson(lam*AREA[geo]/(np.pi*r*r))
    X=sample(geo,rng,n); P=edges(geo,X,r); return clustering(P,n)
