"""Planted edge in R^2 with Poisson intensity n*rho(x), rho(x)=1+a*cos(2*pi*x1), midpoint at x1=0.25.
mode 'fixed': edge iff |x-y|<=r ; mode 'adapt': edge iff |x-y| <= r*(rho(x)rho(y))^(-1/4) (oracle)."""
import numpy as np
from scipy.spatial import cKDTree
from bfsim import bf_core
def rho(X,a): return 1+a*np.cos(2*np.pi*X[:,0])
def trial(rng,lam,r,t,a,direc,mode):
    n=lam/(np.pi*r*r); m=np.array([0.25,0.0]); e=np.array([1.,0.]) if direc=='par' else np.array([0.,1.])
    rmax=r*(1-a)**(-0.5) if mode=='adapt' else r
    if mode=='adapt':
        # choose p,q so that d(p,q) = t * r_pq exactly (solve by fixed point)
        dl=t*r
        for _ in range(30):
            p=m-dl/2*e; q=m+dl/2*e; rpq=r*(rho(p[None],a)[0]*rho(q[None],a)[0])**-0.25; dl=t*rpq
    else:
        dl=t*r; p=m-dl/2*e; q=m+dl/2*e
    L=dl/2+rmax; lo=m-L; hi=m+L
    N=rng.poisson(n*(1+a)*(2*L)**2); X=rng.uniform(lo,hi,(N,2))
    X=X[rng.random(N)<rho(X,a)/(1+a)]
    rx=rho(X,a)
    if mode=='adapt':
        rp=rho(p[None],a)[0]; rq=rho(q[None],a)[0]
        dp=np.linalg.norm(X-p,axis=1)<=r*(rx*rp)**-0.25; dq=np.linalg.norm(X-q,axis=1)<=r*(rx*rq)**-0.25
    else:
        dp=np.linalg.norm(X-p,axis=1)<=r; dq=np.linalg.norm(X-q,axis=1)<=r
    iU=dp&~dq; jU=dq&~dp; Ui=X[iU]; Uj=X[jU]
    ck=np.zeros(len(Ui),int); cw=np.zeros(len(Uj),int)
    if len(Ui) and len(Uj):
        M=cKDTree(Ui).sparse_distance_matrix(cKDTree(Uj),rmax,output_type='ndarray'); I=M['i']; J=M['j']
        if mode=='adapt': ok=M['v']<=r*(rx[iU][I]*rx[jU][J])**-0.25
        else: ok=M['v']<=r
        ck=np.bincount(I[ok],minlength=len(Ui)); cw=np.bincount(J[ok],minlength=len(Uj))
    return bf_core(dp.sum(),dq.sum(),(dp&dq).sum(),ck,cw)[0]
