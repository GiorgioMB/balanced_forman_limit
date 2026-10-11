"""Joint law of BF on two edges e1=(x,j), e2=(x,k) sharing vertex x (flat R^2, unit r).
Theory: Gaussian limit of cell counts (Cov N(A),N(B) = n Vol(A cap B)) pushed through the linearisation."""
import numpy as np
from scipy.spatial import cKDTree
from bfsim import bf_core, f_lens
def pts(t,th): x=np.zeros(2); j=np.array([t,0.]); k=t*np.array([np.cos(th),np.sin(th)]); return x,j,k
def edge_bf(X,p,q,r=1.0):
    dp=((X-p)**2).sum(1)<=r*r; dq=((X-q)**2).sum(1)<=r*r; Ui=X[dp&~dq]; Uj=X[dq&~dp]
    if len(Ui) and len(Uj):
        ck=np.asarray(cKDTree(Uj).query_ball_point(Ui,r,return_length=True)); cw=np.asarray(cKDTree(Ui).query_ball_point(Uj,r,return_length=True))
    else: ck=np.zeros(len(Ui),int); cw=np.zeros(len(Uj),int)
    return bf_core(dp.sum(),dq.sum(),(dp&dq).sum(),ck,cw)[0]
def trial(rng,lam,t,th):
    x,j,k=pts(t,th); n=lam/np.pi; lo=np.minimum.reduce([x,j,k])-1; hi=np.maximum.reduce([x,j,k])+1
    N=rng.poisson(n*np.prod(hi-lo)); X=rng.uniform(lo,hi,(N,2))
    return edge_bf(np.vstack([X,k]),x,j), edge_bf(np.vstack([X,j]),x,k)
def theory(t,th,M=400000,G=400000,seed=0):
    rng=np.random.default_rng(seed); x,j,k=pts(t,th)
    U=rng.uniform(-1-t,1+t,(M,2)); V=(2+2*t)**2/np.pi   # volume factor (normalised by omega_2 = pi)
    inn=lambda c: ((U-c)**2).sum(1)<=1
    Bx,Bj,Bk=inn(x),inn(j),inn(k); sets=[Bx&Bj,Bx,Bj,Bx&Bk,Bk]   # L1,Bx,Bj,L2,Bk
    S=np.array([[V*np.mean(a&b) for b in sets] for a in sets])
    Z=np.random.default_rng(seed+1).multivariate_normal(np.zeros(5),S,size=G)
    f=float(f_lens(2,t))
    W1=3*Z[:,0]-1.5*f*(Z[:,1]+Z[:,2])-0.5*f*np.abs(Z[:,1]-Z[:,2])
    W2=3*Z[:,3]-1.5*f*(Z[:,1]+Z[:,4])-0.5*f*np.abs(Z[:,1]-Z[:,4])
    return float(np.corrcoef(W1,W2)[0,1]), float(W1.var())
