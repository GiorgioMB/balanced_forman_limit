"""Planted edges on a cylinder of radius R (intrinsically flat), graph built with AMBIENT (chord)
distances in R^3.  Intrinsic coords (u,z), u = R*theta.  Edge direction 'circ' (along u) or 'axial'."""
import numpy as np
from scipy.spatial import cKDTree
from bfsim import bf_core
def emb(U,R): return np.column_stack([R*np.cos(U[:,0]/R),R*np.sin(U[:,0]/R),U[:,1]])
def chord_u(du,R): return 2*R*np.sin(np.abs(du)/(2*R))
def trial(rng,lam,r,t,R,direc,ambient=True):
    n=lam/(np.pi*r*r)       # intensity per unit intrinsic area; lam = n*pi*r^2 (flat normalisation)
    if direc=='axial':
        p=np.array([0.,-t*r/2]); q=np.array([0.,t*r/2])
    else:   # ambient distance t*r along the circle: solve chord(du)=t r
        du=2*R*np.arcsin(t*r/(2*R)) if ambient else t*r
        p=np.array([-du/2,0.]); q=np.array([du/2,0.])
    L=r*1.2+abs(q-p).max()/2
    N=rng.poisson(n*(2*L)**2); X=rng.uniform(-L,L,(N,2))
    E=emb(X,R); Ep=emb(p[None],R)[0]; Eq=emb(q[None],R)[0]
    if ambient:
        dp=np.linalg.norm(E-Ep,axis=1)<=r; dq=np.linalg.norm(E-Eq,axis=1)<=r; Ui=E[dp&~dq]; Uj=E[dq&~dp]
    else:
        dp=np.linalg.norm(X-p,axis=1)<=r; dq=np.linalg.norm(X-q,axis=1)<=r; Ui=X[dp&~dq]; Uj=X[dq&~dp]
    if len(Ui) and len(Uj):
        ck=np.asarray(cKDTree(Uj).query_ball_point(Ui,r,return_length=True)); cw=np.asarray(cKDTree(Ui).query_ball_point(Uj,r,return_length=True))
    else: ck=np.zeros(len(Ui),int); cw=np.zeros(len(Uj),int)
    return bf_core(dp.sum(),dq.sum(),(dp&dq).sum(),ck,cw)[0]
def lens_ratio(r,t,R,direc,M=3000):
    """exact ambient lens ratio by grid integration in intrinsic coords"""
    if direc=='axial': p=np.array([0.,-t*r/2]); q=np.array([0.,t*r/2])
    else:
        du=2*R*np.arcsin(t*r/(2*R)); p=np.array([-du/2,0.]); q=np.array([du/2,0.])
    L=r*1.2+abs(q-p).max()/2; g=(np.arange(M)+0.5)/M*2*L-L
    U,Z=np.meshgrid(g,g,indexing='ij')
    def inb(c): return chord_u(U-c[0],R)**2+(Z-c[1])**2<=r*r
    Bp=inb(p); return (Bp&inb(q)).sum()/Bp.sum()
