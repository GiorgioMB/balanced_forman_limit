"""Bolza surface (genus 2, curvature -1, area 4*pi) as the regular octagon with angles pi/4 in
the hyperboloid model, opposite sides paired by translations of length 2a, cosh a = 1+sqrt2."""
import numpy as np
A_HALF=np.arccosh(1+np.sqrt(2)); R_CIRC=np.arccosh((1+np.sqrt(2))**2)
def boost(theta,s):
    c,sn=np.cos(theta),np.sin(theta)
    Rt=np.array([[1,0,0],[0,c,-sn],[0,sn,c]])
    B=np.array([[np.cosh(s),np.sinh(s),0],[np.sinh(s),np.cosh(s),0],[0,0,1]])
    return Rt@B@Rt.T
GENS=[boost(k*np.pi/4,2*A_HALF) for k in range(8)]
J=np.diag([1.,-1.,-1.])
def hdist0(M):   # distance from origin o=(1,0,0) to M o
    return np.arccosh(max(M[0,0],1.0))
def group_elements(r,maxlen=5):
    """elements g whose tile g(Oct) lies within distance r of Oct (incl. identity)."""
    keep={(): np.eye(3)}; frontier=[np.eye(3)]; seen=[np.eye(3)]
    lim=2*R_CIRC+r
    for L in range(maxlen):
        new=[]
        for M in frontier:
            for G in GENS:
                N=M@G
                if hdist0(N)>lim+2*A_HALF: continue
                if any(np.allclose(N,S,atol=1e-7) for S in seen): continue
                seen.append(N); new.append(N)
        frontier=new
    return [M for M in seen if hdist0(M)<=lim]
def in_octagon(X):
    """X: hyperboloid points (N,3). Klein coords inside regular octagon (vertices at angles (k+1/2)pi/4)."""
    K=X[:,1:]/X[:,:1]; ap=np.tanh(A_HALF)
    ok=np.ones(len(X),bool)
    for k in range(8):
        th=k*np.pi/4; ok&=(K[:,0]*np.cos(th)+K[:,1]*np.sin(th))<=ap
    return ok
def sample(rng,N):
    out=[]
    while sum(len(o) for o in out)<N:
        m=2*N
        a=np.arccosh(1+rng.random(m)*(np.cosh(R_CIRC)-1)); th=2*np.pi*rng.random(m)
        X=np.column_stack([np.cosh(a),np.sinh(a)*np.cos(th),np.sinh(a)*np.sin(th)])
        out.append(X[in_octagon(X)])
    return np.concatenate(out)[:N]
def poincare(X): return X[:,1:]/(1+X[:,:1])
