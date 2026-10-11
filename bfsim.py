"""Exact Balanced Forman curvature of a planted edge in Poisson random geometric graphs.
Geometries: flat R^d (torus, locally), round spheres S^k, hyperbolic plane H^2, S^2 x R.
Only points in B(p,r) U B(q,r) matter for BF of edge {p,q}."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import gamma, betainc
from scipy.integrate import quad

def omega(d): return np.pi**(d/2)/gamma(d/2+1)
def f_lens(d,t): return betainc((d+1)/2, 0.5, 1-np.asarray(t,float)**2/4)
def Psi(d,t): return -2+3*f_lens(d,t)
def kappa(d,t):
    t=np.asarray(t,float); return omega(d-1)/(6*(d+1)*omega(d))*t*(1-t**2/4)**((d+1)/2)
def W_mean(f): return -f*np.sqrt((1-f)/np.pi)
def W_var(f): return 9*f*(1-f)**2 + 4.5*f**2*(1-f) + 0.5*f**2*(1-f)*(1-2/np.pi)

def bf_core(n_p, n_q, n_L, cnt_k, cnt_w):
    """n_p,n_q: # Poisson pts in B(p,r), B(q,r) (planted pts excluded); n_L: # in lens;
    cnt_k[k]: # U_j-points adjacent to k in U_i; cnt_w[w]: # U_i-points adjacent to w in U_j."""
    d_i = 1 + n_p; d_j = 1 + n_q
    dmax = max(d_i, d_j); dmin = min(d_i, d_j)
    if dmin <= 1: return 0.0, 0.0, 0.0
    aff = 2/d_i + 2/d_j - 2
    tri = 2*n_L/dmax + n_L/dmin
    sqi = int(np.count_nonzero(cnt_k)); sqj = int(np.count_nonzero(cnt_w))
    gam = max(int(cnt_k.max()) if len(cnt_k) else 0, int(cnt_w.max()) if len(cnt_w) else 0)
    sq = (sqi+sqj)/(gam*dmax) if gam > 0 else 0.0
    return aff+tri+sq, sq, tri

def _cross_kd(Ui, Uj, rad):
    if len(Ui)==0 or len(Uj)==0:
        return np.zeros(len(Ui),int), np.zeros(len(Uj),int)
    ck = cKDTree(Uj).query_ball_point(Ui, rad, return_length=True)
    cw = cKDTree(Ui).query_ball_point(Uj, rad, return_length=True)
    return np.asarray(ck), np.asarray(cw)

def _cross_brute(Ui, Uj, adj_fn, chunk=2000):
    ck = np.zeros(len(Ui),int); cw = np.zeros(len(Uj),int)
    for a in range(0, len(Ui), chunk):
        A = adj_fn(Ui[a:a+chunk], Uj)   # boolean matrix
        ck[a:a+chunk] = A.sum(1); cw += A.sum(0)
    return ck, cw

# ---------------- flat ----------------
def flat_trial(rng, d, lam, t):
    r = 1.0; n = lam/omega(d); dl = t*r
    lo = np.array([-dl/2-r]+[-r]*(d-1)); hi = -lo
    N = rng.poisson(n*np.prod(hi-lo))
    X = rng.uniform(lo, hi, size=(N,d))
    p = np.zeros(d); p[0] = -dl/2; q = -p
    dp = ((X-p)**2).sum(1) <= r*r; dq = ((X-q)**2).sum(1) <= r*r
    ck, cw = _cross_kd(X[dp&~dq], X[dq&~dp], r)
    return bf_core(dp.sum(), dq.sum(), (dp&dq).sum(), ck, cw)

# ---------------- spheres S^k (radius 1) ----------------
def _sphere_cap_sample(rng, k, amax, N):
    out = np.empty(0)
    while len(out) < N:
        m = int(1.3*(N-len(out)))+50
        a = amax*rng.random(m)**(1/k)
        acc = rng.random(m) < (np.sin(a)/np.where(a>0,a,1))**(k-1)
        out = np.concatenate([out, a[acc]])
    return out[:N]
def _unit(rng, N, k):
    u = rng.standard_normal((N,k)); return u/np.linalg.norm(u,axis=1,keepdims=True)
_capvol_cache = {}
def sphere_capvol(k, amax):
    key=(k,round(amax,12))
    if key not in _capvol_cache:
        Sk1 = 2*np.pi**(k/2)/gamma(k/2)
        _capvol_cache[key] = Sk1*quad(lambda a: np.sin(a)**(k-1), 0, amax)[0]
    return _capvol_cache[key]
def sphere_trial(rng, k, lam, t, r):
    n = lam/(omega(k)*r**k); dl = t*r; amax = r+dl/2
    N = rng.poisson(n*sphere_capvol(k, amax))
    a = _sphere_cap_sample(rng, k, amax, N); u = _unit(rng, N, k)
    X = np.column_stack([np.cos(a), np.sin(a)[:,None]*u])
    p = np.zeros(k+1); p[0]=np.cos(dl/2); p[1]=-np.sin(dl/2)
    q = p.copy(); q[1] = np.sin(dl/2)
    cr = np.cos(r); dp = X@p >= cr; dq = X@q >= cr
    ck, cw = _cross_kd(X[dp&~dq], X[dq&~dp], 2*np.sin(r/2))
    return bf_core(dp.sum(), dq.sum(), (dp&dq).sum(), ck, cw)

# ---------------- hyperbolic plane H^2 (K=-1), hyperboloid model ----------------
def hyp_trial(rng, lam, t, r):
    n = lam/(omega(2)*r**2); dl = t*r; amax = r+dl/2
    N = rng.poisson(n*2*np.pi*(np.cosh(amax)-1))
    a = np.arccosh(1+rng.random(N)*(np.cosh(amax)-1)); th = 2*np.pi*rng.random(N)
    X = np.column_stack([np.cosh(a), np.sinh(a)*np.cos(th), np.sinh(a)*np.sin(th)])
    p = np.array([np.cosh(dl/2), -np.sinh(dl/2), 0.]); q = p.copy(); q[1] = -q[1]
    J = np.array([1.,-1.,-1.]); ch = np.cosh(r)
    dp = (X*J)@p <= ch; dq = (X*J)@q <= ch
    iU = dp&~dq; jU = dq&~dp
    Ui = X[iU]; Uj = X[jU]
    # normal coordinates at m: Euclidean distance there is <= hyperbolic distance (K<=0), so it gives a superset
    Ni = np.column_stack([a[iU]*np.cos(th[iU]), a[iU]*np.sin(th[iU])])
    Nj = np.column_stack([a[jU]*np.cos(th[jU]), a[jU]*np.sin(th[jU])])
    ck = np.zeros(len(Ui),int); cw = np.zeros(len(Uj),int)
    if len(Ui) and len(Uj):
        M = cKDTree(Ni).sparse_distance_matrix(cKDTree(Nj), r, output_type='ndarray')
        I = M['i']; Jx = M['j']
        ok = (Ui[I]*J*Uj[Jx]).sum(1) <= ch
        ck = np.bincount(I[ok], minlength=len(Ui)); cw = np.bincount(Jx[ok], minlength=len(Uj))
    return bf_core(dp.sum(), dq.sum(), (dp&dq).sum(), ck, cw)

# ---------------- S^2 x R (K=1 on S^2 factor) ----------------
def s2xr_trial(rng, lam, t, r, direction):
    n = lam/(omega(3)*r**3); dl = t*r
    if direction == 'R':
        amax = r; slo, shi = -r-dl/2, r+dl/2
        yp = yq = np.array([1.,0.,0.]); sp_, sq_ = -dl/2, dl/2
    else:
        amax = r+dl/2; slo, shi = -r, r
        yp = np.array([np.cos(dl/2), -np.sin(dl/2), 0.]); yq = yp.copy(); yq[1] = -yq[1]
        sp_ = sq_ = 0.0
    N = rng.poisson(n*2*np.pi*(1-np.cos(amax))*(shi-slo))
    a = np.arccos(1-rng.random(N)*(1-np.cos(amax))); th = 2*np.pi*rng.random(N)
    Y = np.column_stack([np.cos(a), np.sin(a)*np.cos(th), np.sin(a)*np.sin(th)])
    S = rng.uniform(slo, shi, N)
    def d2(Y1,S1,y,s): return np.arccos(np.clip(Y1@y,-1,1))**2 + (S1-s)**2
    dp = d2(Y,S,yp,sp_) <= r*r; dq = d2(Y,S,yq,sq_) <= r*r
    iU = dp&~dq; jU = dq&~dp
    Ui = np.column_stack([Y[iU], S[iU]]); Uj = np.column_stack([Y[jU], S[jU]])
    ck = np.zeros(len(Ui),int); cw = np.zeros(len(Uj),int)
    if len(Ui) and len(Uj):
        M = cKDTree(Ui).sparse_distance_matrix(cKDTree(Uj), r, output_type='ndarray')
        I = M['i']; Jx = M['j']
        g2 = np.arccos(np.clip((Ui[I,:3]*Uj[Jx,:3]).sum(1),-1,1))**2 + (Ui[I,3]-Uj[Jx,3])**2
        ok = g2 <= r*r
        ck = np.bincount(I[ok], minlength=len(Ui)); cw = np.bincount(Jx[ok], minlength=len(Uj))
    return bf_core(dp.sum(), dq.sum(), (dp&dq).sum(), ck, cw)
