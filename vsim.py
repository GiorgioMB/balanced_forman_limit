"""Local (Palm) geometric graphs around a planted vertex x on several geometries.

For a planted vertex x (index 0) we sample a Poisson process of intensity n (w.r.t. Riemannian
volume, lambda = n*omega_d*r^d) on the geodesic ball B(x, 2r).  The geometric graph on these
points contains every vertex, edge, triangle and 4-cycle needed to compute the exact BF
curvature of all edges {x, j}.  Also returns tangent coordinates of each point at x.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.special import gamma
from scipy.integrate import quad

def omega(d): return np.pi**(d/2)/gamma(d/2+1)

def _unit(rng, N, k):
    u = rng.standard_normal((N, k)); return u/np.linalg.norm(u, axis=1, keepdims=True)

def _cap_alpha(rng, k, amax, N):
    """polar angle on S^k with density prop. to sin^{k-1} on [0, amax] (rejection)."""
    out = np.empty(0)
    while len(out) < N:
        m = int(1.3*(N-len(out)))+50
        a = amax*rng.random(m)**(1/k)
        acc = rng.random(m) < (np.sin(a)/np.where(a > 0, a, 1))**(k-1)
        out = np.concatenate([out, a[acc]])
    return out[:N]

_cache = {}
def _capvol(k, amax):
    key = (k, round(amax, 12))
    if key not in _cache:
        _cache[key] = 2*np.pi**(k/2)/gamma(k/2)*quad(lambda a: np.sin(a)**(k-1), 0, amax)[0]
    return _cache[key]

def sample_local(rng, geo, lam, r, R=None):
    """Return dict with: emb (embedding for kd-tree superset query), rad (query radius),
    exact (function(I,J)->bool adjacency on candidate pairs), tan (tangent coords at x), d."""
    R = 2*r if R is None else R
    if geo.startswith('flat'):
        d = int(geo[4:]); n = lam/(omega(d)*r**d)
        N = rng.poisson(n*omega(d)*R**d)
        X = _unit(rng, N, d)*(R*rng.random(N)**(1/d))[:, None]
        X = np.vstack([np.zeros(d), X])
        return dict(d=d, emb=X, rad=r, exact=None, tan=X)
    if geo in ('S2', 'S3'):
        k = int(geo[1]); n = lam/(omega(k)*r**k)
        N = rng.poisson(n*_capvol(k, R))
        a = _cap_alpha(rng, k, R, N); u = _unit(rng, N, k)
        a = np.concatenate([[0.0], a]); u = np.vstack([np.eye(k)[0], u])
        Y = np.column_stack([np.cos(a), np.sin(a)[:, None]*u])
        return dict(d=k, emb=Y, rad=2*np.sin(r/2), exact=None, tan=a[:, None]*u)
    if geo == 'H2':
        n = lam/(omega(2)*r**2)
        N = rng.poisson(n*2*np.pi*(np.cosh(R)-1))
        a = np.arccosh(1+rng.random(N)*(np.cosh(R)-1)); th = 2*np.pi*rng.random(N)
        a = np.concatenate([[0.0], a]); th = np.concatenate([[0.0], th])
        H = np.column_stack([np.cosh(a), np.sinh(a)*np.cos(th), np.sinh(a)*np.sin(th)])
        T = np.column_stack([a*np.cos(th), a*np.sin(th)])   # normal coords: 1-Lipschitz superset
        J = np.array([1., -1., -1.]); ch = np.cosh(r)
        ex = lambda I, Jj: ((H[I]*J)*H[Jj]).sum(1) <= ch
        return dict(d=2, emb=T, rad=r, exact=ex, tan=T)
    if geo == 'S2xR':
        n = lam/(omega(3)*r**3)
        N = rng.poisson(n*2*np.pi*(1-np.cos(R))*2*R)
        a = np.arccos(1-rng.random(N)*(1-np.cos(R))); th = 2*np.pi*rng.random(N)
        s = rng.uniform(-R, R, N)
        keep = a**2+s**2 <= R**2
        a, th, s = a[keep], th[keep], s[keep]
        a = np.concatenate([[0.0], a]); th = np.concatenate([[0.0], th]); s = np.concatenate([[0.0], s])
        Y = np.column_stack([np.cos(a), np.sin(a)*np.cos(th), np.sin(a)*np.sin(th)])
        E = np.column_stack([Y, s])
        ex = lambda I, Jj: np.arccos(np.clip((Y[I]*Y[Jj]).sum(1), -1, 1))**2+(s[I]-s[Jj])**2 <= r*r
        T = np.column_stack([a*np.cos(th), a*np.sin(th), s])
        return dict(d=3, emb=E, rad=r, exact=ex, tan=T)
    raise ValueError(geo)

def adjacency(loc):
    P = cKDTree(loc['emb']).query_pairs(loc['rad'], output_type='ndarray')
    if loc['exact'] is not None and len(P):
        P = P[loc['exact'](P[:, 0], P[:, 1])]
    N = len(loc['emb'])
    I = np.concatenate([P[:, 0], P[:, 1]]); J = np.concatenate([P[:, 1], P[:, 0]])
    return csr_matrix((np.ones(len(I), np.int32), (I, J)), shape=(N, N))

def bf_vertex_edges(A, x=0, js=None):
    """Exact BF curvature of edges {x,j} for j in js (default: all neighbours of x).
    Returns arrays BF, S (4-cycle term), tri."""
    ip, ix = A.indptr, A.indices
    Nx = ix[ip[x]:ip[x+1]]
    if js is None: js = Nx
    N = A.shape[0]
    inx = np.zeros(N, bool); inx[Nx] = True
    out = np.zeros((len(js), 3))
    dx = len(Nx)
    for e, j in enumerate(js):
        Nj = ix[ip[j]:ip[j+1]]
        inj = np.zeros(N, bool); inj[Nj] = True
        tri = int(inx[Nj].sum())
        dj = len(Nj)
        Ux = Nx[~inj[Nx]]; Ux = Ux[Ux != j]
        Uj = Nj[~inx[Nj]]; Uj = Uj[Uj != x]
        dmax, dmin = max(dx, dj), min(dx, dj)
        if dmin <= 1:
            continue
        sq = 0.0
        if len(Ux) and len(Uj):
            mUj = np.zeros(N); mUj[Uj] = 1.0
            mUx = np.zeros(N); mUx[Ux] = 1.0
            ck = A[Ux] @ mUj; cw = A[Uj] @ mUx
            gam = max(ck.max(), cw.max())
            if gam > 0:
                sq = (np.count_nonzero(ck)+np.count_nonzero(cw))/(gam*dmax)
        b = 2/dx+2/dj-2+2*tri/dmax+tri/dmin+sq
        out[e] = (b, sq, tri)
    return out
