# E5: (a) RGG on T^2 with lambda = c log n ; (c) union-kNN graphs.  Each call: build graph (seeded), process
# a slice of edges for worker W, save.  Graph rebuilt identically from seed each call.
import numpy as np, sys, os, time, glob, zlib
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from allbf import bf_edges
kind=sys.argv[1]; n=int(sys.argv[2]); par=float(sys.argv[3]); W=int(os.environ['W']); NW=4; TL=float(os.environ.get('TL','4.5'))
tag=f'{kind}_n{n}_p{par:g}'; D='../results/e5'
rng=np.random.default_rng(zlib.crc32(tag.encode())); X=rng.random((n,2)); T=cKDTree(X,boxsize=1.0)
if kind=='rgg':
    r=np.sqrt(par/(n*np.pi)); P=T.query_pairs(r,output_type='ndarray')
else:
    k=int(par); dd,ii=T.query(X,k+1); I=np.repeat(np.arange(n),k); J=ii[:,1:].ravel()
    P=np.unique(np.sort(np.column_stack([I,J]),axis=1),axis=0); r=np.sqrt(k/(n*np.pi))
A=csr_matrix((np.ones(2*len(P)),(np.r_[P[:,0],P[:,1]],np.r_[P[:,1],P[:,0]])),shape=(n,n)); A.sort_indices()
# edge order: random permutation (so partial completion = random subsample)
perm=np.random.default_rng(1).permutation(len(P)); P=P[perm]
done=sorted(glob.glob(f'{D}/{tag}_w{W}_s*.npz')); start=sum(len(np.load(f)['bf']) for f in done)
mine=np.arange(W,len(P),NW)
t0=time.time(); res=[]; idx=[]; k0=start
while k0<len(mine) and time.time()-t0<TL:
    sl=mine[k0:k0+200]; res.append(bf_edges(A,P[sl])); idx.append(sl); k0+=len(sl)
if res:
    sl=np.concatenate(idx); dv=X[P[sl,0]]-X[P[sl,1]]; dv-=np.round(dv); tt=np.linalg.norm(dv,axis=1)/r
    R=np.concatenate(res); np.savez(f'{D}/{tag}_w{W}_s{len(done)}.npz',bf=R[:,0],sq=R[:,1],t=tt,E=len(P))
