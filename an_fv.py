import numpy as np,glob,collections,json
D=collections.defaultdict(list)
for f in glob.glob('/tmp/w/res/fv_*.txt'):
    p=f.split('/')[-1].split('_'); k=(int(p[1]),int(p[2]))
    try: a=np.loadtxt(f,ndmin=2)
    except Exception: continue
    if a.size: D[k].append(a)
out=[]
for k in sorted(D):
    a=np.vstack(D[k]); a=a[np.unique(a[:,0],return_index=True)[1]]
    S=a[:,1::2]; N=a[:,2::2]; m=S.sum(0)/np.maximum(N.sum(0),1)
    F=(S-N*m).sum(1); G=len(F); lam,n0=k
    v=F.var(ddof=1); m4=((F-F.mean())**4).mean(); se=np.sqrt(max(m4-v*v,0)/G)
    # bootstrap s.e.
    rng=np.random.default_rng(0); bs=[F[rng.integers(0,G,G)].var(ddof=1) for _ in range(400)]
    r=np.sqrt(lam/(np.pi*n0))
    print('lam=%4d n0=%5d r=%.3f graphs=%5d  Var/(n0 lam)=%.3f ± %.3f (boot %.3f)  meanE=%.0f'%(lam,n0,r,G,v/(n0*lam),se/(n0*lam),np.std(bs)/(n0*lam),N.sum(1).mean()))
    out.append(dict(lam=lam,n0=n0,r=r,graphs=G,ratio=v/(n0*lam),se=np.std(bs)/(n0*lam)))
json.dump(out,open('/tmp/w/res/var_table.json','w'),indent=1)
