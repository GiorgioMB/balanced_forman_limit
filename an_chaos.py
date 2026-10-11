import numpy as np,glob,collections
D=collections.defaultdict(list)
for f in glob.glob('/tmp/w/res/c_*.txt'):
    p=f.split('/')[-1].split('_'); k=(int(p[1]),float(p[2]))
    a=np.loadtxt(f,ndmin=2)
    if a.size: D[k].append(a[:,1:])
for lam in sorted({k[0] for k in D}):
    ks=sorted(e for (l,e) in D if l==lam)
    if len(ks)<8: continue
    s=np.array([0.0]+ks+[3.0]); G=[]
    for e in ks: G.append(np.vstack(D[(lam,e)]).mean(0))
    G=np.array(G); G=np.vstack([G[0],G,np.zeros(3)])
    for c,name in [(0,'full'),(2,'J')]:
        y=G[:,c]**2*s; print(lam,name,'Var2/(n lam)=%.4f'%np.trapz(y,s))
