import numpy as np, glob, json
from bfsim import Psi, kappa
D='../results/e1'
def load(pref):
    fs=sorted(glob.glob(f'{D}/{pref}_c*.npz')); Z=[np.load(f) for f in fs]
    return {k:np.concatenate([z[k] for z in Z]) for k in ['t','bf','sq','v','vid']}
def bias_fit(Z,d,mask=None):
    m=np.ones(len(Z['t']),bool) if mask is None else mask
    return np.polyfit(Z['t'][m],Z['bf'][m]-Psi(d,Z['t'][m]),3)
def scal_hat(Z,d,r,coef):
    y=Z['bf']-Psi(d,Z['t'])-np.polyval(coef,Z['t']); k=kappa(d,Z['t'])
    cr=2*(d+1)/(d*(d+2)); vids=np.unique(Z['vid'])
    num=np.bincount(np.searchsorted(vids,Z['vid']),weights=y); den=np.bincount(np.searchsorted(vids,Z['vid']),weights=k)
    return num/(3*r*r*cr*den)
out={}
cfg=[('flat2_l2000_r1.0',2,1.0,'flat2_l2000_r1.0',0),('S2_l2000_r0.5',2,0.5,'flat2_l2000_r1.0',2),('H2_l2000_r0.5',2,0.5,'flat2_l2000_r1.0',-2),
     ('flat3_l600_r1.0',3,1.0,'flat3_l600_r1.0',0),('S3_l600_r0.6',3,0.6,'flat3_l600_r1.0',6),('S2xR_l600_r0.6',3,0.6,'flat3_l600_r1.0',2),
     ('flat2_l500_r1.0',2,1.0,'flat2_l500_r1.0',0),('S2_l500_r0.3',2,0.3,'flat2_l500_r1.0',2),('S2_l500_r0.5',2,0.5,'flat2_l500_r1.0',2),('S2_l500_r0.7',2,0.7,'flat2_l500_r1.0',2)]
fits={}
for name,d,r,ref,S in cfg:
    Z=load(name); lam=float(name.split('_l')[1].split('_')[0])
    if name==ref:   # cross-fit: fit on even vertices, evaluate on odd (r is arbitrary for flat; use r=0.5 scale)
        ev=(Z['vid']%2==0); c=bias_fit(Z,d,ev); fits[ref]=bias_fit(Z,d)
        Zo={k:v[~ev] for k,v in Z.items()}; sh=scal_hat(Zo,d,0.5,c)
    else:
        sh=scal_hat(Z,d,r,fits[ref])
    rec=dict(d=d,lam=lam,r=r,true=S,nv=len(sh),mean=float(sh.mean()),se=float(sh.std()/np.sqrt(len(sh))),sd=float(sh.std()),
             rmse=float(np.sqrt(np.mean((sh-S)**2))),lr4=lam*r**4)
    out[name]=rec
    print(f"{name:18s} true {S:+d}  mean Scal_hat {rec['mean']:+.3f} ± {rec['se']:.3f}  per-vertex sd {rec['sd']:.2f}  rmse {rec['rmse']:.2f}  lam r^4 {rec['lr4']:.1f}  nv {len(sh)}")
# Ricci tensor recovery on S2xR and S3 (pooled weighted LS)
for name,ref,d,r in [('S2xR_l600_r0.6','flat3_l600_r1.0',3,0.6),('S3_l600_r0.6','flat3_l600_r1.0',3,0.6),('S2_l2000_r0.5','flat2_l2000_r1.0',2,0.5),('H2_l2000_r0.5','flat2_l2000_r1.0',2,0.5)]:
    Z=load(name); k=kappa(d,Z['t']); y=(Z['bf']-Psi(d,Z['t'])-np.polyval(fits[ref],Z['t']))/(3*k*r*r)
    v=Z['v']; idx=[(a,b) for a in range(d) for b in range(a,d)]
    X=np.column_stack([(1 if a==b else 2)*v[:,a]*v[:,b] for a,b in idx]); w=k**2
    beta=np.linalg.lstsq(X*np.sqrt(w)[:,None],y*np.sqrt(w),rcond=None)[0]
    # vertex-cluster bootstrap for SE
    vids=np.unique(Z['vid']); rng=np.random.default_rng(0); B=[]
    grp={u:np.where(Z['vid']==u)[0] for u in vids}
    for _ in range(200):
        s=np.concatenate([grp[u] for u in rng.choice(vids,len(vids))])
        B.append(np.linalg.lstsq(X[s]*np.sqrt(w[s])[:,None],y[s]*np.sqrt(w[s]),rcond=None)[0])
    B=np.array(B)
    def mat(b):
        A=np.zeros((d,d))
        for (a,c),x in zip(idx,b): A[a,c]=A[c,a]=x
        return A
    A=mat(beta); tr=np.trace(A); Ric=A-tr/(2*(d+1))*np.eye(d); Sc=(d+2)/(2*(d+1))*tr
    Ab=np.array([mat(b) for b in B]); Rb=np.array([a-np.trace(a)/(2*(d+1))*np.eye(d) for a in Ab]); Sb=(d+2)/(2*(d+1))*np.trace(Ab,axis1=1,axis2=2)
    out['ricci_'+name]=dict(A=A.tolist(),A_se=Ab.std(0).tolist(),Ric=Ric.tolist(),Ric_se=Rb.std(0).tolist(),Scal=float(Sc),Scal_se=float(Sb.std()))
    np.set_printoptions(precision=3,suppress=True)
    print(name,'\n A_hat=\n',A,'\n se=\n',Ab.std(0),'\n Ric_hat=\n',Ric,'\n se\n',Rb.std(0),'\n Scal_hat=%.3f ± %.3f'%(Sc,Sb.std()))
json.dump(out,open('../results/e1_summary.json','w'),indent=1)
