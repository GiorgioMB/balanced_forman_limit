import numpy as np, glob, json, sys
sys.path.insert(0,'/tmp/bf'); from bfsim import Psi, f_lens, kappa
EX=json.load(open('/tmp/bf/res/exact.json')); r=0.5
GEO={'S2':(2,1.5),'H2':(2,-1.5),'S3':(3,3.2),'S2xR_S':(3,1.4),'S2xR_R':(3,0.4)}
def load(g,t):
    fs=sorted(glob.glob(f'/tmp/bf/res/curved/c_{g}_{t}_2000_*.npy')); A=np.concatenate([np.load(f) for f in fs]); assert len(fs)==20 and A.shape==(40000,2); return A[:,0]
out=[]; flat={}
for d in [2,3]:
    for t in [0.5,0.75,1.0]:
        b=load(f'flat{d}',t); flat[(d,t)]=(b.mean(),b.std(ddof=1)/np.sqrt(len(b)))
        out.append(dict(geo=f'flat{d}',t=t,mean=float(b.mean()),se=float(flat[(d,t)][1]),offset=float(b.mean()-Psi(d,t)),offset_z=float((b.mean()-Psi(d,t))/flat[(d,t)][1])))
for g,(d,rho) in GEO.items():
    for t in [0.5,0.75,1.0]:
        b=load(g,t); m,se=b.mean(),b.std(ddof=1)/np.sqrt(len(b)); fm,fse=flat[(d,t)]
        diff=m-fm; s=np.sqrt(se**2+fse**2); ex=3*(EX[f'r05_{g}_{t}']-float(f_lens(d,t))); lead=3*float(kappa(d,t))*rho*r*r
        out.append(dict(geo=g,t=t,diff=float(diff),se=float(s),lead=lead,exact=float(ex),z=float((diff-ex)/s),relerr_lead=float((diff-lead)/lead),zflat=float(diff/s)))
json.dump(out,open('/tmp/bf/res/curved_summary.json','w'),indent=1)
for x in out: print({k:(round(v,5) if isinstance(v,float) else v) for k,v in x.items()})
cs=[x for x in out if 'z' in x]; zs=[x['z'] for x in cs]; re=[abs(x['relerr_lead']) for x in cs]
print('z range',min(zs),max(zs),'max rel',max(re),'mean rel',np.mean(re))
print('S2xR_R zflat',[round(x['zflat'],1) for x in cs if x['geo']=='S2xR_R'])
