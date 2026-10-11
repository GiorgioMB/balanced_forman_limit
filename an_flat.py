import numpy as np, glob, json, sys
sys.path.insert(0,'/tmp/bf'); from bfsim import Psi, f_lens, W_mean, W_var
TS=[0.25,0.5,0.75,1.0]; LAMS=[100,200,400,800,1600,3200,6400,12800]
summ={}
for D in [2,3]:
    Z={}
    for t in TS:
        for lam in LAMS:
            fs=sorted(glob.glob(f'/tmp/bf/res/flat/f_{D}_{lam}_{t}_*.npy'))
            Z[f"{D}_{lam}_{t}"]=np.concatenate([np.load(f) for f in fs])
    np.savez_compressed(f'/tmp/bf/res/flat_edges_d{D}.npz',**Z)
    for t in TS:
        f=float(f_lens(D,t)); ps=float(Psi(D,t)); L=np.array(LAMS,float)
        rms=[];vr=[];z=[];lsq=[]
        for lam in LAMS:
            A=Z[f"{D}_{lam}_{t}"]; b=A[:,0]; sq=A[:,1]; N=len(b)
            rms.append(np.sqrt(((b-ps)**2).mean())); vr.append(lam*b.var(ddof=1)/W_var(f))
            bc=np.sqrt(lam)*(b.mean()-ps-(4-3*f)/lam-sq.mean()); se=np.sqrt(lam)*np.std(b-sq,ddof=1)/np.sqrt(N)
            z.append((bc-W_mean(f))/se); lsq.append(lam*sq.mean())
        m=L>=400; slope=np.polyfit(np.log(L[m]),np.log(np.array(rms)[m]),1)[0]
        summ[f"{D}_{t}"]=dict(slope=float(slope),varratio=[float(x) for x in vr],z=[float(x) for x in z],lamES=[float(x) for x in lsq],rms=[float(x) for x in rms])
json.dump(summ,open('/tmp/bf/res/flat_summary.json','w'),indent=1)
sl=[v['slope'] for v in summ.values()]; vr=np.concatenate([v['varratio'] for v in summ.values()])
zz=np.concatenate([np.array(v['z'])[2:] for v in summ.values()])
print('slopes',min(sl),max(sl),np.mean(sl)); print('varratio',vr.min(),vr.max(),vr.mean(),len(vr))
print('z (lam>=400)',len(zz),zz.mean(),zz.std(ddof=1),np.abs(zz).max())
for k,v in summ.items(): print(k,'1600:',round(v['lamES'][4],2),'12800:',round(v['lamES'][7],2))
