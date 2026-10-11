import numpy as np, glob, json
from bfsim import kappa
def L(c): return np.concatenate([np.load(f) for f in glob.glob(f'../results/oq2/{c}_w*')])
out=[]
for c,ref,d,ric,scal in [('S2','flat2',2,1,2),('H2','flat2',2,-1,-2),('S2xR_S','flat3',3,1,2),('S2xR_R','flat3',3,0,2)]:
    X=L(c); F=L(ref); r=0.6
    rec=dict(geo=c,n=len(X),ric_vv=ric,scal=scal)
    for k,nm in [(1,'BF'),(2,'ORC')]:
        dm=X[:,k].mean()-F[:,k].mean(); se=np.sqrt(X[:,k].var()/len(X)+F[:,k].var()/len(F)); sd=F[:,k].std()
        rec[nm]=dict(shift=float(dm),se=float(se),sd_edge=float(sd),snr_edge=float(dm/sd))
    rec['BF_pred']=float(3*kappa(d,0.75)*(ric+scal/(d+2))*r*r)
    rec['ORC_ollivier_ref']=float(r*r*ric/(2*(d+2)))
    out.append(rec)
    print(f"{c:7s} BF shift {rec['BF']['shift']:+.4f}±{rec['BF']['se']:.4f} (pred {rec['BF_pred']:+.4f}) SNR/edge {rec['BF']['snr_edge']:+.2f} | ORC shift {rec['ORC']['shift']:+.4f}±{rec['ORC']['se']:.4f} (eps^2Ric/2(d+2)={rec['ORC_ollivier_ref']:+.4f}) SNR/edge {rec['ORC']['snr_edge']:+.2f}")
for ref in ['flat2','flat3']:
    F=L(ref); print(ref,'mean BF-Psi %.4f  mean ORC %.4f  sd BF %.4f sd ORC %.4f corr %.3f'%(F[:,1].mean(),F[:,2].mean(),F[:,1].std(),F[:,2].std(),np.corrcoef(F[:,1],F[:,2])[0,1]))
json.dump(out,open('../results/oq2_summary.json','w'),indent=1)
