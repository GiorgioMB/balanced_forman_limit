import numpy as np, glob, json
ex=json.load(open('../results/e2_exact.json')); C2=ex['C2']; g2=0.017229153  # gamma_2 = 3 kappa_2(1)/4
rows=[]
for geo,r,chi,A,exk in [('bolza',0.6,-2,4*np.pi,'H2_0.6'),('S2',0.6,2,4*np.pi,'S2_0.6'),('T2',0.16925687506432687,0,1.0,None),('S2',0.45,2,4*np.pi,'S2_0.45'),('S2',0.3,2,4*np.pi,'S2_0.3')]:
    tag=f'{geo}_l60_r{r:.4f}'; x=np.concatenate([np.load(f) for f in glob.glob(f'../results/e2/{tag}_w*')])
    c=x[:,0]; m,se=c.mean(),c.std()/np.sqrt(len(c))
    lead=C2+g2*(4*np.pi*chi/A)*r*r; exact=ex[exk] if exk else C2
    k=A/(4*np.pi*g2*r*r); chihat=(m-C2)*k; chise=se*k
    rows.append(dict(geo=geo,r=r,chi=chi,A=A,reps=len(c),mean=m,se=se,lead=lead,exact=exact,chihat=chihat,chise=chise,meandeg=float(x[:,2].mean())))
    print(f"{tag:16s} reps {len(c)} mean C {m:.5f}±{se:.5f}  leading {lead:.5f}  exact {exact:.5f}  chi_hat {chihat:+.3f}±{chise:.3f} (chi={chi})  r^2/A={r*r/A:.4f}")
json.dump(rows,open('../results/e2_summary.json','w'),indent=1)
