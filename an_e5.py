import numpy as np, glob, json
from bfsim import Psi, f_lens, W_mean, W_var
out={'regime':[],'fixed':[],'knn':[]}
for n in [1000,3000,10000]:
    for lab,p in [('2logn',2*np.log(n)),('log2n',np.log(n)**2)]:
        tag=f'rgg_n{n}_p{round(p,4):g}'; fs=glob.glob(f'../results/e5/{tag}_w*')
        Z=[np.load(f) for f in fs]; bf=np.concatenate([z['bf'] for z in Z]); t=np.concatenate([z['t'] for z in Z])
        dev=np.abs(bf-Psi(2,t)); rate=np.sqrt(np.log(n)/p)
        rec=dict(n=n,scaling=lab,lam=float(p),E=len(bf),maxdev=float(dev.max()),meandev=float(dev.mean()),rate=float(rate),ratio=float(dev.max()/rate))
        out['regime'].append(rec); print(rec)
Z=[np.load(f) for f in glob.glob('../results/e5/fixedlam_w*')]
for lam in ['2','5','10','20','50','200']:
    x=np.concatenate([z[lam] for z in Z]); f=float(f_lens(2,0.75))
    rec=dict(lam=int(lam),N=len(x),mean=float(x.mean()),sd=float(x.std()),psi=float(Psi(2,0.75)),zero_frac=float(np.mean(x==0)),
             clt_mean=float(Psi(2,0.75)+W_mean(f)/np.sqrt(int(lam))),clt_sd=float(np.sqrt(W_var(f)/int(lam))))
    out['fixed'].append(rec); print(rec)
bins=np.linspace(0,1.6,17)
for tag in ['knn_n20000_p20','knn_n20000_p50','rgg_n20000_p50']:
    Z=[np.load(f) for f in glob.glob(f'../results/e5/{tag}_w*')]; bf=np.concatenate([z['bf'] for z in Z]); t=np.concatenate([z['t'] for z in Z])
    k=np.digitize(t,bins)-1; mb=[float(bf[k==i].mean()) if (k==i).sum()>30 else None for i in range(16)]
    frac_long=float(np.mean(t>1)); rec=dict(tag=tag,E=len(bf),bins=bins.tolist(),mean_by_bin=mb,frac_t_gt_1=frac_long,mean=float(bf.mean()))
    out['knn'].append(rec); print(tag,len(bf),'frac t>1',round(frac_long,3),'mean BF',round(bf.mean(),4),[None if m is None else round(m,3) for m in mb])
print('Psi at bin centers',[round(float(Psi(2,c)),3) if c<=1 else None for c in (bins[:-1]+bins[1:])/2])
json.dump(out,open('../results/e5_summary.json','w'),indent=1)
