import numpy as np, json, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from bfsim import Psi, f_lens, kappa
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':9,'legend.fontsize':7})
R='../results/'; which=sys.argv[1]
if which=='e23':
    E2=json.load(open(R+'e2_summary.json')); E3=json.load(open(R+'e3_summary.json'))
    fig,ax=plt.subplots(1,2,figsize=(6.6,2.4))
    lab={'bolza':'Bolza (genus 2)','S2':'$S^2$','T2':'$T^2$'}
    for i,x in enumerate([e for e in E2 if abs(e['r']*e['r']/e['A']-0.0286)<1e-3]):
        ax[0].errorbar(i,x['chihat'],yerr=2*x['chise'],fmt='o',color='C0',capsize=2); ax[0].plot(i,x['chi'],'kx',ms=7)
    ax[0].set_xticks([0,1,2]); ax[0].set_xticklabels([lab[e['geo']] for e in E2 if abs(e['r']*e['r']/e['A']-0.0286)<1e-3]); ax[0].set_ylabel(r'$\hat\chi$')
    ax[0].set_title(r'(a) Euler characteristic from mean clustering',fontsize=8.5); ax[0].axhline(0,color='k',lw=0.4)
    for t,c in [(0.75,'C3'),(1.0,'C1')]:
        P=[x for x in E3 if x['t']==t and x['mode']=='fixed' and x['dir']=='par']; rr=[x['r'] for x in P]
        ax[1].errorbar(rr,[x['diff'] for x in P],yerr=[2*x['se'] for x in P],fmt='o',color=c,capsize=2,label=f'parallel, $t={t}$')
        ax[1].plot(rr,[x['pred'] for x in P],'--',color=c,lw=0.8); ax[1].plot(rr,[x['pred_folded'] for x in P],':',color=c,lw=1)
        Q=[x for x in E3 if x['t']==t and x['dir']=='perp']; ax[1].errorbar([x['r'] for x in Q],[x['diff'] for x in Q],yerr=[2*x['se'] for x in Q],fmt='s',mfc='none',color=c,capsize=2)
        A=[x for x in E3 if x['t']==t and x['mode']=='adapt']; ax[1].errorbar([x['r'] for x in A],[x['diff'] for x in A],yerr=[2*x['se'] for x in A],fmt='^',color=c,mfc='none',capsize=2)
    ax[1].axhline(0,color='k',lw=0.4); ax[1].set_xlabel('$r$'); ax[1].set_ylabel(r'$\mathrm{E\,BF}_\varrho-\mathrm{E\,BF}_{\rm flat}$')
    ax[1].set_title(r'(b) density gradient',fontsize=8); ax[1].legend(frameon=False,loc='lower left')
    fig.tight_layout(pad=0.3); fig.savefig('../figs/fig_e2e3.pdf')
if which=='e45':
    E4=json.load(open(R+'e4_summary.json')); E5=json.load(open(R+'e5_summary.json'))
    fig,ax=plt.subplots(1,3,figsize=(7.0,2.3))
    for i,x in enumerate(E4):
        ax[0].errorbar(i,x['diff'],yerr=2*x['se'],fmt='o',color='C0',capsize=2); ax[0].plot(i,x['pred'],'kx',ms=6)
    ax[0].set_xticks(range(len(E4))); ax[0].set_xticklabels([("C" if x['dir']=='circ' else "A")+f"\n{x['r']}\n{x['t']}" for x in E4],fontsize=6); ax[0].axhline(0,color='k',lw=0.4)
    ax[0].set_title('(a) cylinder, ambient distances',fontsize=8.5); ax[0].set_xlabel('dir (C/A), $r$, $t$',fontsize=7)
    F=E5['fixed']; lam=np.array([x['lam'] for x in F])
    ax[1].errorbar(lam,[x['mean'] for x in F],yerr=[x['sd'] for x in F],fmt='o',color='C3',capsize=2,label='mean $\\pm$ sd')
    ax[1].axhline(Psi(2,0.75),color='k',lw=0.6,ls='--',label=r'$\Psi_2(0.75)$'); ax[1].set_xscale('log'); ax[1].set_xlabel(r'$\lambda$ (fixed)')
    ax[1].set_title(r'(b) fixed $\lambda$, $t=0.75$',fontsize=8.5); ax[1].legend(frameon=False)
    b=np.array(E5['knn'][0]['bins']); c=(b[:-1]+b[1:])/2
    for rec,col,lb in zip(E5['knn'],['C0','C2','k'],['union $k$NN, $k=20$','union $k$NN, $k=50$','RGG, $\\lambda=50$']):
        m=[np.nan if v is None else v for v in rec['mean_by_bin']]; ax[2].plot(c,m,'o-',ms=2.5,lw=0.7,color=col,label=lb)
    tt=np.linspace(0,1,100); ax[2].plot(tt,Psi(2,tt),'r--',lw=0.8,label=r'$\Psi_2$'); ax[2].set_xlabel(r'$t=d/r_k$'); ax[2].legend(frameon=False,fontsize=6)
    ax[2].set_title('(c) $k$-NN graphs',fontsize=8.5)
    fig.tight_layout(pad=0.3); fig.savefig('../figs/fig_e4e5.pdf')
if which=='e6oq':
    E6=json.load(open(R+'e6_summary.json')); P=json.load(open(R+'oq1_pair.json')); O=json.load(open(R+'oq2_summary.json'))
    fig,ax=plt.subplots(1,3,figsize=(7.0,2.3))
    Bs=['0','50','100','200','400']
    for s,c,lb in [('random','0.5','random'),('raw','C3','raw BF'),('cen','C0','centred BF'),('oracle','C2','corridor (oracle)')]:
        ax[0].plot([int(b) for b in Bs],[E6['rewire'][s][b]['mean'] for b in Bs],'o-',ms=3,color=c,label=lb)
    ax[0].set_xlabel('added edges'); ax[0].set_ylabel(r'mean $R_{\rm eff}$'); ax[0].legend(frameon=False,fontsize=6); ax[0].set_title('(a) rewiring two rooms',fontsize=8.5)
    th=[x['theta'] for x in P]; ax[1].errorbar(th,[x['corr'] for x in P],yerr=[2*x['se'] for x in P],fmt='o',color='C0',capsize=2,label='simulation')
    ax[1].plot(th,[x['theory'] for x in P],'k--',lw=0.8,label='Gaussian cell limit'); ax[1].axhline(0,color='k',lw=0.4)
    ax[1].set_xlabel(r'angle between edges (deg)'); ax[1].set_ylabel('correlation'); ax[1].legend(frameon=False); ax[1].set_title('(b) two edges sharing a vertex',fontsize=8.5)
    x=np.arange(len(O)); w=0.35
    ax[2].bar(x-w/2,[o['BF']['shift'] for o in O],w,yerr=[2*o['BF']['se'] for o in O],color='C3',label='BF',capsize=2)
    ax[2].bar(x+w/2,[o['ORC']['shift'] for o in O],w,yerr=[2*o['ORC']['se'] for o in O],color='C0',label='Ollivier (data cloud)',capsize=2)
    ax[2].plot(x-w/2,[o['BF_pred'] for o in O],'kx'); ax[2].plot(x+w/2,[o['ORC_ollivier_ref'] for o in O],'k+')
    ax[2].set_xticks(x); ax[2].set_xticklabels(['$S^2$','$H^2$',r'$S^2\!\times\!\mathbb{R}$'+'\n$TS^2$',r'$S^2\!\times\!\mathbb{R}$'+'\n$T\mathbb{R}$'],fontsize=6.5)
    ax[2].axhline(0,color='k',lw=0.4); ax[2].legend(frameon=False,fontsize=6); ax[2].set_title('(c) curved minus flat shift',fontsize=8.5)
    fig.tight_layout(pad=0.3); fig.savefig('../figs/fig_e6oq.pdf')
