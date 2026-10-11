"""fig_more.pdf: (a) Euler characteristic from mean clustering, (b) planted-edge BF at fixed lambda,
(c) correlation of BF on two edges sharing a vertex against the exact Gaussian limit of Thm joint.
Inputs: ../results/e2_summary.json, ../results/e5_summary.json, ../results/oq1_pair.json.
Panel (c) computes the limit with fig_7c_rerun.limit_corr; the 'theory' field of oq1_pair.json is not used."""
import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from bfsim import Psi
from fig_7c_rerun import limit_corr
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':9,'legend.fontsize':7})
R='../results/'; E2=json.load(open(R+'e2_summary.json')); E5=json.load(open(R+'e5_summary.json')); P=json.load(open(R+'oq1_pair.json'))
T_PAIR=0.75   # normalised length of both edges in run_oq1pair.py
fig,ax=plt.subplots(1,3,figsize=(7.0,2.3))
lab={'bolza':'Bolza ($g=2$)','S2':'$S^2$','T2':'$T^2$'}
sel=[e for e in E2 if abs(e['r']*e['r']/e['A']-0.0286)<1e-3]
for i,x in enumerate(sel):
    ax[0].errorbar(i,x['chihat'],yerr=2*x['chise'],fmt='o',color='C0',capsize=2); ax[0].plot(i,x['chi'],'kx',ms=7)
ax[0].set_xticks(range(len(sel))); ax[0].set_xticklabels([lab[e['geo']] for e in sel]); ax[0].set_ylabel(r'$\hat\chi$'); ax[0].axhline(0,color='k',lw=0.4)
ax[0].set_title('(a) Euler characteristic',fontsize=8.5)
F=E5['fixed']; lam=np.array([x['lam'] for x in F])
ax[1].errorbar(lam,[x['mean'] for x in F],yerr=[x['sd'] for x in F],fmt='o',color='C3',capsize=2,label='mean $\\pm$ sd')
ax[1].axhline(Psi(2,0.75),color='k',lw=0.6,ls='--',label=r'$\Psi_2(0.75)$'); ax[1].set_xscale('log'); ax[1].set_xlabel(r'$\lambda$ (fixed)')
ax[1].set_title(r'(b) fixed $\lambda$, $t=0.75$',fontsize=8.5); ax[1].legend(frameon=False)
th=[x['theta'] for x in P]; ax[2].errorbar(th,[x['corr'] for x in P],yerr=[2*x['se'] for x in P],fmt='o',color='C0',capsize=2,label='simulation')
grid=np.linspace(min(th),max(th),200)
ax[2].plot(grid,[limit_corr(g,T_PAIR,T_PAIR) for g in grid],'k--',lw=0.8,label='Gaussian limit'); ax[2].axhline(0,color='k',lw=0.4)
ax[2].set_xlabel('angle between edges (deg)'); ax[2].set_ylabel('correlation'); ax[2].legend(frameon=False); ax[2].set_title('(c) edges sharing a vertex',fontsize=8.5)
fig.tight_layout(pad=0.3); fig.savefig('../figs/fig_more.pdf')