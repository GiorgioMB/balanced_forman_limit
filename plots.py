import numpy as np, sys, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
sys.path.insert(0,'/tmp/bf'); from bfsim import *
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':9,'axes.labelsize':9,'legend.fontsize':7.5,
                     'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':0.6,'lines.linewidth':1.0})
TS=[0.25,0.5,0.75,1.0]; COL={0.25:'#1b9e77',0.5:'#d95f02',0.75:'#7570b3',1.0:'#e7298a'}
LAMS=[100,200,400,800,1600,3200,6400,12800]
E1={D:np.load(f'/tmp/bf/res/flat_edges_d{D}.npz') for D in [2,3]}
SQ=json.load(open('/tmp/bf/res/sq_const.json'))
def sn_pdf(w,f):
    s=np.sqrt(f*(1-f)*(9-4*f)); a=-np.sqrt(f/(2-f))/3
    return 2/s*norm.pdf(w/s)*norm.cdf(a*w/s)
# ---------- Fig: rate / variance / bias ----------
fig,ax=plt.subplots(1,3,figsize=(7.0,2.35))
for D,mk,ls in [(2,'o','-'),(3,'s','--')]:
    for t in TS:
        f=float(f_lens(D,t)); ps=float(Psi(D,t))
        rms=[];v=[];bc=[];bse=[]
        for lam in LAMS:
            A=E1[D][f"{D}_{lam}_{t}"]; b=A[:,0]; sq=A[:,1]
            rms.append(np.sqrt(((b-ps)**2).mean())); v.append(lam*b.var())
            bc.append(np.sqrt(lam)*(b.mean()-ps-(4-3*f)/lam-sq.mean())); bse.append(np.sqrt(lam)*b.std()/np.sqrt(len(b)))
        L=np.array(LAMS)
        ax[0].loglog(L,rms,mk,ms=2.8,color=COL[t],mfc='none' if D==3 else COL[t],mew=0.7)
        ax[0].loglog(L,np.sqrt((W_var(f)+W_mean(f)**2)/L),ls,color=COL[t],lw=0.7)
        ax[1].semilogx(L,np.array(v)/W_var(f),mk+ls,ms=2.8,color=COL[t],mfc='none' if D==3 else COL[t],mew=0.7,lw=0.6)
        ax[2].errorbar(L*(1.04 if D==3 else 0.96),np.array(bc)/W_mean(f),yerr=2*np.array(bse)/abs(W_mean(f)),fmt=mk,ms=2.8,color=COL[t],
                       mfc='none' if D==3 else COL[t],mew=0.7,elinewidth=0.5,capsize=0)
ax[0].set_xlabel(r'$\lambda$'); ax[0].set_ylabel(r'RMS of $\mathrm{BF}-\Psi_d(t)$')
ax[0].set_title(r'(a) rate $\lambda^{-1/2}$',fontsize=9)
ax[1].axhline(1,color='k',lw=0.6); ax[1].set_ylim(0.8,1.2); ax[1].set_xlabel(r'$\lambda$'); ax[1].set_ylabel(r'$\lambda\,\mathrm{Var}(\mathrm{BF})\,/\,\mathrm{Var}(W)$')
ax[1].set_title(r'(b) variance',fontsize=9)
ax[2].set_xscale('log'); ax[2].axhline(1,color='k',lw=0.6); ax[2].set_ylim(0,2); ax[2].set_xlabel(r'$\lambda$')
ax[2].set_ylabel(r'$\sqrt{\lambda}\,(\mathrm{bias}-\frac{4-3f}{\lambda}-\mathrm{E}\,S)\,/\,\mathrm{E}W$'); ax[2].set_title(r'(c) $\lambda^{-1/2}$ bias',fontsize=9)
from matplotlib.lines import Line2D
h=[Line2D([],[],color=COL[t],marker='o',ls='',ms=3,label=f'$t={t}$') for t in TS]+[Line2D([],[],color='gray',marker='o',ls='-',ms=3,label='$d=2$'),Line2D([],[],color='gray',marker='s',mfc='none',ls='--',ms=3,label='$d=3$')]
ax[0].legend(handles=h,loc='lower left',frameon=False,ncol=2,handlelength=1.2,columnspacing=0.6)
fig.tight_layout(pad=0.3,w_pad=0.6); fig.savefig('/tmp/bf/fig/fig_rate.pdf'); plt.close(fig)
# ---------- Fig: CLT ----------
fig,ax=plt.subplots(1,2,figsize=(6.0,2.3))
for k,(D,t) in enumerate([(2,0.5),(3,1.0)]):
    f=float(f_lens(D,t)); ws=[]
    for lam in [1600,3200,6400]:
        A=E1[D][f"{D}_{lam}_{t}"]; b=A[:,0]; sq=A[:,1]
        ws.append(np.sqrt(lam)*(b-Psi(D,t)-(4-3*f)/lam-sq))
    w=np.concatenate(ws)
    ax[k].hist(w,bins=45,density=True,color='0.82',edgecolor='0.55',lw=0.35,label=f'simulation ($N={len(w)}$)')
    x=np.linspace(-5,5,400)
    ax[k].plot(x,sn_pdf(x,f),'k',lw=1.1,label='skew-normal limit $W$')
    ax[k].plot(x,norm.pdf(x,0,np.sqrt(W_var(f))),'k:',lw=0.8,label='$N(0,\\mathrm{Var}\\,W)$')
    ax[k].axvline(W_mean(f),color='k',lw=0.5,ls='--')
    ax[k].set_xlim(-5,5); ax[k].set_ylim(0,0.48)
    ax[k].set_title(f'$d={D}$, $t={t}$, $\\lambda\\in\\{{1600,3200,6400\\}}$',fontsize=8.5)
    ax[k].set_xlabel(r'$\sqrt{\lambda}\,(\mathrm{BF}-\Psi_d(t)-S-(4-3f_d)/\lambda)$')
ax[0].legend(frameon=False,loc='upper left',fontsize=6.5)
fig.tight_layout(pad=0.3); fig.savefig('/tmp/bf/fig/fig_clt.pdf'); plt.close(fig)
# ---------- Fig: 4-cycle term ----------
fig,ax=plt.subplots(1,2,figsize=(6.0,2.2),sharey=False)
for k,D in enumerate([2,3]):
    for t in TS:
        y=[LAMS[i]*E1[D][f"{D}_{lam}_{t}"][:,1].mean() for i,lam in enumerate(LAMS)]
        ax[k].semilogx(LAMS,y,'o-',ms=2.8,color=COL[t],lw=0.7,label=f'$t={t}$')
        ax[k].axhline(SQ[f"{D}_{t}"]['lim'],color=COL[t],ls=':',lw=0.9)
    ax[k].set_xlabel(r'$\lambda$'); ax[k].set_ylabel(r'$\lambda\,\mathrm{E}\,S$'); ax[k].set_title(f'$d={D}$',fontsize=9)
ax[0].legend(frameon=False,loc='lower right',ncol=2)
fig.tight_layout(pad=0.3); fig.savefig('/tmp/bf/fig/fig_sq.pdf'); plt.close(fig)
# ---------- Fig: global ----------
def phi_inf(y,d):
    om=omega(d)/omega(d-1)
    tau=np.array([brentq(lambda s: Psi(d,s)-yy,0,1) for yy in y])
    return d*om/3*tau**(d-1)*(1-tau**2/4)**(-(d-1)/2)
fig,ax=plt.subplots(1,3,figsize=(7.0,2.3))
for k,(D,s) in enumerate([(2,'n10000_lam400'),(3,'n25000_lam200')]):
    Z=np.load(f'/tmp/bf/res/global_d{D}_{s}.npz'); b=Z['bf']
    ax[k].hist(b,bins=60,density=True,color='0.8',edgecolor='0.55',lw=0.3,label=f'all $|E|={len(b):,}$ edges')
    y=np.linspace(Psi(D,1.0)+1e-6,1-1e-6,300); ax[k].plot(y,phi_inf(y,D),'k',lw=1.1,label=r'$\varphi_\infty$')
    ax[k].set_title(f'$d={D}$, $n={int(Z["n"])}$, $\\lambda={int(Z["lam"])}$',fontsize=9); ax[k].set_xlabel('BF curvature')
    ax[k].legend(frameon=False,loc='upper right',fontsize=7)
S=json.load(open('/tmp/bf/res/global_summary.json'))
for D,mk in [(2,'o'),(3,'s')]:
    R=[x for x in S if x['d']==D]; L=np.array([x['lam'] for x in R])
    ax[2].loglog(L,[x['maxdev'] for x in R],mk+'-',color='C3',ms=3,lw=0.7,mfc='none' if D==3 else 'C3')
    ax[2].loglog(L,[x['meandev'] for x in R],mk+'-',color='C0',ms=3,lw=0.7,mfc='none' if D==3 else 'C0')
    ax[2].loglog(L,[x['W1'] for x in R],mk+'-',color='C2',ms=3,lw=0.7,mfc='none' if D==3 else 'C2')
L=np.array([25,400]); ax[2].loglog(L,1.9*np.sqrt(np.log(1e4)/L),'k--',lw=0.7); ax[2].loglog(L,1.0/np.sqrt(L),'k:',lw=0.7)
ax[2].text(140,0.75,r'$\max_e|\mathrm{BF}-\Psi|$',color='C3',fontsize=7); ax[2].text(26,0.062,r'mean$_e|\mathrm{BF}-\Psi|$',color='C0',fontsize=7)
ax[2].text(26,0.0075,r'$W_1(\mu_n,\mu_\infty)$',color='C2',fontsize=7)
ax[2].set_xlabel(r'$\lambda$'); ax[2].set_title(r'dashed: $1.9\sqrt{\log n/\lambda}$; dotted: $\lambda^{-1/2}$',fontsize=7.5)
fig.tight_layout(pad=0.3,w_pad=0.5); fig.savefig('/tmp/bf/fig/fig_global.pdf'); plt.close(fig)
# ---------- Fig: curved ----------
E3=json.load(open('/tmp/bf/res/curved_summary.json')); R0=0.5
GEO={'S2':(2,1.5,r'$S^2$ ($\rho=3/2$)','C3','o'),'H2':(2,-1.5,r'$H^2$ ($\rho=-3/2$)','C0','v'),
     'S3':(3,3.2,r'$S^3$ ($\rho=16/5$)','C1','s'),'S2xR_S':(3,1.4,r'$S^2\!\times\!\mathbb{R}$, $v\in TS^2$ ($\rho=7/5$)','C2','D'),
     'S2xR_R':(3,0.4,r'$S^2\!\times\!\mathbb{R}$, $v\in T\mathbb{R}$ ($\rho=2/5$)','C4','^')}
fig,ax=plt.subplots(1,1,figsize=(4.6,2.9))
tt=np.linspace(0,1,200)
for g,(d,rho,lab,c,mk) in GEO.items():
    ax.plot(tt,3*kappa(d,tt)*rho*R0**2,color=c,lw=0.8)
    pts=[x for x in E3 if x['geo']==g]
    ax.errorbar([x['t'] for x in pts],[x['diff'] for x in pts],yerr=[2*x['se'] for x in pts],fmt=mk,color=c,ms=3.5,capsize=1.5,elinewidth=0.7,label=lab)
    ax.plot([x['t'] for x in pts],[x['exact'] for x in pts],'x',color='k',ms=3.5,mew=0.7)
ax.axhline(0,color='k',lw=0.5); ax.set_xlabel(r'$t=\delta/r$'); ax.set_ylabel(r'$\mathrm{E\,BF}_{M}-\mathrm{E\,BF}_{\mathrm{flat}}$')
ax.legend(frameon=False,fontsize=6.6,loc='lower left',bbox_to_anchor=(0.0,0.0))
ax.set_title(r'$r=0.5$, $\lambda=1500$; lines: $3\kappa_d(t)\rho\,r^2$; $\times$: exact lens',fontsize=8)
fig.tight_layout(pad=0.3); fig.savefig('/tmp/bf/fig/fig_curved.pdf'); plt.close(fig)
print('ok')
