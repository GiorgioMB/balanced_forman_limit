import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import an_v1_lib as L
from bfsim import Psi
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':9})
S=json.load(open('../results/e1_summary.json'))
fig,ax=plt.subplots(1,3,figsize=(7.0,2.3))
fits={}
for ref,d in [('flat2_l2000_r1.0',2),('flat3_l600_r1.0',3),('flat2_l500_r1.0',2)]:
    fits[ref]=L.bias_fit(L.load(ref),d)
for name,ref,d,r,c,lab in [('S2_l2000_r0.5','flat2_l2000_r1.0',2,0.5,'C3','$S^2$ (Scal $=2$)'),('H2_l2000_r0.5','flat2_l2000_r1.0',2,0.5,'C0','$H^2$ (Scal $=-2$)'),('S3_l600_r0.6','flat3_l600_r1.0',3,0.6,'C1','$S^3$ (Scal $=6$)')]:
    sh=L.scal_hat(L.load(name),d,r,fits[ref]); ax[0].hist(sh,bins=30,histtype='step',color=c,density=True,label=lab)
ax[0].set_xlabel(r'$\widehat{\mathrm{Scal}}_i$'); ax[0].legend(frameon=False,fontsize=6.5); ax[0].set_title('(a) per-vertex estimates',fontsize=8.5)
pts=[(v['lr4'],v['rmse']) for k,v in S.items() if k.startswith('S2_l')]
x,y=np.array(pts).T; ax[1].loglog(x,y,'o',color='C3',ms=4,label='$S^2$')
for k,c in [('S3_l600_r0.6','C1'),('S2xR_l600_r0.6','C2'),('H2_l2000_r0.5','C0')]:
    ax[1].loglog(S[k]['lr4'],S[k]['rmse'],'s',color=c,ms=4)
xx=np.array([3,150]); ax[1].loglog(xx,11*xx**-0.5,'k--',lw=0.7)
ax[1].set_xlabel(r'$\lambda r^4$'); ax[1].set_ylabel('RMSE'); ax[1].set_title(r'(b) error $\propto(\lambda r^4)^{-1/2}$',fontsize=8.5)
R=S['ricci_S2xR_l600_r0.6']; Ric=np.array(R['Ric']); se=np.array(R['Ric_se'])
lbl=['$R_{11}$','$R_{22}$','$R_{33}$','$R_{12}$','$R_{13}$','$R_{23}$']; ij=[(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]; tru=[1,1,0,0,0,0]
ax[2].errorbar(range(6),[Ric[a] for a in ij],yerr=[2*se[a] for a in ij],fmt='o',color='C2',ms=4,capsize=2)
ax[2].plot(range(6),tru,'kx',ms=6); ax[2].set_xticks(range(6)); ax[2].set_xticklabels(lbl,fontsize=7)
ax[2].set_title(r'(c) Ricci tensor, $S^2\times\mathbb{R}$',fontsize=8.5); ax[2].axhline(0,color='k',lw=0.4)
fig.tight_layout(pad=0.3); fig.savefig('../figs/fig_e1.pdf')
