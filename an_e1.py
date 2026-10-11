import numpy as np, sys
sys.path.insert(0,'/tmp/bf'); from bfsim import *
for D in [2,3]:
    Z=np.load(f'/tmp/bf/res/e1_d{D}_100-200-400-800-1600-3200-6400-12800.npz')
    print(f"===== d={D} =====")
    for t in [0.25,0.5,0.75,1.0]:
        f=float(f_lens(D,t)); ps=float(Psi(D,t))
        print(f" t={t}  Psi={ps:.4f}  E[W]={W_mean(f):.4f}  Var[W]={W_var(f):.4f}")
        for lam in [100,200,400,800,1600,3200,6400,12800]:
            A=Z[f"{D}_{lam}_{t}"]; b=A[:,0]; sq=A[:,1]
            m=b.mean()-ps; se=b.std()/np.sqrt(len(b))
            print(f"   lam={lam:6d} N={len(b):5d} mean(BF-Psi)={m:+.5f}±{se:.5f}  sqrt(lam)*mean={np.sqrt(lam)*m:+.4f}  "
                  f"sqrt(lam)*(mean-4/lam-Sq)={np.sqrt(lam)*(m-4/lam-sq.mean()):+.4f}  lam*Var={lam*b.var():.4f}  lam*E[Sq]={lam*sq.mean():.3f}  "
                  f"rms={np.sqrt(((b-ps)**2).mean()):.4f}")
