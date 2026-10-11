"""Exact clustering C(p,r) on S^2, H^2, S^3 by quadrature (Cor. clustering check)."""
import numpy as np, json, sys
sys.path.insert(0,'/tmp/bf'); from bfsim import f_lens, kappa
from exact import sphere_lens, sphere_ball, hyp_lens, hyp_ball
from scipy.integrate import quad
r=0.05; out={}
for name,d,Scal,lens,ball,dens in [('S2',2,2.0,lambda a:sphere_lens(2,r,a),sphere_ball(2,r),lambda a:2*np.pi*np.sin(a)),
                                    ('H2',2,-2.0,lambda a:hyp_lens(r,a),hyp_ball(r),lambda a:2*np.pi*np.sinh(a)),
                                    ('S3',3,6.0,lambda a:sphere_lens(3,r,a),sphere_ball(3,r),lambda a:4*np.pi*np.sin(a)**2)]:
    C=quad(lambda a: dens(a)*lens(a),0,r,limit=200,epsabs=1e-16,epsrel=1e-11)[0]/ball**2
    dev=(C-1.5*f_lens(d,1.0))/r**2; pred=3*kappa(d,1.0)/(d+2)*Scal
    out[name]=dict(C=C,dev=float(dev),pred=float(pred)); print(name,f"{dev:.6f}",f"{pred:.6f}")
json.dump(out,open('/tmp/bf/res/clust_exact.json','w'),indent=1)
