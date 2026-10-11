import numpy as np, sys
sys.path.insert(0,'/tmp/bf'); from bfsim import omega, f_lens, kappa
from exact import sphere_lens, sphere_ball, hyp_lens, hyp_ball
from scipy.integrate import quad
def gamma_d(d):
    Ek=quad(lambda t: d*t**(d-1)*kappa(d,t),0,1)[0]
    Ef=quad(lambda t: d*t**(d-1)*f_lens(d,t)*(t*t/d-1/(d+2)),0,1)[0]
    return 2*(d+1)/(d*(d+2))*Ek - Ef/6, Ek, Ef
for d in [2,3]:
    g,Ek,Ef=gamma_d(d); print(f"d={d}: gamma_d={g:.6f}  E[kappa]={Ek:.6f}  E[f(T)(T^2/d-1/(d+2))]={Ef:.6f}")
g2=gamma_d(2)[0]; g3=gamma_d(3)[0]
C2=1.5*f_lens(2,1.0); C3=1.5*f_lens(3,1.0)
for r in [0.2,0.1,0.05]:
    AB=sphere_ball(2,r)
    C=quad(lambda a: sphere_lens(2,r,a)*2*np.pi*np.sin(a),0,r,limit=200,epsabs=1e-15,epsrel=1e-11)[0]/AB**2
    AH=hyp_ball(r)
    CH=quad(lambda a: hyp_lens(r,a)*2*np.pi*np.sinh(a),0,r,limit=200,epsabs=1e-15,epsrel=1e-11)[0]/AH**2
    VB=sphere_ball(3,r)
    C3s=quad(lambda a: sphere_lens(3,r,a)*4*np.pi*np.sin(a)**2,0,r,limit=200,epsabs=1e-15,epsrel=1e-11)[0]/VB**2
    print(f"r={r}: S2 (C-C2)/r^2={(C-C2)/r**2:+.6f} pred {g2*2:+.6f} | H2 {(CH-C2)/r**2:+.6f} pred {-g2*2:+.6f} | S3 {(C3s-C3)/r**2:+.6f} pred {g3*6:+.6f}")
