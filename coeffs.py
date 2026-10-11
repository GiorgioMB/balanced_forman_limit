import numpy as np
from scipy.integrate import quad
from scipy.special import betainc, gamma
def omega(d): return np.pi**(d/2)/gamma(d/2+1)
def f(d,t): return betainc((d+1)/2, 0.5, 1-t**2/4)
def m_par(d,t):
    return 2*omega(d-1)*quad(lambda s:(s-t/2)**2*(1-s*s)**((d-1)/2), t/2, 1)[0]
def m_perp(d,t):
    return 2*omega(d-1)/(d+1)*quad(lambda s:(1-s*s)**((d+1)/2), t/2, 1)[0]
def a(d,t): return t**2*f(d,t)/24 + (m_perp(d,t)-m_par(d,t))/(6*omega(d))
def b(d,t): return f(d,t)/(6*(d+2)) - m_perp(d,t)/(6*omega(d))
