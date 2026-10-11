import numpy as np, glob, json
from bfsim import Psi, kappa
D='../results/e1'
def load(pref):
    fs=sorted(glob.glob(f'{D}/{pref}_c*.npz')); Z=[np.load(f) for f in fs]
    return {k:np.concatenate([z[k] for z in Z]) for k in ['t','bf','sq','v','vid']}
def bias_fit(Z,d,mask=None):
    m=np.ones(len(Z['t']),bool) if mask is None else mask
    return np.polyfit(Z['t'][m],Z['bf'][m]-Psi(d,Z['t'][m]),3)
def scal_hat(Z,d,r,coef):
    y=Z['bf']-Psi(d,Z['t'])-np.polyval(coef,Z['t']); k=kappa(d,Z['t'])
    cr=2*(d+1)/(d*(d+2)); vids=np.unique(Z['vid'])
    num=np.bincount(np.searchsorted(vids,Z['vid']),weights=y); den=np.bincount(np.searchsorted(vids,Z['vid']),weights=k)
    return num/(3*r*r*cr*den)
