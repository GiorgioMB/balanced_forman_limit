"""Exact BF curvature for arbitrary edges of a graph given as symmetric CSR adjacency."""
import numpy as np
def bf_edges(A, E):
    ip,ix=A.indptr,A.indices; N=A.shape[0]; out=np.zeros((len(E),2))
    mi=np.zeros(N,bool); mj=np.zeros(N,bool); vUi=np.zeros(N); vUj=np.zeros(N)
    for e,(i,j) in enumerate(E):
        Ni=ix[ip[i]:ip[i+1]]; Nj=ix[ip[j]:ip[j+1]]; di,dj=len(Ni),len(Nj)
        if min(di,dj)<=1: continue
        mi[Ni]=True; mj[Nj]=True
        tri=int(mj[Ni].sum())
        Ui=Ni[~mj[Ni]]; Ui=Ui[Ui!=j]; Uj=Nj[~mi[Nj]]; Uj=Uj[Uj!=i]
        sq=0.0
        if len(Ui) and len(Uj):
            vUj[Uj]=1; vUi[Ui]=1
            ck=A[Ui]@vUj; cw=A[Uj]@vUi
            g=max(ck.max(),cw.max())
            if g>0: sq=(np.count_nonzero(ck)+np.count_nonzero(cw))/(g*max(di,dj))
            vUj[Uj]=0; vUi[Ui]=0
        mi[Ni]=False; mj[Nj]=False
        out[e]=(2/di+2/dj-2+2*tri/max(di,dj)+tri/min(di,dj)+sq, sq)
    return out
