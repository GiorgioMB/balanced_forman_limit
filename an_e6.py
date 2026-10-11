import numpy as np, glob, json, os, sys
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import splu
from scipy.stats import spearmanr
from e6lib import build, region, edge_len, r, PX, Lw, w
from bfsim import Psi
X,P,A=build(2026); n=len(X)
Z=[np.load(f) for f in glob.glob('../results/e6/bf_w*')]; idx=np.concatenate([z['idx'] for z in Z]); bfv=np.concatenate([z['bf'] for z in Z])
BF=np.zeros(len(P)); BF[idx]=bfv; t=edge_len(X,P)/r
reg=region(X); x=X[:,0]%PX
# distance to the walls (in x) for room points
dwall=np.minimum.reduce([np.abs(x-1),np.abs(x-(1+Lw)),np.abs(x-(2+Lw)),np.abs(x-PX),np.abs(x)])
interior=(reg[P[:,0]]<2)&(reg[P[:,1]]<2)&(dwall[P[:,0]]>2*r)&(dwall[P[:,1]]>2*r)
coef=np.polyfit(t[interior],BF[interior]-Psi(2,t[interior]),3)
cen=BF-Psi(2,t)-np.polyval(coef,t)
corr=(reg[P[:,0]]==2)|(reg[P[:,1]]==2)
face=(~corr)&((dwall[P[:,0]]<=r)|(dwall[P[:,1]]<=r))
def describe(score,q=0.01):
    k=int(q*len(P)); sel=np.argsort(score)[:k]
    return dict(k=k,frac_corridor=float(corr[sel].mean()),frac_wallface=float(face[sel].mean()),frac_interior=float(interior[sel].mean()),mean_t=float(t[sel].mean()))
mode=sys.argv[1] if len(sys.argv)>1 else 'describe'
if mode=='describe':
    out=dict(n=n,E=len(P),base_corridor=float(corr.mean()),base_wallface=float(face.mean()),
             spearman_raw_t=float(spearmanr(BF,t)[0]),spearman_cen_t=float(spearmanr(cen,t)[0]),
             raw=describe(BF),cen=describe(cen),raw_01=describe(BF,0.001),cen_01=describe(cen,0.001),coef=coef.tolist())
    print(json.dumps(out,indent=1)); json.dump(out,open('../results/e6_describe.json','w'),indent=1)
else:
    strat=mode; rng=np.random.default_rng(int(os.environ.get('SEED','0')))
    ip,ix=A.indptr,A.indices; Nb=[set(ix[ip[i]:ip[i+1]]) for i in range(n)]
    order={'raw':np.argsort(BF),'cen':np.argsort(cen),'random':rng.permutation(len(P)),'oracle':np.r_[rng.permutation(np.where(corr)[0]),rng.permutation(np.where(~corr)[0])]}[strat] if strat!='none' else []
    pr=np.random.default_rng(5); As=np.where(reg==0)[0]; Bs=np.where(reg==1)[0]
    As=As[dwall[As]>0.3]; Bs=Bs[dwall[Bs]>0.3]; pairs=list(zip(pr.choice(As,30),pr.choice(Bs,30)))
    res={}
    added=[]; ptr=0
    for B in [0,50,100,200,400]:
        while len(added)<B and ptr<len(order):
            i,j=P[order[ptr]]; ptr+=1
            K=list(Nb[i]-Nb[j]-{j}); Lc=list(Nb[j]-Nb[i]-{i})
            if not K or not Lc: continue
            for _ in range(20):
                k=K[rng.integers(len(K))]; l=Lc[rng.integers(len(Lc))]
                if l not in Nb[k] and k!=l:
                    Nb[k].add(l); Nb[l].add(k); added.append((k,l)); break
        E2=np.vstack([P,np.array(added).reshape(-1,2)]) if added else P
        M=csr_matrix((np.ones(2*len(E2)),(np.r_[E2[:,0],E2[:,1]],np.r_[E2[:,1],E2[:,0]])),shape=(n,n))
        L=diags(np.asarray(M.sum(1)).ravel())-M
        g=0; keep=np.arange(1,n); lu=splu(L[keep][:,keep].tocsc())
        R=[]
        for a,b in pairs:
            rhs=np.zeros(n); rhs[a]+=1; rhs[b]-=1; xs=np.zeros(n); xs[keep]=lu.solve(rhs[keep]); R.append(xs[a]-xs[b])
        ad=np.array(added).reshape(-1,2)
        cross=int(np.sum(reg[ad[:,0]]!=reg[ad[:,1]])) if len(ad) else 0
        res[B]=dict(Reff=float(np.mean(R)),cross_region_added=cross)
    json.dump(res,open(f'../results/e6_rew_{strat}_s{os.environ.get("SEED","0")}.json','w')); print(strat,res)
