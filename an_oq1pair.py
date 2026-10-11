import numpy as np, glob, json
from oq1pair import theory
rows=[]
for th in [10,30,60,90,120,150,180]:
    X=np.concatenate([np.load(f) for f in glob.glob(f'../results/oq1/pair_th{th}_w*')]); c=np.corrcoef(X.T)[0,1]; n=len(X)
    se=(1-c*c)/np.sqrt(n-3)
    ct,_=theory(0.75,np.deg2rad(th),M=300000,G=300000)
    rows.append(dict(theta=th,n=n,corr=float(c),se=float(se),theory=ct)); print(th,n,'sim %.3f ± %.3f   theory %.3f'%(c,se,ct))
json.dump(rows,open('../results/oq1_pair.json','w'),indent=1)
