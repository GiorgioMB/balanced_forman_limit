import numpy as np,glob,collections
D=collections.defaultdict(list)
for f in glob.glob('/tmp/w/res/c_*.txt'):
    p=f.split('/')[-1].split('_'); k=(int(p[1]),p[2])
    try: a=np.loadtxt(f,ndmin=2)
    except Exception: continue
    if a.size: D[k].append(a[:,1:])
for k in sorted(D,key=lambda k:(k[1],k[0])):
    a=np.vstack(D[k]); m=a.mean(0); s=a.std(0)/np.sqrt(len(a)); rest=a[:,0]-a[:,1]-a[:,2]
    print('lam=%4d eps=%s n=%6d  tot %.3f±%.3f  S %.3f±%.3f  J %.3f±%.3f (J/sqrt=%.4f)  rest %.3f±%.3f'%(k[0],k[1],len(a),m[0],s[0],m[1],s[1],m[2],s[2],m[2]/np.sqrt(k[0]),rest.mean(),rest.std()/np.sqrt(len(a))))
