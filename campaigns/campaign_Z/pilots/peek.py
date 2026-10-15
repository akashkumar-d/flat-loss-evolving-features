import torch, glob, numpy as np, sys, json, gzip, os
def smooth(L, w=201):
    pad=w//2; Lp=np.concatenate([np.full(pad,L[0]),L,np.full(pad,L[-1])]); return np.convolve(Lp,np.ones(w)/w,mode='valid')
for f in sorted(glob.glob('/home/claude/kt/pilotZ/runs/*.ckpt.pt')) + sorted(glob.glob('/home/claude/kt/pilotZ/runs/*.json.gz')):
    if f.endswith('.json.gz'):
        st=json.load(gzip.open(f,'rt')); tn=len(st['loss'])
    else:
        st=torch.load(f, weights_only=False); tn=st['t_next']
    obs=st['obs']; L=np.array(st['loss'][:tn]); cm=smooth(L) if len(L)>250 else L
    every=int(sys.argv[1]) if len(sys.argv)>1 else 1000
    print('==', os.path.basename(f), 'updates', tn)
    for o in obs:
        if o['step'] % every: continue
        t=o['step']; ah=o.get('AH_full_resid')
        print(f"  t={t:6d} CE201={cm[min(t,len(cm)-1)]:.3f} train={o['train_acc']:.4f} all={o['test_acc']:.3f} nm={o.get('test_acc_nm',float('nan')):.3f} g={o.get('gain',float('nan')):.3f}" + (f" AH={ah:.3f}" if ah is not None else '') + (f" AH1={o['AH_full_h1']:.3f}" if 'AH_full_h1' in o else '') + (f" AH2={o['AH_full_h2']:.3f}" if 'AH_full_h2' in o else ''))
