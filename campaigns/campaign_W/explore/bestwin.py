import glob, gzip, json, os, sys, math
import numpy as np
ROOTS = {'T': '/mnt/user-data/uploads/modular arithmetic/natural_plateau_shared/campaign_T/collected/results',
         'U': '/mnt/user-data/uploads/modular arithmetic/natural_plateau_shared/campaign_U/collected/results',
         'V': '/home/claude/kt/campaign_V/collected/results',
         'P': '/home/claude/kt/campaign_W/explore/runs'}
def best(L, st, te, mm, tol, strict, key='test'):
    n = len(st); bmax = np.empty(n); bmin = np.empty(n)
    for i in range(n):
        e = st[i + 1] if i + 1 < n else st[i] + 1
        seg = L[st[i]:min(e, len(L))]; bmax[i] = seg.max(); bmin[i] = seg.min()
    c = np.clip(te, 0.1, 0.9); bestq = 0.0; arg = None
    for a in range(n):
        if strict and mm[a] <= 0: continue
        mx = np.maximum.accumulate(bmax[a:]); mn = np.minimum.accumulate(bmin[a:])
        ok = mx / mn - 1 <= tol
        if strict:
            bad = np.where(mm[a:] <= 0)[0]
            if len(bad): ok[bad[0]:] = False
        stop = np.where(~ok)[0]; m = stop[0] if len(stop) else len(ok)
        if m == 0: continue
        q = (c[a:a + m] - c[a]) / 0.8; j = int(np.argmax(q))
        if q[j] > bestq: bestq = float(q[j]); arg = (int(st[a]), int(st[a + j]))
    return bestq, arg
pat = sys.argv[1]; tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
print(f"{'run':54s} {'q_fit':>6s} {'window':>15s} {'q_nofit':>7s} {'window':>15s} final")
for camp, root in ROOTS.items():
    for d in sorted(glob.glob(os.path.join(root, '*'))):
        rid = os.path.basename(d).replace('.json.gz', '')
        if not any(x in rid for x in pat.split(',')): continue
        fn = os.path.join(d, 'run.json.gz') if os.path.isdir(d) else d
        if not fn.endswith('.json.gz') or not os.path.exists(fn): continue
        r = json.load(gzip.open(fn, 'rt')); L = np.array(r['loss'])
        obs = [o for o in r['obs'] if o['step'] < len(L)]
        st = np.array([o['step'] for o in obs]); te = np.array([o['test_acc'] for o in obs]); mm = np.array([o['train_margin_min'] for o in obs])
        q1, w1 = best(L, st, te, mm, tol, True); q0, w0 = best(L, st, te, mm, tol, False)
        print(f"{rid[:54]:54s} {q1:6.2f} {str(w1):>15s} {q0:7.2f} {str(w0):>15s} {te[-1]:.3f}")
