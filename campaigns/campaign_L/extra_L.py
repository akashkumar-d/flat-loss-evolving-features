import glob, json, math, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import endpoints_i as E
LRES = 'collected/results'; KRES = '/mnt/user-data/uploads/modular arithmetic/natural_plateau_shared/campaign_K/collected/results'
def first(s, v, thr):
    i = np.where(v >= thr)[0]; return int(s[i[0]]) if len(i) else None
def stats(fn):
    d = E.load(fn); c = d['cfg']; p = c['p']; L = np.asarray(d['loss'], float); lp = math.log(p); cm = (L[:-1] + L[1:]) / 2
    s, te = E.series(d['obs'], 'test_acc'); _, tr = E.series(d['obs'], 'train_acc'); sr, rf = E.series(d['obs'], 'v1_ridge_acc_te')
    ntr = int(c['frac'] * p * p)
    a = first(sr, rf, 0.2); b = first(s, te, 0.9)
    o = dict(p=p, D=c['depth'], final=float(te[-1]), refit_final=float(rf[-1]), a=a, b=b, horizon=len(L) - 1)
    if a is not None and b is not None and b > a:
        seg = cm[a:b]; o['ce_fall'] = float(1 - seg[-1] / seg[0]); o['v_cm'] = float(seg.max() / seg.min() - 1)
        m = (s >= a) & (s <= b); o['max_mis'] = int(round(float(np.max(1 - tr[m])) * ntr)); o['ce_a'] = float(cm[a] / lp)
    o['ce_peak'] = float(cm[int(np.argmin(cm[:20000])):].max() / lp) if len(cm) > 100 else None
    return o
R = json.load(open('analysis/runs_L.json'))['runs']
rows = []
for rid, r in sorted(R.items()):
    o = stats(os.path.join(LRES, rid, 'run.json.gz')); o['id'] = rid; o['rule'] = r['dial']['lam_v_rule']; o['lw'] = r['dial']['lam_w']
    kfn = os.path.join(KRES, 'K_' + rid[2:], 'run.json.gz')
    if os.path.exists(kfn): o['K'] = stats(kfn)
    rows.append(o)
json.dump(rows, open('analysis/extra_L.json', 'w'), indent=1)
big = [o for o in rows if o['p'] >= 47]
print('p>=47 runs:', len(big), '| final >= 0.99:', sum(o['final'] >= 0.99 for o in big), '| refit final >= 0.99:', sum(o['refit_final'] >= 0.99 for o in big))
for D in (3, 4):
    for p in (47, 61, 101):
        g = [o for o in big if o['D'] == D and o['p'] == p and 'ce_fall' in o]
        gk = [o['K'] for o in big if o['D'] == D and o['p'] == p and 'K' in o and 'ce_fall' in o['K']]
        if not g: continue
        f = lambda xs, k, sc=100: f"{min(x[k] for x in xs) * sc:.0f}-{max(x[k] for x in xs) * sc:.0f}"
        print(f"D{D} p{p}: CE fall from refit 0.2 to net 0.9 at 0.5: {f(g, 'ce_fall')}% (K 0.8: {f(gk, 'ce_fall') if gk else '-'}%) | "
              f"max misclassified in that span: {min(o['max_mis'] for o in g)}-{max(o['max_mis'] for o in g)} (K: {min(o['max_mis'] for o in gk) if gk else '-'}-{max(o['max_mis'] for o in gk) if gk else '-'}) | n {len(g)}")
small = [o for o in rows if o['p'] == 23]
print('p=23: final held-out', min(o['final'] for o in small), '-', max(o['final'] for o in small), '| refit final', round(min(o['refit_final'] for o in small), 3), '-', round(max(o['refit_final'] for o in small), 3), '| horizon', set(o['horizon'] for o in small))
