import glob, json, math, os, sys
sys.path.insert(0, '/home/claude/kt/campaign_L')
import numpy as np
import endpoints_i as E

def first(s, v, thr):
    i = np.where(v >= thr)[0]; return int(s[i[0]]) if len(i) else None

def one(fn):
    d = E.load(fn); c = d['cfg']; p = c['p']; lp = math.log(p)
    L = np.asarray(d['loss'], float) / lp; cm = (L[:-1] + L[1:]) / 2; T = len(cm)
    s, te = E.series(d['obs'], 'test_acc'); st, tr = E.series(d['obs'], 'train_acc')
    sr, rf = E.series(d['obs'], 'v1_ridge_acc_te')
    sn, nv = E.series(d['obs'], 'normV'); ki = np.where(nv >= 0.99 / c['lam_v'])[0]; knee = int(sn[ki[0]]) if len(ki) else 0
    pk = knee + int(np.argmax(cm[knee:])); cpk = float(cm[pk])
    dip = int(np.argmin(cm[:pk + 1])); cdip = float(cm[dip])
    out = dict(p=p, D=c['depth'], lam_v=c['lam_v'], lam_w=c['lam_w'], frac=c['frac'], T=T, knee=knee, t_dip=dip, ce_dip=cdip,
               t_pk=pk, ce_pk=cpk)
    for nm, (x, y), thrs in (('rf', (sr, rf), (0.2, 0.5, 0.9)), ('te', (s, te), (0.1, 0.5, 0.9))):
        for th in thrs:
            t = first(x, y, th); out[f'{nm}{int(th*100)}'] = t
            out[f'ce_{nm}{int(th*100)}'] = float(cm[min(t, T - 1)]) / cpk if t is not None else None
    for f_ in (0.97, 0.9):
        i = np.where(cm[pk:] <= f_ * cpk)[0]; out[f't_down{int(round((1-f_)*100))}'] = int(pk + i[0]) if len(i) else None
    a, b = out['rf20'], out['te90']
    if a is not None and b is not None and b > a:
        m = (st >= a) & (st <= b); out['min_tr_rise'] = float(tr[m].min()) if m.any() else None
        out['ce_fall_rise'] = 1 - float(cm[min(b, T-1)]) / float(cm[a])
    out['final_te'] = float(te[-1]); out['final_rf'] = float(rf[-1]) if len(rf) else None
    return out

if __name__ == '__main__':
    rows = {}
    for root in sys.argv[1:-1]:
        for fn in sorted(glob.glob(os.path.join(root, '*', 'run.json.gz'))):
            rid = os.path.basename(os.path.dirname(fn)); rows[rid] = one(fn); print(rid, flush=True)
    json.dump(rows, open(sys.argv[-1], 'w'), indent=0)
