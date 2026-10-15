import glob, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import paperdefs_tf as P

def relaxed(d, kmax, tol=0.1):
    key = P.te_key(d); L = np.asarray(d['loss'], float); obs = [o for o in d['obs'] if o['step'] < len(L) and key in o]
    c = P.cfg_of(d); ntr = int(c['frac'] * c['p'] ** 2)
    st = np.array([o['step'] for o in obs]); te = np.array([o[key] for o in obs])
    mis = np.array([round((1 - o['train_acc']) * ntr) for o in obs]); n = len(st)
    bmax = np.empty(n); bmin = np.empty(n)
    for i in range(n):
        e = st[i + 1] if i + 1 < n else st[i] + 1
        seg = L[st[i]:min(e, len(L))]; bmax[i] = seg.max(); bmin[i] = seg.min()
    cl = np.clip(te, 0.1, 0.9); bq, arg = 0.0, None
    for a in range(n):
        if mis[a] > kmax: continue
        mx = np.maximum.accumulate(bmax[a:]); mn = np.minimum.accumulate(bmin[a:]); ok = mx / mn - 1 <= tol
        bad = np.where(mis[a:] > kmax)[0]
        if len(bad): ok[bad[0]:] = False
        stop = np.where(~ok)[0]; m = stop[0] if len(stop) else len(ok)
        if m == 0: continue
        q = (cl[a:a + m] - cl[a]) / 0.8; j = int(np.argmax(q))
        if q[j] > bq: bq, arg = float(q[j]), (int(st[a]), int(st[a + j]))
    return bq, arg

out = {}
for fn in sorted(glob.glob(os.path.join(sys.argv[1], 'W_R[145]_*', 'run.json.gz'))):
    rid = os.path.basename(os.path.dirname(fn)); d = P.load(fn); c = P.cfg_of(d); ntr = int(c['frac'] * c['p'] ** 2)
    key = P.te_key(d); s, te = P.series(d['obs'], key); _, tr = P.series(d['obs'], 'train_acc')
    ff = P.first_fit(d, key); t90 = P.first_cross(s, te, 0.9); t20 = P.first_cross(s, te, 0.2)
    m = (s >= (ff.get('T_fit') or 0)) & (s <= (t90 if t90 else s[-1]))
    mis = np.round((1 - tr[m]) * ntr); frac_bad = float(np.mean(mis > 0)) if m.any() else None
    row = dict(ntr=ntr, T_fit=ff.get('T_fit'), t20=t20, t90=t90, max_mis_fit_to_t90=int(mis.max()) if m.any() else None,
               share_logs_with_mis=frac_bad)
    for k in (0, 1, 3, 10, int(0.01 * ntr)):
        row[f'q_k{k}'] = relaxed(d, k)
    out[rid] = row
    print(f"{rid[2:]:42s} ntr {ntr} T_fit {row['T_fit']} t20 {t20} t90 {t90} | max mis {row['max_mis_fit_to_t90']} (logs with any: {frac_bad:.2f}) | "
          + ' '.join(f"k{k}: {row[f'q_k{k}'][0]:.2f}" for k in (0, 1, 3, 10, int(0.01 * ntr))))
json.dump(out, open(sys.argv[2], 'w'), indent=1)

print('\nrelaxed fit (<= 10 misclassified training pairs at every log), other conditions as fixed in PLAN.md:')
for fn in sorted(glob.glob(os.path.join(sys.argv[1], 'W_R[145]_*', 'run.json.gz'))):
    rid = os.path.basename(os.path.dirname(fn)); d = P.load(fn); key = P.te_key(d); nm = '_nm' if key == 'test_acc_nm' else ''
    q, I = out[rid]['q_k10']
    s_all, te_all = P.series(d['obs'], 'test_acc'); final = float(te_all[-1])
    if I:
        a, b = I; sr, rf = P.series(d['obs'], 'refit_test_acc' + nm); sp, pb = P.series(d['obs'], 'probe_test_acc' + nm)
        sa, ah = P.series(d['obs'], 'AH_full_resid'); L = np.asarray(d['loss'], float); lp = math.log(P.cfg_of(d)['p'])
        g_r = P.inside_gain(sr, rf, a, b); g_p = P.inside_gain(sp, pb, a, b); dAH = P.at(sa, ah, b) - P.at(sa, ah, a)
        ce = float(np.median(L[a:b + 1]) / lp); v = float(L[a:b + 1].max() / L[a:b + 1].min() - 1)
    else: g_r = g_p = dAH = ce = v = float('nan')
    ok = bool(I and q >= 0.75 and g_r >= 0.2 and dAH >= 0.02 and final >= 0.99)
    out[rid]['relaxed10'] = dict(q=q, I=I, ridge_gain=g_r, probe_gain=g_p, dAH=dAH, ce=ce, v=v, final=final, meets=ok)
    print(f"{rid[2:]:42s} q {q:.2f} I {I} CE/lp {ce:.3f} v {v:.3f} ridge +{g_r:.2f} probe +{g_p:.2f} dAH {dAH:.3f} final {final:.3f} -> {'MEETS' if ok else 'no'}")
json.dump(out, open(sys.argv[2], 'w'), indent=1)
