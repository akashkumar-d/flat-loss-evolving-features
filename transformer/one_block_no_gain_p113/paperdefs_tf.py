#!/usr/bin/env python3
import gzip, json, math
from collections import deque
import numpy as np

BAND, SHARE_MIN, FINAL_MIN, DAH_MIN, GAIN = 1.03, 0.75, 0.99, 0.02, 0.2


def load(fn):
    with (gzip.open(fn, 'rt') if fn.endswith('.gz') else open(fn)) as f: return json.load(f)


def series(obs, key):
    s = [(o['step'], o[key]) for o in obs if key in o and o[key] is not None]
    return np.array([x[0] for x in s], dtype=float), np.array([x[1] for x in s], dtype=float)


def at(s, v, t): return float(v[int(np.argmin(np.abs(s - t)))]) if len(s) else float('nan')


def first_cross(s, v, thr):
    i = np.where(v >= thr)[0]
    return int(s[i[0]]) if len(i) else None


def first_persistent(s, v, thr):
    bad = np.where(v < thr)[0]
    if not len(bad): return int(s[0]) if len(s) else None
    return int(s[bad[-1] + 1]) if bad[-1] + 1 < len(s) else None


def cfg_of(d): return d.get('cfg_full') or d['cfg']


def knee_of(d, mode='cap'):
    c = cfg_of(d); sn, nv = series(d['obs'], 'normV'); cap = 1.0 / c['lam_v']
    if mode == 'paper': ki = np.where(nv >= 0.99 * cap)[0]
    else: ki = np.where((sn > 0) & (np.abs(nv - cap) <= 0.01 * cap))[0]
    return int(sn[ki[0]]) if len(ki) else None


def primary(d, ah='resid'):
    c = cfg_of(d); L = np.asarray(d['loss'], dtype=float); obs = d['obs']; p = c['p']; D = c.get('n_layers', 1)
    s, te = series(obs, 'test_acc'); _, tra = series(obs, 'train_acc'); sm, mtr = series(obs, 'train_margin_min')
    out = dict(final_test=float(te[-1]), t10=first_cross(s, te, 0.1), t90=first_persistent(s, te, 0.9),
               t90_first=first_cross(s, te, 0.9), knee=knee_of(d))
    k = out['knee']; cm = (L[:-1] + L[1:]) / 2
    if k is None or k >= len(cm):
        out.update(primary=False, why_not='no knee'); return out
    ts = int(np.argmax(cm[k:]) + k); ok = cm >= cm[ts] / BAND
    a = ts
    while a > 0 and ok[a - 1]: a -= 1
    b = ts
    while b < len(cm) - 1 and ok[b + 1]: b += 1
    clip = lambda x: min(max(at(s, te, x), 0.1), 0.9)
    share = (clip(b) - clip(a)) / 0.8
    key = 'AH_full_resid' if ah == 'resid' else f'AH_full_h{D}'
    ahs, ahv = series(obs, key); dAH = at(ahs, ahv, b) - at(ahs, ahv, a)
    m = (sm >= a) & (sm <= b); fit = bool(m.any() and np.all(mtr[m] > 0))
    seg = L[a:b + 1]
    out.update(W=[a, b], W_len=b - a, t_peak=ts, CE_peak_over_logp=float(cm[ts]) / math.log(p), share=float(share),
               v_raw=float(seg.max() / seg.min() - 1), dAH=float(dAH), fit_strict=fit)
    checks = dict(share=share >= SHARE_MIN, final=out['final_test'] >= FINAL_MIN, dAH=dAH >= DAH_MIN, fit=fit)
    out['checks'] = checks; out['primary'] = bool(all(checks.values()))
    out['why_not'] = ', '.join(k_ for k_, v_ in checks.items() if not v_) or None
    return out


def stretches(cv, start, tol=BAND):
    n = len(cv); amin = np.full(n, -1, int); dq = deque(); a = start
    for t in range(start, n):
        while dq and cv[dq[-1]] <= cv[t]: dq.pop()
        dq.append(t)
        while cv[dq[0]] > tol * cv[t]:
            a += 1
            while dq[0] < a: dq.popleft()
        amin[t] = a
    return amin


def new_window(cv, knee):
    start = int(min(max(knee or 0, 0), len(cv) - 1)); amin = stretches(cv, start); n = len(cv)
    S = [(int(amin[t]), t) for t in range(start, n) if t == n - 1 or amin[t + 1] != amin[t]]
    return max(S, key=lambda s_: (float(np.max(cv[s_[0]:s_[1] + 1]) - cv[s_[0]]), s_[1] - s_[0]))


def inside_gain(x, y, a, b):
    m = (np.asarray(x) >= a) & (np.asarray(x) <= b)
    return 0.0 if m.sum() < 2 else float(np.asarray(y)[m][-1] - np.asarray(y)[m][0])


def posthoc(d):
    c = cfg_of(d); lp = math.log(c['p']); L = np.asarray(d['loss'], float) / lp; cm = (L[:-1] + L[1:]) / 2
    k = knee_of(d)
    if k is None: return dict(W2=None, ok2=False, why='no knee')
    t, te = series(d['obs'], 'test_acc'); tf, rf = series(d['obs'], 'refit_test_acc')
    a, b = new_window(cm, k)
    g_te, g_rf = inside_gain(t, te, a, b), inside_gain(tf, rf, a, b)
    ts, tr = series(d['obs'], 'train_acc'); m = (ts >= a) & (ts <= b); ntr = int(c['frac'] * c['p'] ** 2)
    return dict(W2=[a, b], ok2=bool(g_te >= GAIN or g_rf >= GAIN), g_te=g_te, g_rf=g_rf, knee=k,
                mis_in_W=int(round(float(np.max(1 - tr[m])) * ntr)) if m.any() else None)


def raw10(d, tol=0.1, key='test_acc'):
    L = np.asarray(d['loss'], float); obs = [o for o in d['obs'] if o['step'] < len(L) and key in o]
    st = np.array([o['step'] for o in obs]); te = np.array([o[key] for o in obs])
    mm = np.array([o['train_margin_min'] for o in obs]); n = len(st)
    bmax = np.empty(n); bmin = np.empty(n)
    for i in range(n):
        e = st[i + 1] if i + 1 < n else st[i] + 1
        seg = L[st[i]:min(e, len(L))]; bmax[i] = seg.max(); bmin[i] = seg.min()
    cl = np.clip(te, 0.1, 0.9); res = {}
    for strict in (True, False):
        bq, arg = 0.0, None
        for a in range(n):
            if strict and mm[a] <= 0: continue
            mx = np.maximum.accumulate(bmax[a:]); mn = np.minimum.accumulate(bmin[a:]); ok = mx / mn - 1 <= tol
            if strict:
                bad = np.where(mm[a:] <= 0)[0]
                if len(bad): ok[bad[0]:] = False
            stop = np.where(~ok)[0]; m = stop[0] if len(stop) else len(ok)
            if m == 0: continue
            q = (cl[a:a + m] - cl[a]) / 0.8; j = int(np.argmax(q))
            if q[j] > bq: bq, arg = float(q[j]), (int(st[a]), int(st[a + j]))
        res['fit' if strict else 'nofit'] = (bq, arg)
    return res


def te_key(d):
    return 'test_acc_nm' if any('test_acc_nm' in o for o in d['obs'][:3]) else 'test_acc'


def first_fit(d, key='test_acc'):
    s, mm = series(d['obs'], 'train_margin_min'); _, te = series(d['obs'], key)
    pos = np.where(mm > 0)[0]
    if not len(pos): return dict(T_fit=None)
    i = pos[0]; t90 = first_cross(s, te, 0.9)
    seg = (s >= s[i]) & (s <= (t90 if t90 is not None else s[-1]))
    return dict(T_fit=int(s[i]), te_at_fit=float(te[i]), relapse=bool(np.any(mm[seg] <= 0)))
