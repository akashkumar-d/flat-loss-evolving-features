import gzip, json, math
import numpy as np

K1 = 0.03 * 0.015
BAND, SHARE_MIN, FINAL_MIN, DAH_MIN = 1.03, 0.75, 0.99, 0.02


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


def fields(d):
    c = d['cfg']; L = np.asarray(d['loss'], dtype=float); obs = d['obs']; T = len(L) - 1; D = c['depth']; p = c['p']
    s, te = series(obs, 'test_acc'); _, tra = series(obs, 'train_acc'); sm, mtr = series(obs, 'train_margin_min')
    rate = c['eta_layers'][-1] * c['lam_layers'][-1] / K1 if 'eta_layers' in c else float('nan')
    out = dict(depth=D, horizon=T, rate=rate, final_test=float(te[-1]), final_train=float(tra[-1]),
               t10=first_cross(s, te, 0.1), t90=first_persistent(s, te, 0.9), t90_first=first_cross(s, te, 0.9))
    out['x'] = rate * K1 * out['t10'] if out['t10'] is not None and not math.isnan(rate) else None
    sn, nv = series(obs, 'normV'); ki = np.where(nv >= 0.99 / c['lam_v'])[0]
    k = int(sn[ki[0]]) if len(ki) else None; out['knee'] = k
    cm = (L[:-1] + L[1:]) / 2
    if k is None or k >= len(cm):
        out['primary'] = False; out['why_not'] = 'no knee'; return out
    ts = int(np.argmax(cm[k:]) + k); ok = cm >= cm[ts] / BAND
    a = ts
    while a > 0 and ok[a - 1]: a -= 1
    b = ts
    while b < len(cm) - 1 and ok[b + 1]: b += 1
    clip = lambda x: min(max(at(s, te, x), 0.1), 0.9)
    share = (clip(b) - clip(a)) / 0.8
    ahs, ahv = series(obs, f'AH_full_h{D}'); dAH = at(ahs, ahv, b) - at(ahs, ahv, a)
    m = (sm >= a) & (sm <= b); fit = bool(m.any() and np.all(mtr[m] > 0))
    seg = L[a:b + 1]
    out.update(W=[a, b], W_len=b - a, t_peak=ts, CE_peak=float(cm[ts]), CE_peak_over_logp=float(cm[ts]) / math.log(p),
               share=float(share), v_min=float((seg.max() - seg.min()) / seg.min()), dAH_last=float(dAH), fit_strict=fit)
    checks = dict(share=share >= SHARE_MIN, final=out['final_test'] >= FINAL_MIN, F02=dAH >= DAH_MIN, fit=fit)
    out['primary_checks'] = checks; out['primary'] = bool(all(checks.values()))
    out['why_not'] = ', '.join(k_ for k_, v_ in checks.items() if not v_) or None
    return out


def wilson(k, n, z=1.959963984540054):
    if n == 0: return (float('nan'), float('nan'))
    ph = k / n; den = 1 + z * z / n; c = (ph + z * z / (2 * n)) / den; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    m = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n)
