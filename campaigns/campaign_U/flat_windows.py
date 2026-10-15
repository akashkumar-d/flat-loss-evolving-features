#!/usr/bin/env python3
"""Flatness at several averaging windows, and the key message checked at each window (campaign U, claude/1hl-figures,
2026-10-08).

    python3 -I flat_windows.py RESULTS_DIR OUT.json [--delta 0.1] [--readout refit|probe]

RESULTS_DIR holds <run>/run.json.gz (campaign T or U layout). The key message (Akash, 2026-10-08): "in a flat or rising
training loss, the refit gets better, and either within the same window or later the held-out accuracy goes up".

Smoothing. The training CE is recorded at every update. For a window of w updates, x_w is its centred moving average, either
arithmetic ('arith': the mean CE over the window, dominated by spikes) or geometric ('geo': exp of the mean log CE, the typical
level between spikes). w = 1 is the raw CE.

Flat stretch at scale w. A stretch [a, b] in which x_w never falls more than delta below its running maximum since a
(x_w(t) >= (1 - delta) max_{a <= s <= t} x_w(s)); rising is allowed. Starts a are the updates with a refit observation; each
start is extended as far as the condition holds.

Key message at scale w ("pass"): some flat stretch [a, b] at scale w, at least max(2w, 500) updates long, in which
  (i)   the refit's held-out accuracy rises by at least 0.2 (from the first refit observation at or after a to the last at or
        before b),
  (ii)  the held-out accuracy rises by at least 0.2 from a to the end of the run (in the same stretch or later), and
  (iii) the final held-out accuracy is at least 0.99.
"pass3": pass with training accuracy >= 0.97 at every logged step of the stretch (at most 3% of training pairs misclassified).
Readout (campaign U): 'refit' is the train-only ridge refit (refit_test_acc, every heavy observation); 'probe' is the
train-only logistic probe (probe_test_acc, recorded by tf_runner 1.2 every probe_every updates).
Reported per run, window and kind: pass, pass3, the stretch (the passing one with the largest refit gain, else the stretch with
the largest refit gain), its refit and held-out gains, its CE level (geometric mean of x_w over the stretch) and its minimum
training accuracy.

Flatness profile. With I the feature-learning interval (from the first time the refit or the held-out accuracy reaches 0.1 to
the later of their first times at 0.9), the drawdown of x_w over I (largest relative fall below the running maximum) and the
ratio x_w(end of I) / x_w(start of I); w* = the smallest window whose drawdown over I is at most 0.3. A window as long as I
itself averages the event away, so only w* well below |I| marks a plateau.
"""
import glob, gzip, json, math, os, sys
import numpy as np

WINDOWS = (1, 100, 500, 2000, 5000, 10000)
KINDS = ('arith', 'geo')
GAIN, FINAL, TRAIN3, MIN_LEN = 0.2, 0.99, 0.97, 500


def load(fn):
    with gzip.open(fn, 'rt') as f: return json.load(f)


def series(obs, key):
    s = [(o['step'], o[key]) for o in obs if o.get(key) is not None]
    return np.array([a for a, _ in s], int), np.array([b for _, b in s], float)


def first(s, v, thr):
    i = np.where(v >= thr)[0]; return int(s[i[0]]) if len(i) else None


def smooth(L, w, kind):
    if w <= 1: return L.copy()
    y = np.log(np.maximum(L, 1e-300)) if kind == 'geo' else L
    c = np.concatenate([[0.0], np.cumsum(y)]); n = len(y); h = w // 2
    lo = np.clip(np.arange(n) - h, 0, n); hi = np.clip(np.arange(n) - h + w, 0, n)
    m = (c[hi] - c[lo]) / np.maximum(hi - lo, 1)
    return np.exp(m) if kind == 'geo' else m


def stretch_end(x, a, delta):
    seg = x[a:]; rm = np.maximum.accumulate(seg); bad = np.where(seg < (1 - delta) * rm)[0]
    return a + (int(bad[0]) - 1 if len(bad) else len(seg) - 1)


def violation_sources(x, delta):
    import bisect
    c = 1.0 / (1.0 - delta); n = len(x); p = np.full(n, -1, dtype=np.int64)
    st_i, st_neg = [], []
    for t in range(n):
        v = x[t]
        k = bisect.bisect_left(st_neg, -c * v)
        if k: p[t] = st_i[k - 1]
        while st_neg and -st_neg[-1] <= v: st_neg.pop(); st_i.pop()
        st_i.append(t); st_neg.append(-v)
    return p


def stretch_ends(x, starts, delta):
    p = violation_sources(x, delta); n = len(x); out = {}
    for a in starts:
        hit = np.nonzero(p[a:] >= a)[0]
        out[int(a)] = int(a + hit[0] - 1) if len(hit) else n - 1
    return out


def at_or_after(s, v, t):
    i = np.searchsorted(s, t, 'left'); return (int(s[i]), float(v[i])) if i < len(s) else (None, None)


def at_or_before(s, v, t):
    i = np.searchsorted(s, t, 'right') - 1; return (int(s[i]), float(v[i])) if i >= 0 else (None, None)


def key_message(d, x, w, delta, L_n, readout='refit', ends=None):
    s, te = series(d['obs'], 'test_acc'); st, tr = series(d['obs'], 'train_acc')
    sr, rf = series(d['obs'], 'refit_test_acc' if readout == 'refit' else 'probe_test_acc')
    final = float(te[-1]); best = None; best_pass = None
    if len(sr) == 0:
        return dict(a=None, b=None, g_rf=None, g_te=None, level=None, train_acc_min=None, final=final, **{'pass': False, 'pass3': False})
    if ends is None: ends = stretch_ends(x, [int(a) for a in sr if a < L_n - 1], delta)
    for a in sr:
        a = int(a)
        if a >= L_n - 1: break
        b = ends[a] if a in ends else stretch_end(x, a, delta)
        if b - a < max(2 * w, MIN_LEN): continue
        ra, r0 = at_or_after(sr, rf, a); rb, r1 = at_or_before(sr, rf, b)
        if ra is None or rb is None or rb <= ra: continue
        g_rf = r1 - r0; _, te_a = at_or_before(s, te, a); g_te = final - te_a
        k = (st >= a) & (st <= b); trmin = float(tr[k].min()) if k.any() else float('nan')
        lvl = float(np.exp(np.mean(np.log(np.maximum(x[a:b + 1], 1e-300)))))
        rec = dict(a=a, b=b, g_rf=g_rf, g_te=g_te, level=lvl, train_acc_min=trmin)
        ok = g_rf >= GAIN and g_te >= GAIN and final >= FINAL
        rec['pass'] = bool(ok); rec['pass3'] = bool(ok and trmin >= TRAIN3)
        if best is None or g_rf > best['g_rf']: best = rec
        if ok and (best_pass is None or (rec['pass3'], g_rf) > (best_pass['pass3'], best_pass['g_rf'])): best_pass = rec
    out = best_pass or best or dict(a=None, b=None, g_rf=None, g_te=None, level=None, train_acc_min=None, **{'pass': False, 'pass3': False})
    out['final'] = final
    return out


def profile(d, L, xs):
    s, te = series(d['obs'], 'test_acc'); sr, rf = series(d['obs'], 'refit_test_acc')
    r10, r90, h10, h90 = first(sr, rf, 0.1), first(sr, rf, 0.9), first(s, te, 0.1), first(s, te, 0.9)
    starts = [t for t in (r10, h10) if t is not None]; ends = [t for t in (r90, h90) if t is not None]
    out = dict(refit10=r10, refit90=r90, heldout10=h10, heldout90=h90)
    if not starts or not ends or max(ends) <= min(starts): return out
    a, b = min(starts), min(max(ends), len(L) - 1); out['I'] = [a, b]
    for kind in KINDS:
        prof = {}
        for w in WINDOWS:
            seg = xs[(kind, w)][a:b + 1]; rm = np.maximum.accumulate(seg)
            prof[str(w)] = dict(drawdown=float(np.max(1 - seg / rm)), ratio=float(seg[-1] / seg[0]),
                                level=float(np.exp(np.mean(np.log(np.maximum(seg, 1e-300))))))
        out[kind] = prof
        flat = [w for w in WINDOWS if prof[str(w)]['drawdown'] <= 0.3]
        out[f'w_star_{kind}'] = flat[0] if flat else None
    return out


def analyse(fn, delta, readouts=('refit',)):
    d = load(fn); L = np.array(d['loss'], float); n = len(L)
    xs = {(k, w): smooth(L, w, k) for k in KINDS for w in WINDOWS}
    rid = os.path.basename(os.path.dirname(fn)); c = d.get('cfg_full', d['cfg'])
    rec = dict(id=rid, optimizer=c.get('optimizer'), n_layers=c.get('n_layers'), final_norm=c.get('final_norm', 'none'),
               frac=c.get('frac'), lam_w=c.get('lam_w'), eta_w=c.get('eta_w'), lam_v=c.get('lam_v'), wd=c.get('wd'),
               updates=n - 1, stop_reason=d.get('stop_reason'), delta=delta, log_p=math.log(c['p']))
    rec['profile'] = profile(d, L, xs)
    starts = sorted({int(o['step']) for o in d['obs'] if o.get('refit_test_acc') is not None or o.get('probe_test_acc') is not None})
    starts = [a for a in starts if a < n - 1]
    for ro in readouts:
        key = {}
        for k in KINDS:
            for w in WINDOWS:
                ends = stretch_ends(xs[(k, w)], starts, delta)
                key[f'{k}_{w}'] = key_message(d, xs[(k, w)], w, delta, n, ro, ends)
        rec['key' if ro == 'refit' else 'key_' + ro] = key
    return rec


def main(res, out, delta=0.1, readouts=('refit',)):
    rows = []
    for fn in sorted(glob.glob(os.path.join(res, '*', 'run.json.gz'))):
        try: rows.append(analyse(fn, delta, readouts))
        except Exception as e: print('skip', fn, type(e).__name__, e, flush=True)
    json.dump(dict(rule=__doc__.split('\n\n')[2:6], delta=delta, windows=WINDOWS, readouts=list(readouts), runs=rows), open(out, 'w'), indent=1)
    for r in rows:
        k = r['key']; cells = []
        for kind in KINDS:
            cells.append(kind[0] + ':' + ''.join(('P' if k[f'{kind}_{w}']['pass3'] else 'p' if k[f'{kind}_{w}']['pass'] else '.') for w in WINDOWS))
        best = [k[f'geo_{w}'] for w in WINDOWS if k[f'geo_{w}']['pass']]
        lv = f"{best[0]['level']:.1e}" if best else '-'
        print(f"{r['id']:48s} {' '.join(cells)}  level(geo, first pass) {lv}  w*(geo) {r['profile'].get('w_star_geo')}  final {k['geo_1']['final']:.3f}", flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]; dl = 0.1; ro = ('refit',)
    if '--delta' in a: i = a.index('--delta'); dl = float(a[i + 1]); del a[i:i + 2]
    if '--readout' in a: i = a.index('--readout'); ro = tuple(a[i + 1].split(',')); del a[i:i + 2]
    main(a[0], a[1], dl, ro)
