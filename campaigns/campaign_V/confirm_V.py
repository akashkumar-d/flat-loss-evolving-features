#!/usr/bin/env python3
"""Campaign V, confirmation rule (fixed before the runs; claude/1hl-figures, 2026-10-09).

    python3 -I confirm_V.py OUT.json RUN_DIR [RUN_DIR ...]      (each RUN_DIR holds run.json.gz)

The key message (Akash, 2026-10-08): in a flat or rising training loss, the refit gets better, and either within the same
window or later the held-out accuracy goes up. A run PASSES if, for w = 500 or w = 2000 updates, there is a stretch [a, b]
with:
  (1) flat or rising loss: the centred geometric moving average of the training CE over w updates never falls more than 10%
      below its running maximum since a (rising is allowed), and b - a >= 2w;
  (2) the training set stays fit: the training accuracy, averaged over 500 updates (centred, over the logged values), is
      >= 0.97 at every logged step of [a, b] (so single loss spikes do not decide it);
  (3) the ridge refit's held-out accuracy rises by >= 0.2 from the first refit observation at or after a to the last at or
      before b;
  (4) the held-out accuracy rises by >= 0.2 from a to the end of the run, and its final value is >= 0.99, where the final
      value is the median held-out accuracy over the last 2000 updates of the run.
Starts a are the updates with a refit observation; each start is extended as far as (1) holds.

A setting is CONFIRMED if at least 8 of its 10 fresh seeds pass. Also reported (secondary): the same rule with the logistic
probe in place of the ridge refit, and campaign U's two earlier versions (pass: (2) dropped; pass3: (2) on the unsmoothed
training accuracy), at the same windows.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import flat_windows as F

WINDOWS = (500, 2000)
DELTA, GAIN, FINAL, TRAIN, TRAIN_AVG, FINAL_SPAN = 0.1, 0.2, 0.99, 0.97, 500, 2000


def centred_mean(v, k):
    if k <= 1: return v.astype(float).copy()
    c = np.concatenate([[0.0], np.cumsum(v, dtype=float)]); n = len(v); h = k // 2
    lo = np.clip(np.arange(n) - h, 0, n); hi = np.clip(np.arange(n) - h + k, 0, n)
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def evaluate(d):
    L = np.array(d['loss'], float); n = len(L); obs = d['obs']
    s, te = F.series(obs, 'test_acc'); st, tr = F.series(obs, 'train_acc')
    every = int(np.median(np.diff(st))) if len(st) > 1 else 1
    trs = centred_mean(tr, max(1, TRAIN_AVG // every))
    last = s >= s[-1] - FINAL_SPAN; final = float(np.median(te[last]))
    out = dict(final=final, updates=n - 1, train_avg_every=every)
    for ro, key in (('ridge', 'refit_test_acc'), ('probe', 'probe_test_acc')):
        sr, rf = F.series(obs, key)
        res = {}
        for w in WINDOWS:
            x = F.smooth(L, w, 'geo'); starts = [int(a) for a in sr if a < n - 1]
            ends = F.stretch_ends(x, starts, DELTA) if starts else {}
            best = None
            for a in starts:
                b = ends[a]
                if b - a < max(2 * w, 500): continue
                ra, r0 = F.at_or_after(sr, rf, a); rb, r1 = F.at_or_before(sr, rf, b)
                if ra is None or rb is None or rb <= ra: continue
                _, te_a = F.at_or_before(s, te, a)
                m = (st >= a) & (st <= b)
                rec = dict(a=a, b=b, g_rf=r1 - r0, g_te=final - te_a, train_min_smoothed=float(trs[m].min()),
                           train_min_raw=float(tr[m].min()), level=float(np.exp(np.mean(np.log(np.maximum(x[a:b + 1], 1e-300))))))
                base = rec['g_rf'] >= GAIN and rec['g_te'] >= GAIN and final >= FINAL
                rec['pass_loose'] = bool(base); rec['pass_raw3'] = bool(base and rec['train_min_raw'] >= TRAIN)
                rec['pass'] = bool(base and rec['train_min_smoothed'] >= TRAIN)
                if best is None or (rec['pass'], rec['pass_raw3'], rec['pass_loose'], rec['g_rf']) > \
                        (best['pass'], best['pass_raw3'], best['pass_loose'], best['g_rf']): best = rec
            res[str(w)] = best
        ok = [v for v in res.values() if v]
        out[ro] = dict(by_window=res, **{k: any(v[k] for v in ok) for k in ('pass', 'pass_raw3', 'pass_loose')})
    out['pass'] = out['ridge']['pass']
    return out


def main(out, dirs):
    rows = {}
    for d_ in dirs:
        rid = os.path.basename(os.path.normpath(d_))
        rows[rid] = evaluate(F.load(os.path.join(d_, 'run.json.gz')))
        r = rows[rid]
        b = max((v for v in r['ridge']['by_window'].values() if v), key=lambda v: (v['pass'], v['g_rf']), default=None)
        desc = (f"[{b['a']},{b['b']}] CE {b['level']:.2g} ridge +{b['g_rf']:.2f} held-out +{b['g_te']:.2f} "
                f"train(avg500) min {b['train_min_smoothed']:.3f} raw min {b['train_min_raw']:.2f}") if b else 'no stretch'
        print(f"{rid[:46]:46s} PASS={str(r['pass']):5s} probe={str(r['probe']['pass']):5s} loose={str(r['ridge']['pass_loose']):5s} "
              f"final {r['final']:.3f} | {desc}", flush=True)
    json.dump(dict(rule=__doc__, runs=rows), open(out, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
