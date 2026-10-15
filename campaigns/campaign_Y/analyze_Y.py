#!/usr/bin/env python3
"""Campaign Y analysis, fixed in PLAN.md section 4 (claude/1hl-figures, 2026-10-11).

Usage:  python3 analyze_Y.py Y_RESULTS_DIR X_RESULTS_DIR OUT.json
  Y_RESULTS_DIR   campaign_Y/collected/results (one folder per run with run.json.gz)
  X_RESULTS_DIR   campaign_X/collected/results (its R5 runs, seeds 11-20, are the no-gain contrast)
Needs campaign X's analyze_X.py and paperdefs_tf.py (path: ../campaign_X, or CAMPAIGN_X_DIR).
"""
import glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('CAMPAIGN_X_DIR', os.path.join(HERE, '..', 'campaign_X')))
import analyze_X as AX

AVG = 201
DROP = 0.03
SHARE, GAIN, DAH, FINAL, TOL = 0.75, 0.2, 0.02, 0.99, 10


def smooth(L, w=AVG):
    pad = w // 2; Lp = np.concatenate([np.full(pad, L[0]), L, np.full(pad, L[-1])])
    return np.convolve(Lp, np.ones(w) / w, mode='valid')


def at(s, v, t): return float(v[int(np.argmin(np.abs(s - t)))])


def share(s, v, a, b):
    clip = lambda t: min(max(at(s, v, t), 0.1), 0.9)
    return (clip(b) - clip(a)) / 0.8


def paper_window(cm, start, s, v, step=20):
    best = (0.0, None); n = len(cm)
    for a in range(int(start), n - 1, step):
        rm = np.maximum.accumulate(cm[a:]); bad = np.where(cm[a:] < (1 - DROP) * rm)[0]
        b = a + (int(bad[0]) - 1 if len(bad) else n - 1 - a)
        if b <= a: continue
        q = share(s, v, a, b)
        if q > best[0] + 1e-12: best = (q, (a, b))
    return best


def gain_inside(s, v, a, b): return at(s, v, b) - at(s, v, a)


def per_run(fn):
    rec = AX.per_run(fn, 'tf')
    d = AX.load(fn); obs = d['obs']; c = d.get('cfg_full') or d['cfg']; p = c['p']; ntr = int(c['frac'] * p * p)
    L = np.asarray(d['loss'], float); cm = smooth(L)
    s, v = AX.series(obs, 'test_acc_nm'); st, tr = AX.series(obs, 'train_acc')
    mis = np.round((1 - tr) * ntr).astype(int)
    fit = AX.first_logged(st, tr, 1.0)
    rec.update(gain_init=c.get('gain_init'), eta_g=c.get('eta_g'), first_fit=fit, steps=len(L) - 1,
               final_all=float(AX.series(obs, 'test_acc')[1][-1]))
    a, b = rec['net']['0.1'], rec['net']['0.9']
    if a is not None and b is not None and fit is not None:
        a, b = int(a), int(b)
        seg = cm[a:b + 1]; dd = 1 - seg / np.maximum.accumulate(seg)
        m = (st >= a) & (st <= b)
        rec['rise'] = dict(a=a, b=b, ce_a=float(cm[a]), ce_b=float(cm[b]), ce_change=float(cm[b] / cm[a] - 1),
                           max_drawdown=float(dd.max()), drawdown_at_b=float(dd[-1]),
                           max_mis=int(mis[m].max()) if m.any() else None,
                           ce_at_50=float(cm[int(rec['net']['0.5'])]), ce_at_50_over_logp=float(cm[int(rec['net']['0.5'])] / math.log(p)))
    if fit is not None:
        q, I = paper_window(cm, int(fit), s, v); w = dict(q_te=q, W=I)
        if I:
            A, B = I; mw = (st >= A) & (st <= B)
            for name, key in (('fourier_top8', 'fref_top8_test_acc_nm'), ('paper_refit', 'refit_test_acc_nm')):
                sr, vr = AX.series(obs, key); w[f'{name}_gain'] = gain_inside(sr, vr, A, B)
            sa, ah = AX.series(obs, 'AH_full_resid'); w['dAH'] = at(sa, ah, B) - at(sa, ah, A)
            w['max_mis'] = int(mis[mw].max()) if mw.any() else None
            w['ce_over_logp'] = float(np.median(cm[A:B + 1]) / math.log(p))
            ok = (q >= SHARE and w['fourier_top8_gain'] >= GAIN and w['dAH'] >= DAH and rec['final_all'] >= FINAL)
            w['pass_Y'] = bool(ok and w['max_mis'] is not None and w['max_mis'] <= TOL)
            w['pass_Y_strict'] = bool(ok and w['max_mis'] == 0)
            w['why'] = [k for k, c_ in (('share', q >= SHARE), ('refit', w['fourier_top8_gain'] >= GAIN), ('A_H', w['dAH'] >= DAH),
                                        ('final', rec['final_all'] >= FINAL),
                                        ('fit', w['max_mis'] is not None and w['max_mis'] <= TOL)) if not c_]
        else:
            w.update(pass_Y=False, pass_Y_strict=False, why=['no window'])
        rec['paper_window'] = w
    if any('gain' in o for o in obs):
        sg, g = AX.series(obs, 'gain')
        rec['gain_at'] = {k: (at(sg, g, rec['net'][k]) if rec['net'][k] is not None else None) for k in ('0.1', '0.5', '0.9')}
    return rec


def summarize(R):
    pw = [r['paper_window'] for r in R if 'paper_window' in r]; rs = [r['rise'] for r in R if 'rise' in r]
    med = lambda x: float(np.median(x)) if len(x) else None
    rng = lambda x: [float(min(x)), float(max(x))] if len(x) else None
    out = dict(n=len(R), pass_Y=sum(w['pass_Y'] for w in pw), pass_Y_strict=sum(w['pass_Y_strict'] for w in pw),
               why={k: sum(k in w['why'] for w in pw) for k in ('share', 'refit', 'A_H', 'final', 'fit', 'no window')},
               q_te=[round(w['q_te'], 3) for w in pw],
               W_strict10=[round(r['window']['strict']['share'], 3) for r in R if 'window' in r])
    for k in ('ce_change', 'max_drawdown', 'drawdown_at_b', 'max_mis', 'ce_at_50', 'ce_at_50_over_logp'):
        x = [r_[k] for r_ in rs if r_.get(k) is not None]; out[k] = dict(median=med(x), range=rng(x), n_rising=sum(v > 0 for v in x) if k == 'ce_change' else None)
    for name in ('fourier_top8', 'paper_refit', 'net_top8', 'probe'):
        for q in ('0.5', '0.9'):
            L = [r['readouts'][name]['lead'][q] for r in R if name in r['readouts'] and r['readouts'][name]['lead'][q] is not None]
            out[f'lead_{name}_{q}'] = dict(n=len(L), n_lead=sum(x > 0 for x in L), median=med(L), range=rng(L))
    for k in ('0.1', '0.5', '0.9'):
        x = [r['net'][k] for r in R if r['net'][k] is not None]; out[f'net_{k}'] = dict(median=med(x), range=rng(x))
        g = [r['gain_at'][k] for r in R if r.get('gain_at') and r['gain_at'].get(k) is not None]
        if g: out[f'gain_at_{k}'] = dict(median=med(g), range=rng(g))
    return out


def main(ydir, xdir, outp):
    groups = {}
    for fn in sorted(glob.glob(os.path.join(ydir, '*', 'run.json.gz'))):
        rid = os.path.basename(os.path.dirname(fn)); groups.setdefault(rid.split('_')[1], []).append(dict(per_run(fn), id=rid))
    seeds = {r['id'].rsplit('_s', 1)[1] for R in groups.values() for r in R}
    for fn in sorted(glob.glob(os.path.join(xdir, 'X_R5_*', 'run.json.gz'))):
        rid = os.path.basename(os.path.dirname(fn))
        if rid.rsplit('_s', 1)[1] in seeds or not seeds: groups.setdefault('X_R5', []).append(dict(per_run(fn), id=rid))
    res = dict(doc=__doc__, groups={g: dict(runs=R, summary=summarize(R)) for g, R in groups.items()})
    json.dump(res, open(outp, 'w'), indent=1, default=float)
    for g, v in res['groups'].items():
        S = v['summary']
        print(f"== {g}: {S['n']} runs | pass_Y {S['pass_Y']}/{S['n']} (strict {S['pass_Y_strict']}) | why {S['why']}")
        print(f"   q_te {S['q_te']} | W raw-CE 10% {S['W_strict10']}")
        for k in ('ce_change', 'max_drawdown', 'drawdown_at_b', 'max_mis', 'ce_at_50', 'ce_at_50_over_logp'): print(f'   {k}: {S[k]}')
        for k in ('fourier_top8', 'paper_refit', 'net_top8', 'probe'):
            print(f"   lead {k}: 0.5 {S[f'lead_{k}_0.5']} | 0.9 {S[f'lead_{k}_0.9']}")
        print('   net crossings', {k: S[f'net_{k}'] for k in ('0.1', '0.5', '0.9')}, '| gain', {k: S.get(f'gain_at_{k}') for k in ('0.1', '0.5', '0.9')})


if __name__ == '__main__':
    main(*sys.argv[1:4])
