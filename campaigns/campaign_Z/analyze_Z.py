#!/usr/bin/env python3
"""Campaign Z analysis, fixed in PLAN.md section 5 (claude/1hl-figures, 2026-10-12).

Every measure is computed on BOTH held-out definitions:
  all  every pair outside the training set (test_acc and the readouts' *_test_acc);
  nm   the no-mirror subset: held-out pairs whose mirror (b, a) is also held out (test_acc_nm, *_test_acc_nm).
The mirror-in-training pairs (all minus nm) are reported separately at the first full fit.

Usage:  python3 analyze_Z.py OUT.json Z_RESULTS_DIR [Y_RESULTS_DIR X_RESULTS_DIR]
  Z_RESULTS_DIR   campaign_Z/collected/results (one folder per run with run.json.gz)
  Y_RESULTS_DIR   campaign_Y/collected/results: the Polar runs with the gain (G2, G3; seeds 11-20), cells 'Y_G2', 'Y_G3'
  X_RESULTS_DIR   campaign_X/collected/results: the Polar runs without the gain (R5; seeds 11-20), cell 'X_R5'
Needs campaign X's analyze_X.py and campaign Y's analyze_Y.py (paths ../campaign_X, ../campaign_Y, or CAMPAIGN_X_DIR,
CAMPAIGN_Y_DIR).
"""
import glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('CAMPAIGN_X_DIR', os.path.join(HERE, '..', 'campaign_X')))
sys.path.insert(0, os.environ.get('CAMPAIGN_Y_DIR', os.path.join(HERE, '..', 'campaign_Y')))
sys.path.insert(0, HERE)
import analyze_X as AX
import analyze_Y as AY

SHARE, GAIN, DAH, FINAL, TOL = 0.75, 0.2, 0.02, 0.99, 10
READOUTS = (('fourier_top8', 'fref_top8_test_acc'), ('paper_refit', 'refit_test_acc'), ('net_top8', 'fnet_top8_test_acc'),
            ('probe', 'probe_test_acc'))
DEFS = (('all', ''), ('nm', '_nm'))


def split_counts(c):
    p = c['p']; a = np.repeat(np.arange(p), p); b = np.tile(np.arange(p), p)
    perm = np.random.default_rng(c['split_seed']).permutation(p * p); n = int(c['frac'] * p * p)
    tr = np.sort(perm[:n]); te = np.sort(perm[n:]); intr = np.zeros(p * p, bool); intr[tr] = True
    return n, len(te), int((~intr[b[te] * p + a[te]]).sum())


def cell_of(rid):
    head, _ = rid.rsplit('_s', 1)
    if rid.startswith('Y_') or rid.startswith('X_'): return '_'.join(head.split('_')[:2])
    return head[2:]


def per_def(obs, cm, fit, mis, st, suf):
    s, v = AX.series(obs, 'test_acc' + suf)
    out = dict(final=float(v[-1]) if len(v) else None)
    net = {q: (AX.interp_cross(s, v, q) if len(v) else None) for q in (0.1, 0.5, 0.9)}
    out['net'] = {str(q): t for q, t in net.items()}
    R = {}
    for name, key in READOUTS:
        sr, vr = AX.series(obs, key + suf)
        if not len(sr): continue
        cr = {q: AX.interp_cross(sr, vr, q) for q in (0.1, 0.5, 0.9)}
        R[name] = dict(cross={str(q): t for q, t in cr.items()},
                       lead={str(q): (net[q] - cr[q]) if (net[q] is not None and cr[q] is not None) else None for q in (0.5, 0.9)},
                       final=float(vr[-1]))
    out['readouts'] = R
    if fit is not None and len(v):
        q, I = AY.paper_window(cm, int(fit), s, v); w = dict(q_te=q, W=I)
        if I:
            A, B = I; mw = (st >= A) & (st <= B)
            for name, key in READOUTS[:2]:
                sr, vr = AX.series(obs, key + suf); w[f'{name}_gain'] = AY.gain_inside(sr, vr, A, B) if len(sr) else None
            sa, ah = AX.series(obs, 'AH_full_resid'); w['dAH'] = AY.at(sa, ah, B) - AY.at(sa, ah, A)
            w['max_mis'] = int(mis[mw].max()) if mw.any() else None
            w['ce_over_logp'] = None
            fg = w['fourier_top8_gain'] if w['fourier_top8_gain'] is not None else -1
            ok = (q >= SHARE and fg >= GAIN and w['dAH'] >= DAH)
            w['ok_loose'] = bool(ok and w['max_mis'] is not None and w['max_mis'] <= TOL)
            w['ok_strict'] = bool(ok and w['max_mis'] == 0)
            w['why'] = [k for k, c_ in (('share', q >= SHARE), ('refit', fg >= GAIN), ('A_H', w['dAH'] >= DAH),
                                        ('fit', w['max_mis'] is not None and w['max_mis'] <= TOL)) if not c_]
        else:
            w.update(ok_loose=False, ok_strict=False, why=['no window'])
        out['window'] = w
    return out


def per_run(fn):
    d = AX.load(fn); obs = d['obs']; c = d.get('cfg_full') or d['cfg']; p = c['p']
    ntr, nte, nnm = split_counts(c)
    L = np.asarray(d['loss'], float); cm = AY.smooth(L)
    st, tr = AX.series(obs, 'train_acc'); mis = np.round((1 - tr) * ntr).astype(int)
    fit = AX.first_logged(st, tr, 1.0)
    rec = dict(file=fn, p=p, n_layers=c['n_layers'], optimizer=c['optimizer'], hidden_mode=c.get('hidden_mode'), frac=c['frac'],
               lam_v=c.get('lam_v'), eta_w=c.get('eta_w'), lam_w=c.get('lam_w'), gain_init=c.get('gain_init'),
               eta_g=c.get('eta_g'), tau=(1 / (c['eta_g'] * c['lam_g'])) if c.get('gain_init') and c.get('eta_g') else None,
               n_train=ntr, n_heldout=nte, n_nm=nnm, n_mirror=nte - nnm, first_fit=fit, steps=len(L) - 1,
               stop_reason=d.get('stop_reason'), final_train=float(tr[-1]) if len(tr) else None)
    rec['ce_min'] = dict(t=int(np.argmin(cm)), ce=float(cm.min()))
    for tag, suf in DEFS: rec[tag] = per_def(obs, cm, fit, mis, st, suf)
    if fit is not None:
        s, va = AX.series(obs, 'test_acc'); _, vn = AX.series(obs, 'test_acc_nm')
        aa, an = AY.at(s, va, fit), AY.at(s, vn, fit)
        rec['at_fit'] = dict(all=aa, nm=an, mirror=(aa * nte - an * nnm) / max(nte - nnm, 1))
    a, b = rec['nm']['net']['0.1'], rec['nm']['net']['0.9']
    if a is not None and b is not None:
        a, b = int(a), int(b); seg = cm[a:b + 1]; dd = 1 - seg / np.maximum.accumulate(seg); m = (st >= a) & (st <= b)
        t50 = rec['nm']['net']['0.5']
        rec['rise'] = dict(a=a, b=b, ce_a=float(cm[a]), ce_b=float(cm[b]), ce_change=float(cm[b] / cm[a] - 1),
                           max_drawdown=float(dd.max()), drawdown_at_b=float(dd[-1]), max_mis=int(mis[m].max()) if m.any() else None,
                           ce_at_50=float(cm[int(t50)]), ce_at_50_over_logp=float(cm[int(t50)] / math.log(p)))
    for tag in ('all', 'nm'):
        w = rec[tag].get('window')
        if w and w.get('W'): w['ce_over_logp'] = float(np.median(cm[w['W'][0]:w['W'][1] + 1]) / math.log(p))
    fin = rec['all']['final'] if rec['all']['final'] is not None else 0.0
    for kind in ('loose', 'strict'):
        rec[f'pass_{kind}'] = bool(fin >= FINAL and all(rec[t].get('window', {}).get(f'ok_{kind}', False) for t in ('all', 'nm')))
    rec['why'] = sorted({x for t in ('all', 'nm') for x in rec[t].get('window', {}).get('why', ['no fit'])} |
                        ({'final'} if fin < FINAL else set()))
    sa, ah = AX.series(obs, 'AH_full_resid')
    rec['AH'] = dict(resid_end=float(ah[-1]) if len(ah) else None,
                     blocks_end=[float(AX.series(obs, f'AH_full_h{l + 1}')[1][-1]) for l in range(c['n_layers'])
                                 if len(AX.series(obs, f'AH_full_h{l + 1}')[1])])
    if any('gain' in o for o in obs):
        sg, g = AX.series(obs, 'gain')
        rec['gain_at'] = {k: (AY.at(sg, g, rec['nm']['net'][k]) if rec['nm']['net'][k] is not None else None) for k in ('0.1', '0.5', '0.9')}
    return rec


def summarize(R):
    med = lambda x: float(np.median(x)) if len(x) else None
    rng = lambda x: [float(min(x)), float(max(x))] if len(x) else None
    S = dict(n=len(R), pass_loose=sum(r['pass_loose'] for r in R), pass_strict=sum(r['pass_strict'] for r in R),
             why={k: sum(k in r['why'] for r in R) for k in ('share', 'refit', 'A_H', 'fit', 'final', 'no window', 'no fit')},
             final_all=rng([r['all']['final'] for r in R if r['all']['final'] is not None]),
             first_fit=dict(median=med([r['first_fit'] for r in R if r['first_fit'] is not None]),
                            n_fit=sum(r['first_fit'] is not None for r in R)),
             steps=rng([r['steps'] for r in R]))
    rs = [r['rise'] for r in R if 'rise' in r]
    for k in ('ce_change', 'max_drawdown', 'max_mis', 'ce_at_50', 'ce_at_50_over_logp'):
        x = [r_[k] for r_ in rs if r_.get(k) is not None]
        S[k] = dict(median=med(x), range=rng(x), n=len(x), n_rising=sum(v > 0 for v in x) if k == 'ce_change' else None)
    for tag in ('all', 'nm'):
        T = {}
        ws = [r[tag]['window'] for r in R if 'window' in r[tag]]
        T['q_te'] = rng([w['q_te'] for w in ws]); T['q_te_median'] = med([w['q_te'] for w in ws])
        for k in ('fourier_top8_gain', 'paper_refit_gain', 'dAH'):
            T[k] = med([w[k] for w in ws if w.get(k) is not None])
        for k in ('0.1', '0.5', '0.9'):
            x = [r[tag]['net'][k] for r in R if r[tag]['net'][k] is not None]; T[f'net_{k}'] = dict(median=med(x), range=rng(x), n=len(x))
        for name, _ in READOUTS:
            for q in ('0.5', '0.9'):
                L_ = [r[tag]['readouts'][name]['lead'][q] for r in R if name in r[tag]['readouts'] and r[tag]['readouts'][name]['lead'][q] is not None]
                T[f'lead_{name}_{q}'] = dict(n=len(L_), n_lead=sum(x > 0 for x in L_), median=med(L_), range=rng(L_))
        S[tag] = T
    af = [r['at_fit'] for r in R if 'at_fit' in r]
    S['at_fit'] = {k: med([a[k] for a in af]) for k in ('all', 'nm', 'mirror')} if af else None
    for k in ('0.1', '0.5', '0.9'):
        g = [r['gain_at'][k] for r in R if r.get('gain_at') and r['gain_at'].get(k) is not None]
        if g: S[f'gain_at_{k}'] = dict(median=med(g), range=rng(g))
    S['AH_resid_end'] = med([r['AH']['resid_end'] for r in R if r['AH']['resid_end'] is not None])
    return S


def collect(zdir, ydir=None, xdir=None):
    cells = {}
    for root, pat in ((zdir, 'Z_*'), (ydir, 'Y_*'), (xdir, 'X_R5_*')):
        if not root: continue
        for fn in sorted(glob.glob(os.path.join(root, pat, 'run.json.gz'))):
            rid = os.path.basename(os.path.dirname(fn)); cells.setdefault(cell_of(rid), []).append(dict(per_run(fn), id=rid))
    return cells


def fmt(x, nd=0, pct=False):
    if x is None: return '-'
    return f'{100 * x:+.{nd}f}%' if pct else f'{x:.{nd}f}'


def main(outp, zdir, ydir=None, xdir=None):
    cells = collect(zdir, ydir, xdir)
    res = dict(doc=__doc__, cells={k: dict(runs=R, summary=summarize(R)) for k, R in sorted(cells.items())})
    json.dump(res, open(outp, 'w'), indent=1, default=float)
    for k, v in res['cells'].items():
        S = v['summary']; A, N = S['all'], S['nm']
        print(f"== {k}: {S['n']} runs | pass (both defs) {S['pass_loose']}/{S['n']}, strict {S['pass_strict']} | why {dict((a, b) for a, b in S['why'].items() if b)}")
        print(f"   final held-out (all) {S['final_all']} | first fit median {S['first_fit']['median']} ({S['first_fit']['n_fit']}/{S['n']} fit) | steps {S['steps']}")
        print(f"   CE across the rise (nm 0.1 -> 0.9): median {fmt(S['ce_change']['median'], 0, True)} range {S['ce_change']['range']} up in {S['ce_change']['n_rising']}/{S['ce_change']['n']}; CE at nm 0.5: {fmt(S['ce_at_50']['median'], 3)} nats ({fmt(S['ce_at_50_over_logp']['median'], 3)} log p)")
        for tag, T in (('all', A), ('nm', N)):
            print(f"   [{tag}] q_te {T['q_te']} | net 0.1/0.5/0.9 {[fmt(T[f'net_{q}']['median']) for q in ('0.1', '0.5', '0.9')]} | "
                  f"Fourier top-8 lead 0.5 {T['lead_fourier_top8_0.5']['n_lead']}/{T['lead_fourier_top8_0.5']['n']} {fmt(T['lead_fourier_top8_0.5']['median'])}, "
                  f"0.9 {T['lead_fourier_top8_0.9']['n_lead']}/{T['lead_fourier_top8_0.9']['n']} {fmt(T['lead_fourier_top8_0.9']['median'])} | "
                  f"paper refit lead 0.5 {T['lead_paper_refit_0.5']['n_lead']}/{T['lead_paper_refit_0.5']['n']} {fmt(T['lead_paper_refit_0.5']['median'])} | "
                  f"window gains: Fourier {fmt(T['fourier_top8_gain'], 2)}, dA_H {fmt(T['dAH'], 2)}")
        if S.get('at_fit'): print(f"   at the first fit: held-out all {S['at_fit']['all']:.3f}, no-mirror {S['at_fit']['nm']:.3f}, mirror-in-training {S['at_fit']['mirror']:.3f}")


if __name__ == '__main__':
    main(*sys.argv[1:5])
