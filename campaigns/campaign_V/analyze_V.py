#!/usr/bin/env python3
"""Campaign V: per-run table (claude/1hl-figures, 2026-10-09). Reads confirm_V.py's output and the runs.

    python3 -I analyze_V.py RESULTS_DIR CONFIRM_JSON OUT_JSON

Per run: the confirmation verdicts (confirm_V.py), the order of events and the CE level along the way:
  t_fit    first logged update at which the training accuracy, averaged over 500 updates (as in confirm_V.py), is >= 0.99;
  t10, t50, t90, t99   first logged update at which the held-out accuracy reaches 0.1, 0.5, 0.9, 0.99;
  r50, r90 the same for the ridge refit's held-out accuracy;
  CE level = centred geometric 500-update average of the training CE, divided by log p, at those updates;
  ce_ratio_rise = that average at t90 over its value at t10 (the CE change across the held-out rise);
  t_fit97  first logged update at which the 500-update average of the training accuracy is >= 0.97; context: the held-out
           accuracy then, the training accuracy (500-update average) at t50, and the CE ratio from t_fit97 to t90.
"""
import json, math, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import flat_windows as F
import confirm_V as CV


def first(s, v, thr):
    i = np.where(v >= thr)[0]; return int(s[i[0]]) if len(i) else None


def run_row(res, rid, conf):
    d = F.load(os.path.join(res, rid, 'run.json.gz')); c = d.get('cfg_full', d['cfg'])
    L = np.array(d['loss'], float); n = len(L); logp = math.log(c['p'])
    x = F.smooth(L, 500, 'geo')
    s, te = F.series(d['obs'], 'test_acc'); st, tr = F.series(d['obs'], 'train_acc'); sr, rf = F.series(d['obs'], 'refit_test_acc')
    every = int(np.median(np.diff(st))); trs = CV.centred_mean(tr, max(1, CV.TRAIN_AVG // every))
    ev = dict(t_fit=first(st, trs, 0.99), t_fit_raw=first(st, tr, 0.99), t10=first(s, te, 0.1), t50=first(s, te, 0.5),
              t90=first(s, te, 0.9), t99=first(s, te, 0.99), r50=first(sr, rf, 0.5), r90=first(sr, rf, 0.9))
    ev['t_fit97'] = first(st, trs, 0.97)
    lvl = {k: (float(x[min(v, n - 1)] / logp) if v is not None else None) for k, v in ev.items()}
    lvl['end'] = float(x[-1] / logp)
    rise = (float(x[ev['t90']] / x[ev['t10']]) if ev['t10'] is not None and ev['t90'] is not None else None)
    m = re.match(r'V_(\w\d)_', rid); seed = int(rid.rsplit('_s', 1)[1])
    at = lambda ss, vv, t: (None if t is None else F.at_or_before(ss, vv, t)[1])
    ctx = dict(heldout_at_fit97=at(s, te, ev['t_fit97']), train_avg_at_t50=(None if ev['t50'] is None else float(trs[np.searchsorted(st, ev['t50'])])),
               ce_ratio_fit97_to_t90=(float(x[ev['t90']] / x[ev['t_fit97']]) if ev['t_fit97'] is not None and ev['t90'] is not None else None))
    row = dict(id=rid, arm=m.group(1), setting=rid.rsplit('_s', 1)[0], seed=seed, updates=n - 1, stop_reason=d.get('stop_reason'),
               optimizer=c.get('optimizer'), n_layers=c.get('n_layers'), final_norm=c.get('final_norm'), frac=c.get('frac'),
               lam_w=c.get('lam_w'), eta_w=c.get('eta_w'), lam_v=c.get('lam_v'), wd=c.get('wd'), log_p=logp,
               events=ev, ce_over_logp=lvl, ce_ratio_rise=rise, final_test=conf['final'], context=ctx,
               fit_first=(ev['t_fit'] is not None and ev['t50'] is not None and ev['t_fit'] < ev['t50']),
               refit_final=float(rf[-1]) if len(rf) else None,
               verdict=dict(pass_=conf['pass'], probe=conf['probe']['pass'], loose=conf['ridge']['pass_loose'],
                            raw3=conf['ridge']['pass_raw3'], probe_loose=conf['probe']['pass_loose']))
    best = max((v for v in conf['ridge']['by_window'].values() if v), key=lambda v: (v['pass'], v['pass_raw3'], v['pass_loose'], v['g_rf']), default=None)
    if best:
        w = [k for k, v in conf['ridge']['by_window'].items() if v is best][0]
        row['stretch'] = dict(window=int(w), a=best['a'], b=best['b'], ce=best['level'], ce_over_logp=best['level'] / logp,
                              ce500_a_over_logp=float(x[best['a']] / logp), ce500_b_over_logp=float(x[min(best['b'], n - 1)] / logp),
                              ridge_gain=best['g_rf'], heldout_gain=best['g_te'], train_min_avg500=best['train_min_smoothed'],
                              train_min_raw=best['train_min_raw'])
    if conf['pass']: row['fails_on'] = None
    elif best is None: row['fails_on'] = 'no stretch'
    else:
        why = []
        if best['g_rf'] < CV.GAIN: why.append('refit gain')
        if best['train_min_smoothed'] < CV.TRAIN: why.append('training fit')
        if conf['final'] < CV.FINAL: why.append('final held-out')
        if best['g_te'] < CV.GAIN: why.append('held-out gain')
        row['fails_on'] = ', '.join(why) or 'other'
    return row


def main(res, conf_fn, out):
    C = json.load(open(conf_fn))['runs']
    rows = [run_row(res, rid, C[rid]) for rid in sorted(C)]
    json.dump(dict(doc=__doc__, runs=rows), open(out, 'w'), indent=1)
    f = lambda v: '-' if v is None else (f'{v / 1000:.1f}k' if v >= 1000 else str(v))
    for r in rows:
        e = r['events']; st = r.get('stretch') or {}
        print(f"{r['id'][:46]:46s} pass={str(r['verdict']['pass_'])[0]} fit {f(e['t_fit']):>6s} h50 {f(e['t50']):>6s} h90 {f(e['t90']):>6s} "
              f"h99 {f(e['t99']):>6s} r90 {f(e['r90']):>6s} | CE/logp fit {r['ce_over_logp']['t_fit'] or float('nan'):.3g} "
              f"h50 {r['ce_over_logp']['t50'] or float('nan'):.3g} end {r['ce_over_logp']['end']:.3g} rise x{r['ce_ratio_rise'] or float('nan'):.2f} "
              f"final {r['final_test']:.3f} {('fails: ' + r['fails_on']) if r['fails_on'] else ''}")


if __name__ == '__main__':
    main(*sys.argv[1:4])
