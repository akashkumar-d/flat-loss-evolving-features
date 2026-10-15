#!/usr/bin/env python3
import json, sys
import numpy as np

READOUTS = ['ridge', 'filt_top1', 'filt_top2', 'filt_top4', 'filt_top8', 'filt_top16', 'phase_top4', 'phase_top8', 'proj_top4',
            'proj_top8', 'filt_sig', 'filt_cv', 'ridge_cv', 'probe', 'oracle_filt', 'net_filt_top4', 'net_filt_top8', 'net_filt_oracle']


def cross(ts, vs, q):
    for i, v in enumerate(vs):
        if v is not None and v >= q:
            if i == 0 or vs[i - 1] is None: return float(ts[i])
            t0, t1, v0 = ts[i - 1], ts[i], vs[i - 1]
            return float(t0 + (q - v0) / (v - v0) * (t1 - t0)) if v > v0 else float(t1)
    return None


def analyze(sc):
    tf = sc['kind'] == 'tf'; key = '_te_nm' if tf else '_te'
    table = 'resid' if tf else 'last'
    obs = sc['obs']; ts_o = [o['step'] for o in obs]; vs_o = [o.get('test_acc_nm' if tf else 'test_acc') for o in obs]
    rows = sc['rows']; ts = [r['step'] for r in rows]
    out = dict(kind=sc['kind'], net={}, readouts={}, tables={})
    for q in (0.1, 0.5, 0.9): out['net'][q] = cross(ts_o, vs_o, q)
    for tab in ([table, 'mlp'] if tf else [table]):
        res = {}
        for k in READOUTS:
            vs = [r[tab].get(k + key) for r in rows]
            if all(v is None for v in vs): continue
            c = {q: cross(ts, vs, q) for q in (0.1, 0.5, 0.9)}
            res[k] = dict(cross=c, lead={q: (out['net'][q] - c[q]) if (c[q] is not None and out['net'][q] is not None) else None
                                         for q in (0.5, 0.9)},
                          final=vs[-1])
        out['tables'][tab] = res
        if out['net'][0.5] is not None:
            i = int(np.argmin([abs(t - out['net'][0.5]) for t in ts]))
            out['tables'][tab + '_at_net50'] = dict(step=ts[i], ks_sig=rows[i][tab]['info']['ks_sig'],
                                                    k_cv=rows[i][tab]['info']['k_cv'], AH=rows[i][tab]['AH'])
    return out


if __name__ == '__main__':
    res = {}
    for fn in sys.argv[2:]:
        sc = json.load(open(fn)); name = fn.split('/')[-1].replace('.json', ''); res[name] = analyze(sc)
        a = res[name]; tab = 'resid' if a['kind'] == 'tf' else 'last'
        print(f"== {name}: network 0.1 / 0.5 / 0.9 at {a['net'][0.1]} / {a['net'][0.5]} / {a['net'][0.9]}")
        for t in a['tables']:
            if t.endswith('_at_net50'): print('  ', t, a['tables'][t]); continue
            for k, v in a['tables'][t].items():
                c = v['cross']; l = v['lead']
                f = lambda x: '-' if x is None else f'{x:7.0f}'
                print(f"   {t:5s} {k:10s} 0.5 at {f(c[0.5])} (lead {f(l[0.5])})   0.9 at {f(c[0.9])} (lead {f(l[0.9])})   final {v['final']:.3f}")
    json.dump(res, open(sys.argv[1], 'w'), indent=1)
