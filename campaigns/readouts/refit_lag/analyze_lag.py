import json, sys
import numpy as np

def cross(ts, vs, q):
    for i, v in enumerate(vs):
        if v >= q:
            if i == 0: return float(ts[0])
            t0, v0 = ts[i - 1], vs[i - 1]
            return float(t0 + (q - v0) / (v - v0) * (ts[i] - t0)) if v > v0 else float(ts[i])
    return None

out = {}
for fn in sys.argv[1:]:
    d = json.load(open(fn)); rows = d['rows']; ts = [r['step'] for r in rows]
    tab = 'resid' if d['kind'] == 'tf' else 'last'
    obs = d['obs']; key = 'test_acc_nm' if any('test_acc_nm' in o for o in obs) else 'test_acc'
    to = [o['step'] for o in obs if key in o]; vo = [o[key] for o in obs if key in o]
    net = {q: cross(to, vo, q) for q in (0.5, 0.9)}
    names = [f'ridge_t{x:g}' for x in d['taus']] + ['centroid', 'filt8', 'filt8_t1', 'filt8_t10', 'probe']
    res = dict(net=net, readouts={})
    print(f'== {fn}: network 0.5 / 0.9 at {net[0.5]:.0f} / {net[0.9]:.0f}' if net[0.9] else f'== {fn}: network 0.5 at {net[0.5]}')
    for nm_ in names:
        vs = [r[tab].get(nm_) for r in rows]
        if any(v is None for v in vs): continue
        c = {q: cross(ts, vs, q) for q in (0.5, 0.9)}
        lead = {q: (net[q] - c[q]) if (c[q] is not None and net[q] is not None) else None for q in (0.5, 0.9)}
        res['readouts'][nm_] = dict(cross=c, lead=lead, final=vs[-1])
        f = lambda x: '   never' if x is None else f'{x:8.0f}'
        print(f'   {nm_:12s} 0.5 at {f(c[0.5])} (lead {f(lead[0.5])})   0.9 at {f(c[0.9])} (lead {f(lead[0.9])})   final {vs[-1]:.3f}')
    i = int(np.argmin([abs(t - net[0.5]) for t in ts])); r = rows[i][tab]; fq = r['freq']
    r2 = np.array(fq['r2_ev']); carried = r2 > 0.05
    amp_tr, amp_ev = np.array(fq['amp_tr']), np.array(fq['amp_ev'])
    res['at_net50'] = dict(step=rows[i]['step'], n_carried=int(carried.sum()), K=len(r2),
                           amp_tr_carried=float(amp_tr[carried].mean()), amp_tr_other=float(amp_tr[~carried].mean()),
                           amp_ev_carried=float(amp_ev[carried].mean()), amp_ev_other=float(amp_ev[~carried].mean()),
                           r2_ev_carried=float(r2[carried].mean()), r2_ev_other=float(r2[~carried].mean()),
                           r2_tr_other=float(np.array(fq['r2_tr'])[~carried].mean()),
                           dec={k: v for k, v in r['dec'].items()})
    a = res['at_net50']
    print(f"   at {a['step']}: {a['n_carried']} of {a['K']} frequencies carried (held-out R2 > 0.05)."
          f" ridge clock amplitude train/held-out: carried {a['amp_tr_carried']:.2f}/{a['amp_ev_carried']:.2f},"
          f" others {a['amp_tr_other']:.2f}/{a['amp_ev_other']:.2f}; held-out R2 carried {a['r2_ev_carried']:.2f}, others {a['r2_ev_other']:.2f};"
          f" training R2 others {a['r2_tr_other']:.2f}")
    for k, v in a['dec'].items():
        sh_ev = v['rms_signal'] ** 2 / (v['rms_signal'] ** 2 + v['rms_rest'] ** 2)
        sh_tr = v['rms_signal_tr'] ** 2 / (v['rms_signal_tr'] ** 2 + v['rms_rest_tr'] ** 2)
        print(f"     {k:8s} logit variance on the top-8 frequencies: held-out {sh_ev:.2f}, training {sh_tr:.2f};"
              f" accuracy of the top-8 part {v['acc_signal']:.2f}, of the rest {v['acc_rest']:.2f}, of both {v['acc_full']:.2f}")
    out[fn] = res
json.dump(out, open('analysis_lag.json', 'w'), indent=1)
