#!/usr/bin/env python3
import glob, json, math, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperdefs_tf as P

Q_MIN, GAIN, DAH_MIN, FINAL_MIN = 0.75, 0.2, 0.02, 0.99


def files(root):
    out = sorted(glob.glob(os.path.join(root, '*', 'run.json.gz')))
    return out or sorted(f for f in glob.glob(os.path.join(root, '*.json.gz')) if not f.endswith('.ckpt.pt'))


def spikes(L, t0):
    seg = np.asarray(L[t0:], float)
    return int(np.sum(seg > 10 * np.median(seg))) if len(seg) else None


def per_run(fn):
    d = P.load(fn); c = P.cfg_of(d); L = np.asarray(d['loss'], float); lp = math.log(c['p']); obs = d['obs']
    K = P.te_key(d); nm = '_nm' if K == 'test_acc_nm' else ''
    s, te = P.series(obs, K); _, te_all = P.series(obs, 'test_acc'); sm, mm = P.series(obs, 'train_margin_min')
    sr, rf = P.series(obs, 'refit_test_acc' + nm); sp, pb = P.series(obs, 'probe_test_acc' + nm)
    sa, ahr = P.series(obs, 'AH_full_resid'); sh, ahm = P.series(obs, f"AH_full_h{c.get('n_layers', 1)}")
    ff = P.first_fit(d, K); rw = P.raw10(d, key=K); ph = P.posthoc(d); pr = P.primary(d)
    t = {k: P.first_cross(s, te, k) for k in (0.1, 0.2, 0.5, 0.9)}
    o = dict(id=os.path.basename(os.path.dirname(fn)) if fn.endswith('run.json.gz') else os.path.basename(fn)[:-8],
             p=c['p'], frac=c['frac'], lam_v=c['lam_v'], eta_w=c['eta_w'], lam_w=c['lam_w'], norm=c.get('final_norm', 'none'),
             hidden=c.get('hidden_mode', 'polar'), dtype=c.get('dtype'), updates=len(L) - 1, final=float(te_all[-1]),
             heldout_key=K, final_nm=float(te[-1]),
             T_fit=ff.get('T_fit'), te_at_fit=ff.get('te_at_fit'), fit_relapse=ff.get('relapse'), t=t,
             fit_first=bool(ff.get('T_fit') is not None and ff['te_at_fit'] <= 0.2),
             fit_kept=bool(ff.get('T_fit') is not None and not ff['relapse']))
    if t[0.2] is not None and t[0.9] is not None and t[0.9] > t[0.2]:
        seg = L[t[0.2]:t[0.9] + 1]; o['v_20_90'] = float(seg.max() / seg.min() - 1)
        o['ce_20'], o['ce_90'] = float(L[t[0.2]] / lp), float(L[t[0.9]] / lp)
    q, I = rw['fit']; o['q_fit'], o['I_fit'] = q, I; o['q_nofit'], o['I_nofit'] = rw['nofit']
    if I:
        a, b = I
        o['ce_in_I'] = float(np.median(L[a:b + 1]) / lp)
        o['ridge_gain_I'] = P.inside_gain(sr, rf, a, b); o['probe_gain_I'] = P.inside_gain(sp, pb, a, b) if len(sp) else None
        o['dAH_I'] = P.at(sa, ahr, b) - P.at(sa, ahr, a); o['dAH_mlp_I'] = P.at(sh, ahm, b) - P.at(sh, ahm, a)
    o['meets'] = bool(I and o['q_fit'] >= Q_MIN and o['ridge_gain_I'] >= GAIN and o['dAH_I'] >= DAH_MIN and o['final'] >= FINAL_MIN)
    for name, (ss, vv) in (('ridge', (sr, rf)), ('probe', (sp, pb))):
        for k in (0.5, 0.9):
            x = P.first_cross(ss, vv, k) if len(ss) else None
            o[f'{name}{k}'] = x; o[f'{name}{k}_lead'] = (t[k] - x) if (x is not None and t[k] is not None) else None
    o['posthoc'] = dict(W2=ph.get('W2'), ok2=ph.get('ok2'), g_te=ph.get('g_te'), g_rf=ph.get('g_rf'), knee=ph.get('knee'))
    o['primary'] = dict(ok=pr.get('primary'), why_not=pr.get('why_not'), W=pr.get('W'), share=pr.get('share'))
    o['spikes_after_fit'] = spikes(L, ff['T_fit']) if ff.get('T_fit') is not None else None
    return o


def setting_key(o):
    return (o['p'], o['frac'], o['norm'], o['hidden'], o['eta_w'], o['lam_w'], o['lam_v'])


def main(root, out=None):
    R = [per_run(f) for f in files(root)]
    G = defaultdict(list)
    for o in R: G[setting_key(o)].append(o)
    rows = []
    for k, v in sorted(G.items()):
        med = lambda key: float(np.median([o[key] for o in v if o.get(key) is not None])) if any(o.get(key) is not None for o in v) else None
        rows.append(dict(setting=dict(zip(('p', 'frac', 'norm', 'hidden', 'eta_w', 'lam_w', 'lam_v'), k)), n=len(v),
                         meets=sum(o['meets'] for o in v), fit_first=sum(o['fit_first'] for o in v), fit_kept=sum(o['fit_kept'] for o in v),
                         final99=sum(o['final'] >= FINAL_MIN for o in v), q_fit=med('q_fit'), q_nofit=med('q_nofit'),
                         ce_in_I=med('ce_in_I'), v_20_90=med('v_20_90'), posthoc=sum(bool(o['posthoc']['ok2']) for o in v),
                         ridge_lead_05=med('ridge0.5_lead'), probe_lead_05=med('probe0.5_lead'), runs=[o['id'] for o in v]))
    cands = [r for r in rows if r['n'] >= 3 and r['meets'] == r['n'] and r['setting']['hidden'] == 'polar']
    cands.sort(key=lambda r: (-(r['ce_in_I'] or 0), -(r['q_fit'] or 0)))
    res = dict(runs=R, settings=rows, candidates=[r['setting'] for r in cands])
    if out: json.dump(res, open(out, 'w'), indent=1)
    print(f"{'setting (p, frac, norm, hidden, eta_w, lam_w, lam_v)':58s} {'n':>2s} {'meets':>5s} {'fit1st':>6s} {'kept':>4s} {'fin99':>5s} "
          f"{'q_fit':>5s} {'q_nof':>5s} {'CE/lp':>6s} {'v20-90':>6s} {'post':>4s} {'ridge':>6s} {'probe':>6s}")
    f = lambda x, w=5, d=2: (f'{x:{w}.{d}f}' if isinstance(x, float) else f"{'-':>{w}s}")
    for r in rows:
        s = r['setting']; k = (s['p'], s['frac'], s['norm'], s['hidden'], s['eta_w'], s['lam_w'], s['lam_v'])
        print(f"{str(k):58s} {r['n']:2d} {r['meets']:5d} {r['fit_first']:6d} {r['fit_kept']:4d} {r['final99']:5d} {f(r['q_fit'])} "
              f"{f(r['q_nofit'])} {f(r['ce_in_I'], 6, 3)} {f(r['v_20_90'], 6, 3)} {r['posthoc']:4d} {f(r['ridge_lead_05'], 6, 0)} {f(r['probe_lead_05'], 6, 0)}")
    print('candidates:', res['candidates'])
    return res


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
