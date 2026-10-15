#!/usr/bin/env python3
import gzip, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'campaign_X'))
import analyze_X as AX


def smooth(L, w=201):
    k = np.ones(w) / w; pad = w // 2
    Lp = np.concatenate([np.full(pad, L[0]), L, np.full(pad, L[-1])])
    return np.convolve(Lp, k, mode='valid')


def share(s, v, a, b):
    at = lambda t: float(v[int(np.argmin(np.abs(s - t)))])
    clip = lambda t: min(max(at(t), 0.1), 0.9)
    return (clip(b) - clip(a)) / 0.8


def drawdown_window(cm, start, s, v, tol=0.03):
    best = (0.0, None); n = len(cm)
    starts = range(start, n, 50)
    for a in starts:
        rm = np.maximum.accumulate(cm[a:]); bad = np.where(cm[a:] < (1 - tol) * rm)[0]
        b = a + (bad[0] - 1 if len(bad) else n - 1 - a)
        if b <= a: continue
        q = share(s, v, a, b)
        if q > best[0] + 1e-12: best = (q, (int(a), int(b)))
    return best


def load_any(fn):
    if fn.endswith('.ckpt.pt'):
        import torch
        ck = torch.load(fn, weights_only=False); name = os.path.basename(fn)[:-len('.json.gz.ckpt.pt')]
        cfg = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(fn))), name + '.json')))
        return dict(loss=list(ck['loss'][:ck['t_next']]), obs=[o for o in ck['obs'] if o['step'] < ck['t_next']], cfg=cfg)
    return AX.load(fn)


def per_run(fn):
    d = load_any(fn); obs = d['obs']; c = d.get('cfg_full') or d['cfg']; p = c['p']; ntr = int(c['frac'] * p * p)
    L = np.asarray(d['loss'], float); cm = smooth(L)
    s, v = AX.series(obs, 'test_acc_nm'); st, tr = AX.series(obs, 'train_acc')
    fit = AX.first_logged(st, tr, 1.0)
    rec = dict(file=os.path.basename(os.path.dirname(fn)) if os.path.basename(fn) == 'run.json.gz' else os.path.basename(fn), gain=c.get('gain_init'), eta_g=c.get('eta_g'),
               steps=len(L) - 1, first_fit=fit, net={str(q): AX.first_logged(s, v, q) for q in AX.QS}, final=float(v[-1]))
    for name, key in (('ridge', 'refit_test_acc_nm'), ('fourier_top8', 'fref_top8_test_acc_nm'), ('paper_refit_f', 'fref_all_test_acc_nm')):
        sr, vr = AX.series(obs, key)
        if len(sr):
            x = AX.interp_cross(sr, vr, 0.5); rec[name + '_50'] = x
            rec[name + '_lead_50'] = (rec['net']['0.5'] - x) if (x is not None and rec['net']['0.5'] is not None) else None
    a, b = rec['net']['0.1'], rec['net']['0.9']
    if a is not None and b is not None and fit is not None:
        a, b = int(a), int(b); start = int(fit)
        seg = cm[a:b + 1]; ddr = 1 - seg / np.maximum.accumulate(seg)
        m = (st >= a) & (st <= b)
        rec['rise'] = dict(a=a, b=b, ce_a=float(cm[a]), ce_b=float(cm[b]), ce_change=float(cm[b] / cm[a] - 1),
                           ce_min_in=float(cm[a:b + 1].min()), ce_max_in=float(cm[a:b + 1].max()),
                           max_drawdown_in=float(ddr.max()), drawdown_at_b=float(ddr[-1]),
                           max_mis=int(round((1 - tr[m].min()) * ntr)) if m.any() else None,
                           ce_over_logp=float(np.median(cm[a:b + 1]) / math.log(p)))
        q, I = drawdown_window(cm, start, s, v); rec['paper_window'] = dict(share=q, W=I)
        if I:
            mw = (st >= I[0]) & (st <= I[1]); rec['paper_window']['max_mis'] = int(round((1 - tr[mw].min()) * ntr)) if mw.any() else None
        qs, Is = AX.relaxed(d, 0, 'test_acc_nm'); rec['W_strict10'] = dict(share=qs, I=Is)
    if 'gain' in obs[-1]:
        sg, g = AX.series(obs, 'gain'); rec['gain_at'] = {k: (float(g[int(np.argmin(np.abs(sg - t)))]) if t is not None else None)
                                                          for k, t in (('fit', fit), ('net0.1', a), ('net0.9', b))}
    return rec


if __name__ == '__main__':
    out = sys.argv[1]; R = [per_run(fn) for fn in sys.argv[2:]]
    json.dump(R, open(out, 'w'), indent=1)
    for r in R:
        print(r['file'], '| fit', r['first_fit'], '| net', r['net'], '| final %.3f' % r['final'])
        if 'rise' in r:
            x = r['rise']; print('   rise %d-%d: CE %.3f -> %.3f (%+.1f%%), max drawdown %.1f%%, at 0.9 %.1f%%, max mis %s, CE/logp %.3f' % (
                x['a'], x['b'], x['ce_a'], x['ce_b'], 100 * x['ce_change'], 100 * x['max_drawdown_in'], 100 * x['drawdown_at_b'], x['max_mis'], x['ce_over_logp']))
            print('   paper window', r['paper_window'], '| W strict 10%', r['W_strict10'])
        print('   leads at 0.5: ridge', r.get('ridge_lead_50'), 'fourier top8', r.get('fourier_top8_lead_50'), 'paper refit', r.get('paper_refit_f_lead_50'), '| gain', r.get('gain_at'))
