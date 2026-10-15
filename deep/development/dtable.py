import sys, glob, math, numpy as np
from anat import load, series, knee_step, first_persistent, at
from plateau import onset
def first_cross(obs, key, thr):
    s, v = series(obs, key); i = np.where(v >= thr)[0]; return int(s[i[0]]) if len(i) else None
def rng(c, a, b):
    seg = c[a:b + 1]; return 100 * (seg.max() - seg.min()) / seg.mean(), 100 * (seg[-1] - seg[0]) / seg.mean(), seg.mean()
def score(fn, span=5000):
    d, L, obs, st = load(fn); c = d['cfg']; p = c['p']; r = p - 1; D = c['depth']; T = len(L) - 1
    cm = (L[:-1] + L[1:]) / 2; lp = math.log(p)
    k = knee_step(d, obs) if c['lam_v'] > 0 else None
    o = dict(run=fn.split('/')[-1][:-5], D=D, knee=k)
    o['on2'] = onset(L, k, span, 0.02) if k is not None else None
    o['on1'] = onset(L, k, span, 0.01) if k is not None else None
    o['t10'] = first_cross(obs, 'test_acc', 0.1); o['t50'] = first_cross(obs, 'test_acc', 0.5); o['t90'] = first_persistent(obs, 'test_acc', 0.9)
    s_, te = series(obs, 'test_acc'); o['final'] = float(te[-1])
    o['peak'] = int(np.argmax(cm[k:]) + k) if k is not None else None
    if o['on2'] is not None:
        a = o['on2']; b = o['t90'] or T
        o['R_on'], o['D_on'], o['lev'] = rng(cm, a, max(b, a + 1))
        o['te_on'] = float(at(obs, 'test_acc', a))
    if o['t10'] is not None:
        a = max(o['t10'] - 500, k or 0); b = o['t90'] or T
        o['R_g'], o['D_g'], o['lev_g'] = rng(cm, a, max(b, a + 1)); o['ga'], o['gb'] = a, b
        for key in [f'agop{r}_ho', f'agopH1_{r}_ho', 'W1_four_top1', 'refit_acc'] + [f'AH_full_h{l+1}' for l in range(D)] + ['AH_full_logit']:
            o[key] = (float(at(obs, key, a)), float(at(obs, key, b)))
    hv = [x for x in obs if 'normW' in x]; o['norms_end'] = hv[-1]['normW'] + [hv[-1]['normV']]
    return o
def fmt(o):
    f = lambda t: f"{t[0]:.2f}→{t[1]:.2f}" if isinstance(t, tuple) and not np.isnan(t[0]) else '–'
    s = f"{o['run']:<40} knee {o['knee']} peak {o['peak']} on2 {o['on2']} on1 {o['on1']} | t10 {o['t10']} t50 {o['t50']} t90 {o['t90']} fin {o['final']:.2f}"
    if 'R_on' in o: s += f" | onset→t90: lev {o['lev']:.3g} R {o['R_on']:.1f}% D {o['D_on']:+.1f}% te@on {o['te_on']:.2f}"
    if 'R_g' in o:
        r = [k for k in o if k.startswith('agop') and not k.startswith('agopH1')][0]
        s += f" | grok[{o['ga']},{o['gb']}]: lev {o['lev_g']:.3g} R {o['R_g']:.1f}% D {o['D_g']:+.1f}% | AGOPlast {f(o[r])} AGOP@h1 {f(o[[k for k in o if k.startswith('agopH1')][0]])} AH " + ' '.join(f(o[f'AH_full_h{l+1}']) for l in range(o['D'])) + f" W1four {f(o['W1_four_top1'])}"
    s += ' | norms ' + ' '.join(f"{x:.1f}" for x in o['norms_end'])
    return s
if __name__ == '__main__':
    for fn in sorted(sum([glob.glob(g) for g in sys.argv[1:]], [])):
        try: print(fmt(score(fn)))
        except Exception as e: print(fn, 'ERR', repr(e))
