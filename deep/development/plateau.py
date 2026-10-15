import numpy as np, math
from anat import load, series, knee_step, first_persistent, window_stats, at
def onset(L, start, span=5000, tol=0.02):
    cm = (L[:-1] + L[1:]) / 2; T = len(cm)
    for t in range(start, T - span):
        seg = cm[t:t + span + 1]
        if np.max(np.abs(seg / cm[t] - 1)) <= tol: return t
    return None
def plateau_summary(fn, span=5000, tol=0.02, wlen=None):
    d, L, obs, st = load(fn); c = d['cfg']; p = c['p']; k = knee_step(d, obs)
    out = dict(name=fn.split('/')[-1][:-5], p=p, knee=k)
    if k is None: return out
    tp = onset(L, k, span, tol); out['tp'] = tp
    if tp is None: return out
    T = len(L) - 1; b = T if wlen is None else min(T, tp + wlen)
    w = window_stats(L, tp, b); out.update(w); out['a'] = tp; out['b'] = b
    out['rel'] = w['level'] / (math.log(p) if c['loss'] == 'ce' else 1.0)
    r = p - 1
    for key in ['test_acc', 'refit_acc', f'agop{r}_ho', f'agop{r}_ho_rand', 'AH_full_h1', 'AH_full_logit', f'agopH1_{r}_ho', f'agopH1_{r}_ho_rand', f'AH_full_h{c["depth"]}']:
        out[key] = (at(obs, key, tp), at(obs, key, b))
    s2, tra = series(obs, 'train_acc'); m = (s2 >= tp) & (s2 <= b); out['trmin'] = float(tra[m].min())
    out['t90'] = first_persistent(obs, 'test_acc', 0.9)
    s3, te = series(obs, 'test_acc'); out['test_max'] = float(te[(s3 >= tp) & (s3 <= b)].max())
    return out
def fmt(o):
    if o.get('tp') is None: return f"{o['name']}: knee {o.get('knee')} no plateau onset"
    r = o['p'] - 1; f = lambda t: f"{t[0]:.2f}->{t[1]:.2f}"
    s = (f"{o['name']}: knee {o['knee']} plateau [{o['a']},{o['b']}] CE {o['level']:.3g} ({o['rel']:.2f}x log p) R {o['R']:.2f}% D {o['D']:+.2f}% trmin {o['trmin']:.3f} | "
         f"test {f(o['test_acc'])} (t90 {o['t90']}) refit {f(o['refit_acc'])} AGOP r{r} ho {f(o[f'agop{r}_ho'])} (rand {o[f'agop{r}_ho_rand'][1]:.2f}) LF {f(o['AH_full_logit'])} AHh1 {f(o['AH_full_h1'])}")
    if not np.isnan(o[f'agopH1_{r}_ho'][0]): s += f" AGOP@h1 ho {f(o[f'agopH1_{r}_ho'])} (rand {o[f'agopH1_{r}_ho_rand'][1]:.2f})"
    return s
if __name__ == '__main__':
    import sys, glob
    for fn in sorted(sum([glob.glob(g) for g in sys.argv[1:]], [])):
        try: print(fmt(plateau_summary(fn)))
        except Exception as e: print(fn, 'ERR', repr(e))
