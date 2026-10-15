import json, numpy as np, glob, sys, math
import functools, os
@functools.lru_cache(maxsize=48)
def _load(fn, mtime):
    d = json.load(open(fn)); L = np.array(d['loss']); obs = d['obs']
    steps = np.array([o['step'] for o in obs]); return d, L, obs, steps
def load(fn):
    return _load(fn, os.path.getmtime(fn))
def series(obs, key):
    s = [(o['step'], o[key]) for o in obs if key in o]
    return np.array([x[0] for x in s]), np.array([x[1] for x in s], dtype=float)
def knee_step(d, obs):
    lam_v = d['cfg']['lam_v']; st, nv = series(obs, 'normV')
    idx = np.where(nv >= 0.99 / lam_v)[0]
    return int(st[idx[0]]) if len(idx) else None
def first_persistent(obs, key, thr):
    st, v = series(obs, key)
    for i in range(len(v)):
        if np.all(v[i:] >= thr): return int(st[i])
    return None
def at(obs, key, step):
    st, v = series(obs, key)
    if len(st) == 0: return np.nan
    i = int(np.argmin(np.abs(st - step))); return v[i]
def window_stats(L, a, b):
    seg = L[a:b + 1]; cm = (seg[:-1] + seg[1:]) / 2
    return dict(R=100 * (cm.max() - cm.min()) / cm.mean(), D=100 * (cm[-1] - cm[0]) / cm.mean(), level=cm.mean(),
                osc=100 * np.median(np.abs(seg[1:-1] - (seg[:-2] + seg[2:]) / 2) / np.abs(seg[1:-1])))
def summarize(fn, win_len=None):
    d, L, obs, steps = load(fn); c = d['cfg']; p = c['p']; T = len(L) - 1
    k = knee_step(d, obs); t90 = first_persistent(obs, 'test_acc', 0.9)
    out = dict(name=fn.split('/')[-1][:-5], p=p, knee=k, t90=t90, T=T)
    if k is None: return out
    a = k + 1000
    b = min(T, a + win_len) if win_len else (t90 if (t90 and t90 > a + 500) else T)
    ws = window_stats(L, a, b); out.update(ws); out['a'] = a; out['b'] = b
    out['level_rel'] = ws['level'] / (math.log(p) if c['loss'] == 'ce' else 1.0)
    st, tra = series(obs, 'train_acc'); m = (st >= a) & (st <= b); out['train_min'] = float(tra[m].min()) if m.any() else np.nan
    for key in ['test_acc', 'refit_acc', 'AH_full_h1', 'AH_ho_h1', 'AH_full_logit', 'AH_ho_logit', 'agop4_ho', 'agop4_ho_rand', f'agop{p-1}_ho', f'agop{p-1}_ho_rand', 'agop4_full', f'agop{p-1}_full']:
        out[key] = (at(obs, key, a), at(obs, key, b))
    return out
def fmt(o):
    if o.get('knee') is None: return f"{o['name']}: no knee"
    p = o['p']
    f2 = lambda t: f"{t[0]:.2f}->{t[1]:.2f}"
    return (f"{o['name']}: knee {o['knee']} win [{o['a']},{o['b']}] t90 {o['t90']} | level {o['level']:.3g} ({o['level_rel']:.3f}x trivial) R {o['R']:.1f}% D {o['D']:+.1f}% osc {o['osc']:.3f}% trmin {o['train_min']:.3f} | "
            f"test {f2(o['test_acc'])} refit {f2(o['refit_acc'])} AHh1 {o['AH_full_h1'][0]:.3f}->{o['AH_full_h1'][1]:.3f} AHho_h1 {f2(o['AH_ho_h1'])} AHho_logit {f2(o['AH_ho_logit'])} "
            f"agop4ho {f2(o['agop4_ho'])} (rand {f2(o['agop4_ho_rand'])}) agop{p-1}ho {f2(o[f'agop{p-1}_ho'])} (rand {f2(o[f'agop{p-1}_ho_rand'])})")
if __name__ == '__main__':
    wl = None
    args = sys.argv[1:]
    if args and args[0].startswith('--win='): wl = int(args[0][6:]); args = args[1:]
    for fn in sorted(sum([glob.glob(g) for g in args], [])):
        try: print(fmt(summarize(fn, wl)))
        except Exception as e: print(fn, 'ERR', e)
