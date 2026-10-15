import sys, glob, math, numpy as np
from anat import load, series, knee_step, first_persistent, at
def longest_flat(c, start, tol, stride=20):
    idx = np.arange(start, len(c), stride); x = c[idx]; best = (0, None, None); j = 0
    from collections import deque
    mx, mn = deque(), deque()
    for i in range(len(x)):
        while mx and x[mx[-1]] <= x[i]: mx.pop()
        mx.append(i)
        while mn and x[mn[-1]] >= x[i]: mn.pop()
        mn.append(i)
        while True:
            hi, lo = x[mx[0]], x[mn[0]]; seg_mean = 0.5 * (hi + lo)
            if (hi - lo) / seg_mean <= tol: break
            j += 1
            if mx[0] < j: mx.popleft()
            if mn[0] < j: mn.popleft()
        if i - j > best[0]: best = (i - j, int(idx[j]), int(idx[i]))
    return best[1], best[2]
def flat(fn, tol=0.03):
    d, L, obs, st = load(fn); c = d['cfg']; p = c['p']; r = p - 1; D = c['depth']
    cm = (L[:-1] + L[1:]) / 2; k = knee_step(d, obs) or 0
    a, b = longest_flat(cm, k, tol)
    o = dict(run=fn.split('/')[-1][:-5], a=a, b=b, lev=float(cm[a:b + 1].mean()), rel=float(cm[a:b + 1].mean() / math.log(p)),
             R=float(100 * (cm[a:b + 1].max() - cm[a:b + 1].min()) / cm[a:b + 1].mean()), t90=first_persistent(obs, 'test_acc', 0.9), D=D)
    s_, te = series(obs, 'test_acc'); o['final'] = float(te[-1])
    for key in ['test_acc', 'refit_acc', f'agop{r}_ho', f'agop{r}_ho_rand', f'agopH1_{r}_ho', 'W1_four_top1', 'AH_full_logit'] + [f'AH_full_h{l + 1}' for l in range(D)] + [f'AH_ho_h{l + 1}' for l in range(D)]:
        o[key] = (float(at(obs, key, a)), float(at(obs, key, b)))
    return o
def fmt(o):
    f = lambda t: '–' if np.isnan(t[0]) else f"{t[0]:.2f}→{t[1]:.2f}"
    r = [k for k in o if k.startswith('agop') and k.endswith('_ho') and not k.startswith('agopH1')][0]
    return (f"{o['run']:<34} flat[{o['a']},{o['b']}] len {o['b'] - o['a']:>5} CE {o['lev']:.3g} ({o['rel']:.2f}·logp) R {o['R']:.1f}% | test {f(o['test_acc'])} refit {f(o['refit_acc'])} (t90 {o['t90']}, fin {o['final']:.2f}) | "
            f"AGOPlast {f(o[r])} AGOP@h1 {f(o[[k for k in o if k.startswith('agopH1')][0]])} W1four {f(o['W1_four_top1'])} AH " + ' '.join(f(o[f'AH_full_h{l + 1}']) for l in range(o['D'])) + ' AHho ' + ' '.join(f(o[f'AH_ho_h{l + 1}']) for l in range(o['D'])))
if __name__ == '__main__':
    tol = 0.03; args = sys.argv[1:]
    if args[0].startswith('--tol='): tol = float(args[0][6:]); args = args[1:]
    for fn in sorted(sum([glob.glob(g) for g in args], [])):
        try: print(fmt(flat(fn, tol)))
        except Exception as e: print(fn, 'ERR', repr(e))

def best_grok_window(c, steps, test, start, tol=0.03, stride=20, lo=0.1, hi=0.9):
    from collections import deque
    idx = np.arange(start, len(c), stride); x = c[idx]
    tv = np.clip(np.interp(idx, steps, test), lo, hi)
    best = (-1.0, 0, None, None); j = 0; mx, mn, tm = deque(), deque(), deque()
    for i in range(len(x)):
        while mx and x[mx[-1]] <= x[i]: mx.pop()
        mx.append(i)
        while mn and x[mn[-1]] >= x[i]: mn.pop()
        mn.append(i)
        while tm and tv[tm[-1]] >= tv[i]: tm.pop()
        tm.append(i)
        while True:
            h_, l_ = x[mx[0]], x[mn[0]]
            if (h_ - l_) / (0.5 * (h_ + l_)) <= tol: break
            j += 1
            if mx[0] < j: mx.popleft()
            if mn[0] < j: mn.popleft()
            if tm[0] < j: tm.popleft()
        a = tm[0]; sh = (tv[i] - tv[a]) / (hi - lo)
        if sh > best[0] + 1e-12 or (abs(sh - best[0]) <= 1e-12 and i - j > best[1]):
            a_ = j if tv[j] <= tv[a] + 1e-12 else a
            best = (sh, i - a_, int(idx[a_]), int(idx[i]))
    return best[2], best[3], best[0]
