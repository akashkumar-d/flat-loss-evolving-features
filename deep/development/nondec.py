import numpy as np
from anat import load, series, knee_step
def nondec_window(fn, tol=0.03, delta=0.01, stride=50, lo=0.1, hi=0.9):
    d, L, obs, st = load(fn); c = (L[:-1] + L[1:]) / 2
    k = knee_step(d, obs) or 0
    idx = np.arange(k, len(c), stride); x = c[idx]
    s_, te = series(obs, 'test_acc'); tv = np.clip(np.interp(idx, s_, te), lo, hi)
    best = (-1, 0, None, None)
    n = len(x)
    for i in range(n):
        mx = mn = runmax = x[i]
        for j in range(i, n):
            v = x[j]; mx = max(mx, v); mn = min(mn, v); runmax = max(runmax, v)
            mid = 0.5 * (mx + mn)
            if (mx - mn) / mid > tol or (runmax - v) / mid > delta: break
            sh = (tv[j] - tv[i]) / (hi - lo)
            if sh > best[0] + 1e-12 or (abs(sh - best[0]) <= 1e-12 and j - i > best[1]):
                best = (sh, j - i, int(idx[i]), int(idx[j]))
    sh, _, a, b = best
    seg = c[a:b + 1]
    return a, b, sh, float(seg.mean()), float(100 * (seg[-1] - seg[0]) / seg.mean())
