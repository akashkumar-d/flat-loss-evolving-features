from collections import deque
import numpy as np

def stretches(c, start, tol=1.03):
    n = len(c); amin = np.full(n, -1, int); dq = deque(); a = start
    for t in range(start, n):
        while dq and c[dq[-1]] <= c[t]: dq.pop()
        dq.append(t)
        while c[dq[0]] > tol * c[t]:
            a += 1
            while dq[0] < a: dq.popleft()
        amin[t] = a
    return amin

def window(c, knee, tol=1.03):
    start = int(min(max(knee or 0, 0), len(c) - 1)); amin = stretches(c, start, tol)
    t = np.arange(start, len(c)); L = t - amin[start:]; k = int(np.argmax(L)); longest = (int(amin[start + k]), int(start + k))
    tp = start + int(np.argmax(c[start:]))
    ok = (t >= tp) & (amin[start:] <= tp); bs = t[ok]
    if len(bs):
        Lp = bs - amin[bs]; j = int(np.argmax(Lp)); peak = (int(amin[bs[j]]), int(bs[j]))
    else: peak = (tp, tp)
    return longest, peak
