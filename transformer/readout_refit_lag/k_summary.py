#!/usr/bin/env python3
import json, sys
import numpy as np


def cross(ts, vs, q):
    for i, v in enumerate(vs):
        if v >= q:
            if i == 0: return float(ts[0])
            t0, v0 = ts[i - 1], vs[i - 1]
            return float(t0 + (q - v0) / (v - v0) * (ts[i] - t0)) if v > v0 else float(ts[i])
    return None


def near(rows, t):
    return min(rows, key=lambda r: abs(r['step'] - t))


def one(fn):
    d = json.load(open(fn)); rows = d['rows']; ts = [r['step'] for r in rows]
    Kmax = len(rows[0]['acc_K']); out = dict(file=fn, kind=d['kind'], p=d['cfg']['p'], Kmax=Kmax)
    net = {q: cross(ts, [r['net'] for r in rows], q) for q in (0.5, 0.9)}
    out['net'] = net
    for q in (0.5, 0.9):
        leads = []
        for K in range(1, Kmax + 1):
            c = cross(ts, [r['acc_K'][K - 1] for r in rows], q)
            leads.append(None if (c is None or net[q] is None) else net[q] - c)
        out[f'lead{q}'] = leads
        ok = [l for l in leads if l is not None]
        if ok:
            best = max(ok); out[f'bestlead{q}'] = best
            out[f'bestK_lead{q}'] = int(np.argmax([l if l is not None else -1e9 for l in leads]) + 1)
            out[f'Kband{q}'] = [K for K, l in enumerate(leads, 1) if l is not None and l >= best - 200]
        if net[q] is not None:
            r = near(rows, net[q]); out[f'at_net{q}'] = dict(step=r['step'], carried=r['carried'], bestK=r['bestK'],
                                                            acc8=r['acc_K'][7], acc16=r['acc_K'][min(15, Kmax - 1)],
                                                            acc_all=r['acc_K'][-1], best=max(r['acc_K']))
    c8 = cross(ts, [r['acc_K'][7] for r in rows], 0.5)
    if c8 is not None: out['at_top8_cross'] = dict(step=c8, carried=near(rows, c8)['carried'])
    return out


def main():
    res = [one(f) for f in sys.argv[2:]]
    json.dump(res, open(sys.argv[1], 'w'), indent=1)
    for r in res:
        f = lambda v: 'never' if v is None else f'{v:+.0f}'
        print(f"{r['file']}: {r['kind']} p={r['p']} K={r['Kmax']} net0.5={r['net'][0.5]} net0.9={r['net'][0.9]}")
        for q in (0.5, 0.9):
            L = r[f'lead{q}']
            pick = {K: L[K - 1] for K in (1, 2, 4, 6, 8, 12, 16, 24, 32, r['Kmax']) if K <= r['Kmax']}
            print(f"   lead@{q}: " + ' '.join(f"K{K}:{f(v)}" for K, v in pick.items()),
                  f"| best {f(r.get(f'bestlead{q}'))} at K={r.get(f'bestK_lead{q}')}, within 200: K {min(r.get(f'Kband{q}', [0]))}-{max(r.get(f'Kband{q}', [0]))}")
            if f'at_net{q}' in r: print(f"   at net {q}: {r[f'at_net{q}']}")
        if 'at_top8_cross' in r: print(f"   at top-8 crossing 0.5: {r['at_top8_cross']}")


if __name__ == '__main__':
    main()
