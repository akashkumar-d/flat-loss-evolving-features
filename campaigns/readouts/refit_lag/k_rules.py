#!/usr/bin/env python3
import glob, gzip, json, os, sys
import numpy as np

ROOT = '/home/claude/kt'
GLOBS = [f'{ROOT}/campaign_{c}/collected/results/*/run.json.gz' for c in ('V', 'W', 'X', 'Y', 'Z', 'ZP', 'Q')] + \
        [f'{ROOT}/campaign_ZM/runs/*/run.json.gz']
CS = (2, 3, 5)


def cross(ts, vs, q):
    for t, v in zip(ts, vs):
        if v >= q: return t
    return None


def at(o):
    r2 = np.array(o['fspec_hor2']); K = len(r2); car = set(np.where(r2 > 0.05)[0].tolist())
    row = dict(step=o['step'], K=K, carried=len(car), r2_pos=int((r2 > 0).sum()))
    for name, key in (('E', 'fspec_train'), ('emb', 'fspec_embed'), ('head', 'fspec_head')):
        if key not in o or o[key] is None: continue
        x = np.array(o[key], float); x = x / x.sum(); order = np.argsort(-x)
        for c in CS: row[f'{name}_cnt{c}'] = int((x > c / K).sum())
        row[f'{name}_pr'] = float(1.0 / (x ** 2).sum())
        if name != 'head':
            for k in (4, 8, 16): row[f'{name}_top{k}_hit'] = len(car & set(order[:k].tolist()))
            pos = [int(np.where(order == j)[0][0]) + 1 for j in car]
            row[f'{name}_cover'] = max(pos) if pos else 0
    return row


def one(fn):
    d = json.load(gzip.open(fn)); obs = d['obs']; cfg = d.get('cfg', {})
    nm = any('test_acc_nm' in o for o in obs); key = 'test_acc_nm' if nm else 'test_acc'
    sp = [o for o in obs if 'fspec_hor2' in o and o.get('fspec_train') is not None]
    if not sp: return None
    ts = [o['step'] for o in obs if key in o]; vs = [o[key] for o in obs if key in o]
    out = dict(id=os.path.basename(os.path.dirname(fn)), p=cfg.get('p'), layers=cfg.get('n_layers'), at={})
    for q in (0.1, 0.5, 0.9):
        t = cross(ts, vs, q)
        if t is not None: out['at'][str(q)] = at(min(sp, key=lambda o: abs(o['step'] - t)))
    out['at']['end'] = at(sp[-1])
    return out


def main():
    fns = sorted(f for g in GLOBS for f in glob.glob(g))
    res = [r for r in (one(f) for f in fns) if r]
    json.dump(res, open(sys.argv[1], 'w'))
    print(len(res), 'runs')


if __name__ == '__main__':
    main()
