#!/usr/bin/env python3
import glob, gzip, json, os, re, sys
import numpy as np

GLOBS = ['/home/claude/v7work/data/*/*/run.json.gz'] + \
        [f'/home/claude/kt/campaign_{c}/collected/results/*/run.json.gz' for c in ('K', 'L', 'M', 'N', 'QD')]


def cross(ts, vs, q):
    for i, v in enumerate(vs):
        if v is not None and v >= q:
            j = i - 1
            if j < 0: return float(ts[i])
            t0, v0 = ts[j], vs[j]
            return float(t0 + (q - v0) / (v - v0) * (ts[i] - t0)) if v > v0 else float(ts[i])
    return None


out = []
for fn in sorted(sum((glob.glob(g) for g in GLOBS), [])):
    try:
        d = json.load(gzip.open(fn))
    except Exception as e:
        print('skip', fn, e); continue
    c = d.get('cfg', {}); ob = d.get('obs', [])
    if 'p' not in c: continue
    ts = [o['step'] for o in ob if 'test_acc' in o]; vs = [o['test_acc'] for o in ob if 'test_acc' in o]
    key = 'v1_ridge_acc_te' if any('v1_ridge_acc_te' in o for o in ob) else ('refit_acc' if any('refit_acc' in o for o in ob) else None)
    if key is None: continue
    tr = [o['step'] for o in ob if key in o]; vr = [o[key] for o in ob if key in o]
    rec = dict(id=os.path.basename(os.path.dirname(fn)), src=fn.split('/')[-3], p=c['p'], depth=c.get('depth', 1),
               width=c.get('width'), frac=c.get('frac'), lam_v=c.get('lam_v'), key=key, heavy=c.get('heavy_every'))
    for q in (0.5, 0.9):
        tn, trf = cross(ts, vs, q), cross(tr, vr, q)
        rec[f'net{q}'] = tn; rec[f'ref{q}'] = trf
        rec[f'lead{q}'] = None if (tn is None or trf is None) else tn - trf
    out.append(rec)
json.dump(out, open(sys.argv[1], 'w'), indent=0)
print(len(out), 'runs')
