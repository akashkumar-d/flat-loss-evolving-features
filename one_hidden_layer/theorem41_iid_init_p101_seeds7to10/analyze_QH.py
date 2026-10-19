#!/usr/bin/env python3
"""Campaign QH analysis (akashkumar-d, 2026-10-13): campaign H's per-run endpoints J1-J4 and the joint endpoint (campaign
H's analyze.fields, imported unchanged from the code release) on the 10 runs of the eta_V = 3e-4 arm: seeds 1-6 from campaign H's
archive, seeds 7-10 from QH; the primary arm eta_V = 1e-3 (seeds 1-10) alongside.

    python3 -I analyze_QH.py CAMPAIGN_H_DIR H_ARCHIVE_DIR QH_RESULTS OUT.json
"""
import json, os, sys
import numpy as np
H_DIR, ARCH, QH, OUT = sys.argv[1:5]
sys.path.insert(0, H_DIR)
import analyze as AH

R = AH.runs(ARCH)
for k, v in AH.runs(QH).items(): R[k.replace('QH_', '')] = v
rows = {}
for arm in ('ev3e-4', 'ev1e-3'):
    for s in range(1, 11):
        rid = f'T41_{arm}_s{s}'
        if rid in R:
            f = AH.fields(R[rid]); f.update(arm=arm, seed=s, source='QH' if (arm == 'ev3e-4' and s >= 7) else 'H'); rows[rid] = f
summ = {}
for arm in ('ev3e-4', 'ev1e-3'):
    fs = [f for f in rows.values() if f['arm'] == arm]; n = len(fs)
    s = dict(n=n, **{J: sum(f[J] for f in fs) for J in ('J1', 'J2', 'J3', 'J4', 'joint')})
    s['joint_wilson'] = AH.wilson(s['joint'], n)
    for q in ('v', 'dAH', 'refit_T', 'test_T', 'train_T', 't_refit90', 't_refit50', 't_test50'):
        vals = [f[q] for f in fs if f[q] is not None]
        s[q] = dict(median=float(np.median(vals)), min=float(min(vals)), max=float(max(vals))) if vals else None
    summ[arm] = s
json.dump(dict(doc=__doc__, rows=rows, arms=summ), open(OUT, 'w'), indent=1, default=float)
for arm, s in summ.items():
    print(arm, {k: s[k] for k in ('n', 'J1', 'J2', 'J3', 'J4', 'joint')}, 'v', s['v'], 'test_T', s['test_T'], 'refit 0.9 at', s['t_refit90'])
for rid, f in rows.items():
    if f['arm'] == 'ev3e-4': print(rid, f['source'], round(f['v'], 4), round(f['dAH'], 3), f['t_refit50'], f['t_refit90'], f['t_test50'], round(f['test_T'], 3), round(f['train_T'], 4), f['joint'])
