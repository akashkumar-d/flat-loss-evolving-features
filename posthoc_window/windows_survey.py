#!/usr/bin/env python3
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import ovw, ovw2, win

ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
a = ap.parse_args(); ovw.configure()
rows, seen = [], set()
for camp in 'CDEF':
    for line in open(os.path.join(a.out, camp, 'settings.jsonl')):
        s = json.loads(line)
        if s['camp'] != camp: continue
        for run in s['runs']:
            rid = run['id']
            if rid in seen: continue
            seen.add(rid); r = ovw2.record2(rid)
            (la, lb), (pa, pb) = win.window(r['cm'], r['knee'])
            res = {}
            for name, (x0, x1) in (('longest', (la, lb)), ('peak', (pa, pb)), ('rise', tuple(r['W2']))):
                gte = ovw2.inside_gain(r['t'], r['te'], x0, x1); grf = ovw2.inside_gain(r['tf'], r['rf'], x0, x1)
                res[name] = dict(W=[int(x0), int(x1)], g_te=round(gte, 4), g_rf=round(grf, 4), ok=bool(gte >= ovw2.GAIN or grf >= ovw2.GAIN))
            rows.append(dict(camp=camp, id=rid, knee=r['knee'], stop=r['stop'], **res))
json.dump(rows, open(os.path.join(a.out, 'windows_survey.json'), 'w'), indent=0)
print('| campaign | runs | longest = rise | peak = rise | pass: rise (used) | pass: peak | pass: longest | verdicts longest ≠ rise |')
print('|---|---|---|---|---|---|---|---|')
for camp in 'CDEF':
    R = [x for x in rows if x['camp'] == camp]
    print(f"| {camp} | {len(R)} | {sum(x['longest']['W'] == x['rise']['W'] for x in R)} | {sum(x['peak']['W'] == x['rise']['W'] for x in R)} | "
          f"{sum(x['rise']['ok'] for x in R)} | {sum(x['peak']['ok'] for x in R)} | {sum(x['longest']['ok'] for x in R)} | "
          f"{sum(x['longest']['ok'] != x['rise']['ok'] for x in R)} |")
late = [x for x in rows if x['longest']['W'] != x['rise']['W']]
tail = [x for x in late if x['longest']['W'][0] > x['rise']['W'][1]]
print(f"\nThe literal longest stretch differs from the largest-rise one in {len(late)} of {len(rows)} runs; in {len(tail)} of them it starts "
      f"after the largest-rise stretch ends (the flat tail after grokking).")
