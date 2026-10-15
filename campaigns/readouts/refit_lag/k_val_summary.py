#!/usr/bin/env python3
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k_summary import cross

SETS = ('all', 'top8', 'top16', 'val', 'valK', 'oracle')


def main():
    res = {}
    for fn in sys.argv[2:]:
        d = json.load(open(fn)); rows = d['rows']; ts = [r['step'] for r in rows]; res[fn] = {}
        for f in d['fracs']:
            f = str(f); net = {q: cross(ts, [r[f]['acc']['net'] for r in rows], q) for q in (0.5, 0.9)}
            lead = {}
            for s in SETS:
                for q in (0.5, 0.9):
                    c = cross(ts, [r[f]['acc'][s] for r in rows], q)
                    lead[f'{s}@{q}'] = None if (c is None or net[q] is None) else round(net[q] - c)
            nv = [(r['step'], r[f]['n_val'], r[f]['n_oracle']) for r in rows]
            res[fn][f] = dict(net=net, lead=lead, n_val_rows=rows[-1][f]['n_val_rows'], n_test_rows=rows[-1][f]['n_test_rows'],
                              n_val_at_net05=min(nv, key=lambda x: abs(x[0] - (net[0.5] or 1e9)))[1:] if net[0.5] else None)
            fmt = lambda v: 'never' if v is None else f'{v:+d}'
            print(f"{fn} val fraction {f} ({res[fn][f]['n_val_rows']} validation rows): " +
                  ' '.join(f"{s} {fmt(lead[f'{s}@0.5'])}/{fmt(lead[f'{s}@0.9'])}" for s in SETS),
                  '| (n_val, n_oracle) at net 0.5:', res[fn][f]['n_val_at_net05'])
    json.dump(res, open(sys.argv[1], 'w'), indent=1)


if __name__ == '__main__':
    main()
