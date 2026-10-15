import json, re, collections, sys
import numpy as np
R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'scan_runs.json'))
cell = lambda r: re.sub(r'_s\d+$', '', r['id'])
G = collections.OrderedDict()
for r in R: G.setdefault(cell(r), []).append(r)
def lead(rs, rd, q='0.5'):
    L = [r['readouts'].get(rd, {}).get('lead', {}).get(q) for r in rs]; L = [x for x in L if x is not None]
    return (float(np.median(L)) if L else None, len(L), sum(x >= 0 for x in L))
def med(xs):
    xs = [x for x in xs if x is not None]; return float(np.median(xs)) if xs else None
f = lambda x, fmt='%6.0f': (fmt % x) if x is not None else '   -  '
print(f"{'cell':52s}  n   p L frac  opt     gain  net50 | refit lead50 (n,>=0) | four8 lead50 | probe lead50 | ncar top8sh")
for c, rs in G.items():
    r0 = rs[0]; rl = lead(rs, 'refit'); fl = lead(rs, 'fref_top8'); pl = lead(rs, 'probe')
    nc = med([r.get('at_rise', {}).get('n_carried') for r in rs]); ts = med([r.get('at_rise', {}).get('top8_share') for r in rs])
    print(f"{c[:52]:52s} {len(rs):2d} {r0['p']:3d} {r0['layers']} {r0['frac']:<5} {str(r0['opt'])[:6]:6s} {str(r0['gain']):5s} {f(med([r['net']['0.5'] for r in rs]))} | {f(rl[0])} ({rl[1]:2d},{rl[2]:2d}) | {f(fl[0])} ({fl[1]:2d},{fl[2]:2d}) | {f(pl[0])} ({pl[1]:2d},{pl[2]:2d}) | {f(nc,'%4.1f')} {f(ts,'%.2f')}")
