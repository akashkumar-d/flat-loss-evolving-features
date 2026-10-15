import glob, json, os, math, numpy as np
from make_readme import stats, bestwin
def last_rate(c):
    if 'eta_layers' in c:
        e, l = c['eta_layers'], c['lam_layers']; return e[-1] * l[-1] / (e[0] * l[0])
    if 'eta_deep' in c: return c['eta_deep'] * c['lam_deep'] / (c['eta_w'] * c['lam_w'])
    return None
rows = []; incomplete = []
fns = [f for f in sorted(glob.glob('runs/*.json')) if os.path.basename(f).startswith(('T', 'seedD2', 'd2', 'd3', 'd12_w256', 'd15_w256'))]
for f in fns:
    try: c = json.load(open(f))['cfg']
    except Exception: continue
    if c.get('depth', 1) < 2 or c.get('deep_init_scale', 1) < 4 or c.get('loss') != 'ce': continue
    r = last_rate(c)
    if r is None: continue
    o = stats(f)
    if o['t10'] is None: continue
    if o['t90'] is None:
        incomplete.append((os.path.basename(f)[:-5], o['t10'], round(o['final'], 2))); continue
    a, b, sh, lev, R = bestwin(f)
    kappa0 = c['eta_layers'][0] * c['lam_layers'][0] if 'eta_layers' in c else c['eta_w'] * c['lam_w']
    rows.append((c['depth'], c.get('width', 256), c['p'], os.path.basename(f)[:-5], round(r, 3), o['t10'], o['t90'], r * kappa0 * o['t10'], o['g_D'], sh, c.get('deep_init_scale')))
rows.sort(key=lambda x: (x[2], x[1], x[0], x[4], x[3]))
print(f"{'D':>2} {'w':>4} {'p':>3} {'run':50s} {'r':>5} {'t10':>6} {'t90':>6} {'t10/tau':>7} {'drift':>7} {'share':>5}")
for x in rows: print(f"{x[0]:>2} {x[1]:>4} {x[2]:>3} {x[3][:50]:50s} {x[4]:5.2f} {x[5]:6d} {str(x[6]):>6} {x[7]:7.2f} {x[8]:+7.1f} {100*x[9]:5.0f}")
json.dump(rows, open('rate_rule_rows.json', 'w')); json.dump(incomplete, open('rate_rule_incomplete.json', 'w'))
