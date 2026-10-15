import glob, gzip, json, os, sys
import numpy as np
ROOTS = {'T': '/mnt/user-data/uploads/modular arithmetic/natural_plateau_shared/campaign_T/collected/results',
         'U': '/mnt/user-data/uploads/modular arithmetic/natural_plateau_shared/campaign_U/collected/results',
         'V': '/home/claude/kt/campaign_V/collected/results'}
pat = sys.argv[1] if len(sys.argv) > 1 else ''
def cross(steps, vals, lev):
    for s, v in zip(steps, vals):
        if v is not None and v >= lev: return s
    return None
rows = []
for camp, root in ROOTS.items():
    for d in sorted(glob.glob(os.path.join(root, '*'))):
        rid = os.path.basename(d)
        if pat and not any(x in rid for x in pat.split(',')): continue
        r = json.load(gzip.open(os.path.join(d, 'run.json.gz'), 'rt'))
        obs = r['obs']
        st = [o['step'] for o in obs]; te = [o['test_acc'] for o in obs]
        hv = [o for o in obs if 'refit_test_acc' in o]
        hs = [o['step'] for o in hv]
        out = dict(id=rid)
        for lev in (0.5, 0.9):
            out[f'net{lev}'] = cross(st, te, lev)
            out[f'netH{lev}'] = cross(hs, [o['test_acc'] for o in hv], lev)
            out[f'res{lev}'] = cross(hs, [o.get('refit_test_acc') for o in hv], lev)
            out[f'mlp{lev}'] = cross(hs, [o.get('refit_mlp_test_acc') for o in hv], lev)
            pv = [o for o in hv if 'probe_test_acc' in o]
            out[f'prb{lev}'] = cross([o['step'] for o in pv], [o['probe_test_acc'] for o in pv], lev) if pv else None
        out['final'] = te[-1]
        rows.append(out)
def f(x): return '-' if x is None else str(x)
print(f"{'run':58s} {'net.5':>6s} {'res.5':>6s} {'mlp.5':>6s} {'prb.5':>6s} | {'net.9':>6s} {'res.9':>6s} {'mlp.9':>6s} {'prb.9':>6s}  final")
for o in rows:
    print(f"{o['id'][:58]:58s} {f(o['net0.5']):>6s} {f(o['res0.5']):>6s} {f(o['mlp0.5']):>6s} {f(o['prb0.5']):>6s} | "
          f"{f(o['net0.9']):>6s} {f(o['res0.9']):>6s} {f(o['mlp0.9']):>6s} {f(o['prb0.9']):>6s}  {o['final']:.3f}")
