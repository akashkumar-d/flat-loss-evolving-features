import glob, gzip, json, os, sys, math
import numpy as np
ROOTS = {'T': '/mnt/user-data/uploads/modular arithmetic/natural_plateau_shared/campaign_T/collected/results',
         'U': '/mnt/user-data/uploads/modular arithmetic/natural_plateau_shared/campaign_U/collected/results',
         'V': '/home/claude/kt/campaign_V/collected/results'}
pat = sys.argv[1] if len(sys.argv) > 1 else ''
def cross(steps, vals, lev):
    for s, v in zip(steps, vals):
        if v >= lev: return s
    return None
print(f"{'run':52s} {'Tfit':>6s} {'persist':>7s} {'te@fit':>6s} {'t10':>6s} {'t50':>6s} {'t90':>6s} {'CEfit':>6s} {'CEt90':>6s} {'v_raw':>6s} {'negfrac':>7s} final")
for camp, root in ROOTS.items():
    for d in sorted(glob.glob(os.path.join(root, '*'))):
        rid = os.path.basename(d)
        if pat and not any(x in rid for x in pat.split(',')): continue
        r = json.load(gzip.open(os.path.join(d, 'run.json.gz'), 'rt')); L = np.array(r['loss']); lp = math.log(r['cfg_full']['p'])
        obs = r['obs']; st = np.array([o['step'] for o in obs]); mm = np.array([o['train_margin_min'] for o in obs]); te = np.array([o['test_acc'] for o in obs])
        pos = np.where(mm > 0)[0]
        tfit = int(st[pos[0]]) if len(pos) else None
        t10, t50, t90 = (cross(st, te, x) for x in (0.1, 0.5, 0.9))
        if tfit is not None and t90 is not None and t90 > tfit:
            seg = (st >= tfit) & (st <= t90); negfrac = float((mm[seg] <= 0).mean())
            persist = 'yes' if negfrac == 0 else 'no'
            Ls = L[tfit:t90 + 1]; v = Ls.max() / Ls.min() - 1
            print(f"{rid[:52]:52s} {tfit:6d} {persist:>7s} {te[pos[0]]:6.3f} {t10 or -1:6d} {t50 or -1:6d} {t90:6d} {L[tfit]/lp:6.3f} {L[t90]/lp:6.3f} {v:6.2f} {negfrac:7.2f} {te[-1]:.3f}")
        else:
            print(f"{rid[:52]:52s} {tfit if tfit is not None else -1:6d} {'-':>7s} {'-':>6s} {t10 or -1:6d} {t50 or -1:6d} {t90 or -1:6d} {'-':>6s} {'-':>6s} {'-':>6s} {'-':>7s} {te[-1]:.3f}")
