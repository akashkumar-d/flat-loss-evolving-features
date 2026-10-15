import glob, gzip, json, os, sys, math
import numpy as np
ROOTS = {'T': '/mnt/user-data/uploads/modular arithmetic/natural_plateau_shared/campaign_T/collected/results',
         'U': '/mnt/user-data/uploads/modular arithmetic/natural_plateau_shared/campaign_U/collected/results',
         'V': '/home/claude/kt/campaign_V/collected/results'}
pat = sys.argv[1]
def cross(st, v, lev):
    i = np.where(v >= lev)[0]; return int(st[i[0]]) if len(i) else None
print(f"{'run':52s} {'t10':>6s} {'t90':>6s} {'v_raw':>7s} {'v_2':>7s} {'v_500':>7s} {'fit_all':>7s} {'CE/logp':>8s}")
for camp, root in ROOTS.items():
    for d in sorted(glob.glob(os.path.join(root, '*'))):
        rid = os.path.basename(d)
        if not any(x in rid for x in pat.split(',')): continue
        r = json.load(gzip.open(os.path.join(d, 'run.json.gz'), 'rt')); L = np.array(r['loss']); lp = math.log(r['cfg_full']['p'])
        obs = r['obs']; st = np.array([o['step'] for o in obs]); te = np.array([o['test_acc'] for o in obs]); mm = np.array([o['train_margin_min'] for o in obs])
        t10, t90 = cross(st, te, 0.1), cross(st, te, 0.9)
        if t10 is None or t90 is None: print(f"{rid[:52]:52s} no rise"); continue
        seg = L[t10:t90 + 1]; L2 = 0.5 * (L[:-1] + L[1:]); seg2 = L2[t10:t90]
        k = np.ones(501) / 501; Lm = np.convolve(L, k, mode='same'); segm = Lm[max(t10, 250):min(t90, len(L) - 251) + 1]
        fit = bool((mm[(st >= t10) & (st <= t90)] > 0).all())
        print(f"{rid[:52]:52s} {t10:6d} {t90:6d} {seg.max()/seg.min()-1:7.3f} {seg2.max()/seg2.min()-1:7.3f} {segm.max()/segm.min()-1:7.3f} {str(fit):>7s} {np.median(seg)/lp:8.2e}")
