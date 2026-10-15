import gzip, json, sys, math
import numpy as np
fn = sys.argv[1]; lo = int(sys.argv[2]); hi = int(sys.argv[3]); every = int(sys.argv[4]) if len(sys.argv) > 4 else 250
r = json.load(gzip.open(fn, 'rt')); L = np.array(r['loss']); p = r['cfg_full']['p']; lp = math.log(p)
print(r['cfg_full'].get('final_norm'), 'lam_v', r['cfg_full']['lam_v'], 'eta_w', r['cfg_full']['eta_w'], 'lam_w', r['cfg_full']['lam_w'], 'eta_v', r['cfg_full']['eta_v'], 'n', len(L))
print(f"{'step':>6s} {'CE/logp':>8s} {'ce250':>8s} {'tracc':>6s} {'trmmin':>8s} {'teacc':>6s} {'refit':>6s} {'rmlp':>6s} {'probe':>6s} {'normV':>6s} {'AHr':>5s} {'AHh':>5s}")
for o in r['obs']:
    s = o['step']
    if s < lo or s > hi or s % every: continue
    a = max(0, s - 125); b = min(len(L), s + 125)
    print(f"{s:6d} {L[s]/lp:8.2e} {L[a:b].mean()/lp:8.2e} {o['train_acc']:6.3f} {o['train_margin_min']:8.3f} {o['test_acc']:6.3f} "
          f"{o.get('refit_test_acc', float('nan')):6.3f} {o.get('refit_mlp_test_acc', float('nan')):6.3f} {o.get('probe_test_acc', float('nan')):6.3f} "
          f"{o.get('normV', float('nan')):6.2f} {o.get('AH_full_resid', float('nan')):5.2f} {o.get('AH_full_h1', float('nan')):5.2f}")
