import sys, math, torch, numpy as np
st = torch.load(sys.argv[1], weights_only=False); every = int(sys.argv[2]) if len(sys.argv) > 2 else 100
L = st['loss']; lp = math.log(113)
print('t_next', st['t_next'])
print(f"{'step':>6s} {'CE/logp':>8s} {'tracc':>6s} {'trmmin':>8s} {'teacc':>6s} {'refit':>6s} {'rmlp':>6s} {'probe':>6s} {'normV':>6s} {'AHr':>5s}")
for o in st['obs']:
    s = o['step']
    if s % every: continue
    print(f"{s:6d} {L[s]/lp:8.4f} {o['train_acc']:6.3f} {o['train_margin_min']:8.3f} {o['test_acc']:6.3f} {o.get('refit_test_acc', float('nan')):6.3f} "
          f"{o.get('refit_mlp_test_acc', float('nan')):6.3f} {o.get('probe_test_acc', float('nan')):6.3f} {o.get('normV', float('nan')):6.2f} {o.get('AH_full_resid', float('nan')):5.2f}")
