import sys, glob, os, math
sys.path.insert(0, '/home/claude/kt/campaign_W')
import numpy as np, paperdefs_tf as P
for fn in sys.argv[1:]:
    r = P.load(fn); c = P.cfg_of(r); L = np.array(r['loss']); lp = math.log(c['p'])
    s, te = P.series(r['obs'], 'test_acc'); sm, mm = P.series(r['obs'], 'train_margin_min')
    sr, rf = P.series(r['obs'], 'refit_test_acc'); sp, pb = P.series(r['obs'], 'probe_test_acc')
    ff = P.first_fit(r); rw = P.raw10(r)
    t = {lev: P.first_cross(s, te, lev) for lev in (0.1, 0.2, 0.5, 0.9)}
    tr = {lev: P.first_cross(sr, rf, lev) for lev in (0.2, 0.5, 0.9)}; tp = {lev: P.first_cross(sp, pb, lev) for lev in (0.2, 0.5, 0.9)}
    print(os.path.basename(fn), 'hidden', c.get('hidden_mode', 'polar'), 'lam_v', c['lam_v'])
    print('  first fit (all margins > 0):', ff, ' held-out crossings', t)
    print('  ridge refit crossings', tr, ' probe crossings', tp)
    if t[0.2] and t[0.9]:
        a, b = t[0.2], t[0.9]; seg = L[a:b + 1]
        print(f'  raw CE/log p at held-out 0.2 {L[a]/lp:.4f}, at 0.9 {L[b]/lp:.4f}; v over [t20, t90] = {seg.max()/seg.min()-1:.3f}; '
              f'logs with margin<=0 in it: {int(np.sum(mm[(sm>=a)&(sm<=b)]<=0))}')
    print('  raw10 best share (strict fit, no fit):', rw)
    print('  final held-out', te[-1], ' CE/log p at end', L[-1] / lp)
