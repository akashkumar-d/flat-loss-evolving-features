#!/usr/bin/env python3
"""Campaign Q analysis, fixed in PLAN.md section 5 (akashkumar-d, 2026-10-13).

Every run is scored by campaign Z's per-run analysis (analyze_Z.per_run and summarize, unchanged), with campaign ZM's split
count for the mirror-free split. Cells pool the seeds 11-15 of the earlier campaigns with Q's seeds 16-20:
  P_p61_mf    ZM's ZM_f0.55_s11-15 + Q_M_p61_f0.55_lam_v0.45_s16-20         (across-primes figure, p = 61)
  P_p61       ZP's ZP_P_p61_f0.33_lam_v0.45_s11-15 + Q_P_..._s16-20           (mirror-pair figures, random split)
  P_p89       ZP's ZP_P_p89_f0.25_lam_v0.33_s11-15 + Q's s16-20
  P_p149      ZP's ZP_P_p149_f0.16_lam_v0.23_s11-15 + Q's s16-20
  P_p113      Y's G2, seeds 11-20 (unchanged)
  B_base      Z's Z_B_f0.2_lam_v0.275_tau10k_s11-15 + Q's s16-20
  B_F12, B_E12, B_V12, B_f0.175, B_f0.225: Q's seeds 11-13
Two-block diagnostics per run (after update 12000): the first update with a misclassified training pair, the smallest training
margin, the most misclassified pairs at a log, the growth of the MLP norms (largest value over the value at 12000), the gain at
12000 and at the end; for F12, E12 and V12 also whether the run equals Z's base run of its seed up to update 12000.

Usage:  python3 -I analyze_Q.py OUT.json SHARED
  SHARED  natural_plateau_shared (campaign_{Q,ZP,Z,Y}/collected/results and campaign_ZM/runs)
"""
import glob, gzip, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('CAMPAIGN_Z_DIR', os.path.join(HERE, '..', 'gain_recipe_contrasts_two_blocks')))
os.environ.setdefault('CAMPAIGN_X_DIR', os.path.join(HERE, '..', 'one_block_no_gain_p113'))
os.environ.setdefault('CAMPAIGN_Y_DIR', os.path.join(HERE, '..', 'one_block_gain_p113'))
import analyze_Z as AZ
sys.path.insert(0, HERE)
import tf_runner as R

_random_counts = AZ.split_counts


def split_counts(c):
    if c.get('split_mode', 'random') == 'random': return _random_counts(c)
    p = c['p']; tr = R.mirror_free_train(p, c['frac'], c['split_seed'])
    intr = np.zeros(p * p, bool); intr[tr] = True; te = np.where(~intr)[0]; a, b = te // p, te % p
    return len(tr), len(te), int((~intr[b * p + a]).sum())


AZ.split_counts = split_counts
LATE = 12000


def files(shared):
    Q = os.path.join(shared, 'campaign_Q', 'collected', 'results'); ZP = os.path.join(shared, 'campaign_ZP', 'collected', 'results')
    Z = os.path.join(shared, 'campaign_Z', 'collected', 'results'); Y = os.path.join(shared, 'campaign_Y', 'collected', 'results')
    ZM = os.path.join(shared, 'campaign_ZM', 'runs')
    g = lambda root, pat: sorted(glob.glob(os.path.join(root, pat, 'run.json.gz')))
    return {
        'P_p61_mf': g(ZM, 'ZM_f0.55_s1[1-5]') + g(Q, 'Q_M_p61_f0.55_lam_v0.45_s*'),
        'P_p61': g(ZP, 'ZP_P_p61_f0.33_lam_v0.45_s*') + g(Q, 'Q_P_p61_f0.33_lam_v0.45_s*'),
        'P_p89': g(ZP, 'ZP_P_p89_f0.25_lam_v0.33_s*') + g(Q, 'Q_P_p89_f0.25_lam_v0.33_s*'),
        'P_p149': g(ZP, 'ZP_P_p149_f0.16_lam_v0.23_s*') + g(Q, 'Q_P_p149_f0.16_lam_v0.23_s*'),
        'P_p113': g(Y, 'Y_G2_*'),
        'B_base': g(Z, 'Z_B_f0.2_lam_v0.275_tau10k_s*') + g(Q, 'Q_B_f0.2_lam_v0.275_tau10k_s*'),
        'B_F12': g(Q, 'Q_B_f0.2_lam_v0.275_tau10k_F12_s*'), 'B_E12': g(Q, 'Q_B_f0.2_lam_v0.275_tau10k_E12_s*'),
        'B_V12': g(Q, 'Q_B_f0.2_lam_v0.275_tau10k_V12_s*'),
        'B_f0.175': g(Q, 'Q_B_f0.175_lam_v0.275_tau10k_s*'), 'B_f0.225': g(Q, 'Q_B_f0.225_lam_v0.275_tau10k_s*'),
    }, Z


def seed_of(fn): return int(os.path.basename(os.path.dirname(fn)).rsplit('_s', 1)[1])


def late(fn, ntr, ref=None):
    d = AZ.AX.load(fn); obs = d['obs']; c = d.get('cfg_full') or d['cfg']
    st = np.array([o['step'] for o in obs]); tr = np.array([o.get('train_acc', np.nan) for o in obs])
    mis = np.round((1 - tr) * ntr); after = st > LATE
    out = dict(steps=len(d['loss']) - 1, stop_reason=d.get('stop_reason'))
    k = np.flatnonzero(after & (mis > 0)); out['first_flip'] = int(st[k[0]]) if len(k) else None
    out['max_mis_late'] = int(np.nanmax(mis[after])) if after.any() else None
    mg = np.array([o.get('train_margin_min', np.nan) for o in obs]); out['min_margin_late'] = float(np.nanmin(mg[after])) if after.any() else None
    for key in ('normW', 'normW_W_out1', 'normW_W_in1', 'normW_W_out0', 'normV'):
        v = np.array([o.get(key, np.nan) for o in obs]); ok = np.isfinite(v)
        if not ok.any(): continue
        i12 = np.flatnonzero(ok & (st <= LATE)); after_ok = ok & after
        if len(i12) and after_ok.any(): out[f'{key}_growth'] = float(np.nanmax(v[after_ok]) / v[i12[-1]])
    sg = [(o['step'], o['gain']) for o in obs if 'gain' in o]
    if sg:
        g12 = [g for s, g in sg if s <= LATE]; out['gain_12k'] = g12[-1] if g12 else None; out['gain_end'] = sg[-1][1]
    te = [o['test_acc'] for o in obs if 'test_acc' in o]; out['final_test'] = te[-1] if te else None
    if ref is not None:
        r = AZ.AX.load(ref); n = min(LATE + 1, len(r['loss']), len(d['loss']))
        a, b = np.asarray(d['loss'][:n]), np.asarray(r['loss'][:n])
        out['equal_to_base_until_12k'] = bool(np.all(np.abs(a - b) <= 5e-7 * np.abs(b)))
        out['late_rates'] = c.get('late'); out['late_at'] = c.get('late_at')
    return out


def main(outp, shared):
    F, Zdir = files(shared); cells = {}
    for name, fs in F.items():
        runs = []
        for fn in fs:
            rec = dict(AZ.per_run(fn), id=os.path.basename(os.path.dirname(fn)), seed=seed_of(fn))
            if name.startswith('B_'):
                ref = None
                if name in ('B_F12', 'B_E12', 'B_V12'):
                    ref = os.path.join(Zdir, f"Z_B_f0.2_lam_v0.275_tau10k_s{rec['seed']}", 'run.json.gz')
                rec['late'] = late(fn, rec['n_train'], ref)
            runs.append(rec)
        cells[name] = dict(runs=runs, summary=AZ.summarize(runs) if runs else None)
    res = dict(doc=__doc__, cells=cells)
    json.dump(res, open(outp, 'w'), indent=1, default=float)
    print(f"{'cell':10s} {'n':>3s} {'strict':>6s} {'loose':>5s} {'CE chg (med)':>12s}  seeds")
    for name, c in cells.items():
        if not c['runs']: print(f'{name:10s}   0'); continue
        S = c['summary']
        print(f"{name:10s} {S['n']:3d} {S['pass_strict']:6d} {S['pass_loose']:5d} {S['ce_change']['median'] if S['ce_change']['median'] is not None else float('nan'):12.3f}  "
              f"{sorted(r['seed'] for r in c['runs'])}")
    for name, c in cells.items():
        if not name.startswith('B_'): continue
        for r in c['runs']:
            L = r['late']
            print(f"  {r['id']:42s} flip {L['first_flip']}, max mis {L['max_mis_late']}, min margin {L['min_margin_late']:.2f}, "
                  f"Wout1 x{L.get('normW_W_out1_growth', float('nan')):.2f}, final {L['final_test']:.3f}, strict {r['pass_strict']}, "
                  f"why {r['why']}" + (f", equal to base until 12k: {L['equal_to_base_until_12k']}" if 'equal_to_base_until_12k' in L else ''))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
