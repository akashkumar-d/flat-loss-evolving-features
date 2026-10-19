#!/usr/bin/env python3
"""Campaign ZP analysis, fixed in PLAN.md section 5 (akashkumar-d, 2026-10-12).

Campaign Z's analysis (analyze_Z.py: Y's thresholds, both held-out definitions, the paper's window), unchanged, on the ZP runs,
with these cells alongside for comparison:
  Z's other-p cells  (P_p61_f0.3_*, P_p89_f0.2_*, P_p149_f0.2_*; the fractions ZP replaces)
  Z's R_frac0.3      (p = 113, frac 0.3: a rise that came before the CE dip)
  Y's G2             (p = 113, frac 0.2: the reference recipe)
It adds two things, both fixed before the runs:
  timing    for every cell: x = n_train / (p ln p), the start of the held-out rise (no-mirror accuracy crosses 0.1; median over
            seeds), the CE dip (minimum of the 201-update training CE average), their ratio per run (median), and the rise start
            predicted by PLAN.md section 2, 2300 / (x - 3.9) updates
  appendix  for each p the cell with the most strict passes (ties: more loose passes, then the smaller frac); it 'works' at
            that p if it passes strictly in at least 4 of 5 seeds

Usage:  python3 analyze_ZP.py OUT.json ZP_RESULTS [Z_RESULTS Y_RESULTS]
  ZP_RESULTS  campaign_ZP/collected/results     Z_RESULTS  campaign_Z/collected/results     Y_RESULTS  campaign_Y/collected/results
Needs campaign Z's analyze_Z.py (../campaign_Z, or CAMPAIGN_Z_DIR), which needs campaign X's and Y's analysis scripts.
"""
import glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('CAMPAIGN_Z_DIR', os.path.join(HERE, '..', 'gain_recipe_contrasts_two_blocks')))
import analyze_Z as AZ

X_C, A_RISE = 3.9, 2300.0
WORKS = 4


def cell_of(rid):
    if rid.startswith('ZP_'): return rid.rsplit('_s', 1)[0][3:]
    return AZ.cell_of(rid)


def collect(zpdir, zdir=None, ydir=None):
    cells = {}
    for root, camp, pat in ((zpdir, 'ZP', 'ZP_*'), (zdir, 'Z', 'Z_P_*'), (zdir, 'Z', 'Z_R_frac0.3_*'), (ydir, 'Y', 'Y_G2_*')):
        if not root: continue
        for fn in sorted(glob.glob(os.path.join(root, pat, 'run.json.gz'))):
            rid = os.path.basename(os.path.dirname(fn))
            cells.setdefault(cell_of(rid), []).append(dict(AZ.per_run(fn), id=rid, campaign=camp))
    return cells


def predicted(x):
    return A_RISE / (x - X_C) if x > X_C else None


def timing(cell, R):
    med = lambda v: float(np.median(v)) if len(v) else None
    r0 = R[0]; p = r0['p']; x = r0['n_train'] / (p * math.log(p))
    a = [r['rise']['a'] for r in R if r.get('rise')]
    ratio = [r['rise']['a'] / r['ce_min']['t'] for r in R if r.get('rise') and r['ce_min']['t'] > 0]
    fin = [r['all']['final'] for r in R if r['all']['final'] is not None]
    return dict(cell=cell, campaign=r0['campaign'], p=p, frac=r0['frac'], lam_v=r0['lam_v'], n_train=r0['n_train'], x=round(x, 3),
                predicted_rise_start=predicted(x), rise_start=med(a), n_rise=len(a), n=len(R),
                ce_dip=med([r['ce_min']['t'] for r in R]), ratio=med(ratio), ratio_range=[min(ratio), max(ratio)] if ratio else None,
                pass_strict=sum(r['pass_strict'] for r in R), pass_loose=sum(r['pass_loose'] for r in R),
                final_all=[min(fin), max(fin)] if fin else None, steps=med([r['steps'] for r in R]),
                why={k: v for k, v in AZ.summarize(R)['why'].items() if v})


def appendix(T):
    out = {}
    for p in sorted({t['p'] for t in T}):
        C = [t for t in T if t['p'] == p and t['n'] >= 5]
        if not C: continue
        best = sorted(C, key=lambda t: (-t['pass_strict'], -t['pass_loose'], t['frac']))[0]
        out[str(p)] = dict(cell=best['cell'], campaign=best['campaign'], frac=best['frac'], pass_strict=best['pass_strict'],
                           pass_loose=best['pass_loose'], n=best['n'], works=best['pass_strict'] >= WORKS)
    return out


def main(outp, zpdir, zdir=None, ydir=None):
    AZ.collect = collect
    AZ.main(outp, zpdir, zdir, ydir)
    res = json.load(open(outp)); res['doc'] = __doc__
    for k, v in res['cells'].items(): v['campaign'] = v['runs'][0]['campaign']
    T = [timing(k, v['runs']) for k, v in res['cells'].items()]
    T.sort(key=lambda t: (t['p'], t['frac']))
    res['timing'] = T; res['appendix'] = appendix(T)
    json.dump(res, open(outp, 'w'), indent=1, default=float)
    f = lambda v, nd=0: '-' if v is None else f'{v:.{nd}f}'
    print('\n== timing (PLAN.md section 2) and pass counts')
    print(f"{'cell':24s} {'camp':4s} {'n_tr':>5s} {'x':>5s} {'pred.':>6s} {'rise':>6s} {'dip':>5s} {'ratio':>5s}  {'strict':>6s} {'loose':>5s}  final (all)      why")
    for t in T:
        print(f"{t['cell']:24s} {t['campaign']:4s} {t['n_train']:5d} {t['x']:5.2f} {f(t['predicted_rise_start']):>6s} "
              f"{f(t['rise_start']):>6s} {f(t['ce_dip']):>5s} {f(t['ratio'], 2):>5s}  {t['pass_strict']:>2d}/{t['n']:<3d} {t['pass_loose']:>2d}/{t['n']:<2d}  "
              f"{t['final_all']}  {t['why']}" + (f"  (rise in {t['n_rise']}/{t['n']})" if t['n_rise'] < t['n'] else ''))
    print('\n== appendix choice (most strict passes; ties: more loose passes, then the smaller frac)')
    for p, a in res['appendix'].items():
        print(f"p = {p}: {a['cell']} [{a['campaign']}] strict {a['pass_strict']}/{a['n']}, loose {a['pass_loose']}/{a['n']} -> "
              + ('works' if a['works'] else f'does not reach {WORKS}/5'))


if __name__ == '__main__':
    main(*sys.argv[1:5])
