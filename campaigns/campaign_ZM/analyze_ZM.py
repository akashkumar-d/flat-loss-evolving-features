#!/usr/bin/env python3
"""Campaign ZM analysis, fixed in PLAN.md section 3 (claude/1hl-figures, 2026-10-12).

Every run is scored by campaign Z's per-run analysis (analyze_Z.per_run and summarize, unchanged); its split count is taught
the mirror-free split of tf_runner 1.7 (for split_mode 'random' it is analyze_Z's own). Cells: the ZM runs (mirror-free split)
and, for comparison, campaign ZP's p = 61 random-split cells at fractions 0.33 and 0.35 (same seeds 11-15). Added per run:
a0 = the all-pairs held-out accuracy when the all-pairs window opens, and the bound (0.9 - max(a0, 0.1)) / 0.8.

Usage (from campaign_ZM):  python3 -I analyze_ZM.py OUT.json ZM_RUNS_DIR ZP_RESULTS_DIR
"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('CAMPAIGN_Z_DIR', os.path.join(HERE, '..', 'campaign_Z')))
import analyze_Z as AZ
sys.path.insert(0, HERE)
import tf_runner as R

_split_counts_random = AZ.split_counts


def split_counts(c):
    if c.get('split_mode', 'random') == 'random': return _split_counts_random(c)
    assert c['split_mode'] == 'mirror_free'
    p = c['p']; tr = R.mirror_free_train(p, c['frac'], c['split_seed'])
    intr = np.zeros(p * p, bool); intr[tr] = True; te = np.where(~intr)[0]
    a, b = te // p, te % p
    return len(tr), len(te), int((~intr[b * p + a]).sum())


AZ.split_counts = split_counts


def cell_of(rid):
    head, _ = rid.rsplit('_s', 1)
    return head


def extra(rec):
    d = AZ.AX.load(rec['file']); s, va = AZ.AX.series(d['obs'], 'test_acc'); W = rec['all'].get('window', {}).get('W')
    if W:
        rec['a0_all'] = float(va[int(np.argmin(np.abs(s - W[0])))])
        rec['bound_all'] = min(1.0, (0.9 - max(rec['a0_all'], 0.1)) / 0.8)
    rec['split_mode'] = (d.get('cfg_full') or d['cfg']).get('split_mode', 'random')
    return rec


def main(out, zm_dir, zp_dir):
    cells = {}
    for fn in sorted(glob.glob(os.path.join(zm_dir, '*', 'run.json.gz'))):
        rid = os.path.basename(os.path.dirname(fn)); cells.setdefault(cell_of(rid), []).append(dict(extra(AZ.per_run(fn)), id=rid))
    for f in ('0.33', '0.35'):
        for fn in sorted(glob.glob(os.path.join(zp_dir, f'ZP_P_p61_f{f}_lam_v0.45_s*', 'run.json.gz'))):
            rid = os.path.basename(os.path.dirname(fn))
            cells.setdefault(f'ZP_random_f{f}', []).append(dict(extra(AZ.per_run(fn)), id=rid))
    res = dict(doc=__doc__, cells={k: dict(runs=v, summary=AZ.summarize(v)) for k, v in cells.items()})
    json.dump(res, open(out, 'w'), indent=1, default=float)
    print(f"{'run':34s} {'split':11s} {'n_tr':>5s} {'n_m':>4s} {'fit':>5s} {'dip':>5s} {'rise0.1':>7s} {'0.5':>6s} {'0.9':>6s} "
          f"{'W':>13s} {'a0':>6s} {'q_all':>6s} {'bound':>6s} {'mis':>4s} {'final':>6s} {'CE chg':>7s}  pass  why")
    for k, v in res['cells'].items():
        for r in sorted(v['runs'], key=lambda r: r['id']):
            w = r['all'].get('window', {}); n = r['all']['net']; f_ = lambda x: '-' if x is None else f'{x:.0f}'
            print(f"{r['id']:34s} {r['split_mode']:11s} {r['n_train']:5d} {r['n_mirror']:4d} {f_(r['first_fit']):>5s} {r['ce_min']['t']:5d} "
                  f"{f_(n['0.1']):>7s} {f_(n['0.5']):>6s} {f_(n['0.9']):>6s} {str(w.get('W')):>13s} "
                  f"{r.get('a0_all', float('nan')):6.3f} {w.get('q_te') if w.get('q_te') is None else round(w['q_te'], 3)!s:>6s} "
                  f"{r.get('bound_all', float('nan')):6.3f} {str(w.get('max_mis')):>4s} {r['all']['final']:6.3f} "
                  f"{(100 * r['rise']['ce_change']) if 'rise' in r else float('nan'):+6.0f}%  {'S' if r['pass_strict'] else ('L' if r['pass_loose'] else '-')}     {r['why']}")
        S = v['summary']
        print(f"== {k}: {S['n']} runs | pass strict {S['pass_strict']}/{S['n']}, loose {S['pass_loose']}/{S['n']} | why {S['why']}\n")


if __name__ == '__main__':
    main(*sys.argv[1:4])
