#!/usr/bin/env python3
"""Campaign QD analysis, fixed in PLAN.md section 4 (akashkumar-d, 2026-10-13): campaign C's per-run primary endpoint
(endpoints_i.fields, campaign I's implementation, unchanged) on the 40 depth-2 runs, bias-free (QD_D2) and with a hidden bias
(QD_B_D2), with Wilson intervals and the paired comparison (McNemar on flips).

    python3 -I analyze_QD.py RESULTS_DIR OUT.json
"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import endpoints_i as E


def main(rdir, outp):
    R = {}
    for fn in sorted(glob.glob(os.path.join(rdir, 'QD_*', 'run.json.gz'))):
        rid = os.path.basename(os.path.dirname(fn)); f = E.fields(E.load(fn)); f.pop('primary_checks', None); R[rid] = f
    out = dict(doc=__doc__, runs=R, summary={})
    for fam in ('D2', 'B_D2'):
        ids = [f'QD_{fam}_s{i:03d}' for i in range(1, 41)]; k = sum(R[i]['primary'] for i in ids)
        why = {}
        for i in ids:
            for w in (R[i]['why_not'] or '').split(', '):
                if w: why[w] = why.get(w, 0) + 1
        med = lambda key: float(np.median([R[i][key] for i in ids if R[i].get(key) is not None]))
        out['summary'][fam] = dict(n=len(ids), primary=k, wilson=E.wilson(k, len(ids)), why=why,
                                   final_min=min(R[i]['final_test'] for i in ids), t10_median=med('t10'), t90_median=med('t90'),
                                   share_median=med('share'), CE_peak_over_logp_median=med('CE_peak_over_logp'))
    b = sum(R[f'QD_D2_s{i:03d}']['primary'] and not R[f'QD_B_D2_s{i:03d}']['primary'] for i in range(1, 41))
    c = sum(R[f'QD_B_D2_s{i:03d}']['primary'] and not R[f'QD_D2_s{i:03d}']['primary'] for i in range(1, 41))
    out['summary']['paired'] = dict(lost_with_bias=b, gained_with_bias=c, mcnemar_p=E.mcnemar(b, c))
    json.dump(out, open(outp, 'w'), indent=1, default=float)
    print(json.dumps(out['summary'], indent=1, default=float))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
