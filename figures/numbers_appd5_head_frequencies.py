#!/usr/bin/env python3
"""Appendix D.5 of the paper (version 2): how many label frequencies the transformer's head input carries at the network's
held-out 0.5 crossing, and the held-out R^2 of the frequencies outside the top 8, per setting of Table 8 and for two blocks.
Written for the code release; it computes these numbers from the run records with the rules of
campaigns/readouts/refit_lag/scan_runs.py, which counted them for the paper:
  - the network's crossing: first no-mirror held-out accuracy >= 0.5, interpolated on its 20-update log (scan_runs.cross);
  - the heavy log nearest that crossing; per frequency k there, fspec_hor2 is the held-out R^2 of the refit's prediction of
    e^{i w_k y} (a diagnostic that uses held-out labels); carried: R^2 > 0.05;
  - outside the top 8: the frequencies not in ftop8 (the 8 with the most training energy, the Fourier refit's choice);
    per run their mean R^2; per setting the median over runs.
For the settings it lists, refit_lag/paper_cells_freq.json gives the same medians of the R^2 outside the top 8.

    python3 -I readout_freqs.py ANALYSIS_Q.json OUT.json      (run files: the 'file' fields of ANALYSIS_Q.json)
"""
import gzip, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'transformer', 'readout_refit_lag'))
import scan_runs as SR

CELLS = (('p = 61, mirror-free, fraction 0.55', 'P_p61_mf'), ('p = 61, random split, fraction 0.33', 'P_p61'),
         ('p = 89', 'P_p89'), ('p = 113', 'P_p113'), ('p = 149', 'P_p149'), ('two blocks, p = 113', 'B_base'))


def one(fn):
    obs = json.load(gzip.open(fn, 'rt'))['obs']
    ts = [o['step'] for o in obs if 'test_acc_nm' in o]; vs = [o['test_acc_nm'] for o in obs if 'test_acc_nm' in o]
    t5 = SR.cross(ts, vs, 0.5)
    o = min((o for o in obs if 'fspec_hor2' in o), key=lambda o: abs(o['step'] - t5))
    r2 = np.array(o['fspec_hor2'], float); top = np.isin(np.arange(1, len(r2) + 1), o['ftop8'])
    return dict(t_net=t5, step=o['step'], K=len(r2), carried=int((r2 > 0.05).sum()),
                carried_outside_top8=int(((r2 > 0.05) & ~top).sum()), r2_outside_top8=float(r2[~top].mean()))


def main(aq, out):
    A = json.load(open(aq))['cells']; res = {}
    for label, key in CELLS:
        R = [one(r['file']) for r in A[key]['runs']]; c = [r['carried'] for r in R]
        res[label] = dict(runs=len(R), K=R[0]['K'], carried_min=min(c), carried_median=float(np.median(c)), carried_max=max(c),
                          carried_outside_top8_median=float(np.median([r['carried_outside_top8'] for r in R])),
                          r2_outside_top8_median=float(np.median([r['r2_outside_top8'] for r in R])), per_run=R)
        s = res[label]
        print(f"{label:38s} {s['runs']} runs, K = {s['K']}: carried {s['carried_min']}-{s['carried_max']} (median "
              f"{s['carried_median']:g}), outside the top 8 {s['carried_outside_top8_median']:g}; held-out R^2 outside the "
              f"top 8, median {s['r2_outside_top8_median']:.2f}")
    json.dump(dict(doc=__doc__, cells=res), open(out, 'w'), indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:3])
