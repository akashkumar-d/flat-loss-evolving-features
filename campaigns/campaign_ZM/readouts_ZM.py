#!/usr/bin/env python3
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'campaign_Z'))
import plot_Z as PZ
import plot_X as PX
import analyze_X as AX

READOUTS = (('paper refit (all 30 or 56 freq.)', 'refit_test_acc'), ('Fourier refit top 1', 'fref_top1_test_acc'),
            ('Fourier refit top 2', 'fref_top2_test_acc'), ('Fourier refit top 4', 'fref_top4_test_acc'),
            ('Fourier refit top 8', 'fref_top8_test_acc'), ('Fourier refit top 16', 'fref_top16_test_acc'),
            ('phase vote top 8', 'fphase_top8_test_acc'), ('net logits top 4', 'fnet_top4_test_acc'),
            ('net logits top 8', 'fnet_top8_test_acc'), ('net logits top 16', 'fnet_top16_test_acc'),
            ('logistic probe', 'probe_test_acc'), ('ridge on MLP acts', 'refit_mlp_test_acc'),
            ('Fourier refit MLP top 8', 'fmref_top8_test_acc'),
            ('[oracle] Fourier refit', 'fref_oracle_test_acc'), ('[oracle] net logits', 'fnet_oracle_test_acc'))
CELLS = (('p61 mirror-free f0.55 (ZM)', 'ZM', 'ZM_f0.55'), ('p61 random f0.33 (ZP)', 'ZM', 'ZP_random_f0.33'),
         ('p89 random f0.25 (ZP)', 'ZP', 'P_p89_f0.25_lam_v0.33'), ('p113 random f0.2 (Y G2)', 'ZP', 'Y_G2'),
         ('p149 random f0.16 (ZP)', 'ZP', 'P_p149_f0.16_lam_v0.23'))


def cross(obs, key):
    s, v = PX.series(obs, key)
    if not len(s): return None
    return [AX.interp_cross(s, v, q) for q in (0.1, 0.5, 0.9)] + [float(v[-1])]


def main(out):
    AZM = json.load(open(os.path.join(HERE, 'analysis', 'analysis_ZM.json')))['cells']
    AZP = json.load(open(os.path.join(HERE, '..', 'campaign_ZP', 'analysis', 'analysis_ZP.json')))['cells']
    res = {}
    for label, src, key in CELLS:
        cell = (AZM if src == 'ZM' else AZP)[key]; base = HERE if src == 'ZM' else os.path.join(HERE, '..', 'campaign_ZP')
        rows = {}
        for r in cell['runs']:
            fn = r['file'] if os.path.isabs(r['file']) else os.path.join(base, r['file'])
            obs = PX.load(fn)['obs']; net = cross(obs, 'test_acc_nm')
            for name, k in READOUTS:
                c = cross(obs, k + '_nm')
                if c is None: continue
                rows.setdefault(name, []).append(dict(t=c[:3], final=c[3],
                                                      lead=[(net[i] - c[i]) if (net[i] is not None and c[i] is not None) else None for i in range(3)]))
            rows.setdefault('network', []).append(dict(t=net[:3], final=net[3], lead=[0, 0, 0]))
        res[label] = rows
    json.dump(res, open(out, 'w'), indent=1, default=float)
    med = lambda v: (float(np.median([x for x in v if x is not None])) if any(x is not None for x in v) else None)
    for label, rows in res.items():
        n = len(rows['network'])
        nt = '/'.join('%.0f' % med([r['t'][i] for r in rows['network']]) for i in range(3))
        print(f"\n== {label} ({n} runs): network no-mirror held-out 0.1/0.5/0.9 at {nt}")
        print(f"   {'readout':34s} {'lead@0.1':>8s} {'lead@0.5':>8s} {'lead@0.9':>8s} {'leads@0.5':>9s} {'leads@0.9':>9s} {'final':>6s}")
        for name, _ in READOUTS:
            if name not in rows: continue
            R = rows[name]; L = [[r['lead'][i] for r in R] for i in range(3)]
            f = lambda x: '   never' if x is None else f'{x:+8.0f}'
            n5 = sum(1 for x in L[1] if x is not None and x > 0); n9 = sum(1 for x in L[2] if x is not None and x > 0)
            print(f"   {name:34s} {f(med(L[0]))} {f(med(L[1]))} {f(med(L[2]))} {n5:>6d}/{len(R)} {n9:>6d}/{len(R)} {med([r['final'] for r in R]):6.3f}")


if __name__ == '__main__':
    main(sys.argv[1])
