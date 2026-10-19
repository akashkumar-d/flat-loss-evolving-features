#!/usr/bin/env python3
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'gain_recipe_contrasts_two_blocks'))
import plot_Z as PZ
import plot_X as PX
import analyze_X as AX
from readouts_ZM import CELLS

THR = 0.05


def at(H, t):
    o = min(H, key=lambda o: abs(o['step'] - t))
    r2 = np.array(o['fspec_hor2']); rank = np.argsort(-np.array(o['fspec_train']))
    sig = set(np.where(r2 > THR)[0].tolist())
    return dict(step=o['step'], K=len(r2), n_sig=len(sig), in_top8=len(sig & set(rank[:8].tolist())),
                in_top16=len(sig & set(rank[:16].tolist())), ranks=sorted(int(np.where(rank == k)[0][0]) + 1 for k in sig))


def main(out):
    AZM = json.load(open(os.path.join(HERE, 'analysis', 'analysis_ZM.json')))['cells']
    AZP = json.load(open(os.path.join(HERE, '..', 'one_block_other_primes', 'analysis', 'analysis_ZP.json')))['cells']
    res = {}
    for label, src, key in CELLS:
        cell = (AZM if src == 'ZM' else AZP)[key]; base = HERE if src == 'ZM' else os.path.join(HERE, '..', 'campaign_ZP')
        rows = []
        for r in cell['runs']:
            fn = r['file'] if os.path.isabs(r['file']) else os.path.join(base, r['file'])
            obs = PX.load(fn)['obs']; s, v = PX.series(obs, 'test_acc_nm')
            H = [o for o in obs if 'fspec_hor2' in o and 'fspec_train' in o]
            rows.append({name: at(H, t) for name, t in (('net 0.5', AX.interp_cross(s, v, 0.5)), ('net 0.9', AX.interp_cross(s, v, 0.9)),
                                                        ('end', obs[-1]['step']))})
        res[label] = rows
        print(f"\n== {label}: K = {rows[0]['end']['K']} frequencies, {len(rows)} runs; frequencies with held-out R^2 > {THR} (per run)")
        for name in ('net 0.5', 'net 0.9', 'end'):
            g = lambda f: [x[name][f] for x in rows]
            print(f"   at {name:7s}: {g('n_sig')}; of them in the top 8 by training energy: {g('in_top8')}; in the top 16: {g('in_top16')}")
        print(f"   training-energy ranks of those frequencies at net 0.9: {[x['net 0.9']['ranks'] for x in rows]}")
    json.dump(res, open(out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main(sys.argv[1])
