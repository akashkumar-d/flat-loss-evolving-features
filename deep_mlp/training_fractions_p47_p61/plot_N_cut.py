#!/usr/bin/env python3
import json, math, os, sys
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import acrossp_figs as A
import plot_N as P


def column2_cut(runs, title, ids):
    end = min(int(r['ce'][0][-1]) for r in runs)
    t90 = [r['t90'] or r['W'][1] for r in runs]
    xmax = int(math.ceil((1.15 * max(np.median([r['W'][1] for r in runs]), np.percentile(t90, 90)) + 1500) / 1000) * 1000)
    xmax = min(xmax, end)
    assert max(t90) <= xmax, (title, max(t90), xmax)
    g = np.arange(0, xmax + 1, max(10, xmax // 2000), dtype=float)
    on = A.on_grid
    S = dict(ce=np.array([on(*r['ce'], g) for r in runs]),
             test=np.array([on(*r['test'], g) for r in runs]), train=np.array([on(*r['train'], g) for r in runs]),
             err=np.array([on(*r['err'], g) for r in runs]), refit=np.array([on(*r['refit'], g) for r in runs]),
             ahL=np.array([on(*r['ahL'], g) for r in runs]),
             ah1=np.array([on(*r['ah1'], g) for r in runs]) if runs[0]['ah1'] is not None else None)
    band = (float(np.median([r['W'][0] for r in runs])), float(np.median([r['W'][1] for r in runs])))
    print(f'  {title.splitlines()[0]}: x-axis to {xmax} (shortest run {end})', flush=True)
    return dict(g=g, S=S, band=(band[0], min(band[1], xmax)), xmax=xmax, title=title, stats={}, ntr=max(r['ntr'] for r in runs))


def main(nres, ana, outdir):
    out = Path(outdir) / 'ladder_allseeds'; out.mkdir(parents=True, exist_ok=True); R = json.load(open(ana))
    A.column2 = column2_cut
    for D, p in P.GROUPS:
        cols = []
        for f in P.FRACS:
            c = R['cells'][f'D{D}_p{p}_f{f:g}']
            fin = [x for x in R['runs'].values() if x['depth'] == D and x['p'] == p and x['dial']['frac'] == f and x['state'] == 'finished']
            cols.append(P.column(nres, sorted(fin, key=lambda x: x['seed']), f"fraction {f:g}, r = {c['r']:g}\n{c['pass_M']}/{c['n']} pass_M"))
        P.draw2(cols, out, f'N_ladder_D{D}_p{p}')


if __name__ == '__main__':
    main(*sys.argv[1:4])
