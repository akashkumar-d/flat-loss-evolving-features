#!/usr/bin/env python3
"""The paper's mirror-pair and readout figures (campaign ZM's plot_ZM.py and plot_readouts_ZM.py, unchanged) with 10 seeds per
cell (akashkumar-d, 2026-10-13): seeds 11-15 of campaigns ZP, ZM and Y with seeds 16-20 of campaign Q. Writes an analysis
file in campaign ZM's format (analyze_ZM.extra adds each run's a0 and bound) and the readouts file of readouts_ZM.py for these
cells, then draws both figures.

    python3 -I build_zm10.py ANALYSIS_Q.json OUT_DIR
"""
import json, os, sys
import numpy as np
ZM = os.path.join(os.environ.get('SHARED', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'transformer')), 'one_block_p61_mirror_free')
sys.path.insert(0, ZM)
import analyze_ZM as AZM
import readouts_ZM as RZ
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
import plot_ZM10 as PZM
import plot_readouts_ZM as PR

AQ, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
Q = json.load(open(AQ))['cells']
KEYS = {'ZM_f0.55': 'P_p61_mf', 'ZP_random_f0.33': 'P_p61', 'P_p89_f0.25_lam_v0.33': 'P_p89', 'Y_G2': 'P_p113',
        'P_p149_f0.16_lam_v0.23': 'P_p149'}
cells = {}
for k, q in KEYS.items():
    runs = [AZM.extra(dict(r)) for r in Q[q]['runs']]
    cells[k] = dict(runs=runs, summary=AZM.AZ.summarize(runs))
ana = os.path.join(OUT, 'analysis_ZM10.json')
json.dump(dict(doc=__doc__, cells=cells), open(ana, 'w'), indent=1, default=float)
_load = json.load
def load(f, *a, **k):
    name = getattr(f, 'name', '')
    if name.endswith('analysis_ZM.json') or name.endswith('analysis_ZP.json'): return _load(open(ana))
    return _load(f, *a, **k)
RZ.json.load = load
ro = os.path.join(OUT, 'readouts_ZM10.json'); RZ.main(ro); RZ.json.load = _load
PZM.main(ana, OUT, 'ZM_f0.55')
PR.main(ana, ro, OUT)
for k in KEYS: print(k, cells[k]['summary']['n'], 'strict', cells[k]['summary']['pass_strict'], 'loose', cells[k]['summary']['pass_loose'])
