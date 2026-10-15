#!/usr/bin/env python3
"""The refit's lags quoted in revision 7 (claude/1hl-figures, 2026-10-14): how far the refit (all frequencies) trails the
network at the network's held-out 0.5 crossing, in the transformer figures and in every transformer run on record.

    python3 -I tools/rev7_refit_lags.py ANALYSIS_Z.json ANALYSIS_Q.json [ANALYSIS_ZP.json ANALYSIS_ZM.json] > OUT.json
Leads are campaign Z's analysis (network crossing minus readout crossing; negative = the refit trails); nothing is rescored.
"""
import json, sys
import numpy as np

Z = json.load(open(sys.argv[1]))['cells']; Q = json.load(open(sys.argv[2]))['cells']
rest = [json.load(open(f))['cells'] for f in sys.argv[3:]]


def lag(cell, d='all', q='0.5'):
    L = [r[d]['readouts']['paper_refit']['lead'][q] for r in cell['runs'] if 'paper_refit' in r[d]['readouts']]
    L = [x for x in L if x is not None]
    return dict(n=len(L), n_lead=int(sum(x > 0 for x in L)), median=float(np.median(L)) if L else None)


out = dict(doc=__doc__, figure5={}, across_primes={}, two_blocks={}, all_runs={})
for k, c in (('Polar', Z['Y_G2']), ('NGD', Z['C_NGD_G2']), ('AdamW', Z['C_ADAMW_G2'])):
    out['figure5'][k] = {d: lag(c, d) for d in ('all', 'nm')}
for k in ('P_p61_mf', 'P_p89', 'P_p113', 'P_p149'):
    out['across_primes'][k] = {d: lag(Q[k], d) for d in ('all', 'nm')}
out['two_blocks']['B_base'] = {d: lag(Q['B_base'], d) for d in ('all', 'nm')}
seen, n, lead = {}, 0, []
for cells in [Z, Q] + rest:
    for k, c in cells.items():
        for r in c.get('runs', []):
            rid = r['file'].rstrip('/').split('/')[-2]
            if rid in seen: continue
            seen[rid] = 1
            for d in ('all', 'nm'):
                x = r[d]['readouts'].get('paper_refit', {}).get('lead', {}).get('0.5')
                if x is None: continue
                n += 1
                if x > 0: lead.append(dict(run=rid, definition=d, lead=float(x)))
out['all_runs'] = dict(runs=len(seen), crossings=n, refit_leads=lead)
print(json.dumps(out, indent=1))
