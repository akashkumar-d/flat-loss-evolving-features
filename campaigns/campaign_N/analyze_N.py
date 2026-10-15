#!/usr/bin/env python3
"""Campaign N analysis, fixed in PLAN.md section 5 before any run (claude/1hl-figures, 2026-10-12).

    python3 -I analyze_N.py N_RESULTS STATUS_DIR OUT.json [M_ANALYSIS_JSON]

N_RESULTS holds <run>/run.json.gz (campaign_N/collected/results); STATUS_DIR holds the session status files
(collected/status), used only to label unfinished runs; M_ANALYSIS_JSON (campaign_M/analysis/analysis_M.json, optional) gives
campaign M's development counts for the same cells.

Each run is scored by campaign M's scorer, unchanged (analyze_M.py here is a byte-identical copy of campaign M's):
  pass_M    Akash's refit criterion (refit +0.2 inside the paper's post-hoc window W, held-out +0.2 from the start of W to the
            end of the run, final held-out >= 0.99), and at least 75% of the held-out and of the refit 10% -> 90% rises inside
            W (the paper's clip formula), and Delta A_H of the last hidden layer >= 0.02 across W;
  pass_M3   pass_M with training accuracy >= 0.97 at every logged step of W.
Campaign C's per-run endpoint is reported, not deciding. Held-out accuracy is on all pairs outside the training set.

Per cell (depth, p, fraction; seeds 1-10): pass_M and pass_M3 counts with Wilson 95% intervals; the cell 'works' if pass_M
holds in at least 8 of 10 seeds. Descriptive: the shares q_te and q_rf, Delta A_H, the CE change from the refit's 0.2 crossing
to the held-out 0.9 crossing (and in how many runs it rose), the training CE / log p at the held-out 0.5 crossing (the level
the rise happens at), the CE's drawdown below its running maximum at held-out 0.9, the
most training pairs misclassified in W, timing (held-out 0.5 and 0.9, refit 0.5, refit lead = held-out 0.5 minus refit 0.5),
final held-out accuracy and run length.
"""
import glob, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import analyze_M as AM

WORKS = 8


def wilson(k, n, z=1.96):
    if not n: return None
    ph = k / n; den = 1 + z * z / n; c = (ph + z * z / (2 * n)) / den; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def med(x):
    x = [v for v in x if v is not None]; return float(np.median(x)) if x else None


def rng(x):
    x = [v for v in x if v is not None]; return [float(min(x)), float(max(x))] if x else None


def cell_summary(rows):
    fin = [r for r in rows if r['state'] == 'finished']; S = [r['M'] for r in fin]; n = len(S)
    c50 = [r.get('ce_te50') for r in fin]
    k = sum(s['verdict']['pass_M'] for s in S); k3 = sum(s['verdict']['pass_M3'] for s in S)
    T = [s['timing'] for s in S]; W = [s['window'] for s in S]
    ce = [t.get('ce_change_rf20_te90') for t in T]
    lead = [(t['te50'] - t['rf50']) if t.get('te50') is not None and t.get('rf50') is not None else None for t in T]
    why = {}
    for s in S:
        for w in s['verdict']['why']: why[w] = why.get(w, 0) + 1
    return dict(n=n, of=len(rows), pass_M=k, pass_M3=k3, wilson_M=wilson(k, n), wilson_M3=wilson(k3, n),
                works=bool(n == len(rows) and k >= WORKS), why=why, endpoint_C=sum(bool(s['endpoint'].get('primary')) for s in S),
                q_te=dict(median=med([w['q_te'] for w in W]), range=rng([w['q_te'] for w in W])),
                q_rf=dict(median=med([w['q_rf'] for w in W]), range=rng([w['q_rf'] for w in W])),
                dAH=dict(median=med([w['dAH'] for w in W]), range=rng([w['dAH'] for w in W])),
                max_mis_W=rng([w['max_mis'] for w in W]),
                ce_change=dict(median=med(ce), range=rng(ce), n=sum(c is not None for c in ce), n_rising=sum(c > 0 for c in ce if c is not None)),
                ce_at_te50=dict(median=med(c50), range=rng(c50)),
                drawdown_te90=dict(median=med([t.get('drawdown_te90') for t in T]), max=(rng([t.get('drawdown_te90') for t in T]) or [None, None])[1]),
                te50=med([t.get('te50') for t in T]), te90=med([t.get('te90') for t in T]), rf50=med([t.get('rf50') for t in T]),
                refit_lead=dict(median=med(lead), range=rng(lead)), t_peak=med([t.get('t_pk') for t in T]),
                final=rng([s['final'] for s in S]), steps=dict(median=med([s['steps'] for s in S]), range=rng([s['steps'] for s in S])))


def ce_at(fn, t):
    if t is None: return None
    d = AM.E.load(fn); L = np.asarray(d['loss'], float) / math.log(d['cfg']['p']); cm = (L[:-1] + L[1:]) / 2
    return float(cm[min(int(t), len(cm) - 1)])


def m_counts(mjson, D, p, f, r):
    if not mjson: return None
    sel = [x for x in mjson['runs'].values() if x['depth'] == D and x['p'] == p and x['dial']['lam_v_rule'] == 'lin'
           and x['dial']['frac'] == f and abs(x['dial']['r'] - r) < 1e-9 and x['state'] == 'finished']
    return dict(pass_M=sum(x['M']['verdict']['pass_M'] for x in sel), pass_M3=sum(x['M']['verdict']['pass_M3'] for x in sel), n=len(sel))


def main(nres, st_dir, out, mjson_fn=None):
    man = json.load(open(os.path.join(HERE, 'manifest.json')))
    running, left = AM.status(st_dir) if os.path.isdir(st_dir) else ({}, set())
    mjson = json.load(open(mjson_fn)) if mjson_fn and os.path.exists(mjson_fn) else None
    rows = {}
    for r in man['runs']:
        rid = r['id']; fn = os.path.join(nres, rid, 'run.json.gz')
        row = dict(id=rid, setting=r['setting'], p=r['p'], depth=r['depth'], seed=r['seed'], dial=r['dial'],
                   state='finished' if os.path.exists(fn) else ('running' if rid in running else 'not started'), step=running.get(rid))
        if row['state'] == 'finished':
            row['M'] = AM.score_file(fn); v = row['M']['verdict']; t = row['M']['timing']; w = row['M']['window']
            row['ce_te50'] = ce_at(fn, t.get('te50'))
            print(f"{rid:30s} {'PASS_M' if v['pass_M'] else 'fail (' + ', '.join(v['why']) + ')':44s} {'(pass_M3)' if v['pass_M3'] else '':9s} "
                  f"q_te {w['q_te']:.2f} q_rf {w['q_rf']:.2f} | pk {t['t_pk']} te50 {t['te50']} te90 {t['te90']} final {row['M']['final']:.3f}",
                  flush=True)
        rows[rid] = row
    cells = {}
    for D in (3, 4):
        for p in (47, 61):
            for f in (0.8, 0.7, 0.6, 0.5):
                R = [x for x in rows.values() if x['depth'] == D and x['p'] == p and x['dial']['frac'] == f]
                r_ = R[0]['dial']['r']; key = f'D{D}_p{p}_f{f:g}'
                cells[key] = dict(depth=D, p=p, frac=f, r=r_, **cell_summary(R), M_dev=m_counts(mjson, D, p, f, r_))
    json.dump(dict(doc=__doc__, runs=rows, cells=cells), open(out, 'w'), indent=1, default=float)
    fmt = lambda v, nd=0: '-' if v is None else f'{v:.{nd}f}'
    print('\ncell            r      pass_M  (Wilson)       pass_M3  works | q_te med  CE change med (rising) at CE/log p   drawdown90 max | '
          'te50  te90  lead (k) | final        | M dev')
    for k, c in cells.items():
        ce = c['ce_change']; md = c['M_dev']
        ce_txt = '-' if ce['median'] is None else '%+.0f%%' % (100 * ce['median'])
        lvl = c['ce_at_te50']['range']; lvl_txt = '-' if lvl is None else '%.3f-%.3f' % tuple(lvl)
        k_lead = None if c['refit_lead']['median'] is None else c['refit_lead']['median'] / 1e3
        fin_txt = str([round(x, 3) for x in c['final']]) if c['final'] else '-'
        md_txt = '%d/%d' % (md['pass_M'], md['n']) if md else '-'
        tail = '  [%d/%d finished]' % (c['n'], c['of']) if c['n'] < c['of'] else ''
        works = 'yes' if c['works'] else 'no'
        print(f"{k:15s} {c['r']:<5g}  {c['pass_M']:2d}/{c['n']:<2d} {str(c['wilson_M']):15s} {c['pass_M3']:2d}/{c['n']:<2d}  {works:3s}  | "
              f"{fmt(c['q_te']['median'], 2):5s}  {ce_txt:>6s} ({ce['n_rising']}/{ce['n']}) at {lvl_txt:11s} {fmt(c['drawdown_te90']['max'], 3):6s} | "
              f"{fmt(c['te50'] and c['te50'] / 1e3, 1):5s} {fmt(c['te90'] and c['te90'] / 1e3, 1):5s} {fmt(k_lead, 1):5s} | {fin_txt:13s} | {md_txt}{tail}")
    n = {s_: sum(1 for x in rows.values() if x['state'] == s_) for s_ in ('finished', 'running', 'not started')}
    print('states:', n, '| cells that work (pass_M in >= 8/10):', sum(c['works'] for c in cells.values()), 'of', len(cells))


if __name__ == '__main__':
    main(*sys.argv[1:5])
