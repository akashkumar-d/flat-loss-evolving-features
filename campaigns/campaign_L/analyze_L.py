#!/usr/bin/env python3
"""Campaign L: score the finished runs as campaign K was scored, and pair each with campaign K's run of the same setting and
seed (claude/1hl-figures, 2026-10-09; PLAN.md section "Analysis").

    python3 -I analyze_L.py L_RESULTS K_RESULTS PAPER_RUNS STATUS_DIR OUT.json

L_RESULTS / K_RESULTS hold <run>/run.json.gz (collected/results of campaigns L and K); PAPER_RUNS holds the unpacked campaign
D, F and G runs (the paper's setting at training fraction 0.8, 10 seeds per cell, seeds 5000 + i / 6000 + i, so not paired);
STATUS_DIR holds the session status files (collected/status), used only to label unfinished runs as running or not started.

Per run:
  endpoint   campaign C's per-run endpoint (endpoints_i.fields, unchanged): knee, loss-only band W, share of the held-out
             10% -> 90% rise inside W, final held-out accuracy, Delta A_H of the last hidden layer across W, training margins.
  refit      Akash's refit criterion as in campaign_K/plot_K_refit.py (score() copied unchanged): W = the paper's post-hoc
             window (acrossp_figs.window_record); pass = refit +0.2 inside W, held-out +0.2 from the start of W to the end of the
             run, final held-out >= 0.99; pass3 = pass with training accuracy >= 0.97 at every logged step in W.
  timing     t10, t90 (first persistent) of the held-out rise; the CE plateau level (CE peak in W / log p); the refit's lead
             (update at which the held-out accuracy first reaches 0.5 minus the same for the refit).
"""
import glob, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import acrossp_figs as A
import endpoints_i as E

GAIN, FINAL, TRAIN3 = 0.2, 0.99, 0.97
PRIMES = (23, 47, 61, 101)


def at(s, v, t): return float(v[int(np.argmin(np.abs(s - t)))])


def first(s, v, thr):
    i = np.where(v >= thr)[0]; return int(s[i[0]]) if len(i) else None


def refit_score(d):
    w = A.window_record(d); a, b = w['W2']
    s, te = E.series(d['obs'], 'test_acc'); st, tr = E.series(d['obs'], 'train_acc')
    sr, rf = E.series(d['obs'], 'v1_ridge_acc_te')
    k = (st >= a) & (st <= b); minacc = float(tr[k].min()) if k.any() else float('nan')
    final = float(te[-1]); rise = final - at(s, te, a)
    ok = bool(w['g_rf'] >= GAIN and rise >= GAIN and final >= FINAL)
    return dict(W=[int(a), int(b)], g_rf=w['g_rf'], g_te=w['g_te'], heldout_rise_from_a=rise, final=final, min_train_acc_in_W=minacc,
                ok=ok, ok3=bool(ok and minacc >= TRAIN3), t_refit50=first(sr, rf, 0.5), t_heldout50=first(s, te, 0.5),
                why=[n for n, c in (('refit', w['g_rf'] >= GAIN), ('held-out', rise >= GAIN), ('final', final >= FINAL)) if not c])


def relapses_in_band(d, W):
    s, m = E.series(d['obs'], 'train_margin_min')
    return int(np.sum(m[(s >= W[0]) & (s <= W[1])] <= 0)) if W else None


def score_file(fn):
    d = E.load(fn); f = E.fields(d); f['relapses_in_band'] = relapses_in_band(d, f.get('W'))
    x = refit_score(d); c = d['cfg']
    keep = ('primary', 'why_not', 'share', 'W', 'CE_peak_over_logp', 't10', 't90', 't90_first', 'final_test', 'final_train',
            'dAH_last', 'knee', 'horizon', 'relapses_in_band', 'fit_strict')
    return dict(endpoint={k: f.get(k) for k in keep}, refit=x, frac=c.get('frac'), steps=len(d['loss']) - 1, cfg_steps=c.get('steps'),
                stop_reason=d.get('stop_reason'))


def status(st_dir):
    run, left = {}, set()
    for fn in glob.glob(os.path.join(st_dir, '*', '*.json')):
        if 'selftest' in fn: continue
        s = json.load(open(fn))
        for x in s.get('running', []): run[x['id']] = x.get('step')
        left |= set(s.get('own_left', []))
    return run, left


def paper_runs(root):
    out = {}
    for fn in sorted(glob.glob(os.path.join(root, '**', 'run.json.gz'), recursive=True)):
        d = E.load(fn); c = d['cfg']
        if c.get('lam_v') != 0.2 or c['p'] not in PRIMES or c['depth'] not in (3, 4): continue
        r = round(c['eta_layers'][-1] / (0.03 * 0.015 / 0.07), 2); want = 0.32 if (c['depth'] == 4 and c['p'] == 23) else 0.45
        if abs(r - want) > 1e-6: continue
        out.setdefault(f"D{c['depth']}_p{c['p']}", []).append(fn)
    return out


def main(lres, kres, paper, st_dir, out):
    man = json.load(open(os.path.join(HERE, 'manifest.json')))
    running, left = status(st_dir)
    rows = {}
    for r in man['runs']:
        rid = r['id']; fn = os.path.join(lres, rid, 'run.json.gz')
        row = dict(id=rid, setting=r['setting'], p=r['p'], depth=r['depth'], seed=r['seed'], dial=r['dial'],
                   state='finished' if os.path.exists(fn) else ('running' if rid in running else 'not started'),
                   step=running.get(rid))
        if row['state'] == 'finished': row['L'] = score_file(fn)
        kid = 'K_' + rid[2:]; kfn = os.path.join(kres, kid, 'run.json.gz')
        if row['state'] == 'finished' and r['dial']['lam_v_rule'] != 'paper' and os.path.exists(kfn): row['K'] = score_file(kfn); row['K_id'] = kid
        rows[rid] = row
        if row['state'] == 'finished':
            e = row['L']['endpoint']; x = row['L']['refit']
            ke = row.get('K', {}).get('endpoint', {})
            print(f"{rid:34s} endpoint {'PASS' if e['primary'] else 'fail (' + str(e['why_not']) + ')':28s} refit {'PASS' if x['ok'] else 'fail ' + ','.join(x['why'])} "
                  f"{'(pass3)' if x['ok3'] else ''} | t10 {e['t10']} t90 {e['t90']} final {e['final_test']:.3f} CE/logp {e.get('CE_peak_over_logp') or float('nan'):.3f}"
                  + (f" || K: t10 {ke.get('t10')} t90 {ke.get('t90')} final {ke.get('final_test'):.3f} endpoint {'PASS' if ke.get('primary') else 'fail'}" if ke else ''), flush=True)
    pap = {}
    for cell, fns in sorted(paper_runs(paper).items()):
        pap[cell] = [dict(id=os.path.basename(os.path.dirname(fn)), **score_file(fn)) for fn in fns]
    json.dump(dict(doc=__doc__, runs=rows, paper_frac08=pap), open(out, 'w'), indent=1, default=float)
    n = {s: sum(1 for r in rows.values() if r['state'] == s) for s in ('finished', 'running', 'not started')}
    print('states:', n)


if __name__ == '__main__':
    main(*sys.argv[1:6])
