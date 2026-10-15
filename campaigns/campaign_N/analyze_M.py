#!/usr/bin/env python3
"""Campaign M: score the finished runs (claude/1hl-figures, 2026-10-11; PLAN.md section 4, fixed before any run).

    python3 -I analyze_M.py M_RESULTS L_RESULTS K_RESULTS STATUS_DIR OUT.json

M_RESULTS, L_RESULTS, K_RESULTS hold <run>/run.json.gz (collected/results of campaigns M, L and K); STATUS_DIR holds campaign
M's session status files (collected/status), used only to label unfinished runs as running or not started. Campaign L's runs
with the same cell, lam_v and seed (r = 0.45, fraction 0.5, lam_w 0.015) are scored the same way as the r = 0.45 controls at
fraction 0.5; campaign K's (fraction 0.8, linear rule) are compared with M's own fraction-0.8 runs of the same configuration.

Per run:
  timing    knee (first log with ||V|| >= 0.99 / lam_v); the CE (cycle mean) dip and the post-knee CE peak (update, CE / log p);
            first crossings of the train-only ridge refit (0.2, 0.5, 0.9) and of the held-out accuracy (0.1, 0.5, 0.9; t90 also
            first persistent); the CE's drawdown below its running maximum since the knee at the held-out 0.5 and 0.9 crossings;
            the CE change from the refit's 0.2 crossing to the held-out 0.9 crossing and the most training pairs misclassified
            over that stretch.
  refit     Akash's refit criterion as in campaigns K and L (campaign_K/plot_K_refit.py score(), unchanged): W = the paper's
            post-hoc window (acrossp_figs.window_record); pass = refit +0.2 inside W, held-out +0.2 from the start of W to the
            end of the run, final held-out >= 0.99; pass3 = pass with training accuracy >= 0.97 at every logged step in W.
  window    the shares q_te and q_rf of the held-out and refit 10% -> 90% rises inside W (the paper's clip formula), Delta A_H
            of the last hidden layer across W, the least training accuracy and the strict fit (every logged margin > 0) in W.
  verdicts  pass_M = refit pass and q_te >= 0.75 and q_rf >= 0.75 and Delta A_H >= 0.02; pass_M3 = pass_M with training
            accuracy >= 0.97 at every logged step of W. Campaign C's per-run endpoint (endpoints_i.fields) is reported, not
            deciding.
"""
import glob, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import acrossp_figs as A
import endpoints_i as E

GAIN, FINAL, TRAIN3, SHARE, DAH = 0.2, 0.99, 0.97, 0.75, 0.02
ETA_UNIT = 0.03 * 0.015 / 0.07


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


def share(s, v, a, b):
    clip = lambda t: min(max(at(s, v, t), 0.1), 0.9)
    return (clip(b) - clip(a)) / 0.8


def score_file(fn):
    d = E.load(fn); c = d['cfg']; p = c['p']; D = c['depth']; obs = d['obs']; ntr = int(c['frac'] * p * p)
    L = np.asarray(d['loss'], float) / math.log(p); cm = (L[:-1] + L[1:]) / 2; T = len(cm)
    s, te = E.series(obs, 'test_acc'); st, tr = E.series(obs, 'train_acc'); sm, mm = E.series(obs, 'train_margin_min')
    sr, rf = E.series(obs, 'v1_ridge_acc_te'); sa, ah = E.series(obs, f'AH_full_h{D}')
    sn, nv = E.series(obs, 'normV'); ki = np.where(nv >= 0.99 / c['lam_v'])[0]; knee = int(sn[ki[0]]) if len(ki) else 0
    pk = knee + int(np.argmax(cm[knee:])); dip = int(np.argmin(cm[:pk + 1]))
    runmax = np.maximum.accumulate(cm[knee:])
    dd = lambda t: None if t is None else float(1 - cm[min(t, T - 1)] / runmax[min(t, T - 1) - knee]) if t >= knee else 0.0
    tm = dict(knee=knee, t_dip=dip, ce_dip=float(cm[dip]), t_pk=pk, ce_pk=float(cm[pk]), horizon=T,
              rf20=first(sr, rf, 0.2), rf50=first(sr, rf, 0.5), rf90=first(sr, rf, 0.9),
              te10=first(s, te, 0.1), te50=first(s, te, 0.5), te90=first(s, te, 0.9), t90_persistent=E.first_persistent(s, te, 0.9))
    tm['drawdown_te50'] = dd(tm['te50']); tm['drawdown_te90'] = dd(tm['te90'])
    a_, b_ = tm['rf20'], tm['te90']
    if a_ is not None and b_ is not None and b_ > a_:
        m = (st >= a_) & (st <= b_)
        tm['ce_change_rf20_te90'] = float(cm[min(b_, T - 1)] / cm[min(a_, T - 1)] - 1)
        tm['max_mis_rf20_te90'] = int(round(float(np.max(1 - tr[m])) * ntr)) if m.any() else None
    x = refit_score(d); a, b = x['W']
    mW = (st >= a) & (st <= b); mM = (sm >= a) & (sm <= b)
    win = dict(W=[a, b], q_te=share(s, te, a, b), q_rf=share(sr, rf, a, b), dAH=at(sa, ah, b) - at(sa, ah, a),
               min_train_acc=float(tr[mW].min()) if mW.any() else None,
               max_mis=int(round(float(np.max(1 - tr[mW])) * ntr)) if mW.any() else None,
               fit_strict=bool(mM.any() and np.all(mm[mM] > 0)), ce_W_mean=float(np.mean(cm[a:b + 1])))
    ok = bool(x['ok'] and win['q_te'] >= SHARE and win['q_rf'] >= SHARE and win['dAH'] >= DAH)
    why = [n for n, c_ in (('refit criterion', x['ok']), ('held-out share', win['q_te'] >= SHARE), ('refit share', win['q_rf'] >= SHARE),
                           ('A_H', win['dAH'] >= DAH)) if not c_]
    ver = dict(pass_M=ok, pass_M3=bool(ok and (win['min_train_acc'] or 0) >= TRAIN3), why=why)
    f = E.fields(d)
    endpoint = {k: f.get(k) for k in ('primary', 'why_not', 'share', 'W', 'CE_peak_over_logp', 't10', 't90', 'final_test', 'dAH_last', 'fit_strict')}
    return dict(timing=tm, refit=x, window=win, verdict=ver, endpoint=endpoint, final=float(te[-1]), final_refit=float(rf[-1]),
                steps=T, stop_reason=d.get('stop_reason'), frac=c['frac'], r=round(c['eta_layers'][-1] / ETA_UNIT, 4), lam_v=c['lam_v'])


def status(st_dir):
    run, left = {}, set()
    for fn in glob.glob(os.path.join(st_dir, '*', '*.json')):
        if 'selftest' in fn: continue
        s = json.load(open(fn))
        for x in s.get('running', []): run[x['id']] = x.get('step')
        left |= set(s.get('own_left', []))
    return run, left


def max_loss_diff(f1, f2):
    a = np.asarray(E.load(f1)['loss'], float); b = np.asarray(E.load(f2)['loss'], float); n = min(len(a), len(b))
    return float(np.max(np.abs(a[:n] - b[:n]))), n


def main(mres, lres, kres, st_dir, out):
    man = json.load(open(os.path.join(HERE, 'manifest.json')))
    running, left = status(st_dir) if os.path.isdir(st_dir) else ({}, set())
    rows, controls, kcheck = {}, {}, {}
    for r in man['runs']:
        rid = r['id']; fn = os.path.join(mres, rid, 'run.json.gz'); dl = r['dial']
        row = dict(id=rid, setting=r['setting'], p=r['p'], depth=r['depth'], seed=r['seed'], dial=dl,
                   state='finished' if os.path.exists(fn) else ('running' if rid in running else 'not started'), step=running.get(rid))
        if row['state'] == 'finished':
            row['M'] = score_file(fn); v = row['M']['verdict']; t = row['M']['timing']; w = row['M']['window']
            print(f"{rid:34s} {'PASS_M' if v['pass_M'] else 'fail (' + ', '.join(v['why']) + ')':44s} {'(pass_M3)' if v['pass_M3'] else '':9s} "
                  f"q_te {w['q_te']:.2f} q_rf {w['q_rf']:.2f} | pk {t['t_pk']} te50 {t['te50']} te90 {t['te90']} dd90 "
                  f"{t['drawdown_te90'] if t['drawdown_te90'] is None else round(t['drawdown_te90'], 3)} final {row['M']['final']:.3f}", flush=True)
        rows[rid] = row
        lid = f"L_D{r['depth']}_p{r['p']}_lv{dl['lam_v']:.4f}_lw0.015_s{r['seed']}"; lfn = os.path.join(lres, lid, 'run.json.gz')
        if lid not in controls and os.path.exists(lfn):
            controls[lid] = dict(id=lid, p=r['p'], depth=r['depth'], seed=r['seed'], dial=dict(dl, frac=0.5, r=0.45), M=score_file(lfn))
        if dl['frac'] == 0.8 and dl['lam_v_rule'] == 'lin' and row['state'] == 'finished':
            kid = 'K_' + lid[2:]; kfn = os.path.join(kres, kid, 'run.json.gz')
            if os.path.exists(kfn): kcheck[rid] = dict(K=kid, max_abs_loss_diff=max_loss_diff(fn, kfn))
    json.dump(dict(doc=__doc__, runs=rows, controls_L=controls, K_check=kcheck), open(out, 'w'), indent=1, default=float)
    n = {s_: sum(1 for x in rows.values() if x['state'] == s_) for s_ in ('finished', 'running', 'not started')}
    print('states:', n, '| L controls:', len(controls), '| K checks:', len(kcheck))


if __name__ == '__main__':
    main(*sys.argv[1:6])
