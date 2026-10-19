#!/usr/bin/env python3
import glob, gzip, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paperdefs_tf as P

QS = (0.1, 0.5, 0.9)
READOUTS = {
    'paper_refit': ('fref_all', 'Paper refit (all frequencies)'),
    'fourier_top8': ('fref_top8', 'Fourier-filtered refit, top 8'),
    'fourier_top4': ('fref_top4', 'Fourier-filtered refit, top 4'),
    'fourier_top16': ('fref_top16', 'Fourier-filtered refit, top 16'),
    'phase_top8': ('fphase_top8', 'Phase vote, top 8'),
    'net_top8': ('fnet_top8', "Network's logits restricted to the top 8"),
    'net_top16': ('fnet_top16', "Network's logits restricted to the top 16"),
    'probe': ('probe', 'Logistic probe'),
    'oracle': ('fref_oracle', 'Oracle filter (diagnostic)'),
    'fam_all': ('ffam_all', 'Family alone, all pairs (diagnostic)'),
    'fam_ho': ('ffam_ho', 'Family alone, held-out pairs (diagnostic)'),
    'fam_tr': ('ffam_tr', 'Family alone, training pairs (diagnostic)'),
    'mlp_table_top8': ('fmref_top8', 'Fourier-filtered refit on the MLP activations, top 8'),
}


def load(fn):
    with gzip.open(fn, 'rt') as f: return json.load(f)


def series(obs, key):
    s = [(o['step'], o[key]) for o in obs if o.get(key) is not None]
    return np.array([x[0] for x in s], float), np.array([x[1] for x in s], float)


def first_logged(s, v, q):
    i = np.where(v >= q)[0]
    return float(s[i[0]]) if len(i) else None


def interp_cross(s, v, q):
    i = np.where(v >= q)[0]
    if not len(i): return None
    i = i[0]
    if i == 0 or v[i] <= v[i - 1]: return float(s[i])
    return float(s[i - 1] + (q - v[i - 1]) / (v[i] - v[i - 1]) * (s[i] - s[i - 1]))


def onset(s, v, t50):
    if t50 is None: return None
    i = np.where((v < 0.1) & (s <= t50))[0]
    return float(s[i[-1]]) if len(i) else float(s[0])


def relaxed(d, kmax, key, tol=0.1):
    L = np.asarray(d['loss'], float); obs = [o for o in d['obs'] if o['step'] < len(L) and key in o]
    c = P.cfg_of(d); ntr = int(c['frac'] * c['p'] ** 2)
    st = np.array([o['step'] for o in obs]); te = np.array([o[key] for o in obs])
    mis = np.array([round((1 - o['train_acc']) * ntr) for o in obs]); n = len(st)
    bmax = np.empty(n); bmin = np.empty(n)
    for i in range(n):
        e = st[i + 1] if i + 1 < n else st[i] + 1
        seg = L[st[i]:min(e, len(L))]; bmax[i] = seg.max(); bmin[i] = seg.min()
    cl = np.clip(te, 0.1, 0.9); bq, arg = 0.0, None
    for a in range(n):
        if mis[a] > kmax: continue
        mx = np.maximum.accumulate(bmax[a:]); mn = np.minimum.accumulate(bmin[a:]); ok = mx / mn - 1 <= tol
        bad = np.where(mis[a:] > kmax)[0]
        if len(bad): ok[bad[0]:] = False
        stop = np.where(~ok)[0]; m = stop[0] if len(stop) else len(ok)
        if m == 0: continue
        q = (cl[a:a + m] - cl[a]) / 0.8; j = int(np.argmax(q))
        if q[j] > bq: bq, arg = float(q[j]), (int(st[a]), int(st[a + j]))
    return bq, arg


def per_run(fn, kind):
    d = load(fn); obs = d['obs']; c = d.get('cfg_full') or d['cfg']; p = c['p']
    tf = kind == 'tf'; suf = '_test_acc_nm' if tf else '_test_acc'
    net_key = 'test_acc_nm' if tf else 'test_acc'
    s_net, v_net = series(obs, net_key)
    rec = dict(file=fn, kind=kind, p=p, net={str(q): first_logged(s_net, v_net, q) for q in QS}, readouts={})
    for name, (key, _) in READOUTS.items():
        k = 'probe_test_acc_nm' if (name == 'probe' and tf) else ('probe_test_acc' if name == 'probe' else key + suf)
        s, v = series(obs, k)
        if not len(s): continue
        cr = {str(q): interp_cross(s, v, q) for q in QS}
        lead = {str(q): (rec['net'][str(q)] - cr[str(q)]) if (cr[str(q)] is not None and rec['net'][str(q)] is not None) else None
                for q in QS}
        rec['readouts'][name] = dict(cross=cr, lead=lead, final=float(v[-1]))
    H = [o for o in obs if 'fspec_hor2' in o]
    t50 = rec['net']['0.5']
    if H and t50 is not None:
        o = min(H, key=lambda o: abs(o['step'] - t50))
        r2 = np.array(o['fspec_hor2']); top8 = o['ftop8']
        sig = {int(k) + 1 for k in np.where(r2 > 0.05)[0]}
        head8 = [int(k) + 1 for k in np.argsort(-np.array(o['fspec_head']))[:8]] if 'fspec_head' in o else []
        emb8 = [int(k) + 1 for k in np.argsort(-np.array(o['fspec_embed']))[:8]] if 'fspec_embed' in o else []
        rec['freq_at_net50'] = dict(step=o['step'], n_signal=len(sig), signal=sorted(sig), top8=top8,
                                    top8_in_signal=sum(k in sig for k in top8), head8_overlap=len(set(top8) & set(head8)),
                                    embed8_overlap=len(set(top8) & set(emb8)), AH=float(sum(o['fspec_grid'])), K=len(r2))
    L = np.asarray(d['loss'], float); lp = math.log(p)
    fr = rec['readouts'].get('fourier_top8')
    if fr and fr['cross']['0.5'] is not None and rec['net']['0.9'] is not None:
        s8, v8 = series(obs, 'fref_top8' + suf); a = int(onset(s8, v8, fr['cross']['0.5'])); b = int(rec['net']['0.9'])
        cm = (L[:-1] + L[1:]) / 2
        t = np.arange(a, min(b, len(cm) - 1) + 1); y = cm[t]
        slope = float(np.polyfit(t, np.log(y), 1)[0] * 1000 * 100) if len(t) > 2 else float('nan')
        st_, tr_ = series(obs, 'train_acc'); m = (st_ >= a) & (st_ <= b); ntr = int(c['frac'] * p * p)
        rec['band'] = dict(a=a, b=b, ce_a=float(cm[a]), ce_b=float(cm[min(b, len(cm) - 1)]),
                           ce_rel_change=float(cm[min(b, len(cm) - 1)] / cm[a] - 1), ce_over_logp=float(np.median(y) / lp),
                           slope_pct_per_1k=slope, raw_ratio=float(L[a:b + 1].max() / L[a:b + 1].min() - 1),
                           min_train_acc=float(tr_[m].min()) if m.any() else None,
                           max_mis=int(round((1 - tr_[m].min()) * ntr)) if m.any() else None)
    if tf:
        rec['window'] = {}
        for kmax in (0, 10):
            q, I = relaxed(d, kmax, net_key)
            w = dict(share=q, I=I)
            if I:
                a, b = I
                for name, key in (('ridge', 'refit_test_acc_nm'), ('fourier_top8', 'fref_top8_test_acc_nm')):
                    sr, rr = series(obs, key); w[f'{name}_gain'] = float(P.inside_gain(sr, rr, a, b))
                sa, ah = series(obs, 'AH_full_resid'); w['dAH'] = float(P.at(sa, ah, b) - P.at(sa, ah, a))
                w['ce_over_logp'] = float(np.median(L[a:b + 1]) / lp)
            final = float(series(obs, 'test_acc')[1][-1])
            w['final'] = final
            w['meets'] = bool(I and q >= 0.75 and w.get('ridge_gain', 0) >= 0.2 and w.get('dAH', 0) >= 0.02 and final >= 0.99)
            w['meets_fourier'] = bool(I and q >= 0.75 and w.get('fourier_top8_gain', 0) >= 0.2 and w.get('dAH', 0) >= 0.02
                                      and final >= 0.99)
            rec['window']['strict' if kmax == 0 else 'tol10'] = w
    return rec


def summarize(recs):
    out = {}
    names = sorted({n for r in recs for n in r['readouts']}, key=list(READOUTS).index)
    for name in names:
        row = {}
        for q in ('0.5', '0.9'):
            L = [r['readouts'][name]['lead'][q] for r in recs if name in r['readouts']]
            have = [x for x in L if x is not None]
            row[q] = dict(n=len(L), n_cross=len(have), n_lead=sum(x > 0 for x in have),
                          median=float(np.median(have)) if have else None, lo=min(have) if have else None,
                          hi=max(have) if have else None)
        out[name] = row
    fq = [r['freq_at_net50'] for r in recs if 'freq_at_net50' in r]
    if fq:
        out['_freq'] = dict(n_signal=[f['n_signal'] for f in fq], top8_in_signal=[f['top8_in_signal'] for f in fq],
                            head8_overlap=[f['head8_overlap'] for f in fq], embed8_overlap=[f['embed8_overlap'] for f in fq])
    bd = [r['band'] for r in recs if 'band' in r]
    if bd:
        out['_band'] = {k: [b[k] for b in bd] for k in ('ce_rel_change', 'slope_pct_per_1k', 'ce_over_logp', 'max_mis', 'raw_ratio')}
    wn = [r['window'] for r in recs if 'window' in r]
    if wn:
        out['_window'] = {k: dict(meets=sum(w[k]['meets'] for w in wn), meets_fourier=sum(w[k]['meets_fourier'] for w in wn),
                                  share=[round(w[k]['share'], 3) for w in wn]) for k in ('strict', 'tol10')}
    out['_n'] = len(recs)
    return out


def main(xdir, mdir, outp):
    groups = {}
    for fn in sorted(glob.glob(os.path.join(xdir, '*', 'run.json.gz'))):
        rid = os.path.basename(os.path.dirname(fn)); arm = rid.split('_')[1]
        groups.setdefault(f'transformer {arm}', []).append(dict(per_run(fn, 'tf'), id=rid))
    for fn in sorted(glob.glob(os.path.join(mdir, '*.json.gz'))):
        groups.setdefault('MLP 1HL p=31', []).append(dict(per_run(fn, 'mlp'), id=os.path.basename(fn)))
    res = dict(groups={g: dict(runs=R, summary=summarize(R)) for g, R in groups.items()})
    json.dump(res, open(outp, 'w'), indent=1)
    for g, v in res['groups'].items():
        S = v['summary']; print(f"== {g}: {S['_n']} runs")
        for name in READOUTS:
            if name not in S: continue
            r5, r9 = S[name]['0.5'], S[name]['0.9']
            f = lambda r: (f"{r['n_lead']}/{r['n']} lead, median {r['median']:+.0f} [{r['lo']:+.0f}, {r['hi']:+.0f}]"
                           if r['median'] is not None else f"0/{r['n']} cross")
            print(f"   {READOUTS[name][1]:52s} 0.5: {f(r5):40s} 0.9: {f(r9)}")
        if '_freq' in S: print('   frequencies at the network 0.5 crossing:', S['_freq'])
        if '_band' in S: print('   band [filtered-refit onset, network 0.9]:', {k: [round(x, 3) if isinstance(x, float) else x for x in v_] for k, v_ in S['_band'].items()})
        if '_window' in S: print('   W window test:', S['_window'])


if __name__ == '__main__':
    main(*sys.argv[1:4])
