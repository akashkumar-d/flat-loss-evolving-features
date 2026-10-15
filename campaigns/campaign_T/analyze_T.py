import glob, gzip, json, math, os, statistics as S, sys
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np

V_MAX, DAH_MIN, FINAL_MIN = 0.10, 0.02, 0.99


def load(fn):
    with gzip.open(fn, 'rt') as f: return json.load(f)


def series(obs, key):
    s = [(o['step'], o[key]) for o in obs if o.get(key) is not None]
    return np.array([x[0] for x in s], dtype=float), np.array([x[1] for x in s], dtype=float)


def at(s, v, t): return float(v[int(np.argmin(np.abs(s - t)))]) if len(s) else float('nan')


def first_cross(s, v, thr):
    i = np.where(v >= thr)[0]; return int(s[i[0]]) if len(i) else None


def first_persistent(s, v, thr):
    bad = np.where(v < thr)[0]
    if not len(bad): return int(s[0]) if len(s) else None
    return int(s[bad[-1] + 1]) if bad[-1] + 1 < len(s) else None


def window_v(L, a, b):
    seg = np.asarray(L[a:b + 1], dtype=float)
    return float(seg.max() / seg.min() - 1) if len(seg) and seg.min() > 0 else float('inf')


def fields(d):
    c = d.get('cfg_full', d['cfg']); L = d['loss']; obs = d['obs']; p = c['p']; nl = c['n_layers']
    s, te = series(obs, 'test_acc'); _, tra = series(obs, 'train_acc'); sm, mtr = series(obs, 'train_margin_min')
    t_fit = first_persistent(sm, (mtr > 0).astype(float), 0.5)
    t10, t90 = first_cross(s, te, 0.1), first_persistent(s, te, 0.9)
    out = dict(n_layers=nl, optimizer=c['optimizer'], horizon=len(L) - 1, t_fit=t_fit, t10=t10, t90=t90, t90_first=first_cross(s, te, 0.9),
               final_test=float(te[-1]), final_train=float(tra[-1]), test_at_fit=at(s, te, t_fit) if t_fit is not None else None)
    hs, ah = series(obs, f'AH_full_h{nl}'); _, ahr = series(obs, 'AH_full_resid')
    rs, rf = series(obs, 'refit_test_acc')
    out['refit90'] = first_cross(rs, rf, 0.9)
    if t_fit is not None and t90 is not None and t90 > t_fit:
        out['v_fit_t90'] = window_v(L, t_fit, t90)
        out['CE_fit_over_logp'] = float(L[t_fit]) / math.log(p); out['CE_t90_over_logp'] = float(L[t90]) / math.log(p)
        out['dAH_mlp'] = at(hs, ah, t90) - at(hs, ah, t_fit); out['dAH_resid'] = at(hs, ahr, t90) - at(hs, ahr, t_fit)
    q = dict(fit_before_rise=t_fit is not None and t10 is not None and t_fit <= t10 and (out['test_at_fit'] or 0) <= 0.1,
             final=out['final_test'] >= FINAL_MIN, flat=out.get('v_fit_t90', float('inf')) <= V_MAX,
             features=out.get('dAH_mlp', -1) >= DAH_MIN)
    out['qualify_checks'] = q; out['qualifies'] = bool(all(q.values()))
    out['why_not'] = ', '.join(k for k, v in q.items() if not v) or None
    return out


def ceil500(x): return int(500 * math.ceil(x / 500))


def main(res_dir, out_dir):
    man = json.load(open(os.path.join(HERE, 'manifest.json'))); by_id = {r['id']: r for r in man['runs']}
    rows = {}
    for fn in sorted(glob.glob(os.path.join(res_dir, '*', 'run.json.gz'))):
        rid = os.path.basename(os.path.dirname(fn))
        if rid in by_id: rows[rid] = dict(setting=by_id[rid]['setting'], arm=by_id[rid]['arm'], dial=by_id[rid]['dial'], f=fields(load(fn)))
    settings = {}
    for r in man['runs']: settings.setdefault((r['arm'], r['setting']), []).append(r['id'])
    lines = ['# Campaign T, development stage: results', '', f'{len(rows)} of {len(man["runs"])} runs analysed.', '',
             '| arm | setting | qualify | why not | t_fit | t10 | t90 | v[fit, t90] | CE at fit / log p | dA_H (MLP) | refit 0.9 | final |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    fmt = lambda v, f='%.3g': (f % v) if isinstance(v, (int, float)) and v is not None and math.isfinite(v) else '-'
    per = {}
    for (arm, st), ids in sorted(settings.items()):
        got = [rows[i]['f'] for i in ids if i in rows]
        rec = dict(arm=arm, setting=st, n=len(ids), done=len(got), qualifies=len(got) == len(ids) and all(f['qualifies'] for f in got),
                   max_v=max((f.get('v_fit_t90', float('inf')) for f in got), default=float('inf')),
                   min_dAH=min((f.get('dAH_mlp', -1) for f in got), default=-1), dial=rows[ids[0]]['dial'] if ids[0] in rows else None,
                   max_t_fit=max((f['t_fit'] for f in got if f['t_fit'] is not None), default=None),
                   max_t90=max((f['t90'] for f in got if f['t90'] is not None), default=None))
        per[st] = rec
        for i in ids:
            if i not in rows: continue
            f = rows[i]['f']
            lines.append(f"| {arm} | {i} | {'yes' if f['qualifies'] else 'no'} | {f['why_not'] or ''} | {fmt(f['t_fit'], '%d')} | "
                         f"{fmt(f['t10'], '%d')} | {fmt(f['t90'], '%d')} | {fmt(f.get('v_fit_t90'))} | {fmt(f.get('CE_fit_over_logp'))} | "
                         f"{fmt(f.get('dAH_mlp'))} | {fmt(f['refit90'], '%d')} | {fmt(f['final_test'])} |")
    selection = {}
    for arm in ('P1', 'P2'):
        pool = [r for r in per.values() if r['arm'] == arm and r['qualifies']]
        if not pool:
            selection[arm] = dict(selected=None, note='no setting qualifies in every development seed'); continue
        pool.sort(key=lambda r: (r['max_v'], -r['min_dAH'], r['dial']['frac'] != 0.3, r['setting']))
        b = pool[0]
        selection[arm] = dict(selected=b['setting'], dial=b['dial'], max_v=b['max_v'], min_dAH=b['min_dAH'],
                              window=[ceil500(1.2 * b['max_t_fit']), ceil500(1.2 * b['max_t90'])],
                              qualifying_settings=[r['setting'] for r in pool])
    lines += ['', '## Selection (PLAN.md)', '']
    for arm, s in selection.items():
        lines.append(f"- {arm}: " + (f"{s['selected']}, window I = {s['window']} (max v {s['max_v']:.3f}, min dA_H {s['min_dAH']:.3f}; "
                                      f"{len(s['qualifying_settings'])} settings qualify)" if s['selected'] else s['note']))
    os.makedirs(out_dir, exist_ok=True)
    json.dump(dict(campaign='T', stage='development', rows=rows, settings=per), open(os.path.join(out_dir, 'analysis.json'), 'w'),
              indent=1, default=float)
    json.dump(selection, open(os.path.join(out_dir, 'selection.json'), 'w'), indent=1, default=float)
    open(os.path.join(out_dir, 'SUMMARY.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines[-12:]))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'analysis'))
