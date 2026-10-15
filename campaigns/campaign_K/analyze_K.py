import glob, json, math, os, statistics as S, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import endpoints_i as E


def relapses_in_band(d, W):
    s, m = E.series(d['obs'], 'train_margin_min')
    if W is None: return None
    k = (s >= W[0]) & (s <= W[1])
    return int(np.sum(m[k] <= 0))


def first_persistent_fit(d):
    s, m = E.series(d['obs'], 'train_margin_min')
    bad = np.where(m <= 0)[0]
    if not len(bad): return int(s[0])
    return int(s[bad[-1] + 1]) if bad[-1] + 1 < len(s) else None


def main(res_dir, out_dir):
    man = json.load(open(os.path.join(HERE, 'manifest.json'))); by_id = {r['id']: r for r in man['runs']}
    rows = {}
    for fn in sorted(glob.glob(os.path.join(res_dir, '*', 'run.json.gz'))):
        rid = os.path.basename(os.path.dirname(fn))
        if rid not in by_id: continue
        d = E.load(fn); f = E.fields(d)
        f['relapses_in_band'] = relapses_in_band(d, f.get('W')); f['fit_persistent'] = first_persistent_fit(d)
        f['CE_level_over_logp'] = f.get('CE_peak_over_logp')
        r = by_id[rid]
        rows[rid] = dict(setting=r['setting'], p=r['p'], depth=r['depth'], seed=r['seed'], dial=r['dial'], f=f)
    cells = {}
    for r in man['runs']:
        cells.setdefault((r['p'], r['depth']), {}).setdefault(r['setting'], []).append(r['id'])
    selection = {}; lines = ['# Campaign K, development stage: results', '',
                             f'{len(rows)} of {len(man["runs"])} runs scored with campaign C\'s per-run endpoint.', '']
    for (p, D), settings in sorted(cells.items()):
        lines += [f'## p = {p}, depth {D}', '',
                  '| setting | lam_v | lam_w | passes | failed checks | share (min) | relapses in band (median) | CE peak / log p | t10 | t90 | final |',
                  '|---|---|---|---|---|---|---|---|---|---|---|']
        scored = []
        for st, ids in sorted(settings.items()):
            got = [rows[i] for i in ids if i in rows]
            if len(got) < len(ids):
                lines.append(f'| {st} | | | {len(got)}/{len(ids)} runs finished | | | | | | | |'); continue
            fs = [g['f'] for g in got]; dial = got[0]['dial']
            k = sum(bool(x.get('primary')) for x in fs)
            shares = [x.get('share') for x in fs if x.get('share') is not None]
            rel = [x['relapses_in_band'] for x in fs if x.get('relapses_in_band') is not None]
            med = lambda key: S.median([x[key] for x in fs if isinstance(x.get(key), (int, float))]) if any(isinstance(x.get(key), (int, float)) for x in fs) else None
            why = '; '.join(sorted({x.get('why_not') or 'pass' for x in fs}))
            rec = dict(setting=st, passes=k, n=len(fs), min_share=min(shares) if shares else -1.0,
                       med_relapses=S.median(rel) if rel else float('inf'), sqrt_rule=dial['lam_v_rule'] == 'sqrt',
                       paper_lam_w=dial['lam_w'] == 0.015, dial=dial)
            scored.append(rec)
            fmt = lambda v, f='%.2f': (f % v) if isinstance(v, (int, float)) else '-'
            lines.append(f"| {st} | {dial['lam_v']} | {dial['lam_w']} | {k}/{len(fs)} | {why} | {fmt(rec['min_share'])} | "
                         f"{fmt(rec['med_relapses'], '%.0f')} | {fmt(med('CE_peak_over_logp'))} | {fmt(med('t10'), '%.0f')} | "
                         f"{fmt(med('t90'), '%.0f')} | {fmt(med('final_test'), '%.3f')} |")
        complete = len(scored) == len(settings)
        best = None; step = None
        if complete and any(r['passes'] > 0 for r in scored):
            keys = [('passes', lambda r: r['passes']), ('min share', lambda r: r['min_share']),
                    ('relapses', lambda r: -r['med_relapses']), ('square-root rule', lambda r: r['sqrt_rule']),
                    ('lam_w 0.015', lambda r: r['paper_lam_w'])]
            pool = scored
            for name, key in keys:
                top = max(key(r) for r in pool); pool = [r for r in pool if key(r) == top]
                if len(pool) == 1: step = name; break
            best = pool[0]; step = step or 'all ties (first by name)'
        selection[f'p{p}_D{D}'] = dict(p=p, depth=D, complete=complete, selected=best['setting'] if best else None,
                                       decided_by=step, dial=best['dial'] if best else None,
                                       note=None if best else ('incomplete' if not complete else 'no development run passes'))
        lines += ['', f"Selected: {best['setting']} (decided by: {step})" if best else
                  f"Selected: none ({selection[f'p{p}_D{D}']['note']})", '']
    os.makedirs(out_dir, exist_ok=True)
    json.dump(dict(campaign='K', stage='development', rows=rows), open(os.path.join(out_dir, 'analysis.json'), 'w'), indent=1, default=float)
    json.dump(selection, open(os.path.join(out_dir, 'selection.json'), 'w'), indent=1)
    open(os.path.join(out_dir, 'SUMMARY.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'analysis'))
