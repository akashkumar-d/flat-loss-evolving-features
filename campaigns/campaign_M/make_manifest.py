import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKERS = [f'M{i}' for i in range(1, 25)]
CORES = 4
CONTENTION = 1.05
ETA_UNIT = 0.03 * 0.015 / 0.07
PRIMES = [47, 61]
DEPTHS = [3, 4]
FRACS = [0.8, 0.7, 0.6, 0.5]
R_GRID = {3: {0.8: [0.45], 0.7: [0.45, 0.30], 0.6: [0.45, 0.28, 0.18], 0.5: [0.20, 0.13, 0.08]},
          4: {0.8: [0.45], 0.7: [0.45, 0.28], 0.6: [0.45, 0.20, 0.13], 0.5: [0.14, 0.09, 0.06]}}
HORIZON = {0.8: 100000, 0.7: 150000, 0.6: 200000, 0.5: 250000}
HEAVY = 500
LAM_W = 0.015
RULES = ['paper', 'lin']
SEEDS = [1, 2, 3]
SEC = {(47, 3): 0.052, (61, 3): 0.095, (47, 4): 0.082, (61, 4): 0.125}
TE90 = {(3, 47, 'lin'): {0.8: 20e3, 0.7: 33e3, 0.6: 58e3, 0.5: 115e3}, (3, 47, 'paper'): {0.8: 21e3, 0.7: 37e3, 0.6: 72e3, 0.5: 156e3},
        (3, 61, 'lin'): {0.8: 18e3, 0.7: 27e3, 0.6: 43e3, 0.5: 76e3}, (3, 61, 'paper'): {0.8: 19e3, 0.7: 34e3, 0.6: 67e3, 0.5: 149e3},
        (4, 47, 'lin'): {0.8: 27e3, 0.7: 44e3, 0.6: 79e3, 0.5: 158e3}, (4, 47, 'paper'): {0.8: 29e3, 0.7: 49e3, 0.6: 93e3, 0.5: 197e3},
        (4, 61, 'lin'): {0.8: 24e3, 0.7: 38e3, 0.6: 63e3, 0.5: 116e3}, (4, 61, 'paper'): {0.8: 26e3, 0.7: 44e3, 0.6: 82e3, 0.5: 171e3}}


def lam_v(p, rule): return 0.2 if rule == 'paper' else round(0.2 * 31 / p, 4)


def base_cfg(p, D):
    return dict(frac=0.8, loss='ce', eta_w=0.03, eta_v=0.03, init_scale=24.0, obs_every=20, lam_v=0.2, lam_w=LAM_W,
                observers_v1=True, stop_generalized=dict(acc=0.99, factor=1.5, extra=10000), p=p, depth=D, width=256,
                deep_init_scale=8.0, lam_deep=0.07, steps=100000, heavy_every=HEAVY,
                eta_layers=[0.03] + [ETA_UNIT] * (D - 2) + [0.45 * ETA_UNIT], lam_layers=[LAM_W] + [0.07] * (D - 1))


def setting_id(D, p, rule, f, r): return f'M_D{D}_p{p}_{rule}_f{f:.1f}_r{r:.2f}'


def runs(test=False):
    out = []
    for D in DEPTHS:
        for p in PRIMES:
            for rule in RULES:
                for f in FRACS:
                    for r in R_GRID[D][f]:
                        st = setting_id(D, p, rule, f, r)
                        for i in SEEDS:
                            cfg = dict(base_cfg(p, D), frac=f, steps=HORIZON[f], lam_v=lam_v(p, rule), init_seed=7000 + i,
                                       split_seed=8000 + i)
                            cfg['eta_layers'] = cfg['eta_layers'][:-1] + [r * ETA_UNIT]
                            exp_updates = min(HORIZON[f], 2.0 * TE90[(D, p, rule)][f])
                            out.append(dict(id=f'{st}_s{i}', setting=st, family=f'D{D}', depth=D, p=p, seed=i,
                                            init_seed=7000 + i, split_seed=8000 + i, kind='one', threads=1,
                                            dial=dict(frac=f, r=r, lam_v=lam_v(p, rule), lam_v_rule=rule, lam_w=LAM_W),
                                            cfg=cfg, est_h=SEC[(p, D)] * exp_updates / 3600))
    if test:
        keep = [r for r in out if r['seed'] == 1 and r['p'] == 47 and r['dial']['lam_v_rule'] == 'lin' and r['dial']['frac'] in (0.8, 0.5)]
        for r in keep:
            r['cfg'].update(p=11, steps=300, obs_every=20, heavy_every=100); r['est_h'] = 60 / 3600
        out = keep
    for r in out:
        r['est_core_hours'] = round(CONTENTION * r['est_h'] * r['threads'], 3); del r['est_h']
    load = {w: 0.0 for w in WORKERS}; order = {w: 0 for w in WORKERS}
    for r in sorted(out, key=lambda r: (-r['est_core_hours'], r['id'])):
        w = min(WORKERS, key=lambda w: (load[w], WORKERS.index(w)))
        r['worker'] = w; r['order'] = order[w]; order[w] += 1; load[w] += r['est_core_hours']
    return out


def session_hours(R, w):
    free = [0.0] * CORES
    for r in sorted((r for r in R if r['worker'] == w), key=lambda r: r['order']):
        k = min(r['threads'], CORES); free.sort(); start = free[k - 1]
        for j in range(k): free[j] = start + r['est_core_hours'] / r['threads']
    return round(max(free), 2)


if __name__ == '__main__':
    test = '--test' in sys.argv
    R = runs(test)
    ids = [r['id'] for r in R]; assert len(ids) == len(set(ids))
    man = dict(campaign='M', stage='development (training-fraction ladder with the last-layer rate re-tuned)',
               version='1.0' + ('-test' if test else ''), author='claude/1hl-figures', created='2026-10-11', workers=WORKERS,
               cores_per_worker=CORES, plan='PLAN.md', runner='bias_runner.py', runs=R,
               est_session_hours={w: session_hours(R, w) for w in WORKERS})
    fn = os.path.join(HERE, 'manifest_test.json' if test else 'manifest.json')
    with open(fn, 'w') as f: json.dump(man, f, indent=1)
    print(f'wrote {fn}: {len(R)} runs, {sum(r["est_core_hours"] for r in R):.1f} core-hours; session hours:',
          sorted(man['est_session_hours'].values()))
