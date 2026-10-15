import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKERS = [f'N{i}' for i in range(1, 17)]
CORES = 4
CONTENTION = 1.05
ETA_UNIT = 0.03 * 0.015 / 0.07
PRIMES = [47, 61]
DEPTHS = [3, 4]
FRACS = [0.8, 0.7, 0.6, 0.5]
R_SEL = {3: {0.8: 0.45, 0.7: 0.30, 0.6: 0.18, 0.5: 0.08}, 4: {0.8: 0.45, 0.7: 0.28, 0.6: 0.13, 0.5: 0.06}}
HORIZON = {0.8: 100000, 0.7: 150000, 0.6: 200000, 0.5: 250000}
HEAVY = 500
LAM_W = 0.015
SEEDS = list(range(1, 11))
SEC = {(47, 3): 0.0515, (61, 3): 0.0805, (47, 4): 0.0706, (61, 4): 0.1073}
EXP_STEPS = {(3, 47): {0.8: 33760, 0.7: 43800, 0.6: 72120, 0.5: 87160}, (3, 61): {0.8: 30700, 0.7: 36840, 0.6: 47580, 0.5: 73440},
             (4, 47): {0.8: 46920, 0.7: 68280, 0.6: 94200, 0.5: 208660}, (4, 61): {0.8: 42040, 0.7: 54000, 0.6: 70360, 0.5: 113160}}


def lam_v(p): return round(0.2 * 31 / p, 4)


def base_cfg(p, D):
    return dict(frac=0.8, loss='ce', eta_w=0.03, eta_v=0.03, init_scale=24.0, obs_every=20, lam_v=0.2, lam_w=LAM_W,
                observers_v1=True, stop_generalized=dict(acc=0.99, factor=1.5, extra=10000), p=p, depth=D, width=256,
                deep_init_scale=8.0, lam_deep=0.07, steps=100000, heavy_every=HEAVY,
                eta_layers=[0.03] + [ETA_UNIT] * (D - 2) + [0.45 * ETA_UNIT], lam_layers=[LAM_W] + [0.07] * (D - 1))


def setting_id(D, p, f): return f'N_D{D}_p{p}_f{f:.1f}_r{R_SEL[D][f]:.2f}'


def runs(test=False):
    out = []
    for D in DEPTHS:
        for p in PRIMES:
            for f in FRACS:
                r = R_SEL[D][f]; st = setting_id(D, p, f)
                for i in SEEDS:
                    cfg = dict(base_cfg(p, D), frac=f, steps=HORIZON[f], lam_v=lam_v(p), init_seed=9000 + i, split_seed=9500 + i)
                    cfg['eta_layers'] = cfg['eta_layers'][:-1] + [r * ETA_UNIT]
                    out.append(dict(id=f'{st}_s{i}', setting=st, family=f'D{D}', depth=D, p=p, seed=i, init_seed=9000 + i,
                                    split_seed=9500 + i, kind='one', threads=1,
                                    dial=dict(frac=f, r=r, lam_v=lam_v(p), lam_v_rule='lin', lam_w=LAM_W),
                                    cfg=cfg, est_h=SEC[(p, D)] * EXP_STEPS[(D, p)][f] / 3600))
    if test:
        keep = [r for r in out if r['seed'] in (1, 2) and r['p'] == 47 and r['depth'] in (3, 4) and r['dial']['frac'] in (0.8, 0.5)]
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
        free.sort(); free[0] += r['est_core_hours']
    return round(max(free), 2)


if __name__ == '__main__':
    test = '--test' in sys.argv
    R = runs(test)
    ids = [r['id'] for r in R]; assert len(ids) == len(set(ids))
    man = dict(campaign='N', stage='confirmation on fresh seeds (deep networks across p and training fraction)',
               version='1.0' + ('-test' if test else ''), author='claude/1hl-figures', created='2026-10-12', workers=WORKERS,
               cores_per_worker=CORES, plan='PLAN.md', runner='bias_runner.py', runs=R,
               est_session_hours={w: session_hours(R, w) for w in WORKERS})
    fn = os.path.join(HERE, 'manifest_test.json' if test else 'manifest.json')
    with open(fn, 'w') as f: json.dump(man, f, indent=1)
    print(f'wrote {fn}: {len(R)} runs, {sum(r["est_core_hours"] for r in R):.1f} core-hours; session hours:',
          sorted(man['est_session_hours'].values()))
