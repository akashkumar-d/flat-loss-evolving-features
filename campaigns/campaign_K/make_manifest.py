import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKERS = [f'K{i}' for i in range(1, 21)]
CORES = 4
CONTENTION = 1.15
ETA_UNIT = 0.03 * 0.015 / 0.07
PRIMES = [23, 47, 61, 101]
DEPTHS = [3, 4]
R_CELL = {(23, 3): 0.45, (23, 4): 0.32, (47, 3): 0.45, (47, 4): 0.45, (61, 3): 0.45, (61, 4): 0.45, (101, 3): 0.45, (101, 4): 0.45}
HORIZON = {23: 120000, 47: 100000, 61: 100000, 101: 100000}
THREADS = {23: 1, 47: 1, 61: 1, 101: 4}
HEAVY = {23: 500, 47: 500, 61: 500, 101: 1000}
LAM_W = [0.015, 0.025]
DEV_SEEDS = [1, 2, 3]
EST_H = {(23, 3): 0.58, (23, 4): 0.98, (47, 3): 1.32, (47, 4): 1.69, (61, 3): 1.96, (61, 4): 2.64, (101, 3): 2.9, (101, 4): 3.8}


def lam_v_levels(p):
    return [round(0.2 * (31 / p) ** 0.5, 4), round(0.2 * (31 / p), 4)]


def base_cfg(p, D):
    r = R_CELL[(p, D)]
    eta = [0.03] + [ETA_UNIT] * (D - 2) + [r * ETA_UNIT]
    return dict(frac=0.8, loss='ce', eta_w=0.03, eta_v=0.03, init_scale=24.0, obs_every=20, lam_v=0.2, lam_w=0.015,
                observers_v1=True, stop_generalized=dict(acc=0.99, factor=1.5, extra=10000), p=p, depth=D, width=256,
                deep_init_scale=8.0, lam_deep=0.07, steps=HORIZON[p], heavy_every=HEAVY[p],
                eta_layers=eta, lam_layers=[0.015] + [0.07] * (D - 1))


def runs(test=False):
    out = []
    for p in PRIMES:
        for D in DEPTHS:
            for iv, lv in enumerate(lam_v_levels(p)):
                for lw in LAM_W:
                    setting = f'K_D{D}_p{p}_lv{lv:.4f}_lw{lw:.3f}'
                    for i in DEV_SEEDS:
                        cfg = dict(base_cfg(p, D), lam_v=lv, lam_w=lw, init_seed=7000 + i, split_seed=8000 + i)
                        cfg['lam_layers'] = [lw] + cfg['lam_layers'][1:]
                        out.append(dict(id=f'{setting}_s{i}', setting=setting, family=f'D{D}', depth=D, p=p, seed=i,
                                        init_seed=7000 + i, split_seed=8000 + i, kind='one', threads=THREADS[p],
                                        dial=dict(lam_v=lv, lam_v_rule=['sqrt', 'linear'][iv], lam_w=lw, r=R_CELL[(p, D)]),
                                        cfg=cfg, est_h=EST_H[(p, D)]))
    if test:
        keep = [r for r in out if r['seed'] == 1 and r['dial']['lam_w'] == 0.015 and r['p'] in (23, 47)]
        for r in keep:
            r['cfg'].update(p=11, steps=300, obs_every=20, heavy_every=100); r['threads'] = 1; r['est_h'] = 60 / 3600
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
    man = dict(campaign='K', stage='development', version='1.0' + ('-test' if test else ''), author='claude/1hl-figures',
               created='2026-10-07', workers=WORKERS, cores_per_worker=CORES, plan='PLAN.md', runner='bias_runner.py', runs=R,
               est_session_hours={w: session_hours(R, w) for w in WORKERS})
    fn = os.path.join(HERE, 'manifest_test.json' if test else 'manifest.json')
    with open(fn, 'w') as f: json.dump(man, f, indent=1)
    print(f'wrote {fn}: {len(R)} runs, {sum(r["est_core_hours"] for r in R):.1f} core-hours; session hours:',
          sorted(man['est_session_hours'].values()))
