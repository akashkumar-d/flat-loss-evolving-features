import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKERS = [f'S{i}' for i in range(1, 17)]
CORES = 4
ETA_UNIT = 0.03 * 0.015 / 0.07
CONTENTION = 1.15

H_CFG = dict(depth=1, eta_v=0.03, eta_w=0.03, frac=0.8, heavy_every=500, init_scale=24.0, lam_v=0.2, lam_w=0.015, loss='ce',
             obs_every=20, observers_v1=True, p=31, steps=40000, stop_generalized=dict(acc=0.99, extra=10000, factor=1.5), width=512)
D2_CFG = dict(deep_init_scale=8.0, depth=2, eta_deep=0.003214285714285714, eta_v=0.03, eta_w=0.03, frac=0.8, heavy_every=500,
              init_scale=24.0, lam_deep=0.07, lam_v=0.2, lam_w=0.015, loss='ce', obs_every=20, p=31, steps=40000, width=256)
D4_CFG = dict(frac=0.8, loss='ce', eta_w=0.03, eta_v=0.03, init_scale=24.0, obs_every=20, heavy_every=500, lam_v=0.2, lam_w=0.015,
              deep_init_scale=8.0, lam_deep=0.07, width=256, p=31, observers_v1=True, depth=4, steps=100000,
              lam_layers=[0.015, 0.07, 0.07, 0.07])
D4_R0, D4_X = 0.35, 3.85
SEC = dict(H=368.0, D2=891.0, D4=6816.0)


def d4_eta(rate): return [0.03, ETA_UNIT, ETA_UNIT, rate * ETA_UNIT]


def runs(test=False):
    out = []
    for i in range(1, 11):
        cfg = dict(H_CFG, init_seed=5000 + i, split_seed=6000 + i, bias=True)
        out.append(dict(id=f'B_H_p31_lw0.0150_s{i}', family='H', depth=1, seed=i, init_seed=5000 + i, split_seed=6000 + i,
                        kind='one', cfg=cfg, paired_with=f'H_p31_lw0.0150_s{i}', est_s=SEC['H']))
    for i in range(0, 10):
        cfg = dict(D2_CFG, init_seed=101 + i, split_seed=201 + i, bias=True)
        out.append(dict(id=f'B_D2_r0.5_s{i}', family='D2', depth=2, seed=i, init_seed=101 + i, split_seed=201 + i,
                        kind='one', cfg=cfg, paired_with=f'seedD2_r0.5_s{i}', est_s=SEC['D2']))
    for i in range(1, 41):
        a, b = 1000 + i, 2000 + i
        tmpl = dict(D4_CFG, init_seed=a, split_seed=b, eta_layers=d4_eta(D4_R0), bias=True)
        spec = dict(r0=D4_R0, x_target=D4_X, pilot_cfg=dict(tmpl, stop_at_test=0.1), main_template=tmpl)
        out.append(dict(id=f'B_D4_s{i:03d}', family='D4', depth=4, seed=i, init_seed=a, split_seed=b, kind='chain', spec=spec,
                        paired_with=f"{'C2' if i <= 20 else 'C1'}_D4_s{i:03d}", est_s=SEC['D4']))
    if test:
        keep = [r for r in out if r['seed'] in (1, 2) or (r['family'] == 'D2' and r['seed'] in (0, 1))]
        for r in keep:
            if r['kind'] == 'one': r['cfg'].update(p=11, width=32, steps=300, obs_every=20, heavy_every=100)
            else:
                for c in (r['spec']['pilot_cfg'], r['spec']['main_template']): c.update(p=11, width=32, steps=300, obs_every=20, heavy_every=100)
            r['est_s'] = 60.0
        out = keep
    for r in out:
        r['est_core_hours'] = round(CONTENTION * r['est_s'] / 3600, 3); del r['est_s']
    load = {w: 0.0 for w in WORKERS}; order = {w: 0 for w in WORKERS}
    for r in sorted(out, key=lambda r: (-r['est_core_hours'], r['id'])):
        w = min(WORKERS, key=lambda w: (load[w], WORKERS.index(w)))
        r['worker'] = w; r['order'] = order[w]; order[w] += 1; load[w] += r['est_core_hours']
    return out, load


def session_hours(R, w):
    free = [0.0] * CORES
    for r in sorted((r for r in R if r['worker'] == w), key=lambda r: r['order']):
        i = free.index(min(free)); free[i] += r['est_core_hours']
    return round(max(free), 2)


if __name__ == '__main__':
    test = '--test' in sys.argv
    R, load = runs(test)
    man = dict(campaign='I', version='1.0' + ('-test' if test else ''), author='the authors', created='2026-10-06',
               workers=WORKERS, cores_per_worker=CORES, plan='PLAN.md', runs=R,
               est_session_hours={w: session_hours(R, w) for w in WORKERS})
    fn = os.path.join(HERE, 'manifest_test.json' if test else 'manifest.json')
    with open(fn, 'w') as f: json.dump(man, f, indent=1)
    print(f'wrote {fn}: {len(R)} runs, {sum(r["est_core_hours"] for r in R):.1f} core-hours; runs per worker:',
          {w: sum(1 for r in R if r['worker'] == w) for w in WORKERS})
