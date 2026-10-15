import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKERS = [f'QD{i}' for i in range(1, 5)]
CORES = 4
CONTENTION = 1.15
D2 = dict(deep_init_scale=8.0, depth=2, eta_deep=0.003214285714285714, eta_v=0.03, eta_w=0.03, frac=0.8, heavy_every=500,
          init_scale=24.0, lam_deep=0.07, lam_v=0.2, lam_w=0.015, loss='ce', obs_every=20, p=31, steps=40000, width=256)
HOURS = 0.25


def paired(i):
    return f"C2_D3_s{i:03d} / C2_D4_s{i:03d} / C2_D5_s{i:03d}" if i <= 20 else f"C1_D3_s{i:03d} / C1_D4_s{i:03d} / C1_D5_s{i:03d}"


def runs(test=False):
    out = []
    for i in range(1, 41):
        for fam, extra in (('D2', {}), ('B_D2', dict(bias=True))):
            cfg = dict(D2, observers_v1=True, **extra); cfg.update(init_seed=1000 + i, split_seed=2000 + i)
            out.append(dict(id=f'QD_{fam}_s{i:03d}', family=fam, depth=2, seed=i, init_seed=1000 + i, split_seed=2000 + i,
                            kind='one', threads=1, cfg=cfg, paired_with=paired(i) if fam == 'D2' else f'QD_D2_s{i:03d}',
                            est_h=HOURS, worker=WORKERS[(i - 1) % 4]))
    if test:
        out = [r for r in out if r['seed'] in (1, 2)]
        for r in out:
            r['cfg'].update(p=11, width=32, steps=300, heavy_every=100); r['est_h'] = 20 / 3600
            r['worker'] = 'QD1' if r['seed'] == 1 else 'QD2'
    for r in out:
        r['est_core_hours'] = round(CONTENTION * r['est_h'], 3); del r['est_h']
    order = {}
    for r in out:
        r['order'] = order.get(r['worker'], 0); order[r['worker']] = r['order'] + 1
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
    W = sorted({r['worker'] for r in R}) if test else WORKERS
    man = dict(campaign='QD', stage="depth 2 at p = 31 on the 40 evaluation seeds of depths 3-5, bias-free and with a hidden bias",
               version='1.0' + ('-test' if test else ''), author='claude/1hl-figures', created='2026-10-13', workers=WORKERS,
               cores_per_worker=CORES, plan='PLAN.md', runner='bias_runner.py', runs=R,
               est_session_hours={w: session_hours(R, w) for w in W})
    fn = os.path.join(HERE, 'manifest_test.json' if test else 'manifest.json')
    with open(fn, 'w') as f: json.dump(man, f, indent=1)
    print(f'wrote {fn}: {len(R)} runs, {sum(r["est_core_hours"] for r in R):.1f} core-hours; session hours {man["est_session_hours"]}')
