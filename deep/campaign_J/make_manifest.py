import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
H_CFG = dict(depth=1, eta_v=0.03, eta_w=0.03, frac=0.8, heavy_every=500, init_scale=24.0, lam_v=0.2, lam_w=0.015, loss='ce',
             obs_every=20, observers_v1=True, p=31, steps=40000, stop_generalized=dict(acc=0.99, extra=10000, factor=1.5), width=512,
             bias=True)
ARMS = [('K', dict(bias_after_knee='frozen')), ('KD', dict(bias_after_knee='decay')), ('S', dict(bias_update='separate'))]
SEC = 280.0


def runs():
    out = []
    for arm, extra in ARMS:
        for i in range(1, 11):
            cfg = dict(H_CFG, init_seed=5000 + i, split_seed=6000 + i, **extra)
            out.append(dict(id=f'J{arm}_H_p31_lw0.0150_s{i}', arm=arm, family='H', depth=1, seed=i, init_seed=5000 + i,
                            split_seed=6000 + i, kind='one', cfg=cfg, joint=f'B_H_p31_lw0.0150_s{i}', twin=f'H_p31_lw0.0150_s{i}',
                            est_core_hours=round(SEC / 3600, 3)))
    out.sort(key=lambda r: (r['seed'], [a for a, _ in ARMS].index(r['arm'])))
    for k, r in enumerate(out): r['order'] = k
    return out


if __name__ == '__main__':
    R = runs()
    man = dict(campaign='J', version='1.0', author='the authors', created='2026-10-06', plan='PLAN.md',
               runner='bias_runner.py', where='cloud container, 2 cores, two runs at a time (run_local.py)', runs=R)
    fn = os.path.join(HERE, 'manifest.json')
    with open(fn, 'w') as f: json.dump(man, f, indent=1)
    print(f'wrote {fn}: {len(R)} runs, {sum(r["est_core_hours"] for r in R):.1f} core-hours;',
          {a: sum(r['arm'] == a for r in R) for a, _ in ARMS})
