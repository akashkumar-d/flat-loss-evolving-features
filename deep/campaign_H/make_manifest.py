import json, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
WORKERS = [f'S{i}' for i in range(1, 9)]
SEC_PER_STEP = 0.115
Q_CLK = 0.9 ** (1 / 32)
CLOCK = 1 - Q_CLK

BASE = dict(p=101, width=512, split='qr', split_seed=0, init_scale=24.0, head_init_div=24.0, lam_v=0.0,
            steps=40000, obs_every=50, heavy_every=250, stop_generalized=dict(acc=0.99, factor=1.25, extra=4000))

ARMS = [
    ('ev1e-3', CLOCK, 1.0, 1e-3, range(1, 11)),
    ('ev3e-4', CLOCK, 1.0, 3e-4, range(1, 7)),
    ('ev3e-3', CLOCK, 1.0, 3e-3, range(1, 7)),
    ('ev1e-2', CLOCK, 1.0, 1e-2, range(1, 7)),
    ('ev1e-6', CLOCK, 1.0, 1e-6, range(1, 3)),
    ('paper', 0.03, 0.02, 1e-3, range(1, 3)),
]


def runs(test=False):
    out = []
    arms = [('mini', 0.03, 0.02, 1e-3, range(1, 7))] if test else ARMS
    for arm, ew, lw, ev, seeds in arms:
        for s in seeds:
            cfg = dict(BASE, eta_w=ew, lam_w=lw, eta_v=ev, init_seed=9100 + s)
            if test: cfg.update(p=13, width=32, steps=300, obs_every=20, heavy_every=100)
            out.append(dict(id=f'T41_{arm}_s{s}', arm=arm, seed=s, cfg=cfg,
                            est_core_hours=round(cfg['steps'] * (0.001 if test else SEC_PER_STEP) / 3600, 3)))
    for i, r in enumerate(out):
        r['worker'] = WORKERS[i % len(WORKERS)]; r['order'] = i // len(WORKERS)
    return out


if __name__ == '__main__':
    test = '--test' in sys.argv
    R = runs(test)
    man = dict(campaign='H', version='1.0' + ('-test' if test else ''), author='the authors', created='2026-10-06',
               workers=WORKERS, plan='PLAN.md', runs=R,
               est_session_hours={w: round(sum(r['est_core_hours'] for r in R if r['worker'] == w) / 4, 2) for w in WORKERS})
    fn = os.path.join(HERE, 'manifest_test.json' if test else 'manifest.json')
    with open(fn, 'w') as f: json.dump(man, f, indent=1)
    print(f'wrote {fn}: {len(R)} runs, {sum(r["est_core_hours"] for r in R):.1f} core-hours; per worker:',
          {w: sum(1 for r in R if r['worker'] == w) for w in WORKERS})
