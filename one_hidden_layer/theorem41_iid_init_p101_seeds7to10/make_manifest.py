import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKERS = ['QH1']
CORES = 4
ARM = dict(p=101, width=512, split='qr', split_seed=0, init_scale=24.0, head_init_div=24.0, lam_v=0.0, steps=40000, obs_every=50,
           heavy_every=250, stop_generalized=dict(acc=0.99, factor=1.25, extra=4000), eta_w=0.0032871017270746927, lam_w=1.0,
           eta_v=0.0003)
EST = 1.278


def runs(test=False):
    out = []
    for s in range(7, 11):
        cfg = dict(ARM, init_seed=9100 + s)
        out.append(dict(id=f'QH_T41_ev3e-4_s{s}', arm='ev3e-4', family='ev3e-4', depth=1, seed=s, init_seed=9100 + s, split_seed=0,
                        kind='one', threads=1, cfg=cfg,
                        paired_with=f'T41_ev1e-3_s{s}', est_core_hours=EST, worker='QH1', order=s - 7))
    if test:
        out = out[:2]
        for r in out:
            r['cfg'].update(p=11, width=16, steps=60, obs_every=10, heavy_every=30); r['est_core_hours'] = 0.01
    return out


if __name__ == '__main__':
    test = '--test' in sys.argv
    R = runs(test)
    man = dict(campaign='QH', stage="campaign H's arm eta_V = 3e-4, seeds 7-10", version='1.0' + ('-test' if test else ''),
               author='akashkumar-d', created='2026-10-13', workers=WORKERS, cores_per_worker=CORES, plan='PLAN.md',
               runner='t41_runner.py', runs=R, est_session_hours={'QH1': max(r['est_core_hours'] for r in R)})
    fn = os.path.join(HERE, 'manifest_test.json' if test else 'manifest.json')
    with open(fn, 'w') as f: json.dump(man, f, indent=1)
    print(f'wrote {fn}: {len(R)} runs, {sum(r["est_core_hours"] for r in R):.1f} core-hours')
