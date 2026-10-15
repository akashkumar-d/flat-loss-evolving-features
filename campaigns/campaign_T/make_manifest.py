import itertools, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKERS = [f'T{i}' for i in range(1, 13)]
CORES = 4
CONTENTION = 1.15
P = 113
SEEDS = [1, 2]
COMMON = dict(p=P, d_model=128, n_heads=4, d_mlp=512, dtype='float32', obs_every=20, heavy_every=500, refit_tau=1e-3,
              stop_generalized=dict(acc=0.99, factor=1.5, extra=5000))
P1 = dict(lam_w=[0.01, 0.02, 0.03, 0.05], eta_w=[0.03, 0.1], lam_v=[0.2, 1.0, 3.0], frac=[0.3])
P1_HALF = dict(lam_w=[0.01, 0.02, 0.03], eta_w=[0.1], lam_v=[1.0, 3.0], frac=[0.5])
P2 = dict(lam_w=[0.01, 0.03], eta_w=[0.1], lam_v=[0.2, 1.0, 3.0], frac=[0.3])
ADAMW = dict(lr=1e-3, wd=1.0, betas=[0.9, 0.98])
STEPS = {1: 60000, 2: 60000}
SEC = {('polar', 1): 0.15, ('polar', 2): 0.30, ('adamw', 1): 0.12, ('adamw', 2): 0.25}
EXPECTED_STEPS = {'polar': 45000, 'adamw': 15000}


def runs(test=False):
    out = []
    def add(arm, L, opt, dial, frac):
        tag = '_'.join(f'{k}{v:g}' for k, v in dial.items() if not isinstance(v, list)) if opt == 'polar' else 'adamw'
        setting = f'T_{arm}_f{frac:g}_{tag}' if tag else f'T_{arm}_f{frac:g}'
        for i in SEEDS:
            cfg = dict(COMMON, n_layers=L, optimizer=opt, frac=frac, steps=STEPS[L], init_seed=7100 + i, split_seed=8100 + i, **dial)
            if opt == 'polar': cfg['eta_v'] = 0.03
            out.append(dict(id=f'{setting}_s{i}', setting=setting, arm=arm, family=arm, depth=L, n_layers=L, optimizer=opt, seed=i,
                            init_seed=7100 + i, split_seed=8100 + i, kind='one', threads=1, dial=dict(dial, frac=frac), cfg=cfg,
                            est_h=SEC[(opt, L)] * EXPECTED_STEPS[opt] / 3600))
    for arm, grid, L in (('P1', P1, 1), ('P1', P1_HALF, 1), ('P2', P2, 2)):
        for lw, ew, lv, fr in itertools.product(grid['lam_w'], grid['eta_w'], grid['lam_v'], grid['frac']):
            add(arm, L, 'polar', dict(lam_w=lw, eta_w=ew, lam_v=lv), fr)
    for arm, L in (('A1', 1), ('A2', 2)):
        for fr in (0.3, 0.5):
            add(arm, L, 'adamw', dict(ADAMW), fr)
    if test:
        keep = [r for r in out if r['seed'] == 1 and (r['arm'] in ('A1', 'A2') and r['dial']['frac'] == 0.3
                                                         or r['id'].startswith('T_P1_f0.3_lam_w0.03_eta_w0.1_lam_v1_')
                                                         or r['id'].startswith('T_P2_f0.3_lam_w0.03_eta_w0.1_lam_v1_'))]
        for r in keep:
            r['cfg'].update(p=11, d_model=16, n_heads=2, d_mlp=32, steps=60, obs_every=10, heavy_every=30); r['est_h'] = 30 / 3600
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
    man = dict(campaign='T', stage='development', version='1.0' + ('-test' if test else ''), author='claude/1hl-figures',
               created='2026-10-07', workers=WORKERS, cores_per_worker=CORES, plan='PLAN.md', runner='tf_runner.py', runs=R,
               est_session_hours={w: session_hours(R, w) for w in WORKERS})
    fn = os.path.join(HERE, 'manifest_test.json' if test else 'manifest.json')
    with open(fn, 'w') as f: json.dump(man, f, indent=1)
    print(f'wrote {fn}: {len(R)} runs, {sum(r["est_core_hours"] for r in R):.1f} core-hours; session hours:',
          sorted(man['est_session_hours'].values()))
