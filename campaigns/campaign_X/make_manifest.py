import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKERS = [f'X{i}' for i in range(1, 6)]
CORES = 4
CONTENTION = 1.15
COMMON = dict(d_model=128, n_heads=4, d_mlp=512, dtype='float64', obs_every=20, heavy_every=100, probe_every=200, refit_tau=1e-3,
              eta_v=0.03, observers_fourier=True)
STOP = dict(acc=0.99, factor=1.5, extra=5000)
SEEDS = tuple(range(11, 21))
SEC_PER_UPDATE = 0.20
EXPECTED_STEPS = 14500

GRID = [('R5', 0.2, 'rms', dict(lam_w=0.01, eta_w=0.1, lam_v=0.275)),
        ('R4', 0.2, 'rms', dict(lam_w=0.01, eta_w=0.1, lam_v=0.25))]


def setting_id(arm, frac, fn, dial):
    return f'X_{arm}_f{frac:g}_{fn}_lam_w{dial["lam_w"]:g}_eta_w{dial["eta_w"]:g}_lam_v{dial["lam_v"]:g}'


def runs(test=False):
    out = []
    for arm, frac, fn, dial in GRID:
        st = setting_id(arm, frac, fn, dial)
        for i in SEEDS:
            cfg = dict(COMMON, p=113, n_layers=1, optimizer='polar', hidden_mode='polar', frac=frac, final_norm=fn, steps=30000,
                       init_seed=7100 + i, split_seed=8100 + i, stop_generalized=dict(STOP), **dial)
            out.append(dict(id=f'{st}_s{i}', setting=st, arm=arm, family=arm, depth=1, n_layers=1, optimizer='polar',
                            hidden_mode='polar', final_norm=fn, p=113, seed=i, init_seed=7100 + i, split_seed=8100 + i,
                            kind='one', threads=1, dial=dict(dial, frac=frac), cfg=cfg,
                            est_h=SEC_PER_UPDATE * EXPECTED_STEPS / 3600))
    if test:
        out = [r for r in out if r['seed'] == 11]
        for r in out:
            r['cfg'].update(p=11, d_model=16, n_heads=2, d_mlp=32, steps=60, obs_every=10, heavy_every=30, probe_every=30)
            r['est_h'] = 30 / 3600
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
    man = dict(campaign='X', stage='Fourier readouts on the transformer settings closest to the paper (R5, R4), 10 fresh seeds',
               version='1.0' + ('-test' if test else ''), author='claude/1hl-figures', created='2026-10-11', workers=WORKERS,
               cores_per_worker=CORES, plan='PLAN.md', runner='tf_runner.py', runs=R,
               est_session_hours={w: session_hours(R, w) for w in WORKERS})
    fn = os.path.join(HERE, 'manifest_test.json' if test else 'manifest.json')
    with open(fn, 'w') as f: json.dump(man, f, indent=1)
    print(f'wrote {fn}: {len(R)} runs, {sum(r["est_core_hours"] for r in R):.1f} core-hours; session hours:',
          sorted(man['est_session_hours'].values()))
