import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKERS = [f'W{i}' for i in range(1, 11)]
CORES = 4
CONTENTION = 1.15
COMMON = dict(d_model=128, n_heads=4, d_mlp=512, dtype='float64', obs_every=20, heavy_every=250, probe_every=500, refit_tau=1e-3,
              eta_v=0.03)
STOP = dict(acc=0.99, factor=1.5, extra=5000)
DEV = (1, 2, 3)
def sec(p, frac, L): return (0.17 * (frac * p * p) / 2553 + 0.01) * (2.8 if L == 2 else 1.0)
EXPECTED = {'R1': 30000, 'R2': 25000, 'R3': 45000, 'R4': 12000, 'R5': 12000, 'N1': 30000, 'S1': 15000, 'P1': 14000}

GRID = (
    [('R1', 113, 1, 0.2, 'rms', 'polar', dict(lam_w=0.01, eta_w=0.03, lam_v=lv), 40000, DEV) for lv in (0.25, 0.3)] +
    [('R2', 113, 1, 0.15, 'rms', 'polar', dict(lam_w=0.01, eta_w=0.1, lam_v=lv), 40000, DEV) for lv in (0.3, 0.4, 0.5)] +
    [('R3', 113, 1, 0.1, 'rms', 'polar', dict(lam_w=0.01, eta_w=0.1, lam_v=lv), 60000, DEV) for lv in (0.4, 0.5)] +
    [('R4', 113, 1, 0.2, 'rms', 'polar', dict(lam_w=0.01, eta_w=0.1, lam_v=0.25), 30000, (3, 4, 5))] +
    [('R5', 113, 1, 0.2, 'rms', 'polar', dict(lam_w=0.01, eta_w=0.1, lam_v=0.275), 30000, DEV)] +
    [('N1', 113, 1, 0.2, 'rms', 'ngd', dict(lam_w=0.01, eta_w=0.1, lam_v=lv), 30000, DEV) for lv in (0.25, 0.5)] +
    [('S1', 31, 1, 0.5, 'rms', 'polar', dict(lam_w=0.01, eta_w=0.1, lam_v=lv), 30000, DEV) for lv in (0.5, 1.0)] +
    [('P1', 113, 1, 0.3, 'none', 'polar', dict(lam_w=0.02, eta_w=0.1, lam_v=0.2), 30000, (6, 7, 8))]
)


def setting_id(arm, p, frac, fn, mode, dial):
    norm = '_rms' if fn == 'rms' else ''
    pp = '' if p == 113 else f'_p{p}'
    hm = '_ngd' if mode == 'ngd' else ''
    return f'W_{arm}{pp}_f{frac:g}{norm}{hm}_lam_w{dial["lam_w"]:g}_eta_w{dial["eta_w"]:g}_lam_v{dial["lam_v"]:g}'


def runs(test=False):
    out = []
    for arm, p, L, frac, fn, mode, dial, steps, seeds in GRID:
        st = setting_id(arm, p, frac, fn, mode, dial)
        for i in seeds:
            cfg = dict(COMMON, p=p, n_layers=L, optimizer='polar', hidden_mode=mode, frac=frac, final_norm=fn, steps=steps,
                       init_seed=7100 + i, split_seed=8100 + i, stop_generalized=dict(STOP), **dial)
            out.append(dict(id=f'{st}_s{i}', setting=st, arm=arm, family=arm, depth=L, n_layers=L, optimizer='polar',
                            hidden_mode=mode, final_norm=fn, p=p, seed=i, init_seed=7100 + i, split_seed=8100 + i, kind='one',
                            threads=1, dial=dict(dial, frac=frac), cfg=cfg,
                            est_h=sec(p, frac, L) * min(EXPECTED[arm], steps) / 3600))
    if test:
        keep, seen = [], set()
        for r in out:
            if r['arm'] not in seen and r['seed'] in (1, 3, 6):
                seen.add(r['arm']); keep.append(r)
        for r in keep:
            r['cfg'].update(p=11, d_model=16, n_heads=2, d_mlp=32, steps=60, obs_every=10, heavy_every=30, probe_every=30)
            r['est_h'] = 30 / 3600
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
    man = dict(campaign='W', stage='exploration under the paper definitions (R1-R4, S1), NGD contrast (N1), float64 C1 (P1)',
               version='1.0' + ('-test' if test else ''), author='claude/1hl-figures', created='2026-10-10', workers=WORKERS,
               cores_per_worker=CORES, plan='PLAN.md', runner='tf_runner.py', runs=R,
               est_session_hours={w: session_hours(R, w) for w in WORKERS})
    fn = os.path.join(HERE, 'manifest_test.json' if test else 'manifest.json')
    with open(fn, 'w') as f: json.dump(man, f, indent=1)
    print(f'wrote {fn}: {len(R)} runs, {sum(r["est_core_hours"] for r in R):.1f} core-hours; session hours:',
          sorted(man['est_session_hours'].values()))
