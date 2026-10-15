import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKERS = [f'V{i}' for i in range(1, 13)]
CORES = 4
CONTENTION = 1.15
COMMON = dict(p=113, d_model=128, n_heads=4, d_mlp=512, dtype='float32', obs_every=20, heavy_every=250, probe_every=500,
              refit_tau=1e-3)
STOP_POLAR = dict(acc=0.99, factor=1.5, extra=5000)
STOP_ADAMW = dict(acc=0.99, factor=1.5, extra=20000)
ADAMW = dict(lr=1e-3, betas=[0.9, 0.98], wd=1.0)
SEC = {('polar', 1): 0.13, ('polar', 2): 0.36, ('adamw', 1): 0.12, ('adamw', 2): 0.33}
EXPECTED = {'C1': 13000, 'C2': 40000, 'E1': 40000, 'E2': 40000, 'A1': 45000}
CONFIRM = tuple(range(6, 16)); DEV = (1, 2, 3)

GRID = (
    [('C1', 1, 0.3, 'none', 'polar', dict(lam_w=0.02, eta_w=0.1, lam_v=0.2), 30000, CONFIRM),
     ('C2', 2, 0.3, 'none', 'polar', dict(lam_w=0.01, eta_w=0.1, lam_v=1.0), 40000, CONFIRM)] +
    [('E1', 1, 0.2, 'rms', 'polar', dict(lam_w=0.01, eta_w=ew, lam_v=lv), 60000, DEV) for lv in (0.5, 0.75, 1.0) for ew in (0.03, 0.1)] +
    [('E2', 2, 0.2, 'rms', 'polar', dict(lam_w=0.01, eta_w=0.1, lam_v=lv), 40000, DEV) for lv in (0.5, 1.0)] +
    [('A1', 1, 0.2, fn, 'adamw', dict(ADAMW), 60000, DEV) for fn in ('none', 'rms')]
)


def setting_id(arm, frac, fn, opt, dial):
    norm = '_rms' if fn == 'rms' else ''
    if opt == 'adamw': return f'V_{arm}_f{frac:g}{norm}_adamw_wd{dial["wd"]:g}'
    return f'V_{arm}_f{frac:g}{norm}_lam_w{dial["lam_w"]:g}_eta_w{dial["eta_w"]:g}_lam_v{dial["lam_v"]:g}'


def runs(test=False):
    out = []
    for arm, L, frac, fn, opt, dial, steps, seeds in GRID:
        st = setting_id(arm, frac, fn, opt, dial)
        for i in seeds:
            cfg = dict(COMMON, n_layers=L, optimizer=opt, frac=frac, final_norm=fn, steps=steps, init_seed=7100 + i,
                       split_seed=8100 + i, stop_generalized=dict(STOP_ADAMW if opt == 'adamw' else STOP_POLAR), **dial)
            if opt == 'polar': cfg['eta_v'] = 0.03
            out.append(dict(id=f'{st}_s{i}', setting=st, arm=arm, family=arm, depth=L, n_layers=L, optimizer=opt, final_norm=fn,
                            seed=i, init_seed=7100 + i, split_seed=8100 + i, kind='one', threads=1,
                            dial={k: v for k, v in dict(dial, frac=frac).items() if k != 'betas'}, cfg=cfg,
                            est_h=SEC[(opt, L)] * min(EXPECTED[arm], steps) / 3600))
    if test:
        keep = [r for r in out if r['seed'] in (1, 6) and r['setting'] in (
            'V_C1_f0.3_lam_w0.02_eta_w0.1_lam_v0.2', 'V_C2_f0.3_lam_w0.01_eta_w0.1_lam_v1', 'V_E1_f0.2_rms_lam_w0.01_eta_w0.03_lam_v1',
            'V_E2_f0.2_rms_lam_w0.01_eta_w0.1_lam_v0.5', 'V_A1_f0.2_rms_adamw_wd1')]
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
    man = dict(campaign='V', stage='confirmation (C1, C2) and exploration (E1, E2, A1)', version='1.0' + ('-test' if test else ''),
               author='claude/1hl-figures', created='2026-10-09', workers=WORKERS, cores_per_worker=CORES, plan='PLAN.md',
               runner='tf_runner.py', runs=R, est_session_hours={w: session_hours(R, w) for w in WORKERS})
    fn = os.path.join(HERE, 'manifest_test.json' if test else 'manifest.json')
    with open(fn, 'w') as f: json.dump(man, f, indent=1)
    print(f'wrote {fn}: {len(R)} runs, {sum(r["est_core_hours"] for r in R):.1f} core-hours; session hours:',
          sorted(man['est_session_hours'].values()))
