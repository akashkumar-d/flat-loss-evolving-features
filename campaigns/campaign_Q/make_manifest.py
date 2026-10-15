import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKERS = [f'Q{i}' for i in range(1, 25)]
CORES = 4
CONTENTION = 1.15
COMMON = dict(d_model=128, n_heads=4, d_mlp=512, dtype='float64', obs_every=20, heavy_every=100, probe_every=200, refit_tau=1e-3,
              eta_v=0.03, observers_fourier=True, final_norm='rms')
STOP = dict(acc=0.99, factor=1.5, extra=5000)
BASE = dict(p=113, n_layers=1, frac=0.2, optimizer='polar', hidden_mode='polar', lam_w=0.01, eta_w=0.1, lam_v=0.275, steps=30000)
G2 = dict(gain_init=2.0, eta_g=1.0 / 5000, lam_g=1.0)
LATE_AT = 12000

P_CELLS = (('M_p61_f0.55_lam_v0.45', dict(p=61, frac=0.55, lam_v=0.45, split_mode='mirror_free'), 1.45),
           ('P_p149_f0.16_lam_v0.23', dict(p=149, frac=0.16, lam_v=0.23), 1.15),
           ('P_p61_f0.33_lam_v0.45', dict(p=61, frac=0.33, lam_v=0.45), 0.45),
           ('P_p89_f0.25_lam_v0.33', dict(p=89, frac=0.25, lam_v=0.33), 0.40))
B_BASE = dict(n_layers=2, frac=0.2, lam_v=0.275, gain_init=2.0, eta_g=1e-4, lam_g=1.0)
B_HOURS = 1.8
B_CELLS = (('B_f0.2_lam_v0.275_tau10k', {}, range(16, 21)),
           ('B_f0.2_lam_v0.275_tau10k_F12', dict(late_at=LATE_AT, late=dict(eta_g=0.0)), range(11, 14)),
           ('B_f0.2_lam_v0.275_tau10k_E12', dict(late_at=LATE_AT, late=dict(eta_w=0.03)), range(11, 14)),
           ('B_f0.2_lam_v0.275_tau10k_V12', dict(late_at=LATE_AT, late=dict(eta_v=0.01)), range(11, 14)),
           ('B_f0.175_lam_v0.275_tau10k', dict(frac=0.175), range(11, 14)),
           ('B_f0.225_lam_v0.275_tau10k', dict(frac=0.225), range(11, 14)))
P_WORKER = {('M', 16): 'Q1', ('M', 17): 'Q1', ('M', 18): 'Q1', ('M', 19): 'Q1',
            ('M', 20): 'Q2', ('P149', 16): 'Q2', ('P149', 17): 'Q2', ('P149', 18): 'Q2',
            ('P149', 19): 'Q3', ('P149', 20): 'Q3', ('P61', 16): 'Q3', ('P61', 17): 'Q3', ('P89', 16): 'Q3', ('P89', 17): 'Q3',
            ('P61', 18): 'Q4', ('P61', 19): 'Q4', ('P61', 20): 'Q4', ('P89', 18): 'Q4', ('P89', 19): 'Q4', ('P89', 20): 'Q4'}


def short(cell):
    return 'M' if cell.startswith('M_') else 'P' + cell.split('_')[1][1:]


def runs(test=False):
    out = []
    for cell, ch, h in P_CELLS:
        for i in range(16, 21):
            cfg = dict(COMMON, **BASE); cfg.update(**G2); cfg.update(ch)
            cfg.update(init_seed=7100 + i, split_seed=8100 + i, stop_generalized=dict(STOP))
            out.append(dict(id=f'Q_{cell}_s{i}', setting=f'Q_{cell}', cell=cell, family=cell, part='P', depth=1, n_layers=1,
                            optimizer='polar', hidden_mode='polar', final_norm='rms', p=cfg['p'], seed=i, init_seed=7100 + i,
                            split_seed=8100 + i, kind='one', threads=1, cfg=cfg, est_h=h, worker=P_WORKER[(short(cell), i)]))
    k = 0
    for cell, ch, seeds in B_CELLS:
        for i in seeds:
            cfg = dict(COMMON, **BASE); cfg.update(**B_BASE); cfg.update(ch)
            cfg.update(init_seed=7100 + i, split_seed=8100 + i, stop_generalized=dict(STOP))
            out.append(dict(id=f'Q_{cell}_s{i}', setting=f'Q_{cell}', cell=cell, family=cell, part='B', depth=2, n_layers=2,
                            optimizer='polar', hidden_mode='polar', final_norm='rms', p=113, seed=i, init_seed=7100 + i,
                            split_seed=8100 + i, kind='one', threads=4, cfg=cfg, est_h=B_HOURS * 4, worker=WORKERS[4 + k]))
            k += 1
    assert k == len(WORKERS) - 4
    if test:
        keep = {'Q_P_p61_f0.33_lam_v0.45_s16', 'Q_M_p61_f0.55_lam_v0.45_s16', 'Q_B_f0.2_lam_v0.275_tau10k_F12_s11',
                'Q_B_f0.175_lam_v0.275_tau10k_s11'}
        out = [r for r in out if r['id'] in keep]
        for r in out:
            r['cfg'].update(p=11, d_model=16, n_heads=2, d_mlp=32, steps=60, obs_every=10, heavy_every=30, probe_every=30)
            if 'late_at' in r['cfg']: r['cfg']['late_at'] = 30
            r['est_h'] = 30 / 3600; r['threads'] = 1; r['worker'] = 'Q1' if r['part'] == 'P' else 'Q5'
    for r in out:
        r['est_core_hours'] = round(CONTENTION * r['est_h'], 3); del r['est_h']
    order = {}
    for r in sorted(out, key=lambda r: (-r['est_core_hours'], r['id'])):
        r['order'] = order.get(r['worker'], 0); order[r['worker']] = r['order'] + 1
    return out


def session_hours(R, w):
    free = [0.0] * CORES
    for r in sorted((r for r in R if r['worker'] == w), key=lambda r: r['order']):
        free.sort(); t = r['threads']; start = free[t - 1]
        free[:t] = [start + r['est_core_hours'] / t] * t
    return round(max(free), 2)


if __name__ == '__main__':
    test = '--test' in sys.argv
    R = runs(test)
    ids = [r['id'] for r in R]; assert len(ids) == len(set(ids))
    W = sorted({r['worker'] for r in R}, key=lambda w: int(w[1:])) if test else WORKERS
    man = dict(campaign='Q', stage='ten seeds per prime for the one-block across-primes figure; two-block exploration at p = 113',
               version='1.0' + ('-test' if test else ''), author='claude/1hl-figures', created='2026-10-13', workers=WORKERS,
               cores_per_worker=CORES, plan='PLAN.md', runner='tf_runner.py', runs=R,
               est_session_hours={w: session_hours(R, w) for w in W})
    fn = os.path.join(HERE, 'manifest_test.json' if test else 'manifest.json')
    with open(fn, 'w') as f: json.dump(man, f, indent=1)
    cells = {}
    for r in R: cells.setdefault(r['cell'], []).append(r)
    print(f'wrote {fn}: {len(R)} runs in {len(cells)} cells, {sum(r["est_core_hours"] for r in R):.1f} core-hours; '
          f'longest session {max(man["est_session_hours"].values())} h')
    for k, v in cells.items():
        print(f'  {k:32s} {len(v)} runs, seeds {[r["seed"] for r in v]}, workers {sorted({r["worker"] for r in v}, key=lambda w: int(w[1:]))}')
