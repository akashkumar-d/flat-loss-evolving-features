import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKERS = [f'ZP{i}' for i in range(1, 9)]
CORES = 4
CONTENTION = 1.15
COMMON = dict(d_model=128, n_heads=4, d_mlp=512, dtype='float64', obs_every=20, heavy_every=100, probe_every=200, refit_tau=1e-3,
              eta_v=0.03, observers_fourier=True, final_norm='rms')
STOP = dict(acc=0.99, factor=1.5, extra=5000)
BASE = dict(p=113, n_layers=1, frac=0.2, optimizer='polar', hidden_mode='polar', lam_w=0.01, eta_w=0.1, lam_v=0.275, steps=30000)
G2 = dict(gain_init=2.0, eta_g=1.0 / 5000, lam_g=1.0)
SEEDS = tuple(range(11, 16))
SEC_1B = 0.19

CELLS = ((61, 0.45, (0.33, 0.35), 20000), (89, 0.33, (0.25, 0.275), 14000), (149, 0.23, (0.15, 0.16), 15000))


def timing(p, frac):
    n = int(frac * p * p); x = n / (p * math.log(p))
    return n, x, (2300 / (x - 3.9) if x > 3.9 else float('inf'))


def runs(test=False):
    out = []
    for p, lv, fracs, steps in CELLS:
        for fr in fracs:
            cid = f'P_p{p}_f{fr:g}_lam_v{lv:g}'
            for i in SEEDS:
                cfg = dict(COMMON, **BASE); cfg.update(p=p, frac=fr, lam_v=lv, **G2)
                cfg.update(init_seed=7100 + i, split_seed=8100 + i, stop_generalized=dict(STOP))
                n, x, a = timing(p, fr)
                out.append(dict(id=f'ZP_{cid}_s{i}', setting=f'ZP_{cid}', cell=cid, family=cid, part='P', depth=1, n_layers=1,
                                optimizer='polar', hidden_mode='polar', final_norm='rms', p=p, seed=i, init_seed=7100 + i,
                                split_seed=8100 + i, kind='one', threads=1,
                                dial={k: cfg[k] for k in ('p', 'n_layers', 'frac', 'optimizer', 'hidden_mode', 'eta_w', 'lam_w', 'lam_v',
                                                          'gain_init', 'eta_g', 'lam_g')},
                                predicted=dict(n_train=n, x=round(x, 3), rise_start=round(a)), cfg=cfg,
                                est_h=SEC_1B * (p / 113) ** 2 * (fr / 0.2) * steps / 3600))
    if test:
        keep = {'P_p61_f0.33_lam_v0.45', 'P_p149_f0.16_lam_v0.23'}
        out = [r for r in out if r['seed'] in (11, 12) and r['cell'] in keep]
        for r in out:
            r['cfg'].update(p=11, d_model=16, n_heads=2, d_mlp=32, steps=60, obs_every=10, heavy_every=30, probe_every=30)
            r['est_h'] = 30 / 3600
    for r in out:
        r['est_core_hours'] = round(CONTENTION * r['est_h'], 3); del r['est_h']
    load = {w: 0.0 for w in WORKERS}; order = {w: 0 for w in WORKERS}
    for r in sorted(out, key=lambda r: (-r['est_core_hours'], r['id'])):
        w = min(WORKERS, key=lambda w: (load[w], WORKERS.index(w)))
        r['worker'] = w; r['order'] = order[w]; order[w] += 1; load[w] += r['est_core_hours']
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
    man = dict(campaign='ZP', stage="other p: campaign Z's P cells with the training fraction set per p",
               version='1.0' + ('-test' if test else ''), author='claude/1hl-figures', created='2026-10-12', workers=WORKERS,
               cores_per_worker=CORES, plan='PLAN.md', runner='tf_runner.py', runs=R,
               est_session_hours={w: session_hours(R, w) for w in WORKERS})
    fn = os.path.join(HERE, 'manifest_test.json' if test else 'manifest.json')
    with open(fn, 'w') as f: json.dump(man, f, indent=1)
    cells = {}
    for r in R: cells.setdefault(r['cell'], []).append(r)
    print(f'wrote {fn}: {len(R)} runs in {len(cells)} cells, {sum(r["est_core_hours"] for r in R):.1f} core-hours; session hours:',
          sorted(man['est_session_hours'].values()))
    if not test:
        for k, v in cells.items():
            pr = v[0]['predicted']
            print(f'  {k:24s} {len(v)} runs {sum(r["est_core_hours"] for r in v):5.1f} core-h | n_train {pr["n_train"]}, x {pr["x"]:.2f}, '
                  f'predicted rise start {pr["rise_start"]} | workers {sorted({r["worker"] for r in v})}')
