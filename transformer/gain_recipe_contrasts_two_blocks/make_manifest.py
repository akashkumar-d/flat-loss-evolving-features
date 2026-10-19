import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OWNERS = [f'Z{i}' for i in range(1, 13)]
HELPERS = [f'Z{i}' for i in range(13, 25)]
WORKERS = OWNERS + HELPERS
CORES = 4
CONTENTION = 1.15
COMMON = dict(d_model=128, n_heads=4, d_mlp=512, dtype='float64', obs_every=20, heavy_every=100, probe_every=200, refit_tau=1e-3,
              eta_v=0.03, observers_fourier=True, final_norm='rms')
STOP = dict(acc=0.99, factor=1.5, extra=5000)
BASE = dict(p=113, n_layers=1, frac=0.2, optimizer='polar', hidden_mode='polar', lam_w=0.01, eta_w=0.1, lam_v=0.275, steps=30000)
SEC_1B = 0.19


def gain(g0, tau):
    return dict(gain_init=float(g0), eta_g=(1.0 / tau) if tau else 0.0, lam_g=1.0)


def lam_v_rule(p):
    return round(0.275 * math.sqrt(113) * math.log(113) / (math.sqrt(p) * math.log(p)), 3)


G2 = gain(2, 5000)
EXTRA_B_SEEDS = (14, 15)
ADAMW = dict(optimizer='adamw', lr=1e-3, wd=1.0, betas=[0.9, 0.98], eps=1e-8)
S10, S5, S3 = tuple(range(11, 21)), tuple(range(11, 16)), (11, 12, 13)

CELLS = [
    ('C_NGD_G2', dict(hidden_mode='ngd'), G2, S10, 30000, 1),
    ('C_ADAMW_G2', dict(ADAMW), G2, S10, 15000, 1),
    ('C_NGD_NOGAIN', dict(hidden_mode='ngd'), None, S5, 30000, 1),
    ('C_ADAMW_NOGAIN', dict(ADAMW), None, S5, 15000, 1),
]
for g0 in (1.5, 2, 3):
    for tau in (2500, 5000, 10000, 20000, None):
        if (g0, tau) == (2, 5000): continue
        CELLS.append((f'R_G{g0:g}_' + (f'tau{tau / 1000:g}k' if tau else 'fixed'), {}, gain(g0, tau), S3, 12000, 1))
for name, ov in (('R_lam_v0.25', dict(lam_v=0.25)), ('R_lam_v0.3', dict(lam_v=0.3)), ('R_eta_w0.05', dict(eta_w=0.05)),
                 ('R_eta_w0.2', dict(eta_w=0.2)), ('R_lam_w0.005', dict(lam_w=0.005)), ('R_lam_w0.02', dict(lam_w=0.02)),
                 ('R_frac0.3', dict(frac=0.3))):
    CELLS.append((name, ov, G2, S3, 12000, 1))
for p, fr, lvs, steps in ((61, 0.3, (0.45, 0.6), 10000), (89, 0.2, (0.33, 0.4), 12000), (149, 0.2, (0.23, 0.275), 15000)):
    for lv in lvs:
        CELLS.append((f'P_p{p}_f{fr:g}_lam_v{lv:g}', dict(p=p, frac=fr, lam_v=lv), G2, S5, steps, 1))
for fr, lv, tau, steps in ((0.3, 0.5, 5000, 12000), (0.3, 0.275, 5000, 12000), (0.25, 0.4, 10000, 20000), (0.2, 0.275, 10000, 25000)):
    CELLS.append((f'B_f{fr:g}_lam_v{lv:g}_tau{tau // 1000}k', dict(n_layers=2, frac=fr, lam_v=lv), gain(2, tau), S3 + EXTRA_B_SEEDS, steps, 2))


def sec_per_update(cfg, threads):
    s = SEC_1B * (cfg['p'] / 113) ** 2 * (cfg['frac'] / 0.2)
    if cfg['n_layers'] == 2: s *= 5.0 / (1.6 if threads >= 2 else 1.0)
    return s


def runs(test=False):
    out = []
    for cid, ov, g, seeds, steps, threads in CELLS:
        for i in seeds:
            cfg = dict(COMMON, **BASE); cfg.update(ov)
            if g: cfg.update(g)
            cfg.update(init_seed=7100 + i, split_seed=8100 + i, stop_generalized=dict(STOP))
            out.append(dict(id=f'Z_{cid}_s{i}', setting=f'Z_{cid}', cell=cid, family=cid, part=cid[0], depth=cfg['n_layers'],
                            n_layers=cfg['n_layers'], optimizer=cfg['optimizer'], hidden_mode=cfg['hidden_mode'], final_norm='rms',
                            p=cfg['p'], seed=i, init_seed=7100 + i, split_seed=8100 + i, kind='one', threads=threads,
                            dial={k: cfg[k] for k in ('p', 'n_layers', 'frac', 'optimizer', 'hidden_mode', 'eta_w', 'lam_w', 'lam_v')
                                  if k in cfg} | ({k: cfg[k] for k in ('gain_init', 'eta_g', 'lam_g')} if g else {}),
                            cfg=cfg, est_h=sec_per_update(cfg, threads) * steps / 3600))
    if test:
        keep = {'C_NGD_G2', 'C_ADAMW_G2', 'C_ADAMW_NOGAIN', 'R_G3_fixed', 'P_p61_f0.3_lam_v0.6', 'B_f0.3_lam_v0.5_tau5k'}
        out = [r for r in out if r['seed'] == 11 and r['cell'] in keep]
        for r in out:
            r['cfg'].update(p=11, d_model=16, n_heads=2, d_mlp=32, steps=60, obs_every=10, heavy_every=30, probe_every=30)
            r['est_h'] = 30 / 3600
    for r in out:
        r['est_core_hours'] = round(CONTENTION * r['est_h'] * r['threads'], 3); del r['est_h']
    load = {w: 0.0 for w in OWNERS}; order = {w: 0 for w in OWNERS}
    main = [r for r in out if r['seed'] not in EXTRA_B_SEEDS or r['part'] != 'B']
    for r in sorted(main, key=lambda r: (-r['est_core_hours'], r['id'])):
        w = min(OWNERS, key=lambda w: (load[w], OWNERS.index(w)))
        r['worker'] = w; r['order'] = order[w]; order[w] += 1; load[w] += r['est_core_hours']
    extra = sorted((r for r in out if r['part'] == 'B' and r['seed'] in EXTRA_B_SEEDS), key=lambda r: (r['seed'], r['id']))
    assert len(extra) <= len(HELPERS)
    for r, w in zip(extra, HELPERS):
        r['worker'] = w; r['order'] = 0
    return main + extra


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
    man = dict(campaign='Z', stage='optimizer contrasts with the gain, the one-block recipe, other p, two blocks',
               version='1.0' + ('-test' if test else ''), author='akashkumar-d', created='2026-10-12', workers=WORKERS,
               cores_per_worker=CORES, plan='PLAN.md', runner='tf_runner.py', runs=R,
               est_session_hours={w: session_hours(R, w) for w in WORKERS})
    fn = os.path.join(HERE, 'manifest_test.json' if test else 'manifest.json')
    with open(fn, 'w') as f: json.dump(man, f, indent=1)
    cells = {}
    for r in R: cells.setdefault(r['cell'], []).append(r)
    print(f'wrote {fn}: {len(R)} runs in {len(cells)} cells, {sum(r["est_core_hours"] for r in R):.1f} core-hours; session hours:',
          sorted(man['est_session_hours'].values()))
    if not test:
        for k, v in cells.items(): print(f'  {k:28s} {len(v):3d} runs  {sum(r["est_core_hours"] for r in v):6.1f} core-h  threads {v[0]["threads"]}')
