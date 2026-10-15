import json, hashlib, os, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
WORKERS = [f'S{i}' for i in range(1, 9)]
CORES = 4
P_OLD, P_NEW = 97, 101
SEEDS = range(1, 11)

SETTINGS = [
    ('H', 512, dict(lam_w=0.02)),
    ('D3', 256, dict(r=0.45)),
    ('D4', 256, dict(r=0.45)),
]
MEASURED_P97 = {'H': 1.06, 'D3': 1.88, 'D4': 2.46}
COST_101 = {'H': 1.135, 'D3': 1.105, 'D4': 1.105}


def d_id(fam, p, width, dial, s):
    if fam == 'H': return f'H_p{p}_lw{dial["lam_w"]:.4f}_s{s}'
    return f'{fam}_p{p}_r{dial["r"]:.2f}_s{s}'


def build():
    dman = json.load(open(os.path.join(HERE, '..', 'campaign_D', 'manifest.json')))
    dby = {c['id']: c for c in dman['chains']}
    E, S = [], []
    for fam, width, dial in SETTINGS:
        src = dby[d_id(fam, P_OLD, width, dial, 1)]
        assert src['p'] == P_OLD and src['width'] == width and src['dial'] == dial and src['cfg']['p'] == P_OLD, src['id']
        key = f'{fam}_p{P_NEW}' + (f'_lw{dial["lam_w"]:g}' if 'lam_w' in dial else f'_r{dial["r"]:g}')
        wall = round(MEASURED_P97[fam] * COST_101[fam], 3)
        ids = []
        for s in SEEDS:
            cfg = dict(src['cfg'], p=P_NEW, init_seed=5000 + s, split_seed=6000 + s)
            e = dict(id=d_id(fam, P_NEW, width, dial, s), stage=src['stage'], family=src['family'], cohort=f"{src['cohort'].split('_p')[0]}_p{P_NEW}",
                     setting=key, p=P_NEW, depth=src['depth'], width=width, dial=dial, seed=s, init_seed=5000 + s, split_seed=6000 + s, cfg=cfg,
                     threads=src['threads'], est_wall_hours=wall, est_core_hours=round(wall * src['threads'], 3), config_from=src['id'])
            assert e['id'] not in dby, e['id']
            E.append(e); ids.append(e['id'])
        S.append(dict(setting=key, family=fam, p=P_NEW, depth=src['depth'], width=width, dial=dial, source=f'p = {P_OLD} setting (campaigns D and F)',
                      config_from=src['id'], seed_rule='init 5000 + i, split 6000 + i', prior=[], new=ids,
                      compare_p97=f'{fam}_p{P_OLD}' + (f'_lw{dial["lam_w"]:g}' if 'lam_w' in dial else f'_r{dial["r"]:g}')))
    return E, S


def assign(E):
    load = {w: 0.0 for w in WORKERS}
    for e in sorted(E, key=lambda e: (-e['est_wall_hours'] * e['threads'], e['id'])):
        w = min(WORKERS, key=lambda w: (load[w], w))
        e['worker'] = w; load[w] += e['est_core_hours'] / CORES
    for w in WORKERS:
        q = sorted([e for e in E if e['worker'] == w], key=lambda e: (-e['threads'], -e['est_wall_hours'], e['id']))
        for i, e in enumerate(q): e['order'] = i
    return load


def main():
    E, S = build(); load = assign(E)
    ids = [e['id'] for e in E]; assert len(ids) == len(set(ids)), 'duplicate ids'
    man = dict(campaign='G', version=1, created=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
               author='the authors',
               kind=f'the p = {P_OLD} settings of campaign F, run unchanged at p = {P_NEW} with 10 fresh seeds each (PLAN.md)',
               workers=WORKERS, cores_per_worker=CORES, settings=S, est_session_hours={w: round(v, 2) for w, v in load.items()}, chains=E)
    fn = os.path.join(HERE, 'manifest.json')
    with open(fn, 'w') as f: json.dump(man, f, indent=1)
    tot = sum(e['est_core_hours'] for e in E)
    print(f"{len(E)} runs in {len(S)} settings, {tot:.0f} core-hours at the scaled durations; per session: "
          + ', '.join(f'{w} {v:.1f} h' for w, v in load.items()))
    by = {}
    for e in E: by.setdefault(e['setting'], [0, 0.0]); by[e['setting']][0] += 1; by[e['setting']][1] += e['est_core_hours']
    print('by setting: ' + ', '.join(f'{k} {v[0]} runs / {v[1]:.0f} core-h' for k, v in by.items()))
    print('threads: ' + ', '.join(f'{n}: {sum(1 for e in E if e["threads"] == n)}' for n in (1, 4)))
    print('sha256', hashlib.sha256(open(fn, 'rb').read()).hexdigest())


if __name__ == '__main__':
    main()
