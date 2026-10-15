#!/usr/bin/env python3
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ZP_MANIFEST = os.path.join(HERE, '..', 'campaign_ZP', 'manifest.json')


def base():
    m = json.load(open(ZP_MANIFEST))
    r = [x for x in m['runs'] if x['id'] == 'ZP_P_p61_f0.33_lam_v0.45_s11'][0]
    c = dict(r['cfg']); [c.pop(k) for k in ('init_seed', 'split_seed', 'frac', 'steps')]
    return dict(c, split_mode='mirror_free')


def write(rid, cfg):
    os.makedirs(os.path.join(HERE, 'configs'), exist_ok=True)
    fn = os.path.join(HERE, 'configs', rid + '.json'); json.dump(cfg, open(fn, 'w'), indent=1, sort_keys=True); print(fn)


if __name__ == '__main__':
    b = base()
    if sys.argv[1] == 'pilot':
        for f in (0.33, 0.55): write(f'ZM_dev_f{f:g}_s1', dict(b, frac=f, init_seed=7101, split_seed=8101, steps=15000))
    elif sys.argv[1] == 'main':
        f = float(sys.argv[2])
        for s in range(11, 16): write(f'ZM_f{f:g}_s{s}', dict(b, frac=f, init_seed=7100 + s, split_seed=8100 + s, steps=30000))
