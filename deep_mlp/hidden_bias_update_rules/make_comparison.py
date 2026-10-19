import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def sha(fn): return hashlib.sha256(open(fn, 'rb').read()).hexdigest()


def entry(d, rid, campaign, archive):
    fn = os.path.join(d, rid, 'run.json.gz'); ch = json.load(open(os.path.join(d, rid, 'chain.json')))
    rec = ch['files']['run.json.gz']; h = sha(fn)
    assert h == rec['sha256'] and os.path.getsize(fn) == rec['bytes'], (rid, h, rec)
    return dict(id=rid, campaign=campaign, sha256=h, bytes=rec['bytes'], archive=archive, branch=ch.get('branch'),
                runner_sha256=ch.get('code', {}).get('runner'), host=ch.get('host'))


if __name__ == '__main__':
    J, TD, TF = sys.argv[1:4]
    seeds = []
    for i in range(1, 11):
        tw = (TD, 'D', 'campaign_D/results_archive/campaignD_H.tar') if i <= 2 else (TF, 'F', 'campaign_F/results_archive/campaignF_H.tar')
        seeds.append(dict(seed=i, init_seed=5000 + i, split_seed=6000 + i,
                          joint=entry(J, f'B_H_p31_lw0.0150_s{i}', 'I', 'campaign_I/results_archive/campaignI_H.tar'),
                          twin=entry(tw[0], f'H_p31_lw0.0150_s{i}', tw[1], tw[2])))
    out = dict(campaign='J', note='Paired references, fixed before any campaign-J evaluation run: joint = campaign I (bias trained '
               'with the joint [W | b] Polar step), twin = the bias-free paper run (campaigns D and F).', seeds=seeds)
    with open(os.path.join(HERE, 'comparison.json'), 'w') as f: json.dump(out, f, indent=1)
    print('wrote comparison.json:', len(seeds), 'seeds; every SHA-256 matches its chain.json')
