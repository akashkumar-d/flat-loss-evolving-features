import glob, os, json, sys
import paperdefs_tf as P
ROOTS = {'T': '/mnt/user-data/uploads/modular arithmetic/natural_plateau_shared/campaign_T/collected/results',
         'U': '/mnt/user-data/uploads/modular arithmetic/natural_plateau_shared/campaign_U/collected/results',
         'V': '/home/claude/kt/campaign_V/collected/results'}
rows = []
for camp, root in ROOTS.items():
    for d in sorted(glob.glob(os.path.join(root, '*'))):
        fn = os.path.join(d, 'run.json.gz')
        if not os.path.exists(fn): continue
        r = P.load(fn); c = P.cfg_of(r)
        pr = P.primary(r); ph = P.posthoc(r); rw = P.raw10(r); ff = P.first_fit(r); prm = P.primary(r, ah='mlp')
        rows.append(dict(id=os.path.basename(d), camp=camp, opt=c['optimizer'], L=c.get('n_layers', 1), norm=c.get('final_norm', 'none'),
                         frac=c['frac'], lam_v=c.get('lam_v'), primary=pr.get('primary'), why=pr.get('why_not'), share=pr.get('share'),
                         v_raw=pr.get('v_raw'), dAH=pr.get('dAH'), dAH_mlp=prm.get('dAH'), fit=pr.get('fit_strict'), level=pr.get('CE_peak_over_logp'),
                         posthoc=ph.get('ok2'), g_te=ph.get('g_te'), g_rf=ph.get('g_rf'), raw10_fit=rw['fit'][0], raw10_nofit=rw['nofit'][0],
                         T_fit=ff.get('T_fit'), te_at_fit=ff.get('te_at_fit'), final=pr['final_test']))
json.dump(rows, open('/home/claude/kt/campaign_W/existing_paperdefs.json', 'w'), indent=1)
import collections
g = collections.defaultdict(list)
for r in rows:
    key = (r['opt'], r['L'], r['norm'], r['frac'], r['lam_v'] if r['opt'] == 'polar' else None)
    g[key].append(r)
print(f"{'setting':42s} {'n':>3s} {'primary':>7s} {'posthoc':>7s} {'raw10fit>=.75':>13s} {'raw10nofit>=.75':>15s} {'fit@IC':>6s} {'why (counts)'}")
for k in sorted(g, key=str):
    v = g[k]; n = len(v)
    why = collections.Counter(w for r in v for w in (r['why'] or '').split(', ') if w)
    print(f"{str(k):42s} {n:3d} {sum(bool(r['primary']) for r in v):7d} {sum(bool(r['posthoc']) for r in v):7d} "
          f"{sum(r['raw10_fit'] >= 0.75 for r in v):13d} {sum(r['raw10_nofit'] >= 0.75 for r in v):15d} {sum(bool(r['fit']) for r in v):6d} {dict(why)}")
