import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
IDS = {'R1': ['T_P2_f0.3_lam_w0.03_eta_w0.1_lam_v0.2_s2', 'T_P1_f0.3_lam_w0.03_eta_w0.1_lam_v3_s1', 'T_P1_f0.3_lam_w0.03_eta_w0.1_lam_v3_s2'],
       'R2': ['T_P1_f0.5_lam_w0.03_eta_w0.1_lam_v3_s1', 'T_P1_f0.5_lam_w0.03_eta_w0.1_lam_v3_s2']}
src = json.load(open(os.path.join(HERE, '..', 'campaign_T', 'manifest.json')))
by = {r['id']: r for r in src['runs']}
runs = []
for w, ids in IDS.items():
    for k, rid in enumerate(ids):
        r = dict(by[rid]); r['worker'] = w; r['order'] = k; r['original_worker'] = by[rid]['worker']; runs.append(r)
hours = {w: max(by[i]['est_core_hours'] for i in ids) for w, ids in IDS.items()}
man = dict(campaign='TR', stage='development (rerun of five campaign-T runs)', version='1.0', author='claude/1hl-figures',
           created='2026-10-08', workers=list(IDS), cores_per_worker=4, plan='README.md', runner='tf_runner.py',
           source_manifest_sha256=__import__('hashlib').sha256(open(os.path.join(HERE, '..', 'campaign_T', 'manifest.json'), 'rb').read()).hexdigest(),
           est_session_hours=hours, runs=runs)
json.dump(man, open(os.path.join(HERE, 'manifest.json'), 'w'), indent=1)
print('wrote manifest.json:', len(runs), 'runs;', hours)
