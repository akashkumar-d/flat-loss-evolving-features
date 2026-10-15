import datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ['PLAN.md', 'manifest.json', 'make_manifest.py', 'bias_runner.py', 'tests/test_bias.py', 'tests/test_j.py', 'endpoints_i.py',
         'comparison.json', 'make_comparison.py', 'analyze_J.py', 'run_local.py']


def sha(f): return hashlib.sha256(open(os.path.join(HERE, f), 'rb').read()).hexdigest()


if __name__ == '__main__':
    fn = os.path.join(HERE, 'HASHES.json')
    if '--check' in sys.argv:
        H = json.load(open(fn)); bad = [f for f, h in H['files'].items() if sha(f) != h]
        print('all files match HASHES.json' if not bad else f'CHANGED: {bad}'); sys.exit(1 if bad else 0)
    H = dict(campaign='J', written=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), by='the authors',
             note='Fixed before any evaluation run: the plan, the runner and its tests, the manifest, the paired references and the '
                  'analysis code. endpoints_i.py is campaign I\'s file, unchanged.',
             files={f: sha(f) for f in FILES})
    with open(fn, 'w') as f: json.dump(H, f, indent=1)
    for f, h in H['files'].items(): print(f'{h}  {f}')
