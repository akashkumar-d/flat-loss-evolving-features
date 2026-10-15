#!/usr/bin/env python3
import argparse, datetime, hashlib, json, os, platform, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, 'bias_runner.py')
OUTS = {'one': ['run.json.gz'], 'chain': ['pilot.json.gz', 'main.json.gz']}


def sha(fn): return hashlib.sha256(open(fn, 'rb').read()).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def host():
    import numpy
    return dict(cpus=os.cpu_count(), machine=platform.machine(), numpy=numpy.__version__, python=platform.python_version())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list', action='store_true'); ap.add_argument('--id', action='append')
    ap.add_argument('--family', choices=['H', 'D2', 'D4']); ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--out', default='runs')
    a = ap.parse_args()
    man_fn = os.path.join(HERE, 'manifest.json'); man = json.load(open(man_fn))
    jobs = sorted(man['runs'], key=lambda r: (r['order'], r['id']))
    if a.id: jobs = [r for r in jobs if r['id'] in set(a.id)]
    if a.family: jobs = [r for r in jobs if r['family'] == a.family]
    if a.list or not (a.id or a.family):
        for r in jobs: print(f"{r['id']:28s} {r['kind']:5s} family {r['family']:2s} paired with {r['paired_with']:22s} ~{r['est_core_hours']:.2f} core-h")
        print(f"{len(jobs)} jobs, about {sum(r['est_core_hours'] for r in jobs):.1f} core-hours")
        return
    out_root = os.path.abspath(a.out); os.makedirs(out_root, exist_ok=True)
    code = dict(runner=sha(RUNNER), manifest=sha(man_fn)); H = host(); pending = []
    for r in jobs:
        d = os.path.join(out_root, r['id']); os.makedirs(d, exist_ok=True)
        if all(os.path.exists(os.path.join(d, f)) for f in OUTS[r['kind']] + ['record.json']): continue
        name = 'cfg.json' if r['kind'] == 'one' else 'spec.json'
        with open(os.path.join(d, name), 'w') as f: json.dump(r['cfg'] if r['kind'] == 'one' else r['spec'], f, indent=1)
        pending.append(r)
    print(f'{now()} {len(jobs) - len(pending)} done, {len(pending)} to run, {a.jobs} at a time', flush=True)
    live = {}
    while pending or live:
        while pending and len(live) < a.jobs:
            r = pending.pop(0); d = os.path.join(out_root, r['id'])
            args = (['one', os.path.join(d, 'cfg.json'), os.path.join(d, 'run.json.gz')] if r['kind'] == 'one'
                    else ['chain', os.path.join(d, 'spec.json'), d])
            log = open(os.path.join(d, 'runner.log'), 'a')
            p = subprocess.Popen([sys.executable, '-I', RUNNER, *args], cwd=d, stdout=log, stderr=subprocess.STDOUT)
            live[p] = (r, time.time(), now(), log); print(f'{now()} start {r["id"]}', flush=True)
        time.sleep(5)
        for p in [p for p in live if p.poll() is not None]:
            r, t0, started, log = live.pop(p); log.close(); d = os.path.join(out_root, r['id'])
            outs = [os.path.join(d, f) for f in OUTS[r['kind']]]
            if p.returncode != 0 or not all(os.path.exists(f) for f in outs):
                print(f'{now()} ERROR {r["id"]} exit {p.returncode} (see {d}/runner.log); not retried', flush=True); continue
            rec = dict(id=r['id'], kind=r['kind'], family=r['family'], seed=r['seed'], init_seed=r['init_seed'],
                       split_seed=r['split_seed'], paired_with=r['paired_with'], code=code, host=H, started=started, finished=now(),
                       wall_seconds=round(time.time() - t0, 1),
                       files={os.path.basename(f): dict(bytes=os.path.getsize(f), sha256=sha(f)) for f in outs})
            with open(os.path.join(d, 'record.json'), 'w') as f: json.dump(rec, f, indent=1)
            print(f'{now()} done {r["id"]} in {rec["wall_seconds"]:.0f} s', flush=True)
    print(f'{now()} ALL DONE', flush=True)


if __name__ == '__main__':
    main()
