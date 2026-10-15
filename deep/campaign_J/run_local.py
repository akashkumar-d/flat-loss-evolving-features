import argparse, datetime, gzip, hashlib, json, os, platform, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def sha(fn): return hashlib.sha256(open(fn, 'rb').read()).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def host():
    cpu = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), platform.processor()) \
        if os.path.exists('/proc/cpuinfo') else platform.processor()
    import numpy
    return dict(cpu=cpu, cpus=os.cpu_count(), numpy=numpy.__version__, python=platform.python_version(), node=platform.node()[:8])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', type=int, default=2); ap.add_argument('--arm'); ap.add_argument('--out', default='runs')
    a = ap.parse_args()
    man_fn = os.path.join(HERE, 'manifest.json'); runner = os.path.join(HERE, 'bias_runner.py')
    man = json.load(open(man_fn)); todo = [r for r in sorted(man['runs'], key=lambda r: r['order']) if not a.arm or r['arm'] == a.arm]
    out_root = os.path.join(HERE, a.out); os.makedirs(out_root, exist_ok=True)
    code = dict(manifest=sha(man_fn), runner=sha(runner)); H = host()
    pending = []
    for r in todo:
        d = os.path.join(out_root, r['id'])
        if os.path.exists(os.path.join(d, 'run.json.gz')) and os.path.exists(os.path.join(d, 'record.json')): continue
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'cfg.json'), 'w') as f: json.dump(r['cfg'], f, indent=1)
        pending.append(r)
    print(f'{now()} {len(todo) - len(pending)} done, {len(pending)} to run, {a.jobs} at a time', flush=True)
    live = {}
    while pending or live:
        while pending and len(live) < a.jobs:
            r = pending.pop(0); d = os.path.join(out_root, r['id'])
            log = open(os.path.join(d, 'runner.log'), 'a')
            p = subprocess.Popen([sys.executable, '-I', runner, 'one', os.path.join(d, 'cfg.json'), os.path.join(d, 'run.json.gz')],
                                 stdout=log, stderr=subprocess.STDOUT)
            live[p] = (r, time.time(), now(), log); print(f'{now()} start {r["id"]}', flush=True)
        time.sleep(5)
        for p in [p for p in live if p.poll() is not None]:
            r, t0, started, log = live.pop(p); log.close(); d = os.path.join(out_root, r['id']); fn = os.path.join(d, 'run.json.gz')
            if p.returncode != 0 or not os.path.exists(fn):
                print(f'{now()} ERROR {r["id"]} exit {p.returncode} (see {d}/runner.log); will not retry automatically', flush=True); continue
            with gzip.open(fn, 'rt') as f: res = json.load(f)
            rec = dict(id=r['id'], arm=r['arm'], seed=r['seed'], init_seed=r['init_seed'], split_seed=r['split_seed'], joint=r['joint'],
                       twin=r['twin'], code=code, host=H, started=started, finished=now(), wall_seconds=round(time.time() - t0, 1),
                       runner_seconds=round(res['seconds'], 1), resumed_from=res.get('resumed_from'), steps=len(res['loss']) - 1,
                       stopped_at=res.get('stopped_at'), stop_reason=res.get('stop_reason'), bias_knee=res.get('bias_knee'),
                       final_test=res['obs'][-1]['test_acc'], file=dict(bytes=os.path.getsize(fn), sha256=sha(fn)))
            with open(os.path.join(d, 'record.json'), 'w') as f: json.dump(rec, f, indent=1)
            print(f'{now()} done {r["id"]} in {rec["wall_seconds"]:.0f} s (knee {rec["bias_knee"]}, stopped at {rec["stopped_at"]}, '
                  f'final held-out {rec["final_test"]:.3f})', flush=True)
    print(f'{now()} ALL DONE', flush=True)


if __name__ == '__main__':
    main()
