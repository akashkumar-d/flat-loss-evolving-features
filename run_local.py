#!/usr/bin/env python3
import argparse, datetime, gzip, hashlib, json, math, os, platform, shutil, subprocess, sys, tempfile, time

VERSION = 'run_local.py 1.0 (2026-10-13)'


def sha256(fn):
    h = hashlib.sha256()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def r7(x):
    if isinstance(x, float): return float(f'{x:.7g}') if math.isfinite(x) else x
    if isinstance(x, list): return [r7(v) for v in x]
    if isinstance(x, dict): return {k: r7(v) for k, v in x.items()}
    return x


def host_info():
    out = dict(python=platform.python_version(), platform=platform.platform(), machine=platform.machine())
    try:
        import numpy; out['numpy'] = numpy.__version__
    except Exception: out['numpy'] = None
    try:
        import torch; out['torch'] = torch.__version__
    except Exception: out['torch'] = None
    return out


def load_runs(camp, a):
    if a.selftest:
        cfg = json.load(open(os.path.join(camp, 'campaign.json')))['selftest_cfg']
        return [dict(id=f'selftest_{os.path.basename(camp)}', cfg=cfg, threads=1, kind='one')]
    if a.configs:
        return [dict(id=os.path.splitext(os.path.basename(fn))[0], cfg=json.load(open(fn)), threads=1, kind='one')
                for fn in a.configs]
    man = os.path.join(camp, 'manifest.json')
    if not os.path.exists(man): raise SystemExit(f'{camp} has no manifest.json; pass --configs (see its README)')
    runs = json.load(open(man))['runs']
    if a.ids: runs = [r for r in runs if r['id'] in set(a.ids)]
    if a.match: runs = [r for r in runs if all(m in r['id'] for m in a.match)]
    if not (a.ids or a.match or a.all or a.list): raise SystemExit('choose runs with --ids, --match or --all (or --list)')
    for r in runs:
        if r.get('kind', 'one') != 'one': raise SystemExit(f"{r['id']}: kind {r.get('kind')} is not supported here")
    return runs


def runner_of(camp):
    conf = os.path.join(camp, 'campaign.json')
    if os.path.exists(conf): return os.path.join(camp, json.load(open(conf))['runner'])
    for name in ('tf_runner.py', 'bias_runner.py'):
        if os.path.exists(os.path.join(camp, name)): return os.path.join(camp, name)
    raise SystemExit(f'no runner found in {camp}')


def start(run, runner, work, threads, ckpt_every):
    d = os.path.join(work, run['id']); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'cfg.json'), 'w') as f: json.dump(run['cfg'], f, indent=1)
    k = str(threads or max(1, int(run.get('threads', 1))))
    env = dict(os.environ, OMP_NUM_THREADS=k, OPENBLAS_NUM_THREADS=k, MKL_NUM_THREADS=k, VECLIB_MAXIMUM_THREADS=k,
               NUMEXPR_NUM_THREADS=k, CAMPAIGN_THREADS=k, CAMPAIGN_CKPT_EVERY=str(ckpt_every))
    log = open(os.path.join(d, 'runner.log'), 'a')
    args = [sys.executable, os.path.abspath(runner), 'one', os.path.join(d, 'cfg.json'), os.path.join(d, 'run.json.gz')]
    return subprocess.Popen(args, cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL), time.time()


def finalize(run, runner, camp, work, out, seconds):
    src = os.path.join(work, run['id'], 'run.json.gz')
    with gzip.open(src, 'rt') as f: M = json.load(f)
    assert json.dumps(M['cfg'], sort_keys=True) == json.dumps(run['cfg'], sort_keys=True), 'output does not match its configuration'
    M['loss'] = r7(M['loss']); M['obs'] = r7(M['obs'])
    M['compact'] = f'floats rounded to 7 significant digits by {VERSION}'
    dst = os.path.join(out, 'results', run['id']); os.makedirs(dst, exist_ok=True)
    with gzip.open(os.path.join(dst, 'run.json.gz.tmp'), 'wt', compresslevel=9) as f: json.dump(M, f, separators=(',', ':'))
    os.replace(os.path.join(dst, 'run.json.gz.tmp'), os.path.join(dst, 'run.json.gz'))
    man = os.path.join(camp, 'manifest.json')
    meta = {k: v for k, v in run.items() if k not in ('cfg', 'worker')}
    meta.update(version=VERSION, host=host_info(), seconds=round(seconds, 1),
                code=dict(runner=sha256(runner), manifest=sha256(man) if os.path.exists(man) else None),
                files={'run.json.gz': dict(sha256=sha256(os.path.join(dst, 'run.json.gz')))},
                finished=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), done=True)
    with open(os.path.join(dst, 'chain.json'), 'w') as f: json.dump(meta, f, indent=1)
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('campaign', help='experiment folder, e.g. transformer/one_block_other_primes')
    ap.add_argument('--ids', nargs='+'); ap.add_argument('--match', nargs='+', help='every text must occur in the run id')
    ap.add_argument('--all', action='store_true'); ap.add_argument('--list', action='store_true')
    ap.add_argument('--configs', nargs='+', help='configuration files instead of the manifest')
    ap.add_argument('--selftest', action='store_true', help="campaign.json's tiny configuration, into a temporary folder")
    ap.add_argument('--jobs', type=int, default=1); ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--out', help='default: <campaign>/collected'); ap.add_argument('--work', help='default: <campaign>/_work')
    ap.add_argument('--ckpt-every', type=int, default=2000); ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    camp = os.path.abspath(a.campaign); runner = runner_of(camp); runs = load_runs(camp, a)
    if a.list:
        for r in runs: print(r['id'], json.dumps(r.get('dial', {})), f"{r.get('est_core_hours', '?')} core-h")
        print(f'{len(runs)} runs'); return
    tmp = tempfile.mkdtemp(prefix='run_local_') if a.selftest else None
    out = os.path.abspath(a.out or (tmp if tmp else os.path.join(camp, 'collected')))
    work = os.path.abspath(a.work or (os.path.join(tmp, '_work') if tmp else os.path.join(camp, '_work')))
    todo = [r for r in runs if a.force or not os.path.exists(os.path.join(out, 'results', r['id'], 'run.json.gz'))]
    print(f'{VERSION}: {len(todo)} of {len(runs)} runs to do with {runner}; output in {out}', flush=True)
    live, failed = {}, []
    while todo or live:
        while todo and len(live) < max(1, a.jobs):
            r = todo.pop(0); p, t0 = start(r, runner, work, a.threads, a.ckpt_every); live[r['id']] = (r, p, t0)
            print(f"started {r['id']}", flush=True)
        time.sleep(1.0)
        for rid, (r, p, t0) in list(live.items()):
            if p.poll() is None: continue
            del live[rid]
            if p.returncode != 0:
                failed.append(rid); print(f'FAILED {rid} (exit {p.returncode}); see {os.path.join(work, rid, "runner.log")}', flush=True)
                continue
            dst = finalize(r, runner, camp, work, out, time.time() - t0)
            print(f'done {rid} -> {dst} ({time.time() - t0:.0f} s)', flush=True)
    if a.selftest and not failed:
        with gzip.open(os.path.join(out, 'results', runs[0]['id'], 'run.json.gz'), 'rt') as f: M = json.load(f)
        assert len(M['loss']) == runs[0]['cfg']['steps'] + 1 or M.get('stopped_at'), 'selftest: unexpected length'
        print(f"selftest ok: {len(M['loss']) - 1} updates, final loss {M['loss'][-1]:.4g}")
        shutil.rmtree(tmp, ignore_errors=True)
    if failed: raise SystemExit(f'{len(failed)} runs failed: {failed}')


if __name__ == '__main__':
    main()
