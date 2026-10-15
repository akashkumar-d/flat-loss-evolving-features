#!/usr/bin/env python3
import os, sys


def _set_blas_threads():
    n = '1'
    if '--threads' in sys.argv:
        i = sys.argv.index('--threads')
        if i + 1 < len(sys.argv): n = sys.argv[i + 1]
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[v] = n


_set_blas_threads()

import argparse, datetime, gzip, hashlib, json, math, platform, shutil, tempfile, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import natplat_campaign as NC
import recipe as RC

VERSION = 'campaign-C local driver 1.0'
MANIFEST = os.path.join(HERE, 'manifest.json')
CKPT_EVERY = 2000


def iso(): return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha256(fn):
    h = hashlib.sha256()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def load_gz(fn):
    with gzip.open(fn, 'rt') as f: return json.load(f)


def write_json(fn, obj):
    tmp = fn + '.tmp'
    with open(tmp, 'w') as f: json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, fn)


def r7(x):
    if isinstance(x, float): return float(f'{x:.7g}') if math.isfinite(x) else x
    if isinstance(x, list): return [r7(v) for v in x]
    if isinstance(x, dict): return {k: r7(v) for k, v in x.items()}
    return x


def write_compact(src, dst):
    d = load_gz(src)
    d['loss'] = r7(d['loss']); d['obs'] = r7(d['obs'])
    d['compact'] = 'floats rounded to 7 significant digits by run_campaign_local.py; full precision in raw/'
    tmp = dst + '.tmp'
    with gzip.open(tmp, 'wt', compresslevel=9) as f: json.dump(d, f, separators=(',', ':'))
    os.replace(tmp, dst)


def cfg_key(c):
    c = {k: v for k, v in c.items() if k != 'stop_at_test'}
    return json.dumps(c, sort_keys=True)


def first_cross(obs, thr):
    return next((o['step'] for o in obs if o.get('test_acc', 0) >= thr), None)


def first_persistent(obs, thr):
    te = [(o['step'], o['test_acc']) for o in obs if 'test_acc' in o]
    last_bad = max((i for i, (_, v) in enumerate(te) if v < thr), default=-1)
    return te[last_bad + 1][0] if last_bad + 1 < len(te) else None


def host_info():
    cpu = ''
    try: cpu = next(l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name'))
    except Exception: cpu = platform.processor()
    import numpy as np
    return dict(cpus=os.cpu_count(), cpu=cpu, python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                blas_threads=os.environ.get('OPENBLAS_NUM_THREADS'))


def load_manifest():
    with open(MANIFEST) as f: man = json.load(f)
    man['_by_id'] = {c['id']: c for c in man['chains']}
    return man


def run_chain(ch, root, ckpt_every=CKPT_EVERY, max_steps=None):
    wd = os.path.join(root, ch['id']); raw = os.path.join(wd, 'raw'); os.makedirs(raw, exist_ok=True)
    done_fn = os.path.join(wd, 'chain.json')
    if os.path.exists(done_fn):
        with open(done_fn) as f: return json.load(f)
    R = ch['recipe']
    assert abs(R['K1'] - RC.K1) < 1e-15 and R['slope'] == RC.SLOPE and R['round'] == RC.ROUND and R['clip'] == [RC.R_MIN, RC.R_MAX], \
        'recipe constants in manifest.json differ from recipe.py'
    assert abs(R['eta_unit'] - RC.ETA_UNIT) < 1e-15
    pcfg = json.loads(json.dumps(ch['pilot']['cfg'])); tmpl = json.loads(json.dumps(ch['main']['cfg_template']))
    if max_steps:
        pcfg['steps'] = min(pcfg['steps'], max_steps); tmpl['steps'] = min(tmpl['steps'], max_steps)
    meta = dict(id=ch['id'], stage=ch['stage'], cohort=ch['cohort'], seed=ch['seed'], init_seed=ch['init_seed'],
                split_seed=ch['split_seed'], started=iso(), version=VERSION, host=host_info(),
                code=dict(runner=sha256(os.path.join(HERE, 'natplat_campaign.py')), recipe=sha256(os.path.join(HERE, 'recipe.py')),
                          manifest=sha256(MANIFEST)))
    if max_steps: meta['smoke'] = dict(max_steps=max_steps)
    pout = os.path.join(raw, 'pilot.json.gz')
    if not os.path.exists(pout): NC.run(pcfg, pout, ckpt_every)
    P = load_gz(pout)
    t10 = first_cross(P['obs'], 0.1)
    censored = t10 is None
    t10u = t10 if t10 is not None else pcfg['steps']
    rate = RC.solve_rate(t10u, R['r0'], R['x_target'], R['slope'])
    meta['pilot'] = dict(kind=ch['pilot']['kind'], rate=ch['pilot']['rate'], t10=t10, censored=censored, stopped_at=P.get('stopped_at'),
                         steps_run=len(P['loss']) - 1, seconds=round(P['seconds'], 1), resumed_from=P.get('resumed_from', 0),
                         final_test=P['obs'][-1]['test_acc'])
    meta['recipe'] = dict(x_target=R['x_target'], r0=R['r0'], t10_used=t10u, rate=rate, x_pred=round(rate * RC.K1 * t10u, 4))
    mcfg = tmpl; mcfg['eta_layers'][-1] = rate * RC.ETA_UNIT
    mout = os.path.join(raw, 'main.json.gz'); reused = False
    if not os.path.exists(mout):
        if ch['pilot']['kind'] == 'full' and 'stopped_at' not in P and cfg_key(mcfg) == cfg_key(pcfg):
            shutil.copy(pout, mout); reused = True
        else:
            NC.run(mcfg, mout, ckpt_every)
    M = load_gz(mout)
    reused = reused or (cfg_key(M['cfg']) == cfg_key(pcfg) and ch['pilot']['kind'] == 'full')
    assert cfg_key(M['cfg']) == cfg_key(mcfg), 'main output does not match its configuration'
    t10m = first_cross(M['obs'], 0.1); t90m = first_persistent(M['obs'], 0.9)
    meta['main'] = dict(rate=rate, eta_last=mcfg['eta_layers'][-1], steps=len(M['loss']) - 1, seconds=round(M['seconds'], 1),
                        resumed_from=M.get('resumed_from', 0), reused_pilot=reused, t10=t10m, t90=t90m,
                        x=round(rate * RC.K1 * t10m, 4) if t10m is not None else None, final_test=M['obs'][-1]['test_acc'],
                        final_train=M['obs'][-1]['train_acc'])
    files = {}
    for name, src in (('pilot.json.gz', pout), ('main.json.gz', mout)):
        dst = os.path.join(wd, name); write_compact(src, dst)
        files[name] = dict(sha256=sha256(dst), bytes=os.path.getsize(dst))
    meta['files'] = files; meta['finished'] = iso(); meta['done'] = True
    write_json(done_fn, meta)
    fail = os.path.join(wd, 'FAILED.json')
    if os.path.exists(fail): os.remove(fail)
    return meta


def _one(args):
    ch, root, ckpt_every, max_steps = args
    t = time.time()
    try:
        m = run_chain(ch, root, ckpt_every, max_steps)
        return ch['id'], 'ok', m, time.time() - t
    except Exception as e:
        os.makedirs(os.path.join(root, ch['id']), exist_ok=True)
        write_json(os.path.join(root, ch['id'], 'FAILED.json'), dict(error=repr(e), traceback=traceback.format_exc(), at=iso()))
        return ch['id'], 'FAILED ' + repr(e), None, time.time() - t


def replay(ch, data_dir, steps):
    fn = os.path.join(data_dir, ch['id'], 'pilot.json.gz')
    saved = load_gz(fn)
    cfg = json.loads(json.dumps(ch['pilot']['cfg'])); cfg.pop('stop_at_test', None); cfg['steps'] = steps
    tmp = tempfile.mkdtemp()
    t = time.time(); r = NC.run(cfg, os.path.join(tmp, 'replay.json.gz'), ckpt_every=0); dt = time.time() - t
    shutil.rmtree(tmp, ignore_errors=True)
    n = min(steps + 1, len(saved['loss']))
    a, b = r['loss'][:n], saved['loss'][:n]
    exact = sum(1 for x, y in zip(r7(a), b) if x == y)
    rel = max(abs(x - y) / abs(y) for x, y in zip(a, b) if y)
    print(f"{ch['id']}: replayed {n} loss values in {dt:.0f} s; {exact}/{n} equal to the saved values at 7 significant digits; "
          f"max relative difference {rel:.1e}")
    return exact == n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list', action='store_true', help='print the stages and cohorts with estimated core-hours, then exit')
    ap.add_argument('--chain', action='append', default=[], help='chain id (repeatable)')
    ap.add_argument('--stage', action='append', default=[], help='C1, C2, C3, C4 or C5 (repeatable)')
    ap.add_argument('--cohort', action='append', default=[], help='D3, D4, D5, D4p23, D4w512 or D3ngd (repeatable)')
    ap.add_argument('--all', action='store_true', help='every chain of the campaign')
    ap.add_argument('--out', default='runs_C', help='output folder, one subfolder per chain (default runs_C)')
    ap.add_argument('--workers', type=int, default=1, help='chains run in parallel (one process each)')
    ap.add_argument('--threads', type=int, default=1, help='BLAS threads per chain (read before numpy is imported)')
    ap.add_argument('--max-steps', type=int, default=None, help='truncate every run (smoke test; not comparable)')
    ap.add_argument('--ckpt-every', type=int, default=CKPT_EVERY)
    ap.add_argument('--replay', metavar='CHAIN', help='rerun the start of this chain\'s pilot and compare with --data')
    ap.add_argument('--data', help='folder with saved chains (<chain>/pilot.json.gz), for --replay')
    ap.add_argument('--steps', type=int, default=400, help='steps for --replay (default 400)')
    a = ap.parse_args()
    man = load_manifest(); by_id = man['_by_id']

    if a.list:
        rows = {}
        for c in man['chains']:
            k = (c['stage'], c['cohort']); rows.setdefault(k, [0, 0.0]); rows[k][0] += 1; rows[k][1] += c.get('est_core_hours', 0.0)
        print(f"{'stage':6s} {'cohort':8s} {'chains':>6s} {'est. core-h':>11s}")
        for (s, co), (n, h) in sorted(rows.items()): print(f'{s:6s} {co:8s} {n:6d} {h:11.1f}')
        print(f"{'':6s} {'total':8s} {sum(v[0] for v in rows.values()):6d} {sum(v[1] for v in rows.values()):11.1f}")
        return
    if a.replay:
        if not a.data: sys.exit('--replay needs --data')
        ok = replay(by_id[a.replay], a.data, a.steps)
        sys.exit(0 if ok else 1)

    if a.all: chains = list(man['chains'])
    else:
        chains = [by_id[c] for c in a.chain]
        if a.stage or a.cohort:
            chains += [c for c in man['chains'] if (not a.stage or c['stage'] in a.stage) and (not a.cohort or c['cohort'] in a.cohort)
                       and c['id'] not in a.chain]
    if not chains: sys.exit('nothing selected: use --chain, --stage/--cohort or --all (see --list)')
    chains.sort(key=lambda c: -c.get('est_core_hours', 0.0))
    os.makedirs(a.out, exist_ok=True)
    print(f'{len(chains)} chain(s) -> {os.path.abspath(a.out)}; workers {a.workers}, BLAS threads {a.threads}'
          + (f'; SMOKE TEST, max {a.max_steps} steps' if a.max_steps else ''), flush=True)
    jobs = [(c, a.out, a.ckpt_every, a.max_steps) for c in chains]
    failed = 0
    if a.workers <= 1:
        results = map(_one, jobs)
    else:
        import multiprocessing as mp
        pool = mp.get_context('spawn').Pool(a.workers)
        results = pool.imap_unordered(_one, jobs)
    for cid, msg, m, dt in results:
        extra = ''
        if m: extra = f"rate {m['recipe']['rate']}, final held-out {m['main']['final_test']:.3f}"
        print(f'{cid}: {msg} ({dt / 60:.1f} min) {extra}', flush=True)
        failed += msg != 'ok'
    if a.workers > 1: pool.close(); pool.join()
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
