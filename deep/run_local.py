#!/usr/bin/env python3
import argparse, datetime, gzip, hashlib, json, math, os, platform, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = {'D': os.path.join(HERE, 'campaign_D', 'natplat_campaign.py'), 'F': os.path.join(HERE, 'campaign_F', 'natplat_campaign.py'),
          'E': os.path.join(HERE, 'campaign_E', 'tfm_runner.py'), 'G': os.path.join(HERE, 'campaign_G', 'natplat_campaign.py'),
          'H': os.path.join(HERE, 'campaign_H', 't41_runner.py')}
CAMPAIGNS = ['D', 'E', 'F', 'G', 'H']
DRIVER = 'campaigns D/E/F local driver 1.0'


def iso(): return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


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


def manifest(camp):
    with open(os.path.join(HERE, f'campaign_{camp}', 'manifest.json')) as f: return json.load(f)


def jobs_of(man): return man['chains'] if 'chains' in man else man['runs']


def select(man, ids=(), setting=None, stage=None):
    out = []
    for ch in jobs_of(man):
        if ids and ch['id'] not in ids: continue
        if setting and setting not in (ch.get('setting'), ch.get('cohort'), ch.get('cell'), ch.get('arm')): continue
        if stage and ch.get('stage') != stage: continue
        out.append(ch)
    return out


def host_info(threads):
    cpu = ''
    try: cpu = next(l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name'))
    except Exception: cpu = platform.processor()
    try: import numpy as np; npv = np.__version__
    except Exception: npv = None
    return dict(cpus=os.cpu_count(), cpu=cpu, python=platform.python_version(), numpy=npv, platform=platform.platform(),
                blas_threads=threads)


def first_cross(obs, thr):
    return next((o['step'] for o in obs if o.get('test_acc', 0) >= thr), None)


def run_job(camp, ch, out, threads, manifest_sha):
    wd = os.path.join(out, ch['id']); raw = os.path.join(wd, 'raw'); os.makedirs(raw, exist_ok=True)
    if os.path.exists(os.path.join(wd, 'chain.json')): return ch['id'], 'already done', 0.0
    cfg_fn = os.path.join(raw, 'cfg.json'); run_raw = os.path.join(raw, 'run.json.gz')
    with open(cfg_fn, 'w') as f: json.dump(ch['cfg'], f, indent=1, sort_keys=True)
    n = str(threads or ch.get('threads', 1)); env = dict(os.environ)
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'): env[v] = n
    started = iso(); t = time.time()
    subprocess.run([sys.executable, '-B', RUNNER[camp], 'one', cfg_fn, run_raw], check=True, env=env)
    with gzip.open(run_raw, 'rt') as f: d = json.load(f)
    obs = d['obs']; last = obs[-1]
    d['loss'] = r7(d['loss']); d['obs'] = r7(d['obs'])
    if camp == 'H': d['train_acc'] = r7(d['train_acc'])
    d['compact'] = 'floats rounded to 7 significant digits by run_local.py; full precision in raw/'
    dst = os.path.join(wd, 'run.json.gz')
    with gzip.open(dst + '.tmp', 'wt', compresslevel=9) as f: json.dump(d, f, separators=(',', ':'))
    os.replace(dst + '.tmp', dst)
    meta = {k: v for k, v in ch.items() if k not in ('cfg', 'worker', 'order', 'est_wall_hours', 'est_core_hours')}
    meta.update(version=DRIVER, done=True, started=started, finished=iso(), ran_by='run_local.py', threads=int(n),
                host=host_info(n), code=dict(runner=sha256(RUNNER[camp]), manifest=manifest_sha, driver=sha256(os.path.abspath(__file__))),
                files={'run.json.gz': dict(bytes=os.path.getsize(dst), sha256=sha256(dst))},
                run=dict(final_loss=d['loss'][-1], final_test=last.get('test_acc'), final_train=last.get('train_acc'),
                         horizon=ch['cfg'].get('steps'), steps=len(d['loss']) - 1, seconds=round(d.get('seconds', time.time() - t), 1),
                         resumed_from=d.get('resumed_from', 0), stop_reason=d.get('stop_reason'), stopped_at=d.get('stopped_at'),
                         t10=first_cross(obs, 0.1), t90=first_cross(obs, 0.9)))
    with open(os.path.join(wd, 'chain.json.tmp'), 'w') as f: json.dump(meta, f, indent=1, sort_keys=True)
    os.replace(os.path.join(wd, 'chain.json.tmp'), os.path.join(wd, 'chain.json'))
    return ch['id'], 'ok', time.time() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('list'); p.add_argument('campaign', choices=CAMPAIGNS)
    p = sub.add_parser('run'); p.add_argument('campaign', choices=CAMPAIGNS); p.add_argument('--out', required=True)
    p.add_argument('--id', nargs='*', default=[]); p.add_argument('--setting'); p.add_argument('--stage')
    p.add_argument('--workers', type=int, default=1); p.add_argument('--threads', type=int, default=None)
    a = ap.parse_args(); man = manifest(a.campaign)
    if a.cmd == 'list':
        groups = {}
        for ch in jobs_of(man):
            g = groups.setdefault((ch.get('stage'), ch.get('setting', ch.get('cohort', ch.get('cell', ch.get('arm'))))), [0, 0.0])
            g[0] += 1; g[1] += ch.get('est_core_hours', 0.0)
        for (stage, name), (k, h) in sorted(groups.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
            print(f'{stage or "-":6s} {name:32s} {k:4d} jobs  ~{h:7.1f} core-hours')
        print(f"total: {len(jobs_of(man))} jobs, ~{sum(c.get('est_core_hours', 0.0) for c in jobs_of(man)):.0f} core-hours (estimates)")
        return
    jobs = select(man, a.id, a.setting, a.stage)
    if not jobs: raise SystemExit('no job matches')
    msha = sha256(os.path.join(HERE, f'campaign_{a.campaign}', 'manifest.json'))
    os.makedirs(a.out, exist_ok=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for jid, msg, sec in ex.map(lambda ch: run_job(a.campaign, ch, a.out, a.threads, msha), jobs):
            print(f'{jid}: {msg} ({sec:.0f} s)', flush=True)


if __name__ == '__main__':
    main()
