#!/usr/bin/env python3
import argparse, datetime, gzip, hashlib, json, math, os, platform, secrets, shutil, signal, subprocess, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, 'campaign.json')) as _f: CONF = json.load(_f)
CAMPAIGN = CONF['campaign']
VERSION = f'campaign-{CAMPAIGN} worker 2.0 (2026-10-07)'
BASE_BRANCH = CONF['base_branch']
BRANCH_PREFIX = os.environ.get('CAMPAIGN_BRANCH_PREFIX', 'claude/')
RUNNER = os.path.join(HERE, CONF['runner'])
CKPT_SUFFIX = CONF.get('ckpt_suffix', '.ckpt.npz')
WORK = os.environ.get('CAMPAIGN_WORK', os.path.join(HERE, '_work'))
RESULTS = os.path.join(HERE, 'results')
STATUS = os.path.join(HERE, 'status')
DEFAULT_MANIFEST = os.path.join(HERE, 'manifest.json')
CKPT_EVERY = int(os.environ.get('CAMPAIGN_CKPT_EVERY', 2000))
HEARTBEAT_S = int(os.environ.get('CAMPAIGN_HEARTBEAT_S', 1800))
STALE_S = int(os.environ.get('CAMPAIGN_STALE_S', 5400))
FETCH_MIN_S = int(os.environ.get('CAMPAIGN_FETCH_MIN_S', 300))
RETRY_PUSH_S = int(os.environ.get('CAMPAIGN_RETRY_PUSH_S', 300))
LOOP_S = float(os.environ.get('CAMPAIGN_LOOP_S', 15))
STEAL_GRACE_S = int(os.environ.get('CAMPAIGN_STEAL_GRACE_S', 900))
CLAIM_WAIT_S = float(os.environ.get('CAMPAIGN_CLAIM_WAIT_S', 60))
MAX_ATTEMPTS = 3
GIT_TIMEOUT = 180


def now(): return time.time()
def iso(t=None): return datetime.datetime.fromtimestamp(now() if t is None else t, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
def parse_iso(s):
    try: return datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()
    except Exception: return 0.0
def sha256(fn):
    h = hashlib.sha256()
    with open(fn, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''): h.update(blk)
    return h.hexdigest()
def read_json(fn, default=None):
    try:
        with open(fn) as f: return json.load(f)
    except Exception: return default
def write_json(fn, obj):
    os.makedirs(os.path.dirname(fn), exist_ok=True); tmp = fn + '.tmp'
    with open(tmp, 'w') as f: json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, fn)
def load_gz(fn):
    with gzip.open(fn, 'rt') as f: return json.load(f)
def log(msg):
    os.makedirs(WORK, exist_ok=True)
    with open(os.path.join(WORK, 'daemon.log'), 'a') as f: f.write(f'{iso()} {msg}\n')
def tail(fn, n=30):
    try:
        with open(fn) as f: return ''.join(f.readlines()[-n:])
    except Exception: return ''

def effective_cores():
    n = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    try:
        q, per = open('/sys/fs/cgroup/cpu.max').read().split()[:2]
        if q != 'max': n = min(n, max(1, int(int(q) // int(per))))
    except Exception: pass
    return max(1, n)

def pid_alive(pid):
    try: os.kill(pid, 0)
    except ProcessLookupError: return False
    except PermissionError: return True
    try:
        st = open(f'/proc/{pid}/stat').read().split(')')[-1].split()[0]
        if st == 'Z': return False
    except Exception: pass
    return True

def cmdline(pid):
    try: return open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='replace')
    except Exception: return ''

def r7(x):
    if isinstance(x, float): return float(f'{x:.7g}') if math.isfinite(x) else x
    if isinstance(x, list): return [r7(v) for v in x]
    if isinstance(x, dict): return {k: r7(v) for k, v in x.items()}
    return x


def load_manifest(path):
    m = read_json(path)
    if not m: raise SystemExit(f'cannot read the manifest {path}')
    m['_path'] = path; m['_sha'] = sha256(path)
    m['_by_id'] = {c['id']: c for c in m['runs']}
    m['_queue'] = {}
    for c in m['runs']: m['_queue'].setdefault(c['worker'], []).append(c)
    for w in m['_queue']: m['_queue'][w].sort(key=lambda c: c['order'])
    return m

def check_worker_id(man, W):
    if W not in man['workers']:
        raise SystemExit(f"ERROR: unknown worker ID '{W}'. Use one of {', '.join(man['workers'])} exactly as written in your prompt.")

def check_clone_owner(W):
    os.makedirs(WORK, exist_ok=True); fn = os.path.join(WORK, 'WORKER_ID')
    if os.path.exists(fn):
        old = open(fn).read().strip()
        if old and old != W: raise SystemExit(f'ERROR: this clone already runs worker {old}; it cannot also run {W}. Use --worker {old}, or tell the user.')
    else:
        open(fn, 'w').write(W)

def instance_id():
    fn = os.path.join(WORK, 'INSTANCE')
    if not os.path.exists(fn):
        os.makedirs(WORK, exist_ok=True); open(fn, 'w').write(secrets.token_hex(3))
    return open(fn).read().strip()


class GitError(Exception): pass

class Git:
    def __init__(self, here):
        r = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=here, capture_output=True, text=True, timeout=60)
        if r.returncode != 0: raise GitError('not a git clone: ' + r.stderr.strip())
        self.root = r.stdout.strip()
        self.rel = os.path.relpath(here, self.root)
        self.branch = self.run('rev-parse', '--abbrev-ref', 'HEAD').stdout.strip()
        self._fetched = 0.0; self._remote = {}

    def run(self, *args, check=True, timeout=GIT_TIMEOUT):
        env = dict(os.environ, GIT_TERMINAL_PROMPT='0')
        r = subprocess.run(['git', *args], cwd=self.root, capture_output=True, text=True, timeout=timeout, env=env)
        if check and r.returncode != 0: raise GitError(f"git {' '.join(args[:3])}: {(r.stderr or r.stdout).strip()[-400:]}")
        return r

    @staticmethod
    def identity_args():
        return ['-c', f'user.name=campaign-{CAMPAIGN} worker', '-c', f'user.email=campaign-{CAMPAIGN.lower()}-worker@users.noreply.github.com']

    def commit_push(self, paths, msg):
        paths = [os.path.relpath(p, self.root) for p in paths if os.path.exists(p)]
        if paths:
            self.run('add', '-f', '--', *paths)
            st = self.run('diff', '--cached', '--name-only', '--', *paths).stdout.strip()
            if st: self.run(*self.identity_args(), 'commit', '-q', '-m', msg, '--', *paths)
        r = self.run('push', '-q', 'origin', f'HEAD:refs/heads/{self.branch}', check=False)
        if r.returncode != 0 and any(k in (r.stderr or '').lower() for k in ('non-fast-forward', 'fetch first', 'rejected')):
            self.run('fetch', '--no-tags', '--quiet', 'origin', f'+refs/heads/{self.branch}:refs/remotes/origin/{self.branch}', check=False)
            self.run('-c', 'rebase.autoStash=true', *self.identity_args(), 'rebase', '-X', 'theirs', f'origin/{self.branch}', check=False)
            r = self.run('push', '-q', 'origin', f'HEAD:refs/heads/{self.branch}', check=False)
        return r.returncode == 0, (r.stderr or r.stdout).strip()[-400:]

    def remote_state(self, force=False):
        if not force and now() - self._fetched < FETCH_MIN_S: return self._remote
        r = self.run('fetch', '--no-tags', '--quiet', 'origin', f'+refs/heads/{BRANCH_PREFIX}*:refs/remotes/origin/{BRANCH_PREFIX}*', check=False)
        if r.returncode != 0 and self._remote: return self._remote
        refs = self.run('for-each-ref', '--format=%(refname:short)', f'refs/remotes/origin/{BRANCH_PREFIX}').stdout.split()
        out = {}
        for ref in refs:
            ls = self.run('ls-tree', '-r', '--name-only', ref, '--', f'{self.rel}/results', f'{self.rel}/status', check=False).stdout.split('\n')
            done = {f.split('/')[-2] for f in ls if f.endswith('/chain.json') and '/results/' in f}
            stats = []
            for f in ls:
                if '/status/' in f and f.endswith('.json') and not f.endswith('-selftest.json'):
                    s = self.run('show', f'{ref}:{f}', check=False).stdout
                    try: stats.append(json.loads(s))
                    except Exception: pass
            if done or stats: out[ref[len('origin/'):]] = dict(done=done, status=stats)
        self._remote = out; self._fetched = now()
        return out


def host_info():
    cpu = ''
    try: cpu = next(l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name'))
    except Exception: pass
    import numpy as np
    out = dict(node=hashlib.sha256(platform.node().encode()).hexdigest()[:8], cpus=effective_cores(), cpu=cpu,
               python=platform.python_version(), numpy=np.__version__)
    if CONF.get('needs_torch'):
        try:
            import torch; out['torch'] = torch.__version__
        except Exception: out['torch'] = None
    return out

def run_dir(rid): return os.path.join(WORK, 'runs', rid)

OUTS = {'one': ['run.json.gz'], 'chain': ['pilot.json.gz', 'main.json.gz']}

def kind_of(rid):
    return 'chain' if os.path.exists(os.path.join(run_dir(rid), 'spec.json')) else 'one'

def outputs_done(rid):
    d = run_dir(rid); return all(os.path.exists(os.path.join(d, f)) for f in OUTS[kind_of(rid)])

def threads_of(ch, cores):
    return max(1, min(int(ch.get('threads', 1)), cores))

def start_run(ch, cores=1):
    d = run_dir(ch['id']); os.makedirs(d, exist_ok=True); k = str(threads_of(ch, cores))
    env = dict(os.environ, OMP_NUM_THREADS=k, OPENBLAS_NUM_THREADS=k, MKL_NUM_THREADS=k, VECLIB_MAXIMUM_THREADS=k,
               NUMEXPR_NUM_THREADS=k, CAMPAIGN_THREADS=k, CAMPAIGN_CKPT_EVERY=str(CKPT_EVERY))
    if ch['kind'] == 'chain':
        write_json(os.path.join(d, 'spec.json'), ch['spec'])
        args = [sys.executable, RUNNER, 'chain', os.path.join(d, 'spec.json'), d]
    else:
        write_json(os.path.join(d, 'cfg.json'), ch['cfg'])
        args = [sys.executable, RUNNER, 'one', os.path.join(d, 'cfg.json'), os.path.join(d, 'run.json.gz')]
    lf = open(os.path.join(d, 'runner.log'), 'a')
    return subprocess.Popen(args, cwd=d, env=env, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)

class Adopted:
    def __init__(self, pid, rid): self.pid, self.rid, self.returncode = pid, rid, None
    def poll(self):
        if pid_alive(self.pid): return None
        self.returncode = 0 if outputs_done(self.rid) else 1
        return self.returncode

def find_runner(rid):
    key = run_dir(rid) + os.sep
    for d in os.listdir('/proc'):
        if d.isdigit() and int(d) != os.getpid():
            c = cmdline(int(d))
            if key in c and os.path.basename(RUNNER) in c and pid_alive(int(d)): return int(d)
    return None

def run_progress(rid):
    d = run_dir(rid)
    if os.path.exists(os.path.join(d, 'chain.json')): return 'done', None
    if outputs_done(rid): return 'finished', None
    def ck(name):
        side = os.path.join(d, name + '.ckpt.json')
        if os.path.exists(side): return int(read_json(side, {}).get('t_next', 0))
        fn = os.path.join(d, name + CKPT_SUFFIX)
        if not os.path.exists(fn) or not fn.endswith('.npz'): return 0
        try:
            import numpy as np
            with np.load(fn, allow_pickle=False) as z: return int(z['t_next'])
        except Exception: return 0
    if kind_of(rid) == 'chain':
        if os.path.exists(os.path.join(d, 'pilot.json.gz')): return 'main', ck('main.json.gz')
        return 'pilot', ck('pilot.json.gz')
    return 'run', ck('run.json.gz')

def summarize(M):
    obs = M['obs']; last = obs[-1]
    te = [(o['step'], o['test_acc']) for o in obs]
    t10 = next((st for st, v in te if v >= 0.1), None)
    bad = [i for i, (st, v) in enumerate(te) if v < 0.9]
    t90 = (te[bad[-1] + 1][0] if bad[-1] + 1 < len(te) else None) if bad else te[0][0]
    hv = [o for o in obs if 'normW' in o]
    return dict(steps=len(M['loss']) - 1, seconds=round(M['seconds'], 1), resumed_from=M.get('resumed_from', 0),
                stopped_at=M.get('stopped_at'), t10=t10, t90=t90, final_test=last['test_acc'], final_train=last['train_acc'],
                final_loss=M['loss'][-1], normb_final=hv[-1].get('normb') if hv else None)

def finalize_run(ch, man, W, inst, branch, stolen_from):
    d = run_dir(ch['id']); dst = os.path.join(RESULTS, ch['id']); os.makedirs(dst, exist_ok=True)
    meta = dict(id=ch['id'], setting=ch.get('setting'), family=ch['family'], depth=ch['depth'], seed=ch['seed'],
                init_seed=ch['init_seed'], split_seed=ch['split_seed'], kind=ch['kind'], dial=ch.get('dial'),
                threads=ch.get('threads', 1), owner=ch['worker'], ran_by=W,
                instance=inst, branch=branch, stolen_from=stolen_from, version=VERSION, host=host_info(),
                code=dict(worker=sha256(os.path.abspath(__file__)), runner=sha256(RUNNER), manifest=man['_sha']), files={})
    for name in OUTS[ch['kind']]:
        M = load_gz(os.path.join(d, name))
        if ch['kind'] == 'one':
            assert json.dumps(M['cfg'], sort_keys=True) == json.dumps(ch['cfg'], sort_keys=True), 'output does not match its configuration'
        elif name == 'pilot.json.gz':
            assert json.dumps(M['cfg'], sort_keys=True) == json.dumps(ch['spec']['pilot_cfg'], sort_keys=True), 'pilot does not match'
        meta[name.split('.')[0]] = summarize(M)
        M['loss'] = r7(M['loss']); M['obs'] = r7(M['obs'])
        M['compact'] = 'floats rounded to 7 significant digits by worker.py; full precision stayed in the session'
        tmp = os.path.join(dst, name + '.tmp')
        with gzip.open(tmp, 'wt', compresslevel=9) as f: json.dump(M, f, separators=(',', ':'))
        os.replace(tmp, os.path.join(dst, name))
        meta['files'][name] = dict(sha256=sha256(os.path.join(dst, name)), bytes=os.path.getsize(os.path.join(dst, name)))
    if ch['kind'] == 'chain':
        rec = read_json(os.path.join(d, 'recipe.json')); meta['recipe'] = rec
        t10m = meta['main']['t10']; meta['main']['x'] = round(rec['rate'] * 0.03 * 0.015 * t10m, 4) if t10m is not None else None
        write_json(os.path.join(dst, 'recipe.json'), rec)
    meta['finished'] = iso(); meta['done'] = True
    write_json(os.path.join(dst, 'chain.json'), meta)
    write_json(os.path.join(d, 'chain.json'), meta)
    return dst


class Daemon:
    def __init__(self, man, W, cores=0, push=True, steal=True):
        self.man, self.W, self.push_on, self.steal_on = man, W, push, steal
        self.cores = cores or effective_cores()
        self.inst = instance_id()
        self.git = Git(HERE) if push else None
        self.branch = self.git.branch if self.git else 'none'
        self.procs = {}
        self.attempts = read_json(os.path.join(WORK, 'attempts.json'), {})
        self.unpushed = set(read_json(os.path.join(WORK, 'unpushed.json'), []))
        self.failed = set(read_json(os.path.join(WORK, 'failed.json'), []))
        self.last_beat = 0.0; self.last_push_try = 0.0; self.push_error = ''; self.started = now(); self.claims = {}
        self.status_fn = os.path.join(STATUS, f'{W}-{self.inst}.json')

    def save(self):
        write_json(os.path.join(WORK, 'attempts.json'), self.attempts)
        write_json(os.path.join(WORK, 'unpushed.json'), sorted(self.unpushed))
        write_json(os.path.join(WORK, 'failed.json'), sorted(self.failed))

    def local_done(self):
        return {c['id'] for c in self.man['runs'] if os.path.exists(os.path.join(run_dir(c['id']), 'chain.json'))}

    def remote(self, force=False):
        if not self.git: return {}
        try: return self.git.remote_state(force)
        except Exception as e: log(f'remote state failed: {e!r}'); return {}

    def done_anywhere(self, rem):
        s = set(self.local_done())
        for b in rem.values(): s |= b['done']
        return s

    def running_elsewhere(self, rem):
        ids, fresh = set(), set()
        for b in rem.values():
            for st in b['status']:
                if st.get('instance') == self.inst: continue
                if now() - parse_iso(st.get('heartbeat', '')) > STALE_S: continue
                fresh.add(st.get('worker'))
                ids |= {r['id'] for r in st.get('running', [])}
        return ids, fresh

    def other_claims(self, rem):
        out = {}
        for b in rem.values():
            for st in b['status']:
                if st.get('instance') == self.inst or now() - parse_iso(st.get('heartbeat', '')) > STALE_S: continue
                for c in st.get('claims', []): out.setdefault(c['id'], []).append((c['at'], st.get('instance')))
        return out

    def write_status(self, standby=False, campaign_done=False):
        q = self.man['_queue'].get(self.W, []); ld = self.local_done()
        st = dict(campaign=CAMPAIGN, worker=self.W, instance=self.inst, branch=self.branch, version=VERSION, heartbeat=iso(),
                  cores=self.cores, running=[dict(id=i, step=run_progress(i)[1]) for i in sorted(self.procs)],
                  done=sorted(ld), own_left=[c['id'] for c in q if c['id'] not in ld and c['id'] not in self.procs],
                  failed=sorted(self.failed), standby=standby, campaign_done=campaign_done, push_error=self.push_error,
                  claims=[dict(id=i, at=a) for i, a in sorted(self.claims.items())])
        write_json(self.status_fn, st)
        return st

    def publish(self, paths, msg):
        if not self.git: return True
        try:
            ok, err = self.git.commit_push(paths, msg)
        except Exception as e:
            ok, err = False, repr(e)
        self.push_error = '' if ok else err
        if not ok: log(f'PUSH FAILING: {err}')
        return ok

    def flush(self, standby=False, campaign_done=False, force=False):
        due = force or self.unpushed or now() - self.last_beat > HEARTBEAT_S
        if not due: return
        if self.unpushed and not force and now() - self.last_push_try < RETRY_PUSH_S and self.push_error: return
        self.last_push_try = now()
        self.write_status(standby, campaign_done)
        paths = [os.path.join(RESULTS, i) for i in sorted(self.unpushed)] + [self.status_fn]
        msg = f'campaign {CAMPAIGN} {self.W}: ' + (', '.join(sorted(self.unpushed)) if self.unpushed else 'status')
        if self.publish(paths, msg):
            self.unpushed.clear(); self.last_beat = now(); self.save()

    def candidates(self, rem):
        done = self.done_anywhere(rem); elsewhere, fresh = self.running_elsewhere(rem)
        busy = set(self.procs) | done | self.failed | elsewhere
        own = [c for c in self.man['_queue'].get(self.W, []) if c['id'] not in busy]
        if own: return [(c, None) for c in own]
        if not self.steal_on: return []
        out = []
        for w in (self.man['workers'] if now() - self.started > STEAL_GRACE_S else []):
            if w == self.W or w in fresh: continue
            out += [(c, w) for c in self.man['_queue'].get(w, []) if c['id'] not in busy]
        if out: return out
        for b in rem.values():
            for st in b['status']:
                if st.get('worker') == self.W or now() - parse_iso(st.get('heartbeat', '')) > STALE_S: continue
                left = [i for i in st.get('own_left', []) if i not in busy]
                if len(left) >= 2: out.append((self.man['_by_id'][left[-1]], st.get('worker')))
        return out

    def used(self):
        return sum(threads_of(self.man['_by_id'][i], self.cores) for i in self.procs)

    def start_more(self):
        free = self.cores - self.used()
        if free <= 0: return
        cands = self.candidates(self.remote()); claimed = False
        if cands and cands[0][1] is not None:
            time.sleep(secrets.randbelow(10))
            rem = self.remote(force=True); cands = self.fit(self.candidates(rem), free)
            if not cands: return
            at = iso(); self.claims = {ch['id']: at for ch, _ in cands}; self.flush(force=True)
            time.sleep(CLAIM_WAIT_S)
            rem = self.remote(force=True); busy = self.running_elsewhere(rem)[0] | self.done_anywhere(rem); oc = self.other_claims(rem)
            keep = []
            for ch, frm in cands:
                if ch['id'] in busy: continue
                if any((t, i) < (at, self.inst) for t, i in oc.get(ch['id'], [])): continue
                keep.append((ch, frm))
            self.claims = {}; cands = keep; claimed = True
        started = False
        for ch, frm in self.fit(cands, free):
            if self.attempts.get(ch['id'], 0) >= MAX_ATTEMPTS: self.failed.add(ch['id']); self.save(); continue
            pr = start_run(ch, self.cores); self.procs[ch['id']] = (pr, frm); started = True
            log(f"start {ch['id']}{' (taken over from ' + frm + ')' if frm else ''} pid {pr.pid}")
        if started or claimed: self.flush(force=True)

    def fit(self, cands, free):
        out = []
        for ch, frm in cands:
            k = threads_of(ch, self.cores)
            if k > free: break
            out.append((ch, frm)); free -= k
        return out

    def reap(self):
        for rid in list(self.procs):
            pr, frm = self.procs[rid]
            if pr.poll() is None: continue
            del self.procs[rid]
            d = run_dir(rid)
            if pr.returncode == 0 and outputs_done(rid):
                try:
                    finalize_run(self.man['_by_id'][rid], self.man, self.W, self.inst, self.branch, frm)
                    self.unpushed.add(rid); self.save(); log(f'finished {rid}')
                except Exception as e:
                    log(f'ERROR finalizing {rid}: {e!r}\n{traceback.format_exc()}')
            else:
                lt = tail(os.path.join(d, 'runner.log'), 15)
                if 'Traceback' in lt or 'Error' in lt:
                    self.attempts[rid] = self.attempts.get(rid, 0) + 1
                    with open(os.path.join(d, 'runner.log'), 'a') as f: f.write(f'--- {iso()} error counted by the worker ---\n')
                    log(f'ERROR {rid} exited with code {pr.returncode} (error {self.attempts[rid]} of {MAX_ATTEMPTS}); runner log tail:\n' + lt)
                    if self.attempts[rid] >= MAX_ATTEMPTS: self.failed.add(rid); log(f'ERROR {rid} failed {MAX_ATTEMPTS} times here: given up')
                    self.save()
                else:
                    log(f'{rid}: its runner stopped without an error (code {pr.returncode}); it will resume from its checkpoint')

    def campaign_complete(self):
        rem = self.remote(force=True)
        done = self.done_anywhere(rem)
        return all(c['id'] in done or c['id'] in self.failed for c in self.man['runs']) and not self.unpushed

    def loop(self):
        log(f'daemon start {VERSION} worker {self.W} instance {self.inst} branch {self.branch} cores {self.cores}')
        for c in self.man['runs']:
            d = run_dir(c['id'])
            if os.path.isdir(d) and not os.path.exists(os.path.join(d, 'chain.json')):
                if outputs_done(c['id']):
                    try: finalize_run(c, self.man, self.W, self.inst, self.branch, None); self.unpushed.add(c['id']); self.save()
                    except Exception as e: log(f'ERROR finalizing {c["id"]} on restart: {e!r}')
                elif find_runner(c['id']):
                    pid = find_runner(c['id']); self.procs[c['id']] = (Adopted(pid, c['id']), None)
                    log(f"adopt {c['id']}: its runner (pid {pid}) is still running")
                elif self.used() + threads_of(c, self.cores) <= self.cores and self.attempts.get(c['id'], 0) < MAX_ATTEMPTS and c['id'] not in self.failed:
                    self.procs[c['id']] = (start_run(c, self.cores), None); log(f"resume {c['id']} from step {run_progress(c['id'])[1]}")
        self.flush(force=True)
        last_check = 0.0
        while True:
            self.reap()
            self.start_more()
            if not self.procs:
                if now() - last_check > FETCH_MIN_S or self.unpushed:
                    last_check = now()
                    if self.campaign_complete():
                        self.flush(standby=True, campaign_done=True, force=True)
                        if not self.unpushed and not self.push_error:
                            log('ALL DONE: every run of the campaign is finished'); return
                self.flush(standby=True)
            else:
                self.flush()
            time.sleep(LOOP_S)


def daemon_info():
    return read_json(os.path.join(WORK, 'daemon.json'), {})

def daemon_alive():
    d = daemon_info(); pid = d.get('pid')
    return bool(pid) and pid_alive(pid) and 'worker.py' in cmdline(pid) and ' run ' in cmdline(pid) + ' '

def start_daemon(W, a):
    os.makedirs(WORK, exist_ok=True)
    args = [sys.executable, os.path.abspath(__file__), 'run', '--worker', W, '--manifest', a.manifest]
    if a.cores: args += ['--cores', str(a.cores)]
    if a.no_push: args.append('--no-push')
    if a.no_steal: args.append('--no-steal')
    lf = open(os.path.join(WORK, 'daemon.out'), 'a')
    pr = subprocess.Popen(args, cwd=HERE, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    write_json(os.path.join(WORK, 'daemon.json'), dict(pid=pr.pid, started=iso(), args=args[1:]))
    return pr.pid

def next_command(W):
    return f'python3 {os.path.relpath(os.path.abspath(__file__), os.getcwd())} watch --worker {W}'

def local_status_text(man, W):
    ld = {c['id'] for c in man['runs'] if os.path.exists(os.path.join(run_dir(c['id']), 'chain.json'))}
    q = man['_queue'].get(W, [])
    lines = [f"{W}: {sum(c['id'] in ld for c in q)}/{len(q)} own runs finished here; {len(ld)} finished here in all"]
    for c in man['runs']:
        ph, st = run_progress(c['id'])
        if os.path.isdir(run_dir(c['id'])) and ph in ('run', 'pilot', 'main'):
            lines.append(f"  running {c['id']}: step {st}" + (f"/{c['cfg']['steps']}" if c['kind'] == 'one' else f" ({ph})"))
    st = read_json(os.path.join(STATUS, f'{W}-{instance_id()}.json'), {})
    if st.get('push_error'): lines.append('  PUSH FAILING: ' + st['push_error'][-200:])
    return '\n'.join(lines)

def campaign_text(man, fetch=True):
    try: g = Git(HERE); rem = g.remote_state(force=True) if fetch else {}
    except Exception as e: return f'(cannot read the other branches: {e})'
    done = set()
    for b in rem.values(): done |= b['done']
    fresh = []
    for b, v in sorted(rem.items()):
        for st in v['status']:
            age = (now() - parse_iso(st.get('heartbeat', ''))) / 60
            fresh.append(f"  {st.get('worker')}-{st.get('instance')} on {b}: {len(st.get('done', []))} done, "
                         f"{len(st.get('running', []))} running, heartbeat {age:.0f} min ago" + (' (stale)' if age * 60 > STALE_S else ''))
    return f"campaign {CAMPAIGN}: {len(done & set(man['_by_id']))}/{len(man['runs'])} runs finished on GitHub\n" + '\n'.join(fresh)

def cmd_watch(a):
    man = load_manifest(a.manifest); W = a.worker; check_worker_id(man, W); check_clone_owner(W)
    if read_json(os.path.join(WORK, 'ALL_DONE.json')):
        print(campaign_text(man)); print('ALL DONE'); return
    if not daemon_alive():
        pid = start_daemon(W, a); print(f'{iso()} started the worker daemon (pid {pid})', flush=True)
    t_end = now() + 60 * a.minutes
    while now() < t_end:
        time.sleep(min(30, max(1, t_end - now())))
        if not daemon_alive():
            if 'ALL DONE' in tail(os.path.join(WORK, 'daemon.log'), 3):
                write_json(os.path.join(WORK, 'ALL_DONE.json'), dict(at=iso()))
                print(campaign_text(man)); print(local_status_text(man, W)); print('ALL DONE'); return
            print(f'{iso()} the daemon stopped; log tail:\n' + tail(os.path.join(WORK, 'daemon.log'), 12), flush=True)
            pid = start_daemon(W, a); print(f'{iso()} restarted the worker daemon (pid {pid})', flush=True)
    print(f'{iso()} status'); print(local_status_text(man, W))
    errs = [l for l in tail(os.path.join(WORK, 'daemon.log'), 40).splitlines() if 'ERROR' in l or 'PUSH FAILING' in l]
    if errs: print('recent problems:\n  ' + '\n  '.join(errs[-3:]))
    print(f'Next, run: {next_command(W)}')

def cmd_run(a):
    man = load_manifest(a.manifest); W = a.worker; check_worker_id(man, W); check_clone_owner(W)
    lock = os.path.join(WORK, 'daemon.lock')
    old = read_json(lock, {}).get('pid')
    if old and old != os.getpid() and pid_alive(old) and ' run ' in cmdline(old) + ' ': raise SystemExit(f'a daemon already runs here (pid {old})')
    write_json(lock, dict(pid=os.getpid(), started=iso()))
    write_json(os.path.join(WORK, 'daemon.json'), dict(pid=os.getpid(), started=iso(), args=sys.argv[1:]))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        Daemon(man, W, a.cores, push=not a.no_push, steal=not a.no_steal).loop()
    except SystemExit: raise
    except Exception as e:
        log(f'ERROR daemon crashed: {e!r}\n{traceback.format_exc()}'); raise

def cmd_status(a):
    man = load_manifest(a.manifest)
    if a.worker: print(local_status_text(man, a.worker))
    if a.all or not a.worker: print(campaign_text(man))

def cmd_selftest(a):
    import importlib.util, tempfile
    ok = True; man = load_manifest(a.manifest); W = a.worker; check_worker_id(man, W)
    def check(name, f):
        nonlocal ok
        t = now()
        try: msg = f() or ''; print(f'PASS  {name} ({now() - t:.0f}s) {msg}', flush=True)
        except Exception as e: ok = False; print(f'FAIL  {name}: {e}\n{traceback.format_exc()[-1200:]}', flush=True)
    def py():
        import numpy as np
        assert sys.version_info >= (3, 8), 'python >= 3.8 needed'
        extra = ''
        if CONF.get('needs_torch'):
            import torch; extra = f', torch {torch.__version__}'
        return f'python {platform.python_version()}, numpy {np.__version__}{extra}, {effective_cores()} usable cores'
    def clone(): check_clone_owner(W); return f'this clone runs worker {W}'
    def hashes():
        H = read_json(os.path.join(HERE, 'HASHES.json'))
        assert H, 'HASHES.json is missing: the campaign files are incomplete'
        bad = [f for f, h in H['files'].items() if not os.path.exists(os.path.join(HERE, f)) or sha256(os.path.join(HERE, f)) != h]
        assert not bad, f'files differ from the registered versions: {bad}'
        return f"{len(H['files'])} files match the registered hashes"
    def queue():
        q = man['_queue'].get(W, [])
        return f"{W}: {len(q)} runs, {sum(c['est_core_hours'] for c in q):.1f} core-hours"
    def runner_tests():
        fn = os.path.join(HERE, CONF['tests'])
        spec = importlib.util.spec_from_file_location('campaign_tests', fn)
        T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
        tests = getattr(T, 'TESTS', None) or [v for k, v in vars(T).items() if k.startswith('test_') and callable(v)]
        for f in tests: f()
        return f'{len(tests)} runner tests'
    def mini_run():
        d = tempfile.mkdtemp(prefix=f'camp{CAMPAIGN}_selftest_')
        cfg = CONF['selftest_cfg']; steps = cfg['steps']
        env = dict(os.environ, OMP_NUM_THREADS='1', CAMPAIGN_THREADS='1')
        json.dump(cfg, open(os.path.join(d, 'cfg.json'), 'w'))
        r = subprocess.run([sys.executable, RUNNER, 'one', os.path.join(d, 'cfg.json'), os.path.join(d, 'run.json.gz')], cwd=d,
                           env=env, capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stdout[-600:] + r.stderr[-600:]
        M = load_gz(os.path.join(d, 'run.json.gz')); assert len(M['loss']) == steps + 1
        assert json.dumps(M['cfg'], sort_keys=True) == json.dumps(cfg, sort_keys=True)
        shutil.rmtree(d, ignore_errors=True)
        return f'a {steps}-step toy run (the numbers mean nothing)'
    def git_checks():
        g = Git(HERE)
        assert g.branch != 'HEAD', "detached HEAD: check out this session's branch"
        assert g.branch.startswith(BRANCH_PREFIX), f'this session is on branch {g.branch}, but results are collected only from branches named {BRANCH_PREFIX}*: tell the user'
        fn = os.path.join(STATUS, f'{W}-selftest.json')
        write_json(fn, dict(worker=W, at=iso(), branch=g.branch, version=VERSION, host=host_info()))
        ok_, err = g.commit_push([fn], f'campaign {CAMPAIGN} {W}: selftest')
        assert ok_, "pushing to this session's branch failed, so results could not reach GitHub: " + err
        rem = g.remote_state(force=True)
        return f'branch {g.branch}; test push ok; {len(rem)} branches with campaign-{CAMPAIGN} files visible'
    check('python, numpy' + (' and torch' if CONF.get('needs_torch') else ''), py)
    check('one worker per clone', clone)
    check('registered file hashes', hashes)
    check('worker queue', queue)
    check('runner tests', runner_tests)
    check('one miniature run', mini_run)
    if not a.no_git: check('git (branch, real test push, fetch)', git_checks)
    print(f'SELFTEST PASS. Next, run: {next_command(W)}' if ok else 'SELFTEST FAIL: tell the user which check failed (copy the FAIL lines).')
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    def common(p): p.add_argument('--manifest', default=DEFAULT_MANIFEST); p.add_argument('--worker', required=True); return p
    for name in ('run', 'watch'):
        p = common(sub.add_parser(name)); p.add_argument('--cores', type=int, default=0)
        p.add_argument('--no-push', action='store_true'); p.add_argument('--no-steal', action='store_true')
        if name == 'watch': p.add_argument('--minutes', type=float, default=9.0)
    p = sub.add_parser('status'); p.add_argument('--manifest', default=DEFAULT_MANIFEST); p.add_argument('--worker'); p.add_argument('--all', action='store_true')
    p = common(sub.add_parser('selftest')); p.add_argument('--no-git', action='store_true')
    a = ap.parse_args()
    dict(run=cmd_run, watch=cmd_watch, status=cmd_status, selftest=cmd_selftest)[a.cmd](a)

if __name__ == '__main__':
    main()
