import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import gzip, json, sys, time
import numpy as np
import natplat_server as N


def run_states(cfg, out, every, ref_dir=None):
    R = N.Run(cfg); T = cfg['steps']; t0 = time.time()
    loss = np.empty(T + 1); obs = []; steps, Ws, Vs = [], [], []
    for t in range(0, T + 1):
        L, gWs, gV = R.loss_and_grads(); loss[t] = L
        if t % cfg['obs_every'] == 0 or t == T:
            o = R.observe(heavy=(t % cfg['heavy_every'] == 0 or t == T)); o['step'] = t; o['loss'] = L; obs.append(o)
        if t % every == 0 or t == T:
            steps.append(t); Ws.append(R.Ws[0].copy()); Vs.append(R.V.copy())
        if t == T:
            break
        R.step(gWs, gV, t)
    res = dict(cfg=cfg, loss=loss.tolist(), obs=obs, seconds=time.time() - t0, resumed_from=0)
    if ref_dir:
        name = os.path.basename(out)[:-len('.json.gz')]
        ref = os.path.join(ref_dir, name + '.json')
        if os.path.exists(ref):
            L0 = np.asarray(json.load(open(ref))['loss'])
            res['reference_max_rel_loss_diff'] = float(np.max(np.abs(loss - L0) / np.abs(L0)))
            res['reference_bitwise_equal'] = bool(np.array_equal(loss, L0))
    with gzip.open(out + '.tmp', 'wt') as f:
        json.dump(res, f)
    os.replace(out + '.tmp', out)
    np.savez_compressed(out[:-len('.json.gz')] + '_states.npz', steps=np.asarray(steps), W=np.stack(Ws), V=np.stack(Vs),
                        train_idx=R.tr, test_idx=R.te)
    return res


def _job(args):
    name, cfg, outdir, every, ref_dir = args
    out = os.path.join(outdir, name + '.json.gz')
    if os.path.exists(out):
        return name, 'exists', 0.0, None
    t = time.time()
    try:
        r = run_states(cfg, out, every, ref_dir); msg = 'ok'; d = r.get('reference_max_rel_loss_diff')
    except Exception as e:
        msg = 'ERROR ' + repr(e); d = None
    with open(os.path.join(outdir, 'queue.log'), 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {name} {msg} {time.time() - t:.0f}s ref_diff={d}\n")
    return name, msg, time.time() - t, d


if __name__ == '__main__':
    import multiprocessing as mp
    jobs = json.load(open(sys.argv[1])); outdir = sys.argv[2]; os.makedirs(outdir, exist_ok=True)
    w = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 4
    every = int(sys.argv[sys.argv.index('--every') + 1]) if '--every' in sys.argv else 500
    ref = sys.argv[sys.argv.index('--reference') + 1] if '--reference' in sys.argv else None
    args = [(j['name'], j['cfg'], outdir, every, ref) for j in jobs]
    with mp.get_context('spawn').Pool(w) as pool:
        for name, msg, dt, d in pool.imap_unordered(_job, args):
            print(f"{name}: {msg} ({dt / 60:.1f} min) ref_diff={d}", flush=True)
