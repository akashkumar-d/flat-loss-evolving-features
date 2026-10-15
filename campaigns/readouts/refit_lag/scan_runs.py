#!/usr/bin/env python3
import glob, gzip, json, os, sys
import numpy as np

ROOT = '/home/claude/kt'
GLOBS = [f'{ROOT}/campaign_{c}/collected/results/*/run.json.gz' for c in ('V', 'W', 'X', 'Y', 'Z', 'ZP', 'Q')] + \
        [f'{ROOT}/campaign_ZM/runs/*/run.json.gz']
READ = ['refit', 'fref_all', 'fref_top1', 'fref_top2', 'fref_top4', 'fref_top8', 'fref_top16', 'fphase_top8', 'probe', 'fnet_top8', 'fnet_top16',
        'refit_mlp', 'fmref_top8', 'fref_oracle', 'fnet_top4', 'fnet_oracle', 'ffam_tr', 'ffam_all']


def cross(ts, vs, q):
    for i, v in enumerate(vs):
        if v is None: continue
        if v >= q:
            j = i - 1
            while j >= 0 and vs[j] is None: j -= 1
            if j < 0: return float(ts[i])
            t0, v0 = ts[j], vs[j]
            return float(t0 + (q - v0) / (v - v0) * (ts[i] - t0)) if v > v0 else float(ts[i])
    return None


def one(fn):
    d = json.load(gzip.open(fn)); obs = d['obs']; cfg = d.get('cfg', {})
    rid = os.path.basename(os.path.dirname(fn)); camp = fn.split('/campaign_')[1].split('/')[0]
    nm = any('test_acc_nm' in o for o in obs); key = 'test_acc_nm' if nm else 'test_acc'
    ts = [o['step'] for o in obs if key in o]; vs = [o[key] for o in obs if key in o]
    net = {q: cross(ts, vs, q) for q in (0.1, 0.5, 0.9)}
    out = dict(id=rid, campaign=camp, p=cfg.get('p'), layers=cfg.get('n_layers'), frac=cfg.get('frac'),
               opt=cfg.get('optimizer'), hidden=cfg.get('hidden_mode'), gain=cfg.get('gain_init'), lam_v=cfg.get('lam_v'),
               eta_w=cfg.get('eta_w'), norm=cfg.get('final_norm'), split=cfg.get('split_mode', cfg.get('split')),
               late=cfg.get('late'), nm=nm, net=net, steps=len(d.get('loss', [])) - 1, readouts={})
    for r in READ:
        k = f'{r}_{key}'
        hv = [(o['step'], o[k]) for o in obs if k in o]
        if not hv: continue
        t_r = [a for a, _ in hv]; v_r = [b for _, b in hv]
        c = {q: cross(t_r, v_r, q) for q in (0.5, 0.9)}
        out['readouts'][r] = dict(cross=c, lead={q: (net[q] - c[q]) if (net[q] is not None and c[q] is not None) else None
                                                 for q in (0.5, 0.9)}, final=v_r[-1])
    sp = [o for o in obs if 'fspec_hor2' in o]
    if sp and net[0.5] is not None:
        o = min(sp, key=lambda o: abs(o['step'] - net[0.5]))
        r2 = np.array(o['fspec_hor2']); E = np.array(o['fspec_train']); top = set(o.get('ftop8') or [])
        carried = [int(k) + 1 for k in np.where(r2 > 0.05)[0]]
        out['at_rise'] = dict(step=o['step'], K=len(r2), n_carried=len(carried), n_pos=int((r2 > 0).sum()),
                              carried_in_top8=len(set(carried) & top), top8_share=float(np.sort(E)[::-1][:8].sum() / E.sum()),
                              r2_top=sorted(r2.tolist(), reverse=True)[:12])
    return out


def main():
    files = sorted(sum((glob.glob(g) for g in GLOBS), []))
    res = []
    for fn in files:
        try:
            res.append(one(fn))
        except Exception as e:
            print('skip', fn, repr(e), flush=True)
    json.dump(res, open(sys.argv[1], 'w'), indent=0)
    print(len(res), 'runs')


if __name__ == '__main__':
    main()
