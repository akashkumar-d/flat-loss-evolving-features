import numpy as np, glob, os, sys, json
src, cid = sys.argv[1], sys.argv[2]
runs = []
for f in sorted(glob.glob(os.path.join(src, cid + '_dev*.npz'))):
    z = np.load(f); ce = z['train_ce']; lb = ce.copy(); lb[:-1] = 0.5 * (ce[:-1] + ce[1:])
    runs.append(dict(id=os.path.basename(f)[:-4], ce=ce, lb=lb, acc=z['test_accuracy'], tr=z['train_accuracy'],
                     os=z['obs_step'].astype(int), AH=z['obs_hidden_AH_add']))
print(len(runs), 'dev runs')
def first90(a, t0):
    i = np.flatnonzero(a[t0 + 1:] >= 0.9); return t0 + 1 + int(i[0]) if len(i) else None
cands = []
for t0 in range(1000, 18001, 500):
    for t1 in range(t0 + 2000, 20001, 500):
        ok, dah = True, []
        for r in runs:
            ce, lb = r['ce'][t0:t1 + 1], r['lb'][t0:t1]
            v = ce.max() / ce.min() - 1
            rm = np.maximum.accumulate(lb); dd = np.max((rm - lb) / rm)
            f90 = first90(r['acc'], t0)
            AH0 = r['AH'][r['os'] == t0]; AH1 = r['AH'][r['os'] == t1]
            if not (len(AH0) and len(AH1)): ok = False; break
            d = float(AH1[0] - AH0[0])
            if not (v <= 0.10 and dd <= 0.005 and lb[-1] >= lb[0] and ce.min() >= 0.1 and r['tr'][t0:t1 + 1].min() >= 1.0
                    and r['acc'][t0] <= 0.20 and f90 is not None and t0 < f90 <= t1 - 500 and d >= 0.04):
                ok = False; break
            dah.append(d)
        if ok: cands.append((min(dah), t0, t1))
print(len(cands), 'qualifying windows')
if not cands: sys.exit('none qualifies')
best = max(c[0] for c in cands)
top = sorted([c for c in cands if c[0] >= best - 0.005], key=lambda c: (c[2] - c[1], c[1]))
print('best min dAH %.4f; choice (shortest within 0.005, then earliest):' % best, top[0])
print('top candidates:', top[:8])
ch = top[0]
for r in runs:
    t0, t1 = ch[1], ch[2]; ce = r['ce'][t0:t1 + 1]; lb = r['lb'][t0:t1]; rm = np.maximum.accumulate(lb)
    print(r['id'], 'v %.2f%% dd %.3f%% net %+.2f%% dAH %.4f acc@t0 %.3f f90 %s minCE %.3f final40k %.3f' % (
        100 * (ce.max() / ce.min() - 1), 100 * np.max((rm - lb) / rm), 100 * (lb[-1] / lb[0] - 1),
        float(r['AH'][r['os'] == t1][0] - r['AH'][r['os'] == t0][0]), r['acc'][t0], first90(r['acc'], t0), ce.min(), r['acc'][-1]))
json.dump(dict(choice=dict(min_dAH=ch[0], t0=ch[1], t1=ch[2]), n_qualifying=len(cands), top=top[:20]),
          open(sys.argv[3] if len(sys.argv) > 3 else 'p23_window_selection.json', 'w'), indent=1)
