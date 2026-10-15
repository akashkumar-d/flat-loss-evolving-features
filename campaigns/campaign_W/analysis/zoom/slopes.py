import glob, json, math, os, sys
import numpy as np
sys.path.insert(0, '/home/claude/kt/campaign_W'); import paperdefs_tf as P
sys.path.insert(0, '/home/claude/kt/campaign_L'); import endpoints_i as E; import acrossp_figs as A

def runmed(x, k=201):
    c = np.concatenate([(x[:-1] + x[1:]) / 2, x[-1:]]); h = k // 2
    return np.convolve(np.pad(c, (h, h), mode='edge'), np.ones(k) / k, mode='valid')

def stats(L, a, b):
    a, b = int(a), int(min(b, len(L) - 1))
    if b - a < 50: return None
    s = runmed(L[max(0, a - 300):min(len(L), b + 301)])[a - max(0, a - 300):][:b - a + 1]
    t = np.arange(a, b + 1); k = np.polyfit(t, np.log(s), 1)[0]
    return dict(a=a, b=b, ce_a=float(s[0]), ce_b=float(s[-1]), pct_per_1k=float((math.exp(1000 * k) - 1) * 100),
                total_pct=float((s[-1] / s[0] - 1) * 100), v_raw=float(L[a:b + 1].max() / L[a:b + 1].min() - 1))

def first(s, v, th):
    i = np.where(v >= th)[0]; return int(s[i[0]]) if len(i) else None

rows = []
def add(group, rid, L, windows, p):
    for wname, (a, b) in windows.items():
        if a is None or b is None: continue
        st = stats(np.asarray(L, float), a, b)
        if st: rows.append(dict(group=group, id=rid, window=wname, p=p, **st))

W = '/home/claude/kt/campaign_W/collected/results'; RX = json.load(open('/home/claude/kt/campaign_W/analysis/relaxed_W.json'))
for st in ('W_R5_f0.2_rms_lam_w0.01_eta_w0.1_lam_v0.275', 'W_R4_f0.2_rms_lam_w0.01_eta_w0.1_lam_v0.25', 'W_R1_f0.2_rms_lam_w0.01_eta_w0.03_lam_v0.3',
           'W_N1_f0.2_rms_ngd_lam_w0.01_eta_w0.1_lam_v0.25', 'W_P1_f0.3_lam_w0.02_eta_w0.1_lam_v0.2'):
    for fn in sorted(glob.glob(f'{W}/{st}_s*/run.json.gz')):
        rid = os.path.basename(os.path.dirname(fn)); d = P.load(fn); key = P.te_key(d); s, te = P.series(d['obs'], key)
        win = {'held-out 0.1 -> 0.9': (first(s, te, 0.1), first(s, te, 0.9))}
        if rid in RX and RX[rid]['q_k10'][1]: win['window (<= 10 misclassified)'] = tuple(RX[rid]['q_k10'][1])
        add(st.split('_')[1] + ' ' + st.split('_', 2)[2], rid, d['loss'], win, 113)
for fn in sorted(glob.glob('/home/claude/kt/campaign_V/collected/results/V_C1_f0.3_lam_w0.02_eta_w0.1_lam_v0.2_s[678]/run.json.gz')):
    rid = os.path.basename(os.path.dirname(fn)); d = P.load(fn); s, te = P.series(d['obs'], 'test_acc')
    add('V_C1 float32', rid, d['loss'], {'held-out 0.1 -> 0.9': (first(s, te, 0.1), first(s, te, 0.9))}, 113)
for fn in sorted(glob.glob('/home/claude/v7work/data/F/H_p31_lw0.0150_s*/run.json.gz')):
    rid = os.path.basename(os.path.dirname(fn)); d = E.load(fn); s, te = E.series(d['obs'], 'test_acc')
    add('paper 1HL p31 (F)', rid, d['loss'], {'paper window [4500, 12000]': (4500, 12000), 'held-out 0.1 -> 0.9': (first(s, te, 0.1), first(s, te, 0.9))}, 31)
K = '/mnt/user-data/uploads/modular arithmetic/natural_plateau_shared/campaign_K/collected/results'
for fn in sorted(glob.glob(f'{K}/K_D3_p61_lv0.1016_lw0.015_s*/run.json.gz')) + sorted(glob.glob('/home/claude/v7work/data/F/D3_p61_r0.45_s*/run.json.gz')):
    rid = os.path.basename(os.path.dirname(fn)); d = E.load(fn); s, te = E.series(d['obs'], 'test_acc'); w = A.window_record(d)['W2']
    add('deep D3 p61 frac 0.8' + (' (K linear)' if 'K_' in rid else ' (F paper)'), rid, d['loss'], {'post-hoc window': tuple(w), 'held-out 0.1 -> 0.9': (first(s, te, 0.1), first(s, te, 0.9))}, 61)
json.dump(rows, open(sys.argv[1], 'w'), indent=1)
import collections
G = collections.defaultdict(list)
for r in rows: G[(r['group'], r['window'])].append(r)
print(f"{'group':52s} {'window':30s} n  {'CE start (nats)':>15s} {'%/1k upd':>12s} {'total %':>14s} {'raw v':>12s}")
for (g, w), rs in G.items():
    f = lambda k, d=1: f"{min(r[k] for r in rs):.{d}f}..{max(r[k] for r in rs):.{d}f}"
    print(f"{g:52s} {w:30s} {len(rs)}  {f('ce_a', 3):>15s} {f('pct_per_1k', 2):>12s} {f('total_pct'):>14s} {f('v_raw', 2):>12s}")
