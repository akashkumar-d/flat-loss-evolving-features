#!/usr/bin/env python3
import glob, gzip, json, math, os, re, sys, tarfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'figures', 'lib'))
import _paths as P
import endpoints as EP

ARCH, ANALYSIS, MANIFEST_F, MANIFEST_G = {}, {}, None, None


def configure(ma=None):
    global MANIFEST_F, MANIFEST_G
    for k in 'CDF':
        ARCH[k] = str(P.CAMPAIGN[k]); ANALYSIS[k] = str(P.CAMPAIGN_CODE[k] / 'results' / 'analysis.json')
    ARCH['G'] = str(P.CAMPAIGN['G'])
    ARCH['dev'] = str(P.DEV_RUNS)
    MANIFEST_F = str(P.CODE / 'deep_mlp' / 'across_primes_ten_seeds' / 'manifest.json')
    MANIFEST_G = str(P.CODE / 'deep_mlp' / 'across_primes_p101' / 'manifest.json')
    _STORES.clear(); _ANALYSIS.clear()


K = dict(train="#0072B2", train_acc="#000000", test="#D55E00", refit="#CC79A7", AH="#009E73", window="#EBEBEB",
         grid="#E6E6E6", neutral="#444444", cm="#003C64", zero="#B00020")
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'xtick.labelsize': 8,
         'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
         'axes.edgecolor': '#555555', 'axes.linewidth': 0.7}


class Store:
    def __init__(self, root):
        self.idx, self.tars = {}, {}
        for tf in sorted(glob.glob(os.path.join(root, '*.tar')) + glob.glob(os.path.join(root, '*.tar.gz'))):
            t = tarfile.open(tf); self.tars[tf] = t
            for m in t.getmembers():
                if not m.isfile(): continue
                parts = m.name.split('/')
                if parts[-1] in ('run.json.gz', 'main.json.gz') and len(parts) >= 2: key = parts[-2]
                elif parts[-1] == 'pilot.json.gz' and len(parts) >= 2: key = parts[-2] + ':pilot'
                elif parts[-1].endswith(('.json', '.json.gz')) and parts[-1] != 'chain.json':
                    key = parts[-1].replace('.json.gz', '').replace('.json', '')
                else: continue
                self.idx.setdefault(key, (tf, m))
    def has(self, key): return key in self.idx
    def load(self, key):
        tf, m = self.idx[key]; raw = self.tars[tf].extractfile(m).read()
        if raw[:2] == b'\x1f\x8b': raw = gzip.decompress(raw)
        return json.loads(raw.decode())


_STORES, _ANALYSIS = {}, {}
def store(kind):
    if kind not in _STORES: _STORES[kind] = Store(ARCH[kind])
    return _STORES[kind]

def analysis(kind):
    if kind not in _ANALYSIS: _ANALYSIS[kind] = json.load(open(ANALYSIS[kind]))['rows']
    return _ANALYSIS[kind]


def _series(obs, key):
    s = [(o['step'], o[key]) for o in obs if key in o and o[key] is not None]
    return np.array([x[0] for x in s], float), np.array([x[1] for x in s], float)


def record(d, rid, fields=None):
    c = d['cfg']; p = c['p']; D = c.get('depth') or c.get('n_layers') or 1; lp = math.log(p)
    L = np.asarray(d['loss'], float) / lp
    obs = d['obs']
    t, tr = _series(obs, 'train_acc'); _, te = _series(obs, 'test_acc'); tm, mm = _series(obs, 'train_margin_min')
    th, rms = _series(obs, 'logit_rms')
    key_last = f'AH_full_h{D}' if any(f'AH_full_h{D}' in o for o in obs[:3]) else 'AH_full_h1'
    ta, ahL = _series(obs, key_last); ta1, ah1 = _series(obs, 'AH_full_h1')
    f = fields if fields is not None else EP.run_fields(d)
    return dict(id=rid, p=p, depth=D, width=c.get('width', c.get('d_model')), L=L, cm=(L[:-1] + L[1:]) / 2, t=t, tr=tr, te=te, tm=tm,
                mm=mm, th=th, rms=rms, ta=ta, ahL=ahL, ah1=(ta1, ah1) if D > 1 and key_last != 'AH_full_h1' else None,
                ntr=int(c.get('frac', 0.8) * p * p), W=f.get('W'), share=f.get('share'), primary=bool(f.get('primary')),
                why=f.get('why_not'), knee=f.get('knee'), t10=f.get('t10'), t90=f.get('t90') or f.get('t90_first'),
                fit_first=f.get('T_fit_first'), level=f.get('CE_peak_over_logp'), final=f.get('final_test'),
                stop=len(L) - 1, f=f)


def find(rid):
    if rid.endswith(':pilot'):
        base = rid[:-6]; return store('C').load(rid), analysis('C')[base]['fixed']
    if rid.startswith(('C1_', 'C2_', 'C3_', 'C4_', 'C5_')):
        return store('C').load(rid), analysis('C')[rid]['main']
    if rid.startswith('dev_'):
        return store('E').load(rid), analysis('E')[rid]['f']
    fields = analysis('F')[rid]['f'] if rid in analysis('F') else analysis('D')[rid]['f'] if rid in analysis('D') else None
    for kind in ('F', 'D', 'G', 'dev'):
        if store(kind).has(rid):
            return store(kind).load(rid), fields
    raise KeyError(rid)


def c_prefix(prefixes):
    if isinstance(prefixes, str): prefixes = (prefixes,)
    return sorted(k for k in store('C').idx if k.startswith(tuple(prefixes)) and not k.endswith(':pilot'))


def hold(x, y, g):
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) == 0: return np.full(len(g), np.nan)
    return np.interp(g, x, y, left=y[0], right=y[-1])


def rng(v, nd=2, pct=False):
    v = [x for x in v if x is not None]
    if not v: return '–'
    lo, hi = min(v), max(v); f = (lambda x: f'{100 * x:.0f}%') if pct else (lambda x: f'{x:.{nd}f}')
    return f(lo) if f(lo) == f(hi) else f'{f(lo)}–{f(hi)}'


def fmt_k(t): return f'{t / 1000:.1f}k'


def seed_label(rid):
    m = re.search(r'_s(\d+)', rid)
    return (f'seed {int(m.group(1))}' if m else rid) + (' (fixed-rate arm)' if rid.endswith(':pilot') else '')
