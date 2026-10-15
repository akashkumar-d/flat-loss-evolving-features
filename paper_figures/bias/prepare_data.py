#!/usr/bin/env python3
import argparse, glob, os, sys, tarfile
from pathlib import PurePosixPath

PLAN = [
    ('campaign_C', 'campaignC_C[12]_D3.tar', 'data/C'), ('campaign_C', 'campaignC_C[12]_D4.tar', 'data/C'),
    ('campaign_C', 'campaignC_C[12]_D5.tar', 'data/C5'),
    ('campaign_D', 'campaignD_*.tar', 'data/D'),
    ('campaign_F', 'campaignF_H.tar', 'data/F'), ('campaign_F', 'campaignF_D3.tar', 'data/F'), ('campaign_F', 'campaignF_D4.tar', 'data/F'),
    ('campaign_F', 'campaignF_D2.tar', 'data/F_D2'),
    ('campaign_G', 'campaignG_*.tar', 'data/G'),
    ('campaign_I', 'campaignI_*.tar', 'bias'),
]
DEV2 = [f'seedD2_r0.5_s{i}.json' for i in range(5)]


def safe(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts: raise SystemExit(f'unsafe member name {name!r}')
    return p


def extract(tar_fn, dest, only=None):
    n = 0
    with tarfile.open(tar_fn) as t:
        for m in t.getmembers():
            if not m.isfile(): continue
            p = safe(m.name)
            if only is not None and p.name not in only: continue
            out = os.path.join(dest, *(p.parts if only is None else (p.name,)))
            if os.path.exists(out): raise SystemExit(f'{out} exists twice (from {tar_fn}); use an empty --out')
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, 'wb') as f: f.write(t.extractfile(m).read())
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--paper-data', default=os.environ.get('PAPER_DATA'), help='the data folder (default: $PAPER_DATA)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if not a.paper_data: sys.exit('give --paper-data or set PAPER_DATA')
    deep = os.path.join(os.path.expanduser(a.paper_data), 'deep')
    if os.path.exists(a.out) and os.listdir(a.out): sys.exit(f'{a.out} is not empty')
    for camp, pat, dest in PLAN:
        tars = sorted(glob.glob(os.path.join(deep, camp, pat)))
        if not tars: sys.exit(f'no archive {camp}/{pat} in {deep}')
        for tf in tars: print(f'{camp}/{os.path.basename(tf)}: {extract(tf, os.path.join(a.out, dest))} files -> {dest}', flush=True)
    found = {}
    for tf in sorted(glob.glob(os.path.join(deep, 'development', '*.tar.gz'))):
        with tarfile.open(tf) as t:
            for m in t.getmembers():
                if m.isfile() and safe(m.name).name in DEV2: found.setdefault(safe(m.name).name, []).append(tf)
    for name in DEV2:
        if len(found.get(name, [])) != 1: sys.exit(f'{name}: found in {found.get(name, [])}; expected exactly one development archive')
        extract(found[name][0], os.path.join(a.out, 'dev2'), only={name})
    print(f'dev2: {len(DEV2)} depth-2 development runs')


if __name__ == '__main__':
    main()
