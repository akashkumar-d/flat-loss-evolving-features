import argparse, glob, gzip, hashlib, io, os, sys, tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.dirname(HERE)


def sha(b): return hashlib.sha256(b).hexdigest()


def stem(member):
    base = os.path.basename(member)
    for ext in ('.json.gz', '.json'):
        if base.endswith(ext): return base[:-len(ext)]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=REPORT, help='directory that receives runs/, runs_dup/, runs_ckpt/')
    ap.add_argument('--archives', default=os.path.join(REPORT, 'runs'), help='folder with the run archives (*.tar.gz, *.json.gz)')
    ap.add_argument('--shared', default=None, help='optional folder of per-task result folders (<TASK>/*.json.gz)')
    a = ap.parse_args()
    want = {}
    for line in open(os.path.join(HERE, 'RUN_MANIFEST.tsv')):
        if line.startswith('#') or not line.strip(): continue
        rel, h = line.rstrip('\n').split('\t')[:2]; want[rel] = h
    cands = {rel: [] for rel in want}
    by_stem = {}
    for rel in want:
        if rel.startswith('runs/'): by_stem[rel[len('runs/'):-len('.json')]] = rel
    tars = sorted(glob.glob(os.path.join(a.archives, '*.tar.gz')))
    for t in tars:
        with tarfile.open(t) as tf:
            for m in tf.getmembers():
                if not m.isfile(): continue
                if m.name.startswith(('runs_dup/', 'runs_ckpt/')) and m.name in want: cands[m.name].append(('tar', t, m.name)); continue
                s = stem(m.name)
                if s in by_stem: cands[by_stem[s]].append(('tar', t, m.name))
    for f in sorted(glob.glob(os.path.join(a.archives, '*.json.gz'))):
        s = stem(f)
        if s in by_stem: cands[by_stem[s]].append(('gz', f, None))
    for f in (sorted(glob.glob(os.path.join(a.shared, '*', '*.json.gz'))) if a.shared else []):
        s = stem(f)
        if s in by_stem: cands[by_stem[s]].append(('gz', f, None))
    ok = kept = 0; bad = []
    open_tars = {}
    for rel, h in sorted(want.items()):
        dst = os.path.join(a.out, rel)
        if os.path.exists(dst) and sha(open(dst, 'rb').read()) == h: kept += 1; continue
        done = False
        for kind, src, member in cands[rel]:
            if kind == 'tar':
                tf = open_tars.get(src) or open_tars.setdefault(src, tarfile.open(src))
                data = tf.extractfile(member).read()
                if member.endswith('.gz'): data = gzip.decompress(data)
            else:
                data = gzip.decompress(open(src, 'rb').read())
            if sha(data) == h:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst + '.tmp', 'wb') as g: g.write(data)
                os.replace(dst + '.tmp', dst); ok += 1; done = True; break
        if not done: bad.append((rel, len(cands[rel])))
    print(f'restored {ok}, already present {kept}, missing or mismatched {len(bad)} (of {len(want)}) into {os.path.abspath(a.out)}')
    for rel, n in bad: print(f'  NOT RESTORED {rel}: {n} candidate source(s), none with the manifest hash')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
