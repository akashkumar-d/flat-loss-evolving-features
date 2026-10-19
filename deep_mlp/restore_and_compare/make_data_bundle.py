#!/usr/bin/env python3
import argparse, hashlib, io, json, os, subprocess, sys, tarfile

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DROP = ('branch', 'instance')


def sha256(fn):
    h = hashlib.sha256()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def read_sums(fn):
    want = {}
    with open(fn) as f:
        for line in f:
            if line.strip(): h, name = line.split(); want[name] = h
    return want


def scrub_chain(data):
    meta = json.loads(data)
    for k in DROP: meta.pop(k, None)
    return (json.dumps(meta, indent=1, sort_keys=True) + '\n').encode()


def repack(src_dir, out_dir):
    want = read_sums(os.path.join(ROOT, 'depths3to5_preregistered_p31', 'results', 'SHA256SUMS'))
    os.makedirs(out_dir, exist_ok=True); sums = []
    for name, h in sorted(want.items()):
        src = os.path.join(src_dir, name)
        if not os.path.exists(src) or sha256(src) != h: sys.exit(f'missing or SHA-256 mismatch: {src}')
        dst = os.path.join(out_dir, name)
        with tarfile.open(src) as tf, tarfile.open(dst + '.tmp', 'w', format=tarfile.PAX_FORMAT) as tw:
            for m in tf.getmembers():
                ti = tarfile.TarInfo(m.name); ti.mtime = 0; ti.uid = ti.gid = 0; ti.uname = ti.gname = ''
                if m.isdir():
                    ti.type = tarfile.DIRTYPE; ti.mode = 0o755; tw.addfile(ti); continue
                data = tf.extractfile(m).read()
                if m.name.endswith('chain.json'): data = scrub_chain(data)
                ti.size = len(data); ti.mode = 0o644; tw.addfile(ti, io.BytesIO(data))
        os.replace(dst + '.tmp', dst); sums.append(f'{sha256(dst)}  {name}\n')
        print(f'{name}: repacked')
    with open(os.path.join(out_dir, 'SHA256SUMS'), 'w') as f: f.writelines(sums)
    print(f'wrote {len(sums)} anonymized tars and SHA256SUMS to {out_dir}')


def pack_dev(archives, shared, out_dir):
    import gzip, shutil, tempfile
    tmp = tempfile.mkdtemp(prefix='devpack_')
    cmd = [sys.executable, os.path.join(ROOT, 'development_depths1to5', 'restore_runs.py'), '--archives', archives, '--out', tmp]
    if shared: cmd += ['--shared', shared]
    if subprocess.run(cmd).returncode: sys.exit('pack: not every development file could be restored')
    rels = [l.split('\t')[0] for l in open(os.path.join(ROOT, 'development_depths1to5', 'RUN_MANIFEST.tsv')) if l.strip() and not l.startswith('#')]
    os.makedirs(out_dir, exist_ok=True); dst = os.path.join(out_dir, 'development_runs.tar.gz')
    with open(dst + '.tmp', 'wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=9, mtime=0, filename='') as gz:
            with tarfile.open(fileobj=gz, mode='w', format=tarfile.PAX_FORMAT) as tw:
                for rel in sorted(rels):
                    with open(os.path.join(tmp, rel), 'rb') as f: data = f.read()
                    ti = tarfile.TarInfo(rel); ti.size = len(data); ti.mtime = 0; ti.mode = 0o644
                    ti.uid = ti.gid = 0; ti.uname = ti.gname = ''
                    tw.addfile(ti, io.BytesIO(data))
    os.replace(dst + '.tmp', dst)
    with open(os.path.join(out_dir, 'SHA256SUMS'), 'w') as f: f.write(f'{sha256(dst)}  development_runs.tar.gz\n')
    shutil.rmtree(tmp, ignore_errors=True)
    print(f'wrote {dst} ({len(rels)} files, {os.path.getsize(dst) / 2**20:.1f} MB) and SHA256SUMS')


def campaign(tars_dir, out):
    local = os.path.join(tars_dir, 'SHA256SUMS')
    want = read_sums(local if os.path.exists(local) else os.path.join(ROOT, 'depths3to5_preregistered_p31', 'results', 'SHA256SUMS'))
    dst_root = os.path.join(out, 'campaign_C'); os.makedirs(dst_root, exist_ok=True)
    n_chain = 0
    for name, h in sorted(want.items()):
        fn = os.path.join(tars_dir, name)
        if not os.path.exists(fn): sys.exit(f'missing {fn}')
        if sha256(fn) != h: sys.exit(f'SHA-256 mismatch: {fn}')
        with tarfile.open(fn) as tf:
            for m in tf.getmembers():
                parts = m.name.split('/')
                if m.isdir(): continue
                if not m.isfile() or len(parts) != 2 or '..' in parts or parts[1] not in ('chain.json', 'pilot.json.gz', 'main.json.gz'):
                    sys.exit(f'unexpected member {m.name} in {name}')
                d = os.path.join(dst_root, parts[0]); os.makedirs(d, exist_ok=True)
                data = tf.extractfile(m).read()
                if parts[1] == 'chain.json':
                    data = scrub_chain(data); n_chain += 1
                with open(os.path.join(d, parts[1]), 'wb') as g: g.write(data)
        print(f'{name}: checked and extracted')
    print(f'campaign C: {n_chain} chains in {dst_root}')


def development(archives, shared, out):
    cmd = [sys.executable, os.path.join(ROOT, 'development_depths1to5', 'restore_runs.py'), '--archives', archives, '--out', os.path.join(out, 'development')]
    if shared: cmd += ['--shared', shared]
    r = subprocess.run(cmd)
    if r.returncode: sys.exit('development runs: not every file was restored (see above)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--campaign-tars', help='folder with the nine campaignC_*.tar files')
    ap.add_argument('--dev-archives', help='folder with the development run archives')
    ap.add_argument('--dev-shared', help='optional folder of per-task result folders')
    ap.add_argument('--out', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--repack-campaign', metavar='DIR', help='folder with the original nine tars, to write anonymized copies')
    ap.add_argument('--repack-out', metavar='OUT', help='where --repack-campaign or --pack-dev writes, with SHA256SUMS')
    ap.add_argument('--pack-dev', metavar='DIR', help='development run archives, to pack the 205 files into one archive')
    a = ap.parse_args()
    if a.repack_campaign:
        if not a.repack_out: ap.error('--repack-campaign needs --repack-out')
        repack(a.repack_campaign, a.repack_out); return
    if a.pack_dev:
        if not a.repack_out: ap.error('--pack-dev needs --repack-out')
        pack_dev(a.pack_dev, a.dev_shared, a.repack_out); return
    if not (a.campaign_tars or a.dev_archives): ap.error('give --campaign-tars and/or --dev-archives (or --repack-campaign)')
    if a.campaign_tars: campaign(a.campaign_tars, a.out)
    if a.dev_archives: development(a.dev_archives, a.dev_shared, a.out)


if __name__ == '__main__':
    main()
