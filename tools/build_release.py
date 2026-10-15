#!/usr/bin/env python3
import argparse, hashlib, os, re, shutil, sys, zipfile
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
SELF = "tools/build_release.py"
SKIP_DIRS = {"data", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
TEXT = {".py", ".md", ".json", ".jsonl", ".tsv", ".txt", ".sh", ".sha256", ".csv", ".cfg", ".toml", ".yaml", ".yml", ".log", ""}
PATTERNS = [
    r"/Users/", r"/home/[A-Za-z]", r"[A-Za-z]:\\Users", r"/mnt/user-data", r"~/Downloads", r"https?://", r"www\.",
    r"github\.com", r"gitlab",
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}",
    r"\btoken\b", r"api[_-]?key", r"passw(or)?d", r"\bsecret\b", r"\[AUTHORS",
]
ALLOW = {
    "deep/campaign_E/tfm_runner.py": {"token"}, "deep/campaign_E/PLAN.md": {"token"},
}
CAMPAIGN_ALLOW = {"token"}
CAMPAIGN_NAME_ALLOW = {"worker.py": {"github.com", "-worker@users.noreply"}, "WORKER.md": {"https://", "password"}}
WORKSPACE_PATHS = {
    "campaigns/campaign_M/design/diag_fig.py", "campaigns/campaign_M/design/timing.py",
    "campaigns/campaign_T/analysis/ce64_check/ce64.py", "campaigns/campaign_W/analysis/zoom/slopes.py",
    "campaigns/campaign_W/analysis/zoom/zoom_fig.py", "campaigns/campaign_W/draft_existing.py",
    "campaigns/campaign_W/draft_pilots.py", "campaigns/campaign_W/explore/bestwin.py",
    "campaigns/campaign_W/explore/fitfirst.py", "campaigns/campaign_W/explore/lead.py",
    "campaigns/campaign_W/explore/pilot_report.py", "campaigns/campaign_W/explore/vraw.py",
    "campaigns/campaign_W/score_existing.py", "campaigns/campaign_X/analysis/rise_paperwindow_X.json",
    "campaigns/campaign_X/testdata/make_testdata.py", "campaigns/campaign_Z/pilots/peek.py",
    "campaigns/campaign_Z/prep/heldout_defs.py", "campaigns/readouts/ah_gap/ah_decomp.py",
    "campaigns/campaign_L/analysis/plot_L.log", "campaigns/campaign_L/extra_L.py",
    "campaigns/campaign_W/CAMPAIGN_W_PLAN_20261010.md", "campaigns/campaign_X/analysis/SUMMARY.md",
    "campaigns/campaign_Y/analysis/SUMMARY.md",
    "campaigns/campaign_Q/analysis/analysis_Q.json", "campaigns/campaign_Q/zm10/analysis_ZM10.json",
    "campaigns/campaign_Q/zm10/build_zm10.py", "campaigns/campaign_Q/zm10/plot_ZM10.py",
    "campaigns/readouts/refit_lag/scan_runs.py", "campaigns/readouts/refit_lag/k_rules.py",
    "campaigns/readouts/refit_lag/k_curve.py", "campaigns/readouts/refit_lag/k_val.py",
    "campaigns/readouts/refit_lag/lag_tau.py", "campaigns/readouts/refit_lag/mlp/scan_mlp.py",
    "campaigns/readouts/refit_lag/analyze_lag.log",
}
WORKSPACE_WORDS = {"/home/c", "/mnt/user-data", "~/downloads"}


def allowed(rel):
    a = set(ALLOW.get(rel, set()))
    if rel.startswith("campaigns/"):
        a |= CAMPAIGN_ALLOW | CAMPAIGN_NAME_ALLOW.get(Path(rel).name, set())
        if rel in WORKSPACE_PATHS: a |= WORKSPACE_WORDS
    return a
RX = re.compile("|".join(f"(?:{p})" for p in PATTERNS), re.I)


def files():
    for p in sorted(CODE.rglob("*")):
        rel = p.relative_to(CODE).as_posix()
        if not p.is_file() or p.is_symlink(): continue
        if set(Path(rel).parts) & SKIP_DIRS: continue
        yield rel, p


def scan():
    hits = []
    for rel, p in files():
        if p.suffix not in TEXT or rel == SELF: continue
        t = p.read_text(encoding="utf-8", errors="replace")
        for m in RX.finditer(t):
            w = m.group(0)
            if w.lower() in allowed(rel): continue
            line = t.count("\n", 0, m.start()) + 1
            hits.append((rel, line, w))
    return hits


def without_credentials(b):
    if b[:8] != b"\x89PNG\r\n\x1a\n": return b
    out, i = bytearray(b[:8]), 8
    while i + 8 <= len(b):
        n = int.from_bytes(b[i:i + 4], "big")
        if b[i + 4:i + 8] != b"caBX": out += b[i:i + 12 + n]
        i += 12 + n
    return bytes(out)


def verify():
    bad, credentials, n = [], 0, 0
    for line in (CODE / "SHA256SUMS").read_text().splitlines():
        h, rel = line.split("  ", 1); p = CODE / rel; n += 1
        if not p.is_file(): bad.append(f"MISSING {rel}"); continue
        b = p.read_bytes()
        if hashlib.sha256(b).hexdigest() == h: continue
        if hashlib.sha256(without_credentials(b)).hexdigest() == h: credentials += 1; continue
        bad.append(f"DIFFERS {rel}")
    listed = {l.split("  ", 1)[1] for l in (CODE / "SHA256SUMS").read_text().splitlines()} | {"SHA256SUMS"}
    bad += [f"NOT LISTED {rel}" for rel, _ in files() if rel not in listed]
    for x in bad: print(x)
    print(f"{n} files listed: {n - len([x for x in bad if not x.startswith('NOT')]) - credentials} identical, "
          f"{credentials} PNG identical apart from Content Credentials, {len(bad)} problem(s)")
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true"); ap.add_argument("--out", type=Path); ap.add_argument("--zip", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify: verify()
    hits = scan()
    for rel, line, w in hits: print(f"HIT {rel}:{line}: {w}")
    n = sum(1 for _ in files())
    if hits: sys.exit(f"{len(hits)} hit(s) in {n} files: nothing written")
    print(f"scan clean: {n} files, {len(PATTERNS)} patterns")
    if a.check or not a.out: return
    if a.out.exists() and any(a.out.iterdir()): sys.exit(f"{a.out} is not empty")
    shipped = []
    for rel, p in files():
        if rel == "SHA256SUMS": continue
        dst = a.out / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
        os.chmod(dst, 0o755 if p.stat().st_mode & 0o100 else 0o644)
        shipped.append(rel)
    sums = [f"{hashlib.sha256((a.out / r).read_bytes()).hexdigest()}  {r}\n" for r in sorted(shipped)]
    (a.out / "SHA256SUMS").write_text("".join(sums))
    print(f"wrote {len(shipped)} files and SHA256SUMS to {a.out}")
    if a.zip:
        z = a.out.with_suffix(".zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(a.out.rglob("*")):
                if p.is_file(): zf.write(p, Path(a.out.name) / p.relative_to(a.out))
        print("wrote", z)


if __name__ == "__main__":
    main()
