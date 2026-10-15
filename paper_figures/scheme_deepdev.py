#!/usr/bin/env python3
import argparse, gzip, hashlib, json, math, os, sys, tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_scheme")
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import _paths as P
from flatwin import best_grok_window
from endpoints import series, knee_step
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

K = dict(train="#0072B2", train_acc="#000000", test="#D55E00", refit="#CC79A7", AH="#009E73", agop="#E69F00",
         null="#888888", window="#EBEBEB", grid="#E6E6E6", layer="#444444")
STYLE = {"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 8, "axes.titlesize": 8,
         "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555555", "axes.linewidth": 0.7,
         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300}
KFMT = FuncFormatter(lambda v, _: f"{v / 1000:g}")
FIRST, LAST, MED = "-.", "-", 1.4
CONFIGS = [("Depth 2", "seedD2_r0.5_s"), ("Depth 3", "T6_D3_cascade_r0.9_0.45_s"), ("Depth 4", "T18_D4_r1.0_1.0_0.4_s"),
           ("Depth 5", "T21_D5_r1.0_1.0_1.0_0.35_s")]


def sha(b): return hashlib.sha256(b).hexdigest()


def manifest():
    rows = P.DEV_MANIFEST.read_text().splitlines()
    return {l.split("\t")[0]: l.split("\t")[1].strip() for l in rows if l and not l.startswith("#")}


def runs_of(prefix, M):
    names = sorted((Path(k).name[:-5] for k in M if k.startswith("runs/" + prefix) and k.endswith(".json")),
                   key=lambda s: int(s[len(prefix):].split("_")[0]))
    return names


def read_all(names, data_dir, M, extra=()):
    want = {n: M["runs/" + n + ".json"] for n in names}; got = {}
    for path in sorted(Path(data_dir).rglob("*.tar.gz")):
        with tarfile.open(path) as tf:
            for m in tf.getmembers():
                nm = Path(m.name).name
                key = nm[:-8] if nm.endswith(".json.gz") else nm[:-5] if nm.endswith(".json") else None
                if not m.isfile() or key not in want or key in got:
                    continue
                raw = tf.extractfile(m).read(); data = gzip.decompress(raw) if m.name.endswith(".gz") else raw
                if sha(data) == want[key]:
                    got[key] = dict(d=json.loads(data), archive=path.name, member=m.name, sha256=sha(data))
    for root in extra:
        for path in sorted(list(Path(root).rglob("*.json.gz")) + list(Path(root).rglob("*.json"))):
            nm = path.name; key = nm[:-8] if nm.endswith(".json.gz") else nm[:-5]
            if key not in want or key in got:
                continue
            raw = path.read_bytes(); data = gzip.decompress(raw) if nm.endswith(".gz") else raw
            if sha(data) == want[key]:
                got[key] = dict(d=json.loads(data), archive=str(path.parent.name), member=nm, sha256=sha(data))
    missing = sorted(set(want) - set(got))
    assert not missing, f"not found with the manifest's SHA-256: {missing}"
    return got


def record(d):
    obs = d["obs"]; cfg = d["cfg"]; D = cfg["depth"]; p = cfg["p"]
    L = np.asarray(d["loss"], float); cm = (L[:-1] + L[1:]) / 2
    s, te = series(obs, "test_acc"); k = knee_step(cfg, obs)
    a, b, share = best_grok_window(cm, s, te, k)
    t10 = int(s[np.flatnonzero(te >= .1)[0]]); t90 = int(s[np.flatnonzero(te < .9)[-1] + 1])
    S = lambda key: series(obs, key)
    return dict(p=p, depth=D, cm=cm / math.log(p), stop=len(cm) - 1, W=[int(a), int(b)], share=float(share), t10=t10, t90=t90,
                test=(s, te), train=S("train_acc"), refit=S("refit_acc"), ahL=S(f"AH_full_h{D}"), ah1=S("AH_full_h1"),
                agL=S("agop30_ho"), agL0=S("agop30_ho_rand"), ag1=S("agopH1_4_ho"), ag10=S("agopH1_4_ho_rand"))


def hold(x, y, g):
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(y); x, y = x[ok], y[ok]
    return np.interp(g, x, y, left=y[0], right=y[-1]) if len(x) else np.full(len(g), np.nan)


def xmax_of(R):
    ends = [r["W"][1] for r in R]; t90 = [r["t90"] for r in R]
    xm = int(math.ceil((1.15 * max(np.median(ends), np.percentile(t90, 90)) + 1500) / 1000) * 1000)
    return min(xm, max(r["stop"] for r in R))


def draw(cols, out, stem="deep_development_overview"):
    plt.rcParams.update(matplotlib.rcParamsDefault); plt.rcParams.update(STYLE)
    n = len(cols); rows = ("ce", "acc", "agop", "ah")
    fig, axs = plt.subplots(len(rows), n, figsize=(6.75, 5.6), sharey="row", layout="constrained")
    stats = []
    for j, (title, R) in enumerate(cols):
        X = xmax_of(R); g = np.arange(0, X + 1, max(10, X // 2000), dtype=float)
        band = (float(np.median([r["W"][0] for r in R])), float(np.median([r["W"][1] for r in R])))
        th = dict(lw=0.45, alpha=0.28 if len(R) > 2 else 0.5, zorder=1)
        H = lambda key: np.array([hold(*r[key], g) for r in R])
        for i, row in enumerate(rows):
            ax = axs[i, j]
            ax.axvspan(*band, color=K["window"], lw=0, zorder=0)
            ax.grid(True, axis="y", color=K["grid"], lw=0.5); ax.set_xlim(0, X)
            ax.xaxis.set_major_formatter(KFMT); ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
            if row == "ce":
                C = np.array([hold(np.arange(len(r["cm"])), r["cm"], g) for r in R])
                for y in C: ax.plot(g, y, color=K["train"], **th)
                ax.plot(g, np.median(C, 0), color=K["train"], lw=MED, zorder=3); ax.set_ylim(0, 1.05)
            elif row == "acc":
                T = H("test")
                for y in T: ax.plot(g, y, color=K["test"], **th)
                ax.plot(g, np.median(H("train"), 0), color=K["train_acc"], lw=1.0, ls="--", zorder=3)
                ax.plot(g, np.median(H("refit"), 0), color=K["refit"], lw=MED, ls=":", zorder=4)
                ax.plot(g, np.median(T, 0), color=K["test"], lw=MED, zorder=5); ax.set_ylim(-0.02, 1.04)
            elif row == "agop":
                A = H("agL")
                for y in A: ax.plot(g, y, color=K["agop"], **th)
                ax.plot(g, np.median(H("agL0"), 0), color=K["null"], lw=0.9, ls=":", zorder=2)
                ax.plot(g, np.median(H("ag10"), 0), color=K["null"], lw=0.9, ls="--", zorder=2)
                ax.plot(g, np.median(A, 0), color=K["agop"], lw=MED, ls=LAST, zorder=3)
                ax.plot(g, np.median(H("ag1"), 0), color=K["agop"], lw=1.2, ls=FIRST, zorder=3); ax.set_ylim(0, 0.65)
            else:
                A = H("ahL")
                for y in A: ax.plot(g, y, color=K["AH"], **th)
                ax.plot(g, np.median(A, 0), color=K["AH"], lw=MED, ls=LAST, zorder=3)
                ax.plot(g, np.median(H("ah1"), 0), color=K["AH"], lw=1.2, ls=FIRST, zorder=3); ax.set_ylim(0, None)
            if i < len(rows) - 1:
                ax.tick_params(labelbottom=False)
        sh = [r["share"] for r in R]
        what = f"{sh[0]:.0%} of rise in W" if len(R) == 1 else f"median {np.median(sh):.0%} of rise in W"
        axs[0, j].set_title(f"{title} ({len(R)} run{'s' if len(R) > 1 else ''})\n{what}", loc="center")
        stats.append(dict(title=title, runs=[r["name"] for r in R], windows=[r["W"] for r in R], shares=sh,
                          t10=[r["t10"] for r in R], t90=[r["t90"] for r in R], band=list(band), xmax=X))
    top = max(float(np.nanmax(l.get_ydata())) for ax in axs[3] for l in ax.lines if len(l.get_ydata()) > 2) * 1.08
    for ax in axs[3]: ax.set_ylim(0, top)
    for i, lab in enumerate(["Training CE / log p\n(cycle mean)", "Accuracy", "Hidden AGOP alignment\n(legacy observer)",
                             "Hidden Fourier\n$A_H$"]):
        axs[i, 0].set_ylabel(lab)
    axs[-1, n // 2].set_xlabel("Updates (thousands)")
    h = [Line2D([], [], color=K["train"], lw=MED), Line2D([], [], color=K["train_acc"], lw=1.0, ls="--"),
         Line2D([], [], color=K["test"], lw=MED), Line2D([], [], color=K["refit"], lw=MED, ls=":"),
         Line2D([], [], color=K["agop"], lw=MED), Line2D([], [], color=K["null"], lw=0.9, ls="--"),
         Line2D([], [], color=K["layer"], lw=MED, ls=LAST), Line2D([], [], color=K["layer"], lw=1.2, ls=FIRST)]
    lab = ["Train CE", "Train acc.", "Held-out", "Refit (legacy)", "AGOP alignment", "Random frames", "Last hidden", "First hidden"]
    fig.legend(h, lab, loc="outside lower center", ncol=4, frameon=False, handlelength=1.8, columnspacing=1.0)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf", metadata={"Title": stem, "Author": "", "Creator": "scheme_deepdev.py", "CreationDate": None})
    fig.savefig(out / f"{stem}.png", dpi=200); plt.close(fig)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--development-data", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--development-results", type=Path, default=None, help="optional folder of loose development JSON files")
    a = ap.parse_args(); M = manifest()
    names = {t: runs_of(pre, M) for t, pre in CONFIGS}
    got = read_all([n for v in names.values() for n in v], a.development_data, M,
                   [a.development_results] if a.development_results else [])
    cols = []
    for t, _ in CONFIGS:
        R = []
        for nm in names[t]:
            r = record(got[nm]["d"]); r["name"] = nm; R.append(r)
        cols.append((t, R))
    stats = draw(cols, a.out.resolve())
    prov = json.loads((P.EXPECTED / "development_overview_exemplars.json").read_text())
    by = {r["name"]: r for _, R in cols for r in R}
    check = [dict(run=m["run"], window_same=by[m["run"]]["W"] == m["window"], share_same=abs(by[m["run"]]["share"] - m["share"]) < 1e-12,
                  t10_same=by[m["run"]]["t10"] == m["t10"], t90_same=by[m["run"]]["t90"] == m["t90"]) for m in prov]
    rep = dict(columns=stats, exemplar_check=check,
               sources=[dict(run=k, archive=v["archive"], member=v["member"], sha256=v["sha256"]) for k, v in sorted(got.items())])
    (a.out.resolve() / "scheme_deepdev_stats.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(dict(columns=[(s["title"], len(s["runs"]), s["band"], [round(x, 3) for x in s["shares"]]) for s in stats],
                          exemplar_check=check)))


if __name__ == "__main__":
    main()
