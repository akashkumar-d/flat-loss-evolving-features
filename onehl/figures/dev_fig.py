import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hl1_figs as F
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
src, out, stem, facet = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
runs = {}
for js in sorted(glob.glob(os.path.join(src, "*.json"))):
    m = json.load(open(js))["meta"]; z = np.load(js[:-5] + ".npz")
    runs.setdefault(m["config_id"], dict(cfg=m["config"], z=[]))["z"].append(z)
cols = sorted({(r["cfg"][facet], r["cfg"]["train_fraction"]) for r in runs.values()}, key=lambda g: (-g[0] if facet == "decay_V" else g[0], g[1]))
lws = sorted({r["cfg"]["decay_W"] for r in runs.values()}); evs = sorted({r["cfg"]["eta_V"] for r in runs.values()})
cm = plt.get_cmap("viridis"); colour = {lw: cm(0.9 * i / max(1, len(lws) - 1)) for i, lw in enumerate(lws)}
styles = {ev: ls for ev, ls in zip(sorted(evs, reverse=True), ["-", "--", ":", "-."])}
fig, axs = plt.subplots(3, len(cols), figsize=(F.TEXTW, 4.8), squeeze=False, constrained_layout=True)
for i in (1, 2):
    for j in range(1, len(cols)):
        axs[i, j].sharey(axs[i, 0])
for j, (fv, tf) in enumerate(cols):
    for cid, r in sorted(runs.items(), key=lambda kv: (kv[1]["cfg"]["decay_W"], -kv[1]["cfg"]["eta_V"])):
        c = r["cfg"]
        if (c[facet], c["train_fraction"]) != (fv, tf):
            continue
        t = r["z"][0]["step"]; sl = slice(0, None, max(1, len(t) // 800))
        ce = np.median([np.maximum(z["train_ce"], 1e-300) for z in r["z"]], 0)
        te = np.median([z["test_accuracy"] for z in r["z"]], 0)
        ot = r["z"][0]["obs_step"]; ah = np.median([z["obs_hidden_AH_add"] for z in r["z"]], 0)
        kw = dict(color=colour[c["decay_W"]], ls=styles[c["eta_V"]], lw=0.9)
        axs[0, j].semilogy(t[sl], ce[sl], **kw, label=f"$\\lambda_W$ {c['decay_W']:g}" + (f", $\\eta_V$ {c['eta_V']:g}" if len(evs) > 1 else ""))
        axs[1, j].plot(t[sl], te[sl], **kw); axs[2, j].plot(ot, ah, **kw)
    ttl = (f"p = {fv}" if facet == "p" else f"$\\lambda_V$ = {fv:g}") + f", {int(round(tf * 100))}% train"
    axs[0, j].set_title(ttl, fontsize=7, loc="left")
    for i in range(3):
        axs[i, j].xaxis.set_major_locator(MaxNLocator(3, integer=True)); axs[i, j].xaxis.set_major_formatter(F.kfmt)
    axs[2, j].set_xlabel("update")
axs[0, 0].set_ylabel("training CE"); axs[1, 0].set_ylabel("held-out accuracy"); axs[2, 0].set_ylabel("hidden Fourier $A_H$")
axs[1, 0].set_ylim(-0.02, 1.02)
from matplotlib.lines import Line2D
hc = [Line2D([], [], color=colour[lw], lw=1.2) for lw in lws]; lc = [f"$\\lambda_W$ = {lw:g}" for lw in lws]
if len(evs) > 1:
    hc += [Line2D([], [], color="0.3", ls=styles[ev], lw=1.0) for ev in sorted(evs, reverse=True)]; lc += [f"$\\eta_V$ = {ev:g}" for ev in sorted(evs, reverse=True)]
fig.legend(hc, lc, loc="outside lower center", ncol=min(4, len(hc)), fontsize=6, frameon=False, handlelength=1.8, columnspacing=1.0)
fig.savefig(os.path.join(out, stem + ".pdf")); fig.savefig(os.path.join(out, stem + ".png"), dpi=300)
print("saved", stem)
