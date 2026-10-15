#!/usr/bin/env python3
import argparse
import glob
import json
import os

import numpy as np

T0, T1 = 6500, 16500


def load(src):
    runs = []
    for f in sorted(glob.glob(os.path.join(src, "*.npz"))):
        z = np.load(f)
        ce = z["train_ce"].astype(float)
        lb = ce.copy()
        lb[:-1] = 0.5 * (ce[:-1] + ce[1:])
        runs.append(dict(ce=ce, lb=lb, acc=z["test_accuracy"].astype(float), tr=z["train_accuracy"],
                         os=z["obs_step"].astype(int), AH=z["obs_hidden_AH_add"].astype(float)))
    return runs


def first_after(a, th):
    i = np.flatnonzero(a[T0 + 1:] >= th)
    return T0 + 1 + int(i[0]) if len(i) else None


def nd(r, th):
    T = first_after(r["acc"], th)
    if T is None:
        return None, None, None
    seg = r["lb"][T0:T + 1]
    rm = np.maximum.accumulate(seg)
    return T, 100 * float(np.max((rm - seg) / rm)), 100 * float(seg[-1] / seg[0] - 1)


def f(x, d=1):
    return f"{x:.{d}f}"


def rr(xs, d=1, unit=""):
    lo, hi = min(xs), max(xs)
    return f"${f(lo, d)}$--${f(hi, d)}{unit}$" if f(lo, d) != f(hi, d) else f"${f(lo, d)}{unit}$"


def pct(xs, d=1):
    return rr([100 * x for x in xs], d, r"\%")


def bound(x):
    return next(f"{b:g}" for b in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10) if x < b)


def ints(xs):
    return f"{min(xs):,}--{max(xs):,}".replace(",", "{,}")


def cohort(src, score_path, th):
    R = load(src)
    S = json.load(open(score_path))
    n = len(R)
    assert n == S["n"] == 20, (n, S["n"])
    v = [r["ce"][T0:T1 + 1].max() / r["ce"][T0:T1 + 1].min() - 1 for r in R]
    dAH = [float(r["AH"][r["os"] == T1][0] - r["AH"][r["os"] == T0][0]) for r in R]
    wmin = [r["ce"][T0:T1 + 1].min() for r in R]
    h0 = [r["acc"][T0] for r in R]
    fit = [bool(np.all(r["tr"][T0:T1 + 1] == 1.0)) for r in R]
    pk = [T0 + int(np.argmax(r["lb"][T0:T1])) for r in R]
    hpk = [r["acc"][p] for r, p in zip(R, pk)]
    up = [100 * (r["lb"][p] / r["lb"][T0] - 1) for r, p in zip(R, pk)]
    main = [nd(r, th) for r in R]
    ok = [T is not None and T <= T1 - 500 and dd <= 0.5 and net >= 0 for T, dd, net in main]
    other = dict(P10=all(x <= 0.10 for x in v), FIT=all(fit), H20=all(x <= 0.2 for x in h0), F04=all(x >= 0.04 for x in dAH))
    joint = [o and x <= 0.10 and fi and h <= 0.2 and d >= 0.04 for o, x, fi, h, d in zip(ok, v, fit, h0, dAH)]
    assert sum(joint) == S["counts"]["joint"], (sum(joint), S["counts"]["joint"])
    late = [T is None or T > T1 - 500 for T, _, _ in main]
    never = [T is None for T, _, _ in main]
    dd_fail = [(not l) and (dd > 0.5 or net < 0) for l, (T, dd, net) in zip(late, main)]
    in_time = [(T, dd, net) for (T, dd, net), l in zip(main, late) if not l]
    return dict(n=n, K=sum(joint), wilson=S["joint_wilson95"], v=v, dAH=dAH, wmin=wmin, h0=h0, other=other,
                pk=pk, hpk=hpk, up=up, late=sum(late), never=sum(never), dd_fail=sum(dd_fail),
                dd_fail_vals=[dd for (T, dd, net), b in zip(main, dd_fail) if b],
                net_in_time=[net for T, dd, net in in_time],
                final_never=[r["acc"][-1] for r, nv in zip(R, never) if nv],
                nd50=[nd(r, 0.5) for r in R], nd70=[nd(r, 0.7) for r in R], nd90=main if th == 0.9 else [nd(r, 0.9) for r in R],
                final=[r["acc"][-1] for r in R], clause_fail=dict(
                    P10=sum(x > 0.10 for x in v), FIT=sum(not x for x in fit), H20=sum(x > 0.2 for x in h0),
                    F04=sum(x < 0.04 for x in dAH), ND=sum(not x for x in ok)))


def fragment(A, B=None):
    assert all(A["other"].values()), A["other"]
    logp = np.log(23)
    s = []
    s.append(r"""\subsection{Pre-registered tests of recovery without CE decrease}\label{paper:app:nodecrease}
In the headline cohort the network's held-out rise coincides with a
$1.5$--$3.4\%$ decrease of the cycle-mean CE $\bar L_t=(L_t+L_{t+1})/2$
(Figure~\ref{fig:h-cohort}c). We therefore looked for a recipe whose network
recovers with no CE decrease. After the headline evaluation, we screened the 12
recipes of the second development search (four pairs each). Only one had $\bar L$ higher at
the first $90\%$ held-out crossing than at the first $10\%$ crossing in all four
pairs, never more than $0.25\%$ below its running maximum in between: $p=23$,
width 512, training fraction $0.8$, $c=24$, $\eta_W=\eta_V=0.03$,
$\lambda_W=0.015$ and $\lambda_V=0.2$. A window rule applied to these four pairs
only chose $I=[6500,16500]$; its first version, which forbade a decrease
anywhere in the window, admitted no window and was relaxed once to the recovery
interval.
""")
    s.append(r"""
Before any new run, we locked the recipe, the window, 20 fresh
initialization/split pairs, a 20{,}000-update horizon, the code and a decision
rule. The primary joint endpoint requires $v(I)\le10\%$, perfect training
accuracy on $I$, held-out accuracy at most $20\%$ at $t_0$, $\Delta\AH\ge0.04$,
and no CE decrease during recovery: the first $90\%$ held-out crossing $T_{90}$
occurs by $t_1-500$, $\bar L$ never falls more than $0.5\%$ below its running
maximum on $[t_0,T_{90}]$, and $\bar L(T_{90})\ge\bar L(t_0)$. The cohort was to
replace Figure~\ref{fig:h-cohort} if the joint endpoint held in at least 18 of
the 20 runs.
""")
    lo, hi = A["wilson"]
    parts = []
    if A["dd_fail"]:
        parts.append(f"in {A['dd_fail']}, $\\bar L$ falls {rr(A['dd_fail_vals'], 1, chr(92) + '%')} below its running "
                     f"maximum before $T_{{90}}$")
    if A["late"]:
        t = f"in {A['late']}, $T_{{90}}$ comes after $t_1-500$"
        if A["never"]:
            t += (f" ({A['never']} of them do not reach $90\\%$ within 20{{,}}000 updates, ending at "
                  f"{pct(A['final_never'])})" if A["never"] > 1 else
                  f" (one does not reach $90\\%$ within 20{{,}}000 updates and ends at {pct(A['final_never'])})")
        parts.append(t)
    s.append(f"""
The joint endpoint holds in {A['K']} of 20 runs (95\\% Wilson interval
$[{lo:.2f},{hi:.2f}]$), so the confirmation fails (Figure~\\ref{{fig:p23-confirm}}).
Every run meets the other four clauses: $v(I)$ is {pct(A['v'])}, $\\Delta\\AH$ is
{rr(A['dAH'], 3)}, the window-minimum CE is {rr(A['wmin'], 3)} ($\\approx{np.median(A['wmin']) / logp:.2f}\\log p$), and
held-out accuracy at $t_0$ is at most {pct([max(A['h0'])])}. The no-decrease clause fails in
{20 - A['K']} runs: {'; '.join(parts)}. Where $T_{{90}}\\le t_1-500$, $\\bar L(T_{{90}})$ is still
{rr(A['net_in_time'], 1, chr(92) + '%')} above $\\bar L(t_0)$.
""")
    dd50 = [dd for T, dd, net in A["nd50"]]
    dd70 = [dd for T, dd, net in A["nd70"]]
    assert all(T is not None for T, _, _ in A["nd50"] + A["nd70"])
    s.append(f"""
These failures concern the end of the recovery. Post hoc, and not part of the
locked endpoints: each run's $\\bar L$ rises by {rr(A['up'], 1, chr(92) + '%')} to a peak at updates
{ints(A['pk'])}, when the network's held-out accuracy is already {pct(A['hpk'], 0)}
(median ${100 * np.median(A['hpk']):.0f}\\%$). Up to the first $50\\%$ and $70\\%$ crossings, no run's
$\\bar L$ falls more than ${bound(max(dd50))}\\%$ and ${bound(max(dd70))}\\%$, respectively, below its running maximum. A
pattern noticed on evaluation runs is not a confirmation.""")
    if B is None:
        s.append(r""" We keep the headline cohort in Figure~\ref{fig:h-cohort} and do not claim a
recipe in which the network recovers without any CE decrease.
""")
    else:
        lo2, hi2 = B["wilson"]
        passed = B["K"] >= 18
        okc = lambda trip: sum(T is not None and T <= T1 - 500 and dd <= 0.5 and net >= 0 for T, dd, net in trip)
        T50s = [T for T, _, _ in B["nd50"] if T is not None]
        dd50b = [dd for T, dd, _ in B["nd50"] if T is not None]
        net50b = [net for T, _, net in B["nd50"] if T is not None]
        reach90 = sum(T is not None for T, _, _ in B["nd90"])
        s.append(f"""

We then tested this observation on 20 new pairs under a second lock, written
before they were launched: the same recipe, window, horizon, clauses and
18-of-20 rule, with $T_{{90}}$ replaced by the first $50\\%$ crossing $T_{{50}}$. The
joint endpoint holds in {B['K']} of 20 runs (95\\% Wilson interval
$[{lo2:.2f},{hi2:.2f}]$), so this confirmation {'succeeds' if passed else 'also fails'}
(Figure~\\ref{{fig:p23-confirm}}, bottom, and Figure~\\ref{{fig:h-cohort}}a).""")
        if passed:
            s.append(f""" Held-out accuracy rises from at most {pct([max(B['h0'])])} at $t_0$
to $50\\%$ at updates {ints(T50s)}, while $\\bar L$ never falls more than ${bound(max(dd50b))}\\%$
below its running maximum and ends {rr(net50b, 1, chr(92) + '%')} above $\\bar L(t_0)$. As secondary
endpoints, the same holds up to the first $70\\%$ crossing in {okc(B['nd70'])} runs and up to the
first $90\\%$ crossing in {okc(B['nd90'])}, and {reach90} runs reach $90\\%$ within 20{{,}}000 updates.
The confirmed statement covers recovery to $50\\%$ held-out accuracy for this recipe
and window, not recovery to $90\\%$.
""")
        else:
            miss = ", ".join(f"{k} in {n}" for k, n in B["clause_fail"].items() if n)
            s.append(f""" The failing runs miss these clauses: {miss}. We keep the headline cohort
in Figure~\\ref{{fig:h-cohort}} and do not claim a recipe in which the network recovers without
any CE decrease.
""")
    s.append(r"""
\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{20260930_fig2/p23_confirmA.pdf}""")
    if B is not None:
        s.append(r"""
\includegraphics[width=\textwidth]{20260930_fig2/p23_confirmB.pdf}""")
    which = r"$90\%$ (top) or $50\%$ (bottom)" if B else r"$90\%$"
    counts = (f"{A['K']} of 20 runs (top) and {B['K']} of 20 (bottom)" if B else f"{A['K']} of 20 runs")
    title = "Pre-registered $p=23$ cohorts" if B else "Pre-registered $p=23$ cohort"
    each = " each" if B else ""
    s.append("\n\\caption{\\textbf{" + title + "} (20 fresh pairs" + each + r"""; window
$I=[6500,16500]$ chosen on four development pairs and locked before these
runs). (a)~Cycle-mean training CE relative to its value at $t_0$; dots mark
each run's first """ + which + r""" held-out crossing on $I$. (b)~Held-out
accuracy of the network and of the train-only refit (dotted; 500-update grid).
(c)~$\AH$. Thin curves are runs and thick curves medians. The joint endpoint,
which forbids a CE decrease before the dot, holds in """ + counts + r"""; 18 were
required.}
\label{fig:p23-confirm}
\end{figure}
""")
    return wrap("".join(s))


def wrap(text, width=80):
    import textwrap
    out = []
    for para in text.split("\n\n"):
        if para.lstrip().startswith("\\begin{figure}"):
            out.append(para)
            continue
        lines = para.split("\n")
        head = [lines.pop(0)] if lines and lines[0].startswith("\\subsection") else []
        body = " ".join(l.strip() for l in lines if l.strip())
        filled = textwrap.fill(body, width=width, break_long_words=False, break_on_hyphens=False) if body else ""
        out.append("\n".join(head + ([filled] if filled else [])))
    return "\n\n".join(out).strip() + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", nargs=2, required=True, metavar=("COMPACT", "SCORE"))
    ap.add_argument("--b", nargs=2, metavar=("COMPACT", "SCORE"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--numbers", help="also write the computed numbers as JSON")
    a = ap.parse_args()
    A = cohort(*a.a, 0.9)
    B = cohort(*a.b, 0.5) if a.b else None
    open(a.out, "w").write(fragment(A, B))
    if a.numbers:
        json.dump(dict(A=A, B=B), open(a.numbers, "w"), indent=1, default=float)
    print(open(a.out).read())


if __name__ == "__main__":
    main()
