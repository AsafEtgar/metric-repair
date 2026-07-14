"""Emit Appendix A: the same analysis as Section 5, on the DENSE family -- and why it cannot carry the paper.

THE APPENDIX'S THESIS, AND WHY IT IS NOT A SECOND SECTION 5.

Section 5's spine is the corruption direction. THE DENSE FAMILY HAS NONE. Gamma(n,p) is a G(n,p) topology
carrying i.i.d. random weights: it is non-metric *intrinsically*, not because anything was done to it. There
is no planted corrupted set B and no true weights w0. So:

  * there is no inflate/deflate axis to split on -- Section 5's organising principle does not exist here;
  * Section 6's set x correction cross CANNOT BE RUN, because it needs w0. The dense family can answer
    Question 1 and only Question 1.

That alone would make it a supporting family rather than a spine. What the numbers add is worse.

WHAT THE NUMBERS SAY, AND WHAT EACH GATE PROTECTS.

  A.1  IT IS NOT ONE FAMILY, IT IS THREE, AND THEY DO NOT POOL.

         exp1   geometric            weights Geom(1-p), COUPLED   p in {0.3, 0.5}   |H|/m = 0.168
         exp2a  geometric            weights Geom(1-p), COUPLED   p = n^-alpha      |H|/m = 0.001   <-- METRIC
         exp2b  decoupled_geometric  weights Geom(0.5), FIXED     p = n^-alpha      |H|/m = 0.067

       The coupled model's mean weight is 1/(1-p). Sweep p downward and the weights collapse onto 1 -- and a
       graph whose edges all weigh 1 is metric BY CONSTRUCTION. exp2a IS that collapse: 600 of the small
       grid's 2,460 tasks sit at |H|/m = 0.001. It is the dense family's own jitter sweep, and G2 checks it.

  A.2  THE OPTIMALITY STORY IS AN ARTEFACT OF THE METRIC SWEEP. Pooled, the small dense grid "solves 69.8%
       of instances for GMR and 36.0% for IOMR". Split by sweep:

         sweep   |H|/m    gmr_ilp   iomr_ilp
         exp1    0.168     49.0%       0.0%      <-- the only genuinely dense AND broken sweep
         exp2a   0.001    100.0%     100.0%      <-- converges because there is nothing to solve
         exp2b   0.067     83.3%      47.5%

       iomr_ilp converges on ZERO of the 1,260 exp1 tasks. And the 607 tasks that DO carry an exact IOMR
       optimum sit at |H|/m = 0.0025, against the grid's 0.089. THE IOMR OPTIMUM EXISTS ONLY WHERE THERE IS
       NOTHING TO REPAIR. G3 and G4 check both halves; neither can pass vacuously.

  A.3  WHERE THE GMR OPTIMUM *IS* KNOWN, ONE METHOD SOLVES IT. On exp1, l1sep_gmr's |S|/OPT is 1.000 --
       exactly. It does not approximate the GMR optimum; it finds it. Nothing else comes close (pivot 2.251,
       spc_gmr 2.294, the roundings 2.68-2.69). This is the one clean result the dense family gives, and it
       is a real one. G8 checks it.

  A.4  THE TWO LARGE SWEEPS DO NOT POOL EITHER, AND THE POOLED MEDIAN IS A FICTION. spc_gmr rewrites 0.225 of
       exp1 and 0.418 of exp2b; pivot rewrites 0.457 and 0.278. THEY SWAP. Pooled, pivot "wins" by 0.008 --
       a fact about the mixture and about neither sweep. Same disease as Section 5's pooling; different axis.
       Here it is the WEIGHT MODEL that flips the ranking, not the corruption. G5 requires the flip.

  A.5  AND THE SPARSITY DEFECT IS INVISIBLE HERE, WHICH IS THE POINT. On exp1 the graph carries 298,617 edges
       and its completion carries 780,625 -- a factor of 2.6. On the RGG at n = 3000 the same factor is 258.
       So pivot and left_edge, which complete the graph, look respectable on a dense graph (1.44x DOMR's
       memory) and catastrophic on a sparse one (2.6x). Meanwhile l1sep_gmr -- whose LP carries a variable per
       EDGE -- inverts: 2.40x on dense, 1.04x on sparse. THE TWO FAMILIES ARE THE TWO SIDES OF THE m-VERSUS-
       n^2 TRADE, and a paper that benchmarks only the dense one cannot see either defect. G7 checks it.

  inputs   analysis/rows_with_ratio.csv        small dense: n=100-300, the ILP
           analysis/large/rows_with_ratio.csv  large dense: n=1000-2000
           analysis/rgg_large/rgg_rows_with_ratio.csv   the RGG, for the contrast in A.5
  outputs  <texdir>/tab_dense.tex, <texdir>/appdense_macros.tex
           <figdir>/fig_a1_dense.pdf

  usage    sage -python experiments/appendix_dense.py --texdir "<paper>/tables" --figdir "<paper>/figures"
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

SMALL = "analysis/rows_with_ratio.csv"
LARGE = "analysis/large/rows_with_ratio.csv"
RGG = "analysis/rgg_large/rgg_rows_with_ratio.csv"

GMR = ["l1sep_gmr", "spc_gmr", "pivot", "gmr_bestofk", "gmr_rand", "gmr_thr_naive"]
IOMR = ["l1sep_iomr", "spc_iomr", "left_edge", "iomr_bestofk", "iomr_rand", "iomr_thr_naive",
        "iomr_regiongrow"]
SWEEPS = ["exp1", "exp2a", "exp2b"]
METRIC_SWEEP = "exp2a"          # the coupled p-sweep: weights collapse onto 1, the graph goes metric
DENSE_SWEEP = "exp1"            # the only sweep that is BOTH dense and broken

STY = {"l1sep_gmr": ("#0072B2", "o"), "spc_gmr": ("#009E73", "s"), "pivot": ("#D55E00", "^"),
       "gmr_bestofk": ("#CC79A7", "D"), "gmr_rand": ("#E69F00", "v"), "gmr_thr_naive": ("#56B4E9", "P"),
       "l1sep_iomr": ("#0072B2", "o"), "spc_iomr": ("#009E73", "s"), "left_edge": ("#D55E00", "^"),
       "iomr_bestofk": ("#CC79A7", "D"), "iomr_rand": ("#E69F00", "v"),
       "iomr_thr_naive": ("#56B4E9", "P"), "iomr_regiongrow": ("#999999", "X"), "domr": ("black", None)}
NAME = {"gmr_thr_naive": "gmr_thr", "iomr_thr_naive": "iomr_thr", "iomr_regiongrow": "iomr_rgrow"}
MACNAME = {"l1sep_gmr": "LoneGmr", "l1sep_iomr": "LoneIomr", "gmr_thr_naive": "GmrThr",
           "iomr_thr_naive": "IomrThr", "iomr_regiongrow": "IomrRgrow"}


def nm(a):
    return NAME.get(a, a)


def tex(a):
    return r"\code{%s}" % nm(a).replace("_", r"\_")


def key(a):
    k = MACNAME.get(a) or "".join(w.title() for w in a.split("_"))
    if not k.isalpha():
        raise SystemExit(f"FATAL: macro fragment {k!r} is not pure letters -- LaTeX cannot read it.")
    return k


def _n(x):
    return f"{int(x):,}".replace(",", "{,}")


def load(path):
    if not os.path.exists(path):
        raise SystemExit(f"FATAL: {path} missing.")
    d = pd.read_csv(path, low_memory=False)
    for c in ("n", "p", "alpha", "E", "H", "size", "valid", "ratio", "wall", "peak_mb"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.drop_duplicates(subset=["task", "algo"], keep="first")
    d["sm"] = d["size"] / d.E
    return d


def ok(d):
    return d[d.status.eq("ok") & d.valid.eq(1)]


def hm(d):
    t = d.drop_duplicates("task")
    return float((t.H / t.E).median())


# ----------------------------------------------------------------------------
def gate(S, L, R):
    fails = []

    def chk(c, name, obs):
        print(f"  [{'PASS' if c else 'FAIL'}] {name:<56} {obs}")
        if not c:
            fails.append(name)

    # G1  the control, on both dense grids
    for lab, d in (("small", S), ("large", L)):
        D = ok(d)[ok(d).algo.eq("domr")].dropna(subset=["size", "H"])
        bad = int((D["size"] != D.H).sum())
        chk(len(D) > 0 and bad == 0, f"G1 domr's cover IS H ({lab} dense)",
            f"{len(D)} tasks, {bad} mismatches")

    # G2  exp2a IS the dense family's jitter sweep. The claim is that the COUPLING drives it metric; if
    #     |H|/m there ever stopped being ~0, the appendix's first argument would be wrong.
    h2a, h1 = hm(S[S.exp.eq(METRIC_SWEEP)]), hm(S[S.exp.eq(DENSE_SWEEP)])
    chk(h2a < 0.01 and h1 > 0.10, f"G2 {METRIC_SWEEP} is near-METRIC (the coupling collapses the weights)",
        f"|H|/m: {METRIC_SWEEP} {h2a:.4f} vs {DENSE_SWEEP} {h1:.3f}")

    # G3  *** THE HEADLINE ***  iomr_ilp must converge on ZERO exp1 tasks. If it ever converges there, the
    #     dense family gains an IOMR optimum and this appendix's central claim is void.
    I = S[S.exp.eq(DENSE_SWEEP) & S.algo.eq("iomr_ilp")]
    n_ok = ok(I).task.nunique()
    chk(n_ok == 0, f"G3 iomr_ilp converges on ZERO {DENSE_SWEEP} tasks",
        f"{n_ok} of {I.task.nunique()}")

    # G4  and the exact-IOMR subset is the near-metric one. Anti-vacuity for G3: it is not that IOMR is hard
    #     everywhere, it is that its optimum is available only where there is nothing to repair.
    E = S[S.variant.eq("IOMR") & S.ref_kind.eq("exact")].drop_duplicates("task")
    hE = float((E.H / E.E).median())
    hAll = hm(S)
    chk(hE < 0.25 * hAll, "G4 the exact-IOMR subset is the near-metric one",
        f"|H|/m {hE:.4f} vs the grid's {hAll:.4f}  ({len(E)} tasks)")

    # G5  THE FLIP. spc_gmr and pivot must swap between exp1 and exp2b, or the "do not pool" claim is empty.
    o = ok(L)
    piv = {a: {e: o[o.algo.eq(a) & o.exp.eq(e)].sm.median() for e in ("exp1", "exp2b")}
           for a in ("spc_gmr", "pivot")}
    flip = (piv["spc_gmr"]["exp1"] < piv["pivot"]["exp1"]) and \
           (piv["spc_gmr"]["exp2b"] > piv["pivot"]["exp2b"])
    chk(flip, "G5 the two large sweeps do not pool (spc/pivot SWAP)",
        f"spc_gmr {piv['spc_gmr']['exp1']:.3f}->{piv['spc_gmr']['exp2b']:.3f}   "
        f"pivot {piv['pivot']['exp1']:.3f}->{piv['pivot']['exp2b']:.3f}")

    # G6  |H| IS the GMR optimum on exp1 -- the same fact inflation gives on the RGG, and for the same reason:
    #     intrinsic non-metricity is inflation-like, every heavy edge exceeds its own detour.
    T = S.drop_duplicates("task").set_index("task")[["H", "E", "exp"]]
    ilp = ok(S)[ok(S).algo.eq("gmr_ilp")].dropna(subset=["size"])
    j = T.loc[ilp.task.unique()].copy()
    j["opt"] = ilp.set_index("task")["size"]
    j = j[j.exp.eq(DENSE_SWEEP)]
    oh = float((j.opt / j.H).median())
    chk(abs(oh - 1.0) < 0.02, f"G6 |H| IS the GMR optimum on {DENSE_SWEEP}",
        f"OPT/|H| = {oh:.3f}  ({len(j)} tasks)")

    # G7  THE COMPLETION IS CHEAP HERE. That is why the sparsity defect hides on a dense graph -- and it is
    #     the appendix's argument for why the paper needs the sparse family.
    t1 = L[L.exp.eq(DENSE_SWEEP)].drop_duplicates("task")
    md, nd = float(t1.E.median()), int(t1.n.median())
    blow_d = (nd * (nd - 1) / 2) / md
    tr = R[R.sweep.isin(["S1", "S1d"])].drop_duplicates("task")
    tr = tr[tr.n.eq(tr.n.max())]
    mr, nr = float(tr.E.median()), int(tr.n.median())
    blow_r = (nr * (nr - 1) / 2) / mr
    chk(blow_d < 5 and blow_r > 100, "G7 the completion is CHEAP on dense and ruinous on sparse",
        f"dense {blow_d:.1f}x (m={int(md):,}) vs RGG {blow_r:.0f}x (m={int(mr):,})")

    # G8  l1sep_gmr SOLVES GMR exactly on exp1. The one clean result the dense family gives.
    E1 = ok(S)[ok(S).exp.eq(DENSE_SWEEP) & ok(S).ref_kind.eq("exact")].dropna(subset=["ratio"])
    r = float(E1[E1.algo.eq("l1sep_gmr")].ratio.median())
    chk(abs(r - 1.0) < 1e-9, f"G8 l1sep_gmr's |S|/OPT is EXACTLY 1 on {DENSE_SWEEP}", f"{r:.4f}")
    return fails, dict(blow_d=blow_d, blow_r=blow_r, md=md, nd=nd, mr=mr, nr=nr, oh=oh, piv=piv)


# ----------------------------------------------------------------------------
def emit_tab(S):
    """|S|/OPT on exp1 -- the ONE place the dense family has an optimum on a broken graph."""
    E = ok(S)[ok(S).exp.eq(DENSE_SWEEP) & ok(S).ref_kind.eq("exact")].dropna(subset=["ratio"])
    g = E[E.variant.eq("GMR")].groupby("algo").ratio.median().sort_values()
    g = g[[a for a in g.index if a in GMR]]
    nt = E.task.nunique()
    cap = (r"\caption{\textbf{Where the dense family \emph{does} have an optimum, one method finds it.} "
           r"$|S|/\mathrm{OPT}$ on \code{exp1} --- the only sweep that is both genuinely dense and genuinely "
           r"broken --- over the %s instances where the \GMR{} integer program converged. \code{l1sep\_gmr} "
           r"is exact. \textbf{There is no \IOMR{} column because there is no \IOMR{} optimum:} "
           r"\code{iomr\_ilp} converges on \emph{none} of these instances.}" % _n(nt))
    out = [r"% GENERATED by experiments/appendix_dense.py -- DO NOT EDIT. Regenerate.",
           r"\begin{table}[t]\centering\footnotesize", cap, r"\label{tab:dense}",
           r"\begin{tabular}{@{}lr@{}}", r"\toprule",
           r"algorithm & $|S|/\mathrm{OPT}$ \\", r"\midrule"]
    for a, v in g.items():
        s = r"$\mathbf{%.3f}$" % v if abs(v - 1.0) < 1e-9 else r"$%.3f$" % v
        out.append(r"%s & %s \\" % (tex(a), s))
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def fig(S, L, figdir):
    fig_, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    # --- 1: the three sweeps are three families. |H|/m says so.
    ax = axes[0]
    labs, vals, cols = [], [], []
    for e in SWEEPS:
        labs.append(e)
        vals.append(hm(S[S.exp.eq(e)]))
        cols.append("#D55E00" if e == METRIC_SWEEP else "#0072B2")
    b = ax.bar(labs, vals, color=cols, width=0.55)
    ax.set_ylabel("$|H|/m$   (non-metric fraction)", fontsize=9)
    ax.set_title("three sweeps, three families", fontsize=10)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    ax.text(1.0, 0.075, "coupled weights\ncollapse -> METRIC", ha="center", fontsize=7.5, color="#D55E00")
    ax.set_ylim(0, 0.21)
    ax.grid(alpha=0.25, lw=0.5, axis="y")

    # --- 2: |S|/OPT on exp1, up the n-ladder. The optimum exists only here, and only for GMR.
    ax = axes[1]
    E = ok(S)[ok(S).exp.eq(DENSE_SWEEP) & ok(S).ref_kind.eq("exact") & ok(S).variant.eq("GMR")]
    for a in GMR:
        A = E[E.algo.eq(a)].dropna(subset=["ratio"])
        if A.empty:
            continue
        g = A.groupby("n").ratio.median().sort_index()
        c, m = STY[a]
        ax.plot(g.index, g.values, marker=m, ms=4, lw=1.6, color=c, label=nm(a))
    ax.axhline(1.0, color="black", lw=1.4, ls="--")
    ax.set_yscale("log"); ax.set_xlabel("$n$", fontsize=9)
    ax.set_ylabel(r"$|S| \,/\, \mathrm{OPT}$   $\downarrow$", fontsize=9)
    ax.set_title(r"exp1: $\mathtt{l1sep\_gmr}$ is exact", fontsize=10)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.25, lw=0.5)

    # --- 3: the flip. The WEIGHT MODEL swaps the ranking, exactly as the corruption does on the RGG.
    ax = axes[2]
    o = ok(L)
    for a in GMR + IOMR:
        v = [o[o.algo.eq(a) & o.exp.eq(e)].sm.median() for e in ("exp1", "exp2b")]
        if not np.isfinite(v).all():
            continue
        c, m = STY[a]
        ax.plot([0, 1], v, marker=m, ms=6, lw=2.0, color=c,
                ls=":" if a in IOMR else "-")
        ax.annotate(nm(a), (-0.05, v[0]), fontsize=7, color=c, va="center", ha="right")
    ax.set_xlim(-0.85, 1.25); ax.set_xticks([0, 1])
    ax.set_xticklabels(["exp1\n(coupled, $p{=}0.3,0.5$)", "exp2b\n(decoupled, $p$-sweep)"], fontsize=8)
    ax.set_ylabel(r"$|S|/m$   $\downarrow$", fontsize=9)
    ax.set_title("the weight model swaps the ranking", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5, axis="y")

    fig_.tight_layout()
    for e in ("pdf", "png"):
        fig_.savefig(os.path.join(figdir, f"fig_a1_dense.{e}"), dpi=160, bbox_inches="tight")
    plt.close(fig_)


def emit_macros(S, L, R, X):
    M = [r"% GENERATED by experiments/appendix_dense.py -- DO NOT EDIT. Every number Appendix A quotes."]

    def mac(k, v):
        M.append(r"\newcommand{\app%s}{%s}" % (k, v))

    mac("SmallTasks", _n(S.task.nunique()))
    mac("LargeTasks", _n(L.task.nunique()))
    for e in SWEEPS:
        E = S[S.exp.eq(e)]
        tag = e.replace("exp", "E").title().replace("e", "")     # exp1 -> E1 ; exp2a -> E2A
        tag = {"exp1": "One", "exp2a": "TwoA", "exp2b": "TwoB"}[e]
        mac(tag + "Tasks", _n(E.task.nunique()))
        mac(tag + "Hm", "%.3f" % hm(E))
        for al, k in (("gmr_ilp", "GmrIlp"), ("iomr_ilp", "IomrIlp")):
            A = E[E.algo.eq(al)]
            mac(tag + k, "%.1f" % (100 * ok(A).task.nunique() / max(A.task.nunique(), 1)))
    # the exact-IOMR subset
    E = S[S.variant.eq("IOMR") & S.ref_kind.eq("exact")].drop_duplicates("task")
    mac("IomrExactTasks", _n(len(E)))
    mac("IomrExactHm", "%.4f" % float((E.H / E.E).median()))
    mac("GridHm", "%.3f" % hm(S))
    # |S|/OPT on exp1
    E1 = ok(S)[ok(S).exp.eq(DENSE_SWEEP) & ok(S).ref_kind.eq("exact")].dropna(subset=["ratio"])
    for a in GMR:
        A = E1[E1.algo.eq(a)]
        if len(A):
            mac("Opt" + key(a), "%.3f" % A.ratio.median())
    mac("OptOverH", "%.3f" % X["oh"])
    # the flip
    for a in ("spc_gmr", "pivot"):
        mac("Sm" + key(a) + "One", "%.3f" % X["piv"][a]["exp1"])
        mac("Sm" + key(a) + "TwoB", "%.3f" % X["piv"][a]["exp2b"])
    o = ok(L)
    pooled = {a: o[o.algo.eq(a)].sm.median() for a in ("spc_gmr", "pivot")}
    mac("SmPooledSpc", "%.3f" % pooled["spc_gmr"])
    mac("SmPooledPivot", "%.3f" % pooled["pivot"])
    mac("SmPooledGap", "%.3f" % abs(pooled["spc_gmr"] - pooled["pivot"]))
    # limits on exp1 at scale
    E1L = L[L.exp.eq(DENSE_SWEEP)]
    nt = E1L.task.nunique()
    for a in ("l1sep_gmr", "l1sep_iomr", "gmr_bestofk", "iomr_bestofk"):
        A = E1L[E1L.algo.eq(a)]
        mac("Ret" + key(a), "%.0f" % (100 * len(ok(A)) / max(nt, 1)))
    # the completion, dense vs sparse
    mac("DenseM", _n(X["md"]))
    mac("DenseN", X["nd"])
    mac("DenseBlowup", "%.1f" % X["blow_d"])
    mac("RggM", _n(X["mr"]))
    mac("RggN", X["nr"])
    mac("RggBlowup", "%.0f" % X["blow_r"])
    # memory: the inversion
    for lab, d, sw in (("Dense", L[L.exp.eq(DENSE_SWEEP)], None), ("Rgg", R[R.sweep.isin(["S1", "S1d"])], 1)):
        K = d[d.status.eq("ok")]
        pv = K.pivot_table(index="task", columns="algo", values="peak_mb")
        rel = pv.div(pv["domr"], axis=0).median()
        for a in ("pivot", "l1sep_gmr"):
            if a in rel:
                mac("Mem" + key(a) + lab, "%.2f" % rel[a])
    return "\n".join(M)


def paper_dir(p, want):
    if not os.path.isdir(p):
        raise SystemExit(f"FATAL: '{p}' is not a directory.")
    p = os.path.abspath(p)
    if not os.path.exists(os.path.join(os.path.dirname(p), "story.tex")):
        raise SystemExit(f"FATAL: no story.tex beside '{p}'. That is not the paper.")
    if os.path.basename(p) != want:
        raise SystemExit(f"FATAL: '{p}' is not {want}/.")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texdir", required=True)
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    texdir, figdir = paper_dir(a.texdir, "tables"), paper_dir(a.figdir, "figures")

    S, L, R = load(SMALL), load(LARGE), load(RGG)
    print(f"small dense: {S.task.nunique()} tasks   large dense: {L.task.nunique()} tasks\n")
    print("GATE -- nothing is written until these pass")
    fails, X = gate(S, L, R)
    if fails and not a.force:
        raise SystemExit(f"\n*** {len(fails)} GATE FAILURE(S): {fails}. NOT writing. ***")

    tab = emit_tab(S)
    macros = emit_macros(S, L, R, X)
    if "nan" in tab.lower() or "nan" in macros.lower():
        raise SystemExit("*** GATE FAILED: rendered LaTeX contains nan. NOT writing. ***")
    import re
    bad = [m for m in re.findall(r"\\newcommand\{\\([^}]*)\}", macros) if not m.isalpha()]
    if bad:
        raise SystemExit(f"*** GATE FAILED: macro names not pure letters: {bad}. NOT writing. ***")

    with open(os.path.join(texdir, "tab_dense.tex"), "w") as f:
        f.write(tab + "\n")
    with open(os.path.join(texdir, "appdense_macros.tex"), "w") as f:
        f.write(macros + "\n")
    fig(S, L, figdir)
    print(f"\n  wrote {texdir}/tab_dense.tex")
    print(f"  wrote {texdir}/appdense_macros.tex  ({macros.count('newcommand')} macros)")
    print(f"  wrote {figdir}/fig_a1_dense.pdf")
    print("\nAll gates passed." if not fails else "\n!! UNDER --force.")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
