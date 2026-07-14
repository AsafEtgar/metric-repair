"""Emit everything Section 5 (Benchmarks) quotes: the optimality table, four figures, and every prose macro.

SECTION 5 RESTS ON THE RGG, and this file is why that is defensible:

  * it is the only synthetic family with a PLANTED corrupted set B and TRUE weights w0, so it is the only one
    that can carry Question 2 into Section 6;
  * its exact optimum is available on 96% of the small grid (the dense grid manages 70% for GMR and 36% for
    IOMR), so the approximation ratios below are ratios to OPT, not to a proxy;
  * its weights are Euclidean distances, so density and the weight model are INDEPENDENT by construction --
    the pathology that makes a Gamma(n,p) density sweep so awkward does not exist here.

WHAT THE SECTION FOUND, AND WHAT EACH GATE PROTECTS.

  5.1  THE RANKING INVERTS WITH THE CORRUPTION DIRECTION, MEASURED AGAINST THE TRUE OPTIMUM. This is not a
       heuristic artefact and not a normalisation choice: on the small grid, where OPT is known,
       `spc_gmr` is 2.58x optimal under inflation and 7.48x under deflation, while `l1sep_gmr` is 7.62x under
       inflation and 1.24x under deflation. They swap ends. G3 requires that swap to be present -- if the two
       ever stop crossing, the section has no story and must not pretend otherwise.

  5.1  AND THE HEAVY SET IS ONLY THE OPTIMUM UNDER INFLATION. Theory fixes both numbers in advance:
         inflate: an inflated edge EXCEEDS its detour, so it is heavy. H = B exactly, and since a decrease-only
                  cover is feasible for GMR and no smaller one exists, OPT_GMR = |H|.
         deflate: a deflated edge is a SHORTCUT -- SHORTER than its detour -- so it is NOT heavy and H cannot
                  contain it. H holds the VICTIMS instead: the edges whose detour now runs through the
                  shortcut. Measured: |H| = 4.73 |B|, and OPT_GMR = 0.209 |H| ~ |H|/4.79. Those two numbers
                  are the same fact seen twice, and G2 checks both.
       This is the deepest reason rho_H = |S|/|H| cannot be the paper's axis: under deflation its denominator
       is roughly five times the optimum, so it is not an approximation ratio at all. |S|/m is.

  5.3  THE ALGORITHMS THAT IGNORE SPARSITY DIE ON A SPARSE GRAPH. `pivot` and `left_edge` COMPLETE the graph
       before they start: at n = 3000 that is 4,498,500 edges for a graph that has 17,444 -- a 258x blowup,
       and 2.7x the peak memory DOMR needs. Meanwhile the LP-rounding methods time out on up to half the
       instances. Both facts are only visible at large n ON A SPARSE GRAPH.

  5.4  FRACTION AND MAGNITUDE ARE MEASURED UNDER DEFLATION ONLY, and the prose MUST say so. The published
       sweeps (P2df, P2dm) plant no inflate arm, and the direction inverts the ranking -- so a claim about
       fraction or magnitude made from these rows is a DEFLATE claim wearing a general label. G5 emits the
       disclosure as a macro so the prose cannot quietly generalise. (rgg_scale_harness adds P2if/P2im.)

  inputs   analysis/rgg/rgg_rows_with_ratio.csv        small grid: n=100-500, the exact optimum
           analysis/rgg_large/rgg_rows_with_ratio.csv  large grid: n=1000-3000, the ladder and the sweeps
  outputs  <texdir>/tab_opt.tex, <texdir>/sec5_macros.tex
           <figdir>/fig_s51_opt.pdf, fig_s52_ladder.pdf, fig_s53_limits.pdf, fig_s54_fracmag.pdf

  usage    sage -python experiments/section5.py --texdir "<paper>/tables" --figdir "<paper>/figures"
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

SMALL = "analysis/rgg/rgg_rows_with_ratio.csv"
LARGE = "analysis/rgg_large/rgg_rows_with_ratio.csv"

GMR = ["l1sep_gmr", "spc_gmr", "gmr_bestofk", "gmr_rand", "gmr_thr_naive", "pivot"]
IOMR = ["l1sep_iomr", "spc_iomr", "iomr_bestofk", "iomr_rand", "iomr_thr_naive", "iomr_regiongrow",
        "left_edge"]
DIRS = ["inflate", "deflate"]
# The ONLY n on the small grid that carries BOTH directions. The n-ladder (S1) is inflate-only;
# every deflate sweep (S3d, S4d, P2df, P2dm) sits at n=300. Every direction comparison is made here.
MATCH_N = 300

STY = {"l1sep_gmr": ("#0072B2", "o"), "l1sep_iomr": ("#0072B2", "o"),
       "spc_gmr": ("#009E73", "s"), "spc_iomr": ("#009E73", "s"),
       "pivot": ("#D55E00", "^"), "left_edge": ("#D55E00", "^"),
       "gmr_bestofk": ("#CC79A7", "D"), "iomr_bestofk": ("#CC79A7", "D"),
       "gmr_rand": ("#E69F00", "v"), "iomr_rand": ("#E69F00", "v"),
       "gmr_thr_naive": ("#56B4E9", "P"), "iomr_thr_naive": ("#56B4E9", "P"),
       "iomr_regiongrow": ("#999999", "X"), "domr": ("black", None)}
NAME = {"gmr_thr_naive": "gmr_thr", "iomr_thr_naive": "iomr_thr", "iomr_regiongrow": "iomr_rgrow"}


def nm(a):
    return NAME.get(a, a)


def ls(a):
    """Solid for GMR, dotted for IOMR. The colour says WHICH method; the line style says which variant.
    Without this, spc_gmr and spc_iomr are the same green square and the reader cannot tell them apart."""
    return ":" if a in IOMR else "-"


def tex(a):
    return r"\code{%s}" % nm(a).replace("_", r"\_")


def _n(x):
    return f"{int(x):,}".replace(",", "{,}")


# LaTeX control sequences are LETTERS ONLY. "l1sep".title() gives "L1Sep", and \secOptL1SepGmrInflate is not
# a macro name -- it is a hard error. inversion_macros.tex already spells l1sep as "Lone"; match it, so the
# same algorithm is not called two different things in two macro files.
MACNAME = {"l1sep_gmr": "LoneGmr", "l1sep_iomr": "LoneIomr", "gmr_thr_naive": "GmrThr",
           "iomr_thr_naive": "IomrThr", "iomr_regiongrow": "IomrRgrow"}


def key(a):
    """algo -> a LaTeX-safe CamelCase fragment. Raises rather than emit a macro LaTeX cannot read."""
    k = MACNAME.get(a) or "".join(w.title() for w in a.split("_"))
    if not k.isalpha():
        raise SystemExit(f"FATAL: macro fragment {k!r} for algo {a!r} is not pure letters. LaTeX control "
                         "sequences cannot contain digits; add it to MACNAME.")
    return k


def load(path, planted_only):
    if not os.path.exists(path):
        raise SystemExit(f"FATAL: {path} missing.")
    d = pd.read_csv(path, low_memory=False)
    for c in ("n", "E", "H", "size", "valid", "ratio", "wall", "peak_mb", "frac_q", "magnitude",
              "n_corrupted", "deg"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    # trap 1: the RGG CSVs are LONG (one row per knn_k). Every statistic here is COVER-level.
    d = d.drop_duplicates(subset=["task", "algo"], keep="first")
    if planted_only:
        # JITTER HAS NO CORRUPTED SET. It plants nothing, so |B| does not exist, the graphs sit near-metric,
        # and every |S|/OPT it contributes is a ratio on a graph that needed no repair. Section 5 is about
        # planted corruptions; the jitter rows are dropped, and the count is reported, never hidden.
        before = d.task.nunique()
        d = d[d.break_type.eq("reweight")]
        d.attrs["n_dropped"] = before - d.task.nunique()
    d["sm"] = d["size"] / d.E
    return d


def ok(d):
    return d[d.status.eq("ok") & d.valid.eq(1)]


# ----------------------------------------------------------------------------
# The gate
# ----------------------------------------------------------------------------
def gate(S, L):
    fails = []

    def chk(c, name, obs):
        print(f"  [{'PASS' if c else 'FAIL'}] {name:<54} {obs}")
        if not c:
            fails.append(name)

    # G1  DOMR's cover IS the heavy set, on both grids. One line, two whole pipelines.
    for lab, d in (("small", S), ("large", L)):
        D = ok(d)[ok(d).algo.eq("domr")].dropna(subset=["size", "H"])
        bad = int((D["size"] != D.H).sum())
        chk(len(D) > 0 and bad == 0, f"G1 domr's cover IS H ({lab})", f"{len(D)} tasks, {bad} mismatches")

    # G2  THE ASYMMETRY, FIXED BY THEORY. Under inflation an inflated edge exceeds its own detour, so it IS
    #     heavy: H = B exactly, and OPT_GMR = |H|. Under deflation a shortcut is SHORTER than its detour, so
    #     it is NOT heavy: H cannot contain it and holds the victims instead, so |H| > |B| and OPT_GMR < |H|.
    #     Both halves are checked. Neither can pass vacuously -- they are computed from the planted set.
    T = S.drop_duplicates("task").set_index("task")[["H", "E", "direction", "n_corrupted"]]
    ilp = ok(S)[ok(S).algo.eq("gmr_ilp")].dropna(subset=["size"])
    j = T.loc[ilp.task.unique()].copy()
    j["opt"] = ilp.set_index("task")["size"]
    R = {}
    for dr, g in j.groupby("direction"):
        R[dr] = dict(opt_over_H=float((g.opt / g.H).median()),
                     H_over_B=float((g.H / g.n_corrupted).median()), n=len(g))
    chk(abs(R["inflate"]["opt_over_H"] - 1.0) < 0.02 and abs(R["inflate"]["H_over_B"] - 1.0) < 0.02,
        "G2a inflate: H == B == the GMR optimum",
        f"OPT/|H| = {R['inflate']['opt_over_H']:.3f}, |H|/|B| = {R['inflate']['H_over_B']:.2f} "
        f"({R['inflate']['n']} tasks)")
    chk(R["deflate"]["opt_over_H"] < 0.5 and R["deflate"]["H_over_B"] > 2.0,
        "G2b deflate: H is the VICTIMS, not the culprits",
        f"OPT/|H| = {R['deflate']['opt_over_H']:.3f}, |H|/|B| = {R['deflate']['H_over_B']:.2f} "
        f"({R['deflate']['n']} tasks)")
    # and the two must be the SAME FACT: OPT/|H| ~ 1/(|H|/|B|), because GMR fixes each shortcut with one edit
    inv = 1.0 / R["deflate"]["H_over_B"]
    chk(abs(R["deflate"]["opt_over_H"] - inv) < 0.06,
        "G2c the two deflate numbers are one fact (OPT/|H| ~ |B|/|H|)",
        f"{R['deflate']['opt_over_H']:.3f} vs 1/{R['deflate']['H_over_B']:.2f} = {inv:.3f}")

    # G3  THE REVERSAL. The section's spine. If l1sep and spc ever stop swapping ends between the two
    #     directions, there is no inversion to report and the prose must not claim one. ANTI-VACUITY: this
    #     fails loudly on data that does not contain the finding.
    #
    #     MATCHED ON n. The small grid's n-ladder (S1) is INFLATE ONLY -- deflate lives at n=300 alone. So a
    #     direction comparison pooled over n would put inflate's n=100..500 against deflate's n=300, and the
    #     ratios do drift with n (pivot 6.34 -> 7.67 from n=100 to 500). MATCH_N is the only n that carries
    #     both directions, and every number in tab:opt is computed there. This is the same discipline the
    #     inversion table needed: a comparison across two different populations measures the populations.
    E = ok(S)[ok(S).ref_kind.eq("exact") & ok(S).n.eq(MATCH_N)].dropna(subset=["ratio"])
    both = set(E.direction.dropna())
    if both != set(DIRS):
        raise SystemExit(f"FATAL: n={MATCH_N} carries only {sorted(both)}. The direction comparison needs "
                         "both, and "
                         "no other n has deflate. Refusing to compare across different n.")
    p = E.pivot_table(index="algo", columns="direction", values="ratio", aggfunc="median")
    a, b = "l1sep_gmr", "spc_gmr"
    swap = (p.loc[a, "inflate"] > p.loc[b, "inflate"]) and (p.loc[a, "deflate"] < p.loc[b, "deflate"])
    chk(swap, "G3 the ranking INVERTS with direction (vs the true OPT)",
        f"{a}: {p.loc[a,'inflate']:.2f} -> {p.loc[a,'deflate']:.2f}   "
        f"{b}: {p.loc[b,'inflate']:.2f} -> {p.loc[b,'deflate']:.2f}")

    # G3b  UNDER INFLATION, EVERY METHOD EDITS MORE THAN THE HEAVY SET. H is the optimum there (G2a), so a
    #      method above that line is strictly worse than one all-pairs pass. The prose claims "every"; if it
    #      ever stops being every, this fails and the word has to change.
    K = ok(L)[ok(L).sweep.eq("S1")]
    hm = K[K.algo.eq("domr")].sm.median()
    algos = [a for a in K.algo.unique() if a != "domr"]
    above = [a for a in algos if K[K.algo.eq(a)].sm.median() > hm]
    chk(len(above) == len(algos),
        "G3b inflate: EVERY method edits more than H (= OPT)",
        f"{len(above)} of {len(algos)}" + (f"; below: {sorted(set(algos)-set(above))}"
                                           if len(above) < len(algos) else ""))

    # G3c  MEMORY IS THE ALL-PAIRS PASS. Every cover-producing method except the two graph-completers must sit
    #      close to DOMR's footprint -- that is the section's claim, and it is what makes the panel worth a
    #      figure slot. If one of them ever strays, the claim is false and this fails.
    lad_ = L[L.sweep.isin(["S1", "S1d"]) & L.status.eq("ok")]
    pv_ = lad_.pivot_table(index="task", columns="algo", values="peak_mb")
    rel_ = pv_.div(pv_["domr"], axis=0).median()
    comp_ = ["pivot", "left_edge"]
    bnd_ = ["gmr_lp_naive", "iomr_lp_naive"]
    rest_ = [x for x in rel_.index if x not in comp_ + bnd_ + ["domr"]]
    chk(rel_[rest_].max() < 1.15 and rel_[comp_].min() > 2.0,
        "G3c memory IS the all-pairs pass (except the completers)",
        f"cover methods {rel_[rest_].min():.2f}-{rel_[rest_].max():.2f}x domr; "
        f"pivot/left_edge {rel_[comp_].min():.1f}-{rel_[comp_].max():.1f}x; "
        f"LP bounds {rel_[bnd_].median():.2f}x")

    # G3d  THE MECHANISM. Inflation must produce far more violated cycles PER HEAVY EDGE than deflation --
    #      that is the section's explanation for why the LP family dies under inflation while sailing through
    #      deflation on FOUR TIMES MORE heavy edges. If the gap ever closes, the explanation is wrong.
    C_ = L[L.sweep.isin(["S1", "S1d"]) & L.algo.eq("l1sep_gmr") & L.status.eq("ok")].dropna(subset=["cuts"])
    C_ = C_.assign(cph=lambda x: x.cuts / x.H)
    ci = C_[C_.sweep.eq("S1")].cph.median()
    cd = C_[C_.sweep.eq("S1d")].cph.median()
    hi = L[L.sweep.eq("S1")].drop_duplicates("task").H.median()
    hd = L[L.sweep.eq("S1d")].drop_duplicates("task").H.median()
    chk(ci > 10 * cd and hi < hd,
        "G3d inflation: FEWER heavy edges, MANY more cycles each",
        f"cycles/|H|: inflate {ci:.1f} vs deflate {cd:.1f} ({ci/cd:.0f}x); "
        f"|H|: inflate {int(hi)} vs deflate {int(hd)}")

    # G4  jitter is out, and we say how much of it there was.
    chk(S.attrs.get("n_dropped", 0) > 0, "G4 jitter dropped from the small grid (planted only)",
        f"{S.attrs.get('n_dropped', 0)} tasks dropped -- they plant nothing, so |B| does not exist")

    # G5  the frac/mag sweeps are DEFLATE-ONLY. Not a failure -- a DISCLOSURE the prose must carry.
    fm = L[L.sweep.isin(["P2df", "P2dm"])]
    dirs = sorted(set(fm.direction.dropna()))
    chk(True, "G5 fraction/magnitude sweeps: which directions?", f"{dirs}")
    if dirs != DIRS:
        print(f"         *** ONE DIRECTION ONLY. Any claim about fraction or magnitude from these rows is a")
        print(f"         *** {dirs[0].upper()} claim. The direction INVERTS the ranking, so it does not")
        print(f"         *** generalise. \\secFMDirs records this; the prose must disclose it.")
    return fails, R, p, dirs


# ----------------------------------------------------------------------------
# Table: |S|/OPT by direction, on the small grid
# ----------------------------------------------------------------------------
def emit_tab_opt(S, R, p):
    E = ok(S)[ok(S).ref_kind.eq("exact") & ok(S).n.eq(MATCH_N)].dropna(subset=["ratio"])
    nt = {d: E[E.direction.eq(d)].task.nunique() for d in DIRS}
    ilp = {a: ok(S)[ok(S).algo.eq(a)].task.nunique() for a in ("gmr_ilp", "iomr_ilp")}
    ntask = S.task.nunique()

    cap = (r"\caption{\textbf{The corruption decides --- and here it is measured against the \emph{true} "
           r"optimum.} $|S|/\mathrm{OPT}$ on the small planted \textsc{rgg}s ($n = %d\text{--}%d$, %s planted "
           r"instances), where the exact cover is available: \code{gmr\_ilp} solves %s of them and "
           r"\code{iomr\_ilp} %s. Medians over the runs that returned \emph{and} verified, split by the "
           r"direction of the corruption; bold marks the best and worst in each variant and direction. "
           r"\textbf{The two ends swap:} \code{spc\_gmr} is the best \GMR{} heuristic under inflation and the "
           r"worst under deflation, and \code{l1sep\_gmr} does the reverse. This is not an artefact of a "
           r"normalisation --- it is a ratio to \textsc{opt}. \textbf{And the heavy set is the optimum only "
           r"under inflation}: an inflated edge exceeds its own detour, so it \emph{is} heavy and "
           r"$H = B$ exactly ($\mathrm{OPT}/|H| = %.3f$); a deflated edge is a \emph{shortcut}, shorter than "
           r"its detour, so $H$ cannot contain it and holds the victims instead ($|H| = %.2f\,|B|$, "
           r"$\mathrm{OPT}/|H| = %.3f$).}"
           % (int(S.n.min()), int(S.n.max()), _n(ntask), _n(ilp["gmr_ilp"]), _n(ilp["iomr_ilp"]),
              R["inflate"]["opt_over_H"], R["deflate"]["H_over_B"], R["deflate"]["opt_over_H"]))

    out = [r"% GENERATED by experiments/section5.py -- DO NOT EDIT, DO NOT TRANSCRIBE. Regenerate.",
           r"\begin{table}[t]\centering\small", cap, r"\label{tab:opt}",
           r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
           r"algorithm & inflate & deflate \\",
           r"\midrule"]
    for var, algos in (("GMR", GMR), ("IOMR", IOMR)):
        have = [a for a in algos if a in p.index]
        if not have:
            continue
        best = {d: min(have, key=lambda a: p.loc[a, d]) for d in DIRS}
        worst = {d: max(have, key=lambda a: p.loc[a, d]) for d in DIRS}
        out.append(r"\multicolumn{3}{@{}l}{\textbf{\%s{}}} \\[1pt]" % var)
        for a in sorted(have, key=lambda x: p.loc[x, "inflate"]):
            cells = []
            for d in DIRS:
                v = p.loc[a, d]
                s = ("$\\mathbf{%.3f}$" % v) if a in (best[d], worst[d]) else ("$%.3f$" % v)
                cells.append(s)
            out.append(r"\quad %s & %s \\" % (tex(a), " & ".join(cells)))
        out.append(r"\addlinespace[2pt]")
    out += [r"\midrule",
            r"\quad \DOMR{} \emph{(the heavy set)} & $%.3f$ & $%.3f$ \\"
            % (1.0 / R["inflate"]["opt_over_H"], 1.0 / R["deflate"]["opt_over_H"]),
            r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out), nt


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_opt(S, figdir):
    """5.1 -- the two things the small grid can honestly show.

    LEFT: |S|/OPT up the n-ladder. Inflate only, because the ladder IS inflate only (S1); every deflate sweep
    sits at n=300. Drawing a deflate "curve" from a single n would be a line through one point.

    RIGHT: the reversal, as a SLOPE CHART at n=300 -- the one n that carries both directions. Each algorithm
    is a segment from its inflate ratio to its deflate ratio. THE CROSSING LINES ARE THE FINDING: spc_gmr
    climbs while l1sep_gmr dives, and they change places. A bar chart would hide that; a slope chart is the
    only encoding where "they swap ends" is the thing your eye actually sees.
    """
    E = ok(S)[ok(S).ref_kind.eq("exact")].dropna(subset=["ratio"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))

    # --- left: the ladder (inflate)
    ax = axes[0]
    lad = E[E.sweep.eq("S1") & E.variant.eq("GMR")]
    for a in GMR:
        A = lad[lad.algo.eq(a)]
        if A.empty:
            continue
        g = A.groupby("n").ratio.median().sort_index()
        c, m = STY[a]
        ax.plot(g.index, g.values, marker=m, ms=4, lw=1.6, color=c, label=nm(a))
    ax.axhline(1.0, color="black", lw=1.4, ls="--")
    ax.text(105, 1.03, "domr $=$ OPT", fontsize=7)
    ax.set_yscale("log"); ax.set_xlabel("$n$", fontsize=9)
    ax.set_ylabel(r"$|S| \,/\, \mathrm{OPT}$   $\downarrow$", fontsize=9)
    ax.set_title("inflate: the ratio does not improve with $n$", fontsize=10)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.25, lw=0.5)

    # --- right: the reversal, matched at n = MATCH_N
    ax = axes[1]
    K = E[E.n.eq(MATCH_N) & E.variant.eq("GMR")]
    p = K.pivot_table(index="algo", columns="direction", values="ratio", aggfunc="median")
    # Label on the INFLATE side. Under deflation four methods land within 1.24-1.32 and their labels collide
    # into an unreadable smear; under inflation they are spread over 2.58-7.73. Label where there is room.
    for a in GMR:
        if a not in p.index:
            continue
        c, m = STY[a]
        ax.plot([0, 1], [p.loc[a, "inflate"], p.loc[a, "deflate"]], marker=m, ms=6, lw=2.0, color=c)
        ax.annotate(nm(a), (-0.05, p.loc[a, "inflate"]), fontsize=7.5, color=c, va="center", ha="right")
    ax.axhline(1.0, color="black", lw=1.4, ls="--")
    ax.text(1.02, 1.02, "domr $=$ OPT", fontsize=7)
    ax.set_xlim(-0.75, 1.35); ax.set_xticks([0, 1]); ax.set_xticklabels(["inflate", "deflate"])
    ax.set_yscale("log")
    ax.set_ylabel(r"$|S| \,/\, \mathrm{OPT}$   $\downarrow$", fontsize=9)
    ax.set_title(f"they swap ends  ($n = {MATCH_N}$)", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"fig_s51_opt.{e}"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_ladder(L, figdir):
    """5.2 -- |S|/m up the ladder, one panel per direction. DOMR (= |H|/m) is the dashed reference."""
    lad = ok(L)[ok(L).sweep.isin(["S1", "S1d"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (d, sw) in zip(axes, (("inflate", "S1"), ("deflate", "S1d"))):
        K = lad[lad.sweep.eq(sw)]
        for a in GMR + IOMR:
            A = K[K.algo.eq(a)]
            if A.empty:
                continue
            g = A.groupby("n").sm.median().sort_index()
            c, m = STY[a]
            ax.plot(g.index, g.values, marker=m, ms=3.5, lw=1.3, color=c, ls=ls(a),
                    label=nm(a), alpha=0.9)
        D = K[K.algo.eq("domr")].groupby("n").sm.median().sort_index()
        ax.plot(D.index, D.values, ls="--", lw=1.8, color="black", label="domr $=|H|/m$")
        ax.set_xlabel("$n$", fontsize=9)
        ax.set_title(d, fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel(r"$|S|/m$   (share of the graph rewritten)   $\downarrow$", fontsize=9)
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, fontsize=7, ncol=7, loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"fig_s52_ladder.{e}"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_limits(L, figdir):
    """5.3 -- WHO CAN RUN AT ALL, and why. Three panels: the deaths, the non-deaths, and the mechanism.

    WHY THIS IS NOT A MEMORY FIGURE ANY MORE. The memory panel carried exactly one bit -- whether a method
    completes the graph -- because every cover-producing method must know the shortest paths, and that one
    all-pairs pass dominates everything it then does. Drop pivot and left_edge and eleven curves lie on top of
    each other within 5%. There is no structure to draw, so we say it in a sentence and spend the space here.

    WHAT IS HERE INSTEAD. The corruption direction does not merely invert the ranking; it decides which
    algorithms EXIST. Under deflation the whole suite reaches n=3000. Under inflation the covering-LP family
    is dead: gmr_bestofk returns on 0% of instances from n=1800, iomr_bestofk from n=2600, and gmr_rand is
    down to 25%.

    AND THE THIRD PANEL IS THE REASON, not a guess. The LP's size is the number of VIOLATED CYCLES, not the
    number of heavy edges -- and the two corruptions trade one against the other:

        inflate:  few heavy edges, each violated MANY ways. An inflated edge exceeds its detour, so EVERY
                  path shorter than it is a broken cycle. Measured: 37.6 cycles per heavy edge.
        deflate:  many heavy edges, each violated ONE way. A shortcut is not itself heavy; it makes victims,
                  and a victim is broken essentially only through the shortcut. Measured: 1.2 -- flat at
                  every n on the ladder.

    So inflation is the harder program with FOUR TIMES FEWER heavy edges, and the family that never builds
    the LP (spc) is the family that survives it. That is the same fact Section 5.1 saw from the other side.
    """
    lad = L[L.sweep.isin(["S1", "S1d"])]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    # --- panels 1 and 2: who returns at all, per direction
    for ax, (d, sw) in zip(axes[:2], (("inflate", "S1"), ("deflate", "S1d"))):
        K = lad[lad.sweep.eq(sw)]
        nt = K.groupby("n").task.nunique()
        for a in GMR + IOMR:
            A = K[K.algo.eq(a)]
            if A.empty:
                continue
            r = (A[A.status.eq("ok") & A.valid.eq(1)].groupby("n").task.nunique()
                 .reindex(nt.index).fillna(0) / nt)
            c, m = STY[a]
            ax.plot(r.index, 100 * r.values, marker=m, ms=3.5, lw=1.5, color=c, ls=ls(a), label=nm(a))
        ax.set_ylim(-4, 104)
        ax.set_xlabel("$n$", fontsize=9)
        ax.set_title(d, fontsize=11)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel(r"returned a verified cover (\%)   $\uparrow$", fontsize=9)

    # --- panel 3: THE MECHANISM. Violated cycles per heavy edge -- what the LP actually has to swallow.
    K = lad[lad.algo.eq("l1sep_gmr") & lad.status.eq("ok")].dropna(subset=["cuts"]).copy()
    K["cph"] = K.cuts / K.H
    for d, sw, c in (("inflate", "S1", "#D55E00"), ("deflate", "S1d", "#0072B2")):
        g = K[K.sweep.eq(sw)].groupby("n").cph.median().sort_index()
        axes[2].plot(g.index, g.values, marker="o", ms=4, lw=2.0, color=c, label=d)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("$n$", fontsize=9)
    axes[2].set_ylabel("violated cycles per heavy edge", fontsize=9)
    axes[2].set_title("why: the LP's size is CYCLES, not $|H|$", fontsize=10)
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25, lw=0.5)

    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, fontsize=7, ncol=7, loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"fig_s53_limits.{e}"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_fracmag(L, figdir, dirs):
    """5.4 -- |S|/m against the corruption fraction and its magnitude. DEFLATE ONLY in the published grid,
    and the figure says so on its face rather than in a footnote."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (sw, knob, xl) in zip(axes, (("P2df", "frac_q", "corrupted fraction"),
                                         ("P2dm", "magnitude", "corruption magnitude"))):
        K = ok(L)[ok(L).sweep.eq(sw)]
        for a in GMR + IOMR:
            A = K[K.algo.eq(a)]
            if A.empty:
                continue
            g = A.groupby(knob).sm.median().sort_index()
            c, m = STY[a]
            ax.plot(g.index, g.values, marker=m, ms=4, lw=1.4, color=c, ls=ls(a),
                    label=nm(a), alpha=0.9)
        D = K[K.algo.eq("domr")].groupby(knob).sm.median().sort_index()
        ax.plot(D.index, D.values, ls="--", lw=1.8, color="black", label="domr $=|H|/m$")
        ax.set_xlabel(xl + f"   ({'/'.join(dirs)} only)", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel(r"$|S|/m$   $\downarrow$", fontsize=9)
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, fontsize=7, ncol=7, loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"fig_s54_fracmag.{e}"), dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Macros: every number the prose quotes
# ----------------------------------------------------------------------------
def emit_macros(S, L, R, p, dirs, nt):
    M = [r"% GENERATED by experiments/section5.py -- DO NOT EDIT. Every number Section 5 quotes."]

    def mac(k, v):
        M.append(r"\newcommand{\sec%s}{%s}" % (k, v))

    # --- 5.1 the small grid
    mac("SmallTasks", _n(S.task.nunique()))
    mac("SmallNLo", int(S.n.min()))
    mac("SmallNHi", int(S.n.max()))
    mac("SmallJitterDropped", _n(S.attrs.get("n_dropped", 0)))
    for a, k in (("gmr_ilp", "GmrIlp"), ("iomr_ilp", "IomrIlp")):
        o = ok(S)[ok(S).algo.eq(a)].task.nunique()
        mac(k + "Solved", _n(o))
        mac(k + "Pct", "%.0f" % (100 * o / S.task.nunique()))
    for d in DIRS:
        mac(d.title() + "Tasks", _n(nt[d]))
        mac(d.title() + "OptOverH", "%.3f" % R[d]["opt_over_H"])
        mac(d.title() + "HOverB", "%.2f" % R[d]["H_over_B"])
    # the |S|/OPT numbers the prose names
    for a in ("l1sep_gmr", "spc_gmr", "pivot", "left_edge", "l1sep_iomr"):
        if a in p.index:
            for d in DIRS:
                mac("Opt" + key(a) + d.title(),
                    "%.3f" % p.loc[a, d])

    # --- 5.2 the ladder
    lad = ok(L)[ok(L).sweep.isin(["S1", "S1d"])]
    mac("LadderNLo", int(lad.n.min()))
    mac("LadderNHi", int(lad.n.max()))
    for a in ("l1sep_gmr", "spc_gmr"):
        for d, sw in (("Inflate", "S1"), ("Deflate", "S1d")):
            K = lad[lad.algo.eq(a) & lad.sweep.eq(sw)]
            mac("Sm" + key(a) + d, "%.3f" % K.sm.median())

    # How many methods edit MORE than the heavy set? Under inflation H IS the optimum, so every method above
    # that line is strictly worse than one all-pairs pass. The prose says "every"; this is where it earns it.
    for d, sw in (("Inflate", "S1"), ("Deflate", "S1d")):
        K = ok(L)[ok(L).sweep.eq(sw)]
        hm = K[K.algo.eq("domr")].sm.median()
        algos = [a for a in K.algo.unique() if a != "domr"]
        above = [a for a in algos if K[K.algo.eq(a)].sm.median() > hm]
        mac(d + "AboveH", len(above))
        mac(d + "NAlgos", len(algos))

    # --- 5.3 limits, at the top of the ladder
    N = int(L[L.sweep.isin(["S1", "S1d"])].n.max())
    top = L[L.sweep.isin(["S1", "S1d"]) & L.n.eq(N)]
    m_top = int(top.drop_duplicates("task").E.median())
    comp = N * (N - 1) // 2
    mac("TopN", N)
    mac("TopM", _n(m_top))
    mac("TopCompletion", _n(comp))
    mac("TopBlowup", "%.0f" % (comp / m_top))
    for a in ("pivot", "left_edge", "domr"):
        A = top[top.algo.eq(a)]
        mac("Mem" + key(a), _n(A.peak_mb.median()))

    # MEMORY IS THE ALL-PAIRS PASS, NOT THE ALGORITHM. Every cover-producing method except the two graph
    # completers sits within a few percent of DOMR's footprint; the two LP bounds sit BELOW it because they
    # never materialise the distance matrix (and never return a cover). These are the numbers the prose uses.
    lad2 = L[L.sweep.isin(["S1", "S1d"]) & L.status.eq("ok")]
    pv2 = lad2.pivot_table(index="task", columns="algo", values="peak_mb")
    rel2 = pv2.div(pv2["domr"], axis=0).median()
    COMPLETERS = ["pivot", "left_edge"]
    BOUNDS = ["gmr_lp_naive", "iomr_lp_naive"]
    rest = [x for x in rel2.index if x not in COMPLETERS + BOUNDS + ["domr"]]
    mac("MemRestLo", "%.2f" % rel2[rest].min())
    mac("MemRestHi", "%.2f" % rel2[rest].max())
    mac("MemRestPct", "%.0f" % (100 * (rel2[rest].max() - 1.0)))
    mac("MemPivotRel", "%.1f" % rel2["pivot"])
    mac("MemLeftEdgeRel", "%.1f" % rel2["left_edge"])
    mac("MemBoundRel", "%.2f" % rel2[BOUNDS].median())
    mac("MemDomrLo", _n(pv2.groupby(lad2.drop_duplicates("task").set_index("task").n).domr.median().iloc[0]))
    mac("MemDomrHi", _n(pv2.groupby(lad2.drop_duplicates("task").set_index("task").n).domr.median().iloc[-1]))
    worst = None
    for a in GMR + IOMR:
        A = top[top.algo.eq(a)]
        if A.empty:
            continue
        to = 100 * A.status.eq("timeout").mean()
        if worst is None or to > worst[1]:
            worst = (a, to)
    mac("TopWorstTimeoutAlgo", tex(worst[0]))
    mac("TopWorstTimeoutPct", "%.0f" % worst[1])
    rg = top[top.algo.eq("iomr_regiongrow")]
    import harness as _h
    mac("RgrowHmax", _h.REGION_H_MAX)
    mac("RgrowTopRet", "%.0f" % (100 * len(rg[rg.status.eq("ok") & rg.valid.eq(1)]) / max(top.task.nunique(), 1)))

    # 5.3 -- WHO DIES, AND WHY. Per DIRECTION: the ladder pools inflate and deflate, and a pooled return
    # rate is the mixing ratio again (gmr_bestofk plateaus at "50%", which is 0% inflate + 100% deflate).
    lad3 = L[L.sweep.isin(["S1", "S1d"])]
    NHI = int(lad3.n.max())
    for d, sw in (("Inflate", "S1"), ("Deflate", "S1d")):
        K = lad3[lad3.sweep.eq(sw) & lad3.n.eq(NHI)]
        nt = K.task.nunique()
        # A method REFUSED by its own gate is not a method that failed at scale. iomr_regiongrow declines any
        # graph with |H| > REGION_H_MAX, in BOTH directions, at every n on this ladder -- that is a design
        # limit, not a timeout, and counting it as a death would let a structural exclusion masquerade as a
        # scale limitation. It is reported separately, by name.
        GATED = {"iomr_regiongrow"}
        dead = []
        for a in GMR + IOMR:
            A = K[K.algo.eq(a)]
            if A.empty:
                continue
            r = 100 * len(A[A.status.eq("ok") & A.valid.eq(1)]) / max(nt, 1)
            mac("Ret" + key(a) + d, "%.0f" % r)
            if r < 50 and a not in GATED:
                dead.append(a)
        mac("NDead" + d, len(dead))

    # THE MECHANISM. The LP's size is the number of VIOLATED CYCLES, not the number of heavy edges, and the
    # two corruptions trade one against the other. l1sep's cut count measures it directly.
    C3 = lad3[lad3.algo.eq("l1sep_gmr") & lad3.status.eq("ok")].dropna(subset=["cuts"]).copy()
    C3["cph"] = C3.cuts / C3.H
    for d, sw in (("Inflate", "S1"), ("Deflate", "S1d")):
        G3_ = C3[C3.sweep.eq(sw)]
        mac("Cyc" + d, "%.1f" % G3_.cph.median())
        T_ = L[L.sweep.eq(sw) & L.n.eq(NHI)].drop_duplicates("task")
        mac("Hset" + d, _n(T_.H.median()))
    mac("CycRatio", "%.0f" % (C3[C3.sweep.eq("S1")].cph.median() / C3[C3.sweep.eq("S1d")].cph.median()))

    # --- 5.5 THE POOLED NUMBER, AND WHY IT IS NOT A STATISTIC.
    # l1sep_gmr beats spc_gmr on 0% of inflate tasks and 100% of deflate tasks. Pool them and the win rate
    # comes out at exactly the DEFLATION SHARE OF THE MIXTURE -- it measures the mixture, not the methods.
    # The paper makes that claim; these two macros are the claim, and they are computed, not asserted.
    G = {"S1": "inflate", "S2": "inflate", "S2k": "inflate",
         "P2df": "deflate", "P2dm": "deflate", "S1d": "deflate"}
    O = ok(L).copy()
    O["grp"] = O.sweep.map(G)
    O = O[O.grp.notna()]
    pv = O[O.algo.isin(["l1sep_gmr", "spc_gmr"])].pivot_table(index="task", columns="algo",
                                                              values="sm").dropna()
    pv["grp"] = O.drop_duplicates("task").set_index("task").grp.reindex(pv.index)
    win = float((pv.l1sep_gmr < pv.spc_gmr).mean())
    share = float((pv.grp == "deflate").mean())
    if abs(win - share) > 0.005:
        raise SystemExit(f"FATAL: the pooled win rate ({win:.3f}) is supposed to BE the mixing ratio "
                         f"({share:.3f}) -- that identity is the paper's claim, and it does not hold. "
                         "Either the 0%/100% split has broken or the grouping is wrong.")
    mac("PooledWin", "%.1f" % (100 * win))
    mac("PooledShare", "%.1f" % (100 * share))
    mac("PooledN", _n(len(pv)))
    mac("MatchN", MATCH_N)
    import harness as _hh
    mac("TimeoutCap", _hh.TIMEOUT_S)

    # --- 5.4 fraction and magnitude  (DIRECTION IS A DISCLOSURE, NOT A DETAIL)
    mac("FMDirs", ", ".join(dirs))
    mac("FMOneDirection", "true" if len(dirs) == 1 else "false")
    for sw, knob, tag in (("P2df", "frac_q", "Frac"), ("P2dm", "magnitude", "Mag")):
        K = ok(L)[ok(L).sweep.eq(sw)]
        T = L[L.sweep.eq(sw)].drop_duplicates("task")
        lo, hi = K[knob].min(), K[knob].max()
        mac(tag + "Lo", "%g" % lo)
        mac(tag + "Hi", "%g" % hi)
        for a in ("l1sep_gmr", "spc_gmr"):
            A = K[K.algo.eq(a)]
            mac(tag + key(a) + "Lo", "%.3f" % A[A[knob].eq(lo)].sm.median())
            mac(tag + key(a) + "Hi", "%.3f" % A[A[knob].eq(hi)].sm.median())
        mac(tag + "HmLo", "%.3f" % (T[T[knob].eq(lo)].H / T[T[knob].eq(lo)].E).median())
        mac(tag + "HmHi", "%.3f" % (T[T[knob].eq(hi)].H / T[T[knob].eq(hi)].E).median())
    return "\n".join(M)


def paper_dir(p, want):
    if not os.path.isdir(p):
        raise SystemExit(f"FATAL: --{want}dir '{p}' is not a directory. Refusing to create it.")
    p = os.path.abspath(p)
    root = os.path.dirname(p)
    if not os.path.exists(os.path.join(root, "story.tex")):
        raise SystemExit(f"FATAL: no story.tex beside '{p}'. That is not the paper.")
    if os.path.basename(p) != want:
        raise SystemExit(f"FATAL: --{want}dir '{p}' is in the paper but is not {want}/.")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texdir", required=True)
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    texdir, figdir = paper_dir(a.texdir, "tables"), paper_dir(a.figdir, "figures")

    S = load(SMALL, planted_only=True)
    L = load(LARGE, planted_only=False)
    print(f"small grid: {S.task.nunique()} planted tasks (n={int(S.n.min())}-{int(S.n.max())}), "
          f"{S.attrs['n_dropped']} jitter tasks dropped")
    print(f"large grid: {L.task.nunique()} tasks (n={int(L.n.min())}-{int(L.n.max())})\n")

    print("GATE -- nothing is written until these pass")
    fails, R, p, dirs = gate(S, L)
    if fails and not a.force:
        raise SystemExit(f"\n*** {len(fails)} GATE FAILURE(S): {fails}. NOT writing. The previous tables and "
                         "figures are left on disk. ***")

    tab, nt = emit_tab_opt(S, R, p)
    macros = emit_macros(S, L, R, p, dirs, nt)
    if "nan" in tab.lower() or "nan" in macros.lower():
        raise SystemExit("*** GATE FAILED: rendered LaTeX contains nan. NOT writing. ***")
    # A macro NAME with a digit in it is not a macro, it is a LaTeX error -- and it is silent here and fatal
    # in the paper. "l1sep".title() is "L1Sep", which is exactly how it got in. Check the string that ships.
    import re as _re
    bad = [m for m in _re.findall(r"\\newcommand\{\\\\([^}]*)\}", macros) if not m.isalpha()]
    if bad:
        raise SystemExit(f"*** GATE FAILED: macro names are not pure letters: {bad}. LaTeX control sequences "
                         "cannot contain digits. NOT writing. ***")

    with open(os.path.join(texdir, "tab_opt.tex"), "w") as f:
        f.write(tab + "\n")
    with open(os.path.join(texdir, "sec5_macros.tex"), "w") as f:
        f.write(macros + "\n")
    fig_opt(S, figdir)
    fig_ladder(L, figdir)
    fig_limits(L, figdir)
    fig_fracmag(L, figdir, dirs)

    print(f"\n  wrote {texdir}/tab_opt.tex")
    print(f"  wrote {texdir}/sec5_macros.tex  ({macros.count('newcommand')} macros)")
    print(f"  wrote {figdir}/fig_s5{{1_opt,2_ladder,3_limits,4_fracmag}}.pdf")
    print("\nAll gates passed." if not fails else "\n!! WRITTEN UNDER --force.")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
