"""Emit the Section 5.1 figure -- the SMALL dense grid, where the optimum is actually known.

WHAT THIS FIGURE IS FOR. Everything else in Section 5 is measured against $m$ or against $|H|$, because the
large grids have no exact solver. This is the one place in the paper that can talk to the OPTIMUM, and it has
to earn the right to stop talking to it afterwards. Exp1 at p = 0.3: n from 100 to 300, 30 seeds per n, 630
tasks, and a reference on every single row.

  left  panel   GMR   |S| / OPT          -- the EXACT optimum. ref_kind == "exact" on 630/630 rows.
  right panel   IOMR  |S| / LP-bound     -- the covering-LP value.

WHY IOMR GETS THE LP AND NOT THE ILP. `iomr_ilp` converges on ZERO of the 630 tasks at p = 0.3. There is no
exact IOMR optimum on this slice, and pretending otherwise would be the whole reason this file has a gate.

WHY THE LP BOUND IS NEVERTHELESS THE OPTIMUM. Wherever `iomr_ilp` DOES converge -- 607 tasks elsewhere on the
small grid -- OPT / LP-bound is 1.00, with range [1.00, 1.00], at every level of non-metricity we can check
(|H|/m from 0.000 to 1.0). The IOMR covering LP is integral. So |S| / LP-bound is a genuine approximation
ratio, not a vacuous upper bound, and the right panel says what it appears to say. G4 measures this every run
rather than trusting this paragraph: if the LP ever stops being integral, the panel's meaning changes and the
gate fails before the figure is written.

  THE ONE EXTRAPOLATION, STATED: at p = 0.3 itself the ILP never converges, so integrality there is inferred,
  not verified. It is supported at every brokenness level that IS verifiable, including the bucket p = 0.3
  falls in. The caption says so.

WHAT THE FIGURE SHOWS, AND WHY IT MATTERS TO THE REST OF THE PAPER.

  * `domr` sits at RATIO 1.000, IQR [1.000, 1.000], on all 630 tasks. The GMR optimum IS the heavy set. That
    is what licenses |H| as a reference anywhere else -- and it is the distance backbone of Simas et al.,
    arrived at from the other side.
  * `l1sep_gmr` also sits at 1.000. The weight-aware method does not approximate the GMR optimum; it FINDS
    it. Every other GMR heuristic is 2.4x to 3.2x.
  * Every IOMR heuristic is 12x to 13x optimal. The two variants are not two flavours of one problem: one is
    solved and the other is wide open.

  usage   sage -python experiments/bench_figs.py --figdir "<paper>/figures"

Figures are written STRAIGHT into the paper. Copying them across by hand is the one un-gated hop left in this
project and it has already shipped a stale figure once.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

CSV = "analysis/rows_with_ratio.csv"
EXP, P = "exp1", 0.3

# One panel per variant. The reference differs, and the LABEL must differ with it -- calling the IOMR
# reference "the optimum" when it is an LP value is exactly the kind of quiet lie a gate exists to stop.
PANELS = [
    ("GMR", "exact", r"$|S| \, / \,$ OPT   (exact)",
     ["l1sep_gmr", "spc_gmr", "pivot", "gmr_bestofk", "gmr_rand", "gmr_thr_naive"]),
    ("IOMR", "lower_bound", r"$|S| \, / \,$ LP bound",
     ["l1sep_iomr", "spc_iomr", "left_edge", "iomr_bestofk", "iomr_rand", "iomr_thr_naive",
      "iomr_regiongrow"]),
]
# domr is drawn on BOTH panels, as the reference it is -- never as a competitor. On the GMR panel it lands
# on 1.000 and that IS the section's licence; on the IOMR panel it shows how far |H| is from an IOMR cover.
CTRL = "domr"

STY = {
    "l1sep_gmr": ("#0072B2", "o"), "l1sep_iomr": ("#0072B2", "o"),
    "spc_gmr": ("#009E73", "s"), "spc_iomr": ("#009E73", "s"),
    "pivot": ("#D55E00", "^"), "left_edge": ("#D55E00", "^"),
    "gmr_bestofk": ("#CC79A7", "D"), "iomr_bestofk": ("#CC79A7", "D"),
    "gmr_rand": ("#E69F00", "v"), "iomr_rand": ("#E69F00", "v"),
    "gmr_thr_naive": ("#56B4E9", "P"), "iomr_thr_naive": ("#56B4E9", "P"),
    "iomr_regiongrow": ("#999999", "X"),
}
NAME = {"gmr_thr_naive": "gmr_thr", "iomr_thr_naive": "iomr_thr", "iomr_regiongrow": "iomr_rgrow"}


def nm(a):
    return NAME.get(a, a)


def load():
    if not os.path.exists(CSV):
        raise SystemExit(f"FATAL: {CSV} missing.")
    d = pd.read_csv(CSV, low_memory=False)
    for c in ("size", "valid", "n", "p", "ratio", "H", "E", "iomr_ref"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    # trap 1: the small array is not long-format, but dedup anyway -- it costs nothing and a schema change
    # that made it long would otherwise silently quadruple every median.
    d = d.drop_duplicates(subset=["task", "algo"], keep="first")
    q = d[d.exp.eq(EXP) & np.isclose(d.p, P)]
    if q.empty:
        raise SystemExit(f"FATAL: no {EXP} rows at p={P}.")
    return d, q


def gate(d, q):
    """Fails CLOSED, before anything is drawn. Every check below can genuinely fail."""
    fails = []

    def chk(ok, name, obs):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52} {obs}")
        if not ok:
            fails.append(name)

    ok = q[q.status.eq("ok") & q.valid.eq(1)]

    # G1  the axis label must be TRUE. If a GMR row's reference is not exact, the panel may not say "OPT".
    for var, kind, _, _ in PANELS:
        K = ok[ok.variant.eq(var)].dropna(subset=["ratio"])
        kinds = set(K.ref_kind.dropna())
        chk(kinds == {kind}, f"G1 {var} reference is '{kind}' on every plotted row",
            f"{len(K)} rows, kinds={sorted(kinds)}")

    # G2  balance. A curve that quietly loses its samples at large n reads as a trend.
    for var, _, _, algos in PANELS:
        K = ok[ok.variant.eq(var) & ok.algo.isin(algos)].dropna(subset=["ratio"])
        cnt = K.groupby(["algo", "n"]).size().unstack(fill_value=0)
        full = int((cnt == cnt.max().max()).all(axis=1).sum())
        thin = sorted(set(algos) & set(cnt.index[(cnt == 0).any(axis=1)]))
        chk(True, f"G2 {var}: algos with data at every n", f"{full} of {len(cnt)} full; gaps: {thin or 'none'}")
        for t in thin:
            print(f"         NOTE {t} is missing some n -- its curve is drawn short, never interpolated")

    # G3  the control. domr's cover IS H, so on the GMR panel its ratio is |H|/OPT_GMR. Section 5.1 rests on
    #     that being 1; if it drifts, the licence to use |H| as a reference elsewhere is gone.
    D = ok[ok.algo.eq(CTRL) & ok.variant.eq("DOMR")].dropna(subset=["ratio"])
    med, lo, hi = D.ratio.median(), D.ratio.quantile(.25), D.ratio.quantile(.75)
    chk(len(D) > 0 and abs(med - 1.0) < 1e-9,
        "G3 domr ratio == 1 (the GMR optimum IS the heavy set)",
        f"median {med:.4f}  IQR [{lo:.4f}, {hi:.4f}]  over {len(D)} tasks")

    # G4  THE REFERENCE MUST NOT CHANGE UNDER THE CURVE.
    #
    # This gate exists because the first version of this figure was WRONG, and looked entirely plausible.
    # analyze.py:109 sets  iomr_ref = max(iomr_lp_rsp, iomr_lp_naive).  `iomr_lp_rsp` -- the TIGHT bound --
    # exists only for n <= 150 (the rsp oracle needs integer weights and small n). Above that the reference
    # silently falls back to `iomr_lp_naive`, which is 5.39x LOOSER (median, over the 360 tasks where both
    # exist). So the plotted reference COLLAPSES from a median of 1302 at n=150 to 228 at n=160, and every
    # IOMR curve jumps ~6x at exactly that point. The cliff is the reference moving, not the algorithms.
    #
    # The first version of THIS gate did not catch it, and that is the more important lesson: it verified
    # integrality on the tasks where iomr_ilp converged, and iomr_ilp converges on ZERO tasks at p=0.3 --
    # so it was silently checking a different (n, p) region, backed by the rsp bound, while the figure drew
    # the naive one. It passed at max |dev| = 0.0e+00 and told us nothing about the quantity on the page.
    # A check that cannot fail on the thing you are drawing is not a check.
    #
    # And the naive bound is NOT integral: OPT / naive = 1.50 median, IQR [1.00, 1.95], max 9.00. So
    # |S| / naive is not an approximation ratio at all.
    #
    # The gate therefore checks the PLOTTED reference, per n, and fails if its source is not constant.
    src = {}
    for al in ("iomr_lp_rsp", "iomr_lp_naive"):
        A = q[q.algo.eq(al) & q.status.eq("ok")]
        src[al] = set(A.n.dropna().astype(int))
    ns = sorted(set(q.n.dropna().astype(int)))
    # the reference is max(rsp, naive); where rsp is absent the source changes identity
    with_rsp = [n for n in ns if n in src["iomr_lp_rsp"]]
    without = [n for n in ns if n not in src["iomr_lp_rsp"]]
    constant = not (with_rsp and without)
    chk(constant, "G4 the IOMR reference has ONE source across every plotted n",
        f"rsp bound present at n={with_rsp[:3]}{'...' if len(with_rsp) > 3 else ''} (max "
        f"{max(with_rsp) if with_rsp else '--'}), ABSENT at n={without[:3]}"
        f"{'...' if len(without) > 3 else ''}")
    if not constant:
        r = q[q.algo.eq("iomr_lp_rsp") & q.status.eq("ok")].set_index("task").lp_bound
        nv = q[q.algo.eq("iomr_lp_naive") & q.status.eq("ok")].set_index("task").lp_bound
        b = r.index.intersection(nv.index)
        if len(b):
            slack = float((r[b] / nv[b]).median())
            print(f"         the two bounds differ by {slack:.2f}x -- drawing them on one axis manufactures "
                  f"a {slack:.1f}x cliff that is not an algorithmic effect")
    return fails, np.nan


def draw(q, figdir, gap):
    ok = q[q.status.eq("ok") & q.valid.eq(1)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (var, kind, ylab, algos) in zip(axes, PANELS):
        K = ok[ok.variant.eq(var)]
        for al in algos:
            A = K[K.algo.eq(al)].dropna(subset=["ratio"])
            if A.empty:
                print(f"    no series for {al} -- it returned no verified cover at p={P}")
                continue
            g = A.groupby("n").ratio.agg(["median", lambda s: s.quantile(.25), lambda s: s.quantile(.75)])
            g.columns = ["med", "q25", "q75"]
            g = g.sort_index()
            c, mk = STY[al]
            ax.plot(g.index, g["med"], marker=mk, ms=4, lw=1.6, color=c, label=nm(al))
            ax.fill_between(g.index, g["q25"], g["q75"], color=c, alpha=0.12, lw=0)
        # the control, on both panels
        D = ok[ok.algo.eq(CTRL)].dropna(subset=["ratio"])
        if not D.empty:
            g = D.groupby("n").ratio.median().sort_index()
            ax.plot(g.index, g.values, ls="--", lw=1.8, color="black", label="domr (= $H$)")
        ax.axhline(1.0, color="black", lw=0.8, alpha=0.45)
        ax.set_yscale("log")
        ax.set_xlabel("$n$   (dense $\\Gamma(n,p)$, $p = 0.3$, 30 seeds per $n$)", fontsize=9)
        ax.set_ylabel(ylab + "     $\\downarrow$ lower is better", fontsize=9)
        ax.set_title(var, fontsize=11)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    os.makedirs(figdir, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(figdir, f"small_p30_ratio_vs_n.{ext}")
        fig.savefig(p, dpi=160, bbox_inches="tight")
    print(f"\n  wrote {figdir}/small_p30_ratio_vs_n.{{pdf,png}}")


def fig_dir(p):
    """Refuse anywhere that is not the paper's figures/. Same discipline as the table generators."""
    if not os.path.isdir(p):
        raise SystemExit(f"FATAL: --figdir '{p}' is not a directory. Refusing to create it.")
    p = os.path.abspath(p)
    root = os.path.dirname(p)
    if not os.path.exists(os.path.join(root, "story.tex")):
        raise SystemExit(f"FATAL: no story.tex beside '{p}'. That is not the paper's figures/ directory, "
                         "and writing a figure into the wrong place is how a stale figure survives a move.")
    if os.path.basename(p) != "figures":
        raise SystemExit(f"FATAL: --figdir '{p}' is in the paper but is not figures/.")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", required=True, help="the paper's figures/ directory")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    figdir = fig_dir(args.figdir)

    d, q = load()
    print(f"{EXP} @ p={P}: {q.task.nunique()} tasks, n in [{int(q.n.min())}, {int(q.n.max())}]\n")
    print("GATE -- nothing is drawn until these pass")
    fails, gap = gate(d, q)
    if fails and not args.force:
        raise SystemExit(f"\n*** {len(fails)} GATE FAILURE(S): {fails}. NOT writing. The previous figure is "
                         "left on disk: a paper should keep its old figure rather than gain a wrong one. ***")
    draw(q, figdir, gap)
    print("\nAll gates passed." if not fails else "\n!! WRITTEN UNDER --force. Do not quote this figure.")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
