"""Emit tab:corruption -- what repair buys downstream, and how the answer flips with the corruption.

WHAT THE TABLE SAYS. Two probes, both against an external ground truth, both computed from the SAME cover
on the SAME repaired distance matrix:

    TOPOLOGY   k-NN Jaccard against the truth      (higher is better)
    GEOMETRY   Procrustes disparity of an MDS      (lower is better)

Cross them with the corruption, and the verdict inverts. Under DEFLATION repair wins both axes. Under
INFLATION it loses both, and the median algorithm makes the map an order of magnitude worse -- which is
Corollary (inflate) in the flesh: an inflated edge already exceeds its own detour, so `restore` is the
identity on the metric and cannot undo the damage, while everything it *does* touch it wrecks.

And on the real base -- a New York road network with a planted corruption -- the two axes DISAGREE:
geometry improves while topology is damaged, from the same covers.

WHY THE MEDIAN, AND WHICH MEDIAN. Choosing the BEST algorithm requires the ground truth, which is the very
thing repair is trying to recover. Best-of-suite is therefore an oracle: an upper bound on what repair COULD
do, not what it WILL do. The median is what a practitioner choosing blind actually gets, so the median is
what we report.

  We report the MEDIAN ALGORITHM, by name, and that algorithm's value -- not the interpolated median of the
  values. The two differ whenever an even number of algorithms return, because then no algorithm SITS at
  the median (on rgg_inflate: interpolated 14.8x worse, the median algorithm 13.9x). Naming a method
  obliges us to quote the number that method actually produced. Convention: the LOWER median of the
  algorithms that returned. It is stated in the caption, because a median convention left unstated is a
  number half-defined.

  Naming it also earns the table its second point: the median algorithm is a DIFFERENT algorithm in every
  row, and different again between the two axes. The ranking does not merely shift -- it does not survive
  the change of corruption at all.

  inputs    analysis/summary_rgg_recovery.csv    the planted RGG      (disparity column: disp_smacof)
            analysis/summary_recovery.csv        the planted road net (disparity column: disp)
  output    <texdir>/tab_corruption.tex

  usage     sage -python experiments/corruption_table.py --texdir "<paper>/tables"

!! THE TWO CSVs DO NOT SHARE A SCHEMA. The disparity column is `disp_smacof` in one and `disp` in the
   other. Read the wrong one and pandas raises -- but read a column that HAPPENS to exist under both names
   and it would not. G2 pins the expected column per source so a schema drift cannot pass silently.

THE CONTROL THAT MAKES THIS TABLE TRUSTWORTHY. \\DOMR is an exact control, not an algorithm under test. By
the decrease-only lemma its cover changes NO shortest path, so it must move NEITHER axis -- its effect is
zero BY CONSTRUCTION. A nonzero reading is therefore not a finding, it is a bug in the pipeline, and G3
fails on it. Measured today: 0.00e+00 on the RGG and 3.9e-16 on the road net. That is the strongest gate
available here, because it tests the whole chain -- cover, reweighting, all-pairs shortest paths, MDS,
Procrustes -- against a value theory fixes in advance.

Gate runs BEFORE the write. A failing gate leaves the previous .tex on disk.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

K = "knn20"          # k = 20. Fixed across every table in the paper; see the pbmc3k construction ceiling.
TOL = 1e-9           # the DOMR control is exact in theory; this allows for APSP + MDS float drift only.

# (source csv, disparity column, rows to draw from it, in table order)
#
# ONE RGG SIZE, NOT TWO. The n=300 and n=1000 planted RGGs tell the identical story -- deflation wins both
# axes, inflation loses both -- and printing both spends six rows to say what three say. We keep the larger:
# n=1000 is the harder instance and the less easily dismissed. The n=300 rows remain in
# summary_rgg_recovery.csv for anyone who wants them.
SOURCES = [
    ("analysis/summary_rgg_recovery.csv", "disp_smacof",
     [("rgg_deflate_n1000", "planted RGG", "deflation"),
      ("rgg_inflate_n1000", "", "inflation"),
      ("rgg_mixed_n1000", "", "mixed")]),
    ("analysis/summary_recovery.csv", "disp",
     [("dimacs_ny_d_deflate", r"\code{dimacs\_ny\_d}", "deflation"),
      ("dimacs_ny_d_inflate", "", "inflation"),
      ("dimacs_ny_d_mixed", "", "mixed")]),
]

TEX = {"gmr_bestofk": r"\code{gmr\_bok}", "iomr_bestofk": r"\code{iomr\_bok}",
       "gmr_thr_naive": r"\code{gmr\_thr}", "iomr_thr_naive": r"\code{iomr\_thr}",
       "iomr_regiongrow": r"\code{iomr\_rgrow}"}


def a(name):
    return TEX.get(str(name), r"\code{%s}" % str(name).replace("_", r"\_"))


def load(path, dcol):
    if not os.path.exists(path):
        raise SystemExit(f"FATAL: {path} missing.")
    d = pd.read_csv(path)
    # G2 -- pin the schema. The two sources name the disparity column differently; a drift here would
    # otherwise surface as a KeyError at best, or as the WRONG column at worst.
    if dcol not in d.columns:
        raise SystemExit(f"FATAL: {path} has no column '{dcol}'. Columns: {list(d.columns)}. "
                         "The two recovery CSVs do not share a schema; refusing to guess.")
    if K not in d.columns:
        raise SystemExit(f"FATAL: {path} has no column '{K}'.")
    for c in (dcol, K):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def median_algo(rep, col, higher_better):
    """The algorithm at the MIDDLE RANK, and its value.

    Not the interpolated median of the values: with an even number of returning algorithms no algorithm sits
    there, and naming a method obliges us to quote what that method actually produced. Lower median."""
    s = rep.sort_values(col, ascending=not higher_better)
    r = s.iloc[(len(s) - 1) // 2]
    return str(r.algo), float(r[col])


def collect():
    """One record per (graph, corruption). Every number derived; nothing carried over from a draft."""
    recs = []
    for path, dcol, graphs in SOURCES:
        d = load(path, dcol)
        ok = d[d.status.eq("ok")]
        for g, group_label, corruption in graphs:
            G = ok[ok.graph.eq(g)]
            if G.empty:
                raise SystemExit(f"FATAL: no ok rows for '{g}' in {path}.")
            obs = G[G.algo.eq("observed")]
            dom = G[G.algo.eq("domr")]
            if len(obs) != 1:
                raise SystemExit(f"FATAL: expected exactly 1 'observed' row for {g}, found {len(obs)}. "
                                 "It is the baseline every number on the row is relative to.")
            obs = obs.iloc[0]
            # the pool under test: everything that is neither the baseline nor the control
            all_rep = d[d.graph.eq(g) & (~d.algo.isin(["observed", "domr"]))]
            rep = G[~G.algo.isin(["observed", "domr"])].dropna(subset=[dcol, K])
            if rep.empty:
                raise SystemExit(f"FATAL: no algorithm returned on '{g}'. Nothing to take a median of.")

            g_algo, g_med = median_algo(rep, dcol, higher_better=False)   # geometry: lower is better
            t_algo, t_med = median_algo(rep, K, higher_better=True)       # topology: higher is better
            # n is no longer in the row label (it made the table wide), so it moves to the caption -- but it
            # is still DERIVED from the data rather than typed into the prose.
            ncol = "n" if "n" in G.columns else ("n_used" if "n_used" in G.columns else None)
            recs.append(dict(
                graph=g, group=group_label, corruption=corruption,
                n_nodes=(int(G[ncol].dropna().iloc[0]) if ncol and G[ncol].notna().any() else None),
                n_ret=len(rep), n_all=len(all_rep),
                obs_d=float(obs[dcol]), obs_k=float(obs[K]),
                g_algo=g_algo, g_med=g_med, t_algo=t_algo, t_med=t_med,
                # the DOMR control, carried so the gate can check it and the caption can quote it
                domr_d=(float(dom[dcol].iloc[0]) if len(dom) else np.nan),
                domr_k=(float(dom[K].iloc[0]) if len(dom) else np.nan),
                # the oracle premium -- REPORTED, never quoted in the table unlabelled
                best_d=float(rep[dcol].min()), best_k=float(rep[K].max()),
            ))
    for r in recs:
        # BOTH axes as a MULTIPLIER of the observed value: median / observed. One definition, applied twice.
        #
        # But the two factors READ IN OPPOSITE DIRECTIONS, and that is unavoidable: a disparity is an error
        # (lower is better, so a factor BELOW 1 is a win), a Jaccard is an agreement (higher is better, so a
        # factor ABOVE 1 is a win). The same 0.58x means "better map" in one block and "worse neighbourhoods"
        # in the other. We do not paper over it by inverting one of them -- that would hide the asymmetry
        # rather than state it. Each stacked block carries its own direction arrow, and the caption says it
        # in words.
        r["g_fac"] = r["g_med"] / r["obs_d"]        # geometry: < 1 is BETTER (error shrank)
        r["t_fac"] = r["t_med"] / r["obs_k"]        # topology: > 1 is BETTER (agreement grew)
        r["g_chg"] = 100.0 * (r["obs_d"] - r["g_med"]) / r["obs_d"]      # kept for the gate's report only
        r["t_chg"] = 100.0 * (r["t_med"] - r["obs_k"]) / r["obs_k"]
        r["g_best_fac"] = r["best_d"] / r["obs_d"]
        r["t_best_fac"] = r["best_k"] / r["obs_k"]
    return recs


# ----------------------------------------------------------------------------
# The gate
# ----------------------------------------------------------------------------
def gate(recs):
    fails = []

    def chk(ok, name, obs):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<50} {obs}")
        if not ok:
            fails.append(name)

    print("GATE -- the DOMR control is fixed by theory; everything else is derived from the CSVs")

    # G3  THE CONTROL. Lemma: a decrease-only cover changes no shortest path, so DOMR moves NEITHER axis.
    #     Its effect is zero BY CONSTRUCTION -- a nonzero reading is a bug in the chain (cover, reweighting,
    #     APSP, MDS, Procrustes), not a result. This is the strongest check available: it tests the whole
    #     pipeline against a number theory fixed in advance, and it is the one that would catch a silently
    #     broken repair path while every other cell still looked plausible.
    wd = max(abs(r["domr_d"] - r["obs_d"]) for r in recs if np.isfinite(r["domr_d"]))
    wk = max(abs(r["domr_k"] - r["obs_k"]) for r in recs if np.isfinite(r["domr_k"]))
    chk(wd <= TOL and wk <= TOL, "G3  DOMR control moves neither axis (Lemma)",
        f"max |domr - observed|: disparity {wd:.1e}, kNN {wk:.1e}")

    # G3b ANTI-VACUITY for G3. If DOMR ever stopped being reported, G3 would pass over an empty set and
    #     prove nothing. Require the control to be PRESENT on every row.
    n_dom = sum(1 for r in recs if np.isfinite(r["domr_d"]))
    chk(n_dom == len(recs), "G3b the control is present on every row", f"{n_dom} of {len(recs)}")

    # G4  the named median algorithm must be a real, returning algorithm -- not an artefact of the sort.
    bad = [r["graph"] for r in recs if not r["g_algo"] or not r["t_algo"] or r["n_ret"] < 1]
    chk(not bad, "G4  a real algorithm sits at each median", "all rows" if not bad else str(bad))

    # G5  ranges. kNN Jaccard is a fraction; a disparity is non-negative.
    bad = [r["graph"] for r in recs
           if not (0 <= r["obs_k"] <= 1 and 0 <= r["t_med"] <= 1) or r["obs_d"] < 0 or r["g_med"] < 0]
    chk(not bad, "G5  kNN in [0,1], disparity >= 0", "all in range" if not bad else str(bad))

    # G6  a median over survivors is only a median if enough survived. NOT fatal -- it is a disclosure. The
    #     road net loses most of the suite to the time cap, and the survivors are systematically the cheap
    #     combinatorial methods, so the median is biased toward them. The caption must say so, and it does.
    thin = [(r["graph"], r["n_ret"], r["n_all"]) for r in recs if r["n_ret"] < 0.5 * r["n_all"]]
    print(f"  [{'NOTE' if thin else 'PASS'}] G6  median over survivors                          "
          + ("all rows >= half the suite" if not thin
             else "; ".join(f"{g}: {a_}/{b}" for g, a_, b in thin) + "  <- disclosed in the caption"))

    # REPORT -- the oracle premium. Never quoted in the table without its label: choosing the best algorithm
    # needs the ground truth, so 'best' is an upper bound on what repair COULD do, not what it WILL do.
    #
    # In FACTORS, like the table. A percentage on an unbounded error is the very thing this table stopped
    # printing; there is no reason for the diagnostics to keep lying in a unit the paper has abandoned.
    print("\n  REPORT -- the oracle premium (best cover vs the median a practitioner actually gets):")
    for r in recs:
        bfac = r["g_best_fac"]
        if abs(bfac - r["g_fac"]) > 0.05 or (bfac < 1.0) != (r["g_fac"] < 1.0):
            flip = "  <-- THE ORACLE FLIPS THE VERDICT" if (bfac < 1.0) != (r["g_fac"] < 1.0) else ""
            print(f"      {r['graph']:<22} geometry: best {bfac:5.2f}x   median {r['g_fac']:6.2f}x"
                  f"   (1.00x = no change; below 1 is better){flip}")
    return fails


# ----------------------------------------------------------------------------
# Emit
# ----------------------------------------------------------------------------
def pct(v):
    return r"$%+.1f\%%$" % v


def sci(v):
    """LaTeX scientific notation -- and say EXACTLY ZERO when it is exactly zero.

    The kNN drift of the DOMR control is 0.0, not 1e-99. Printing a floor value to keep a log() happy would
    understate the control: the claim is not 'very small', it is 'identical'."""
    if v == 0.0:
        return r"exactly $0$"
    e = int(np.floor(np.log10(abs(v))))
    return r"$%.0f \times 10^{%d}$" % (v / 10.0 ** e, e)


def _n(x):
    """Thousands separator, LaTeX-math style: 5000 -> 5{,}000."""
    return f"{int(x):,}".replace(",", "{,}")


def stacked(val, algo):
    """The median value, with the name of the algorithm that produced it set ABOVE it.

    THE NAME GOES ON TOP, AND THAT IS THE WHOLE TRICK. \\shortstack builds a vbox whose reference point is
    its LAST line. Put the name last and the NAME lands on the row's baseline while the number floats above
    it -- so the median column's numbers sit a line higher than the obs. and vs.-obs. numbers beside them,
    and nothing in the row lines up. That is what the first version of this table did, and it looked wrong
    for exactly that reason.

    Put the NUMBER last, and the number lands on the baseline: it aligns with every other number in the
    line, decimal point under decimal point, while the name rides above it as the annotation it is. One
    table row per record, no negative leading, no second row of empty cells. The fix is the ordering."""
    return (r"\shortstack[r]{{\tiny\texttt{%s}}\\[-1pt]$%.4f$}"
            % (str(algo).replace("_", r"\_"), val))


def emit(recs):
    ret_lo = min(r["n_ret"] for r in recs)
    ret_hi = max(r["n_ret"] for r in recs)
    n_all = max(r["n_all"] for r in recs)
    thin = [r for r in recs if r["n_ret"] < 0.5 * r["n_all"]]
    wd = max(abs(r["domr_d"] - r["obs_d"]) for r in recs)
    wk = max(abs(r["domr_k"] - r["obs_k"]) for r in recs)

    infl = [r for r in recs if r["corruption"] == "inflation" and r["graph"].startswith("rgg")]
    worst = max(infl, key=lambda r: r["g_fac"])
    defl = [r for r in recs if r["corruption"] == "deflation" and r["graph"].startswith("rgg")]
    road = [r for r in recs if r["graph"] == "dimacs_ny_d_inflate"][0]
    n_distinct = len({r["g_algo"] for r in recs} | {r["t_algo"] for r in recs})

    sizes = {}
    for r in recs:
        if r["n_nodes"]:
            sizes.setdefault(r["group"] or "", r["n_nodes"])
    size_txt = ", ".join(r"%s $n = %s$" % (g.strip() or "the graph", _n(v))
                         for g, v in sizes.items() if g.strip())

    # SHORT. The author edits captions by hand, and a caption that argues the methodology in the float is a
    # caption he deletes: the first version ran 2,682 characters and cost a page of a ten-page limit. Lead,
    # how to read it, the headline number, the one disclosure the table would be dishonest without. Stop.
    # The median convention, the DOMR control and the two-axes disagreement belong in the section prose.
    # Every number here is still DERIVED -- shortening a caption is never a licence to type a value in.
    cap = (
        r"\caption{\textbf{The corruption decides.} Geometry (above) and topology (below), from the "
        r"\emph{same} cover on the \emph{same} repaired matrix, each as a multiplier of the observed "
        r"graph's value; beneath each median sits the algorithm that produced it. A disparity is an "
        r"\emph{error} ($<1$ is better); a Jaccard is an \emph{agreement} ($>1$ is better). Deflation wins "
        r"on both axes; inflation loses on both, the median algorithm making the map $%.1f\times$ worse "
        r"while keeping only $%.2f$ of the neighbourhoods. Sizes: %s; $k = %s$. \emph{ret} is how many of "
        r"the %d repair algorithms returned within the time cap%s.}"
        % (worst["g_fac"], worst["t_fac"], size_txt, K.replace("knn", ""), n_all,
           (r"; on \code{dimacs\_ny\_d} most of the suite times out and the survivors are the cheap "
            r"combinatorial methods, so those medians are biased toward them" if thin else "")))

    out = [r"% GENERATED by experiments/corruption_table.py -- DO NOT EDIT, DO NOT TRANSCRIBE. Regenerate:",
           r"%   sage -python experiments/corruption_table.py --texdir '<paper>/tables'",
           r"% Sources: analysis/summary_rgg_recovery.csv, analysis/summary_recovery.csv",
           r"% ONE COLUMN (table, not table*): stacking the two axes halved the width.",
           r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{2.5pt}"
           r"\renewcommand{\arraystretch}{1.0}", cap,
           r"\label{tab:corruption}",
           # SIX columns. The two axes are STACKED, not set side by side: geometry above, topology below,
           # sharing one set of columns. Side by side cost twice the width to carry the same six fields
           # twice, and the table is read DOWN a corruption anyway, not across an axis.
           r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
           r"graph & corruption & ret & obs. & median & vs.\ obs. \\"]

    def block(title, obs_key, med_key, algo_key, fac_key, win):
        """One stacked half of the table. `win` decides which side of 1.0 is a victory on this axis -- and
        the two axes disagree about that, which is exactly why each block carries its own arrow.

        Each record becomes TWO table rows: the numbers, then the median algorithm's name on its own line
        beneath the median. That keeps every number in a line on one baseline (see algo_line)."""
        rows = [r"\midrule",
                r"\multicolumn{6}{@{}l}{\textbf{%s}} \\[2pt]" % title]
        for i, r in enumerate(recs):
            if r["group"] and i:
                rows.append(r"\addlinespace[2pt]")
            f = r[fac_key]
            cell = (r"$\mathbf{%.2f\times}$" % f) if win(f) else (r"$%.2f\times$" % f)
            # ONE row per record. The algorithm's name rides above its median inside the cell (see stacked),
            # so it needs no row of its own and no negative leading to hold it in place.
            rows.append(r"%s & %s & $%d$ & $%.4f$ & %s & %s \\"
                        % (r["group"], r["corruption"], r["n_ret"], r[obs_key],
                           stacked(r[med_key], r[algo_key]), cell))
        return rows

    # The block headers are \multicolumn{6} text, so their LENGTH sets the table's width. Keep them terse --
    # the arrow and the inequality carry the direction, and the caption carries the argument. A header that
    # explains itself in a sentence would make the table as wide as the sentence.
    #
    # GEOMETRY: a Procrustes disparity is an ERROR. Lower is better, so a factor BELOW 1 is the win.
    out += block(r"geometry: Procrustes disparity ($\downarrow$; ${<}1$ is better)",
                 "obs_d", "g_med", "g_algo", "g_fac", lambda f: f < 0.99)

    # TOPOLOGY: a k-NN Jaccard is an AGREEMENT. Higher is better, so a factor ABOVE 1 is the win. Same
    # definition of the factor (median / observed); opposite direction of victory. Do not "fix" this by
    # inverting one of them: the asymmetry is real, and hiding it would cost the reader more than it saves.
    out += block(r"topology: $k$-NN Jaccard ($\uparrow$; ${>}1$ is better)",
                 "obs_k", "t_med", "t_algo", "t_fac", lambda f: f > 1.01)

    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def paper_dir(p):
    """Refuse a directory that is not where the paper's generated LaTeX lives.

    A --texdir typo must not quietly write a table into a folder the paper never reads: LaTeX would go on
    compiling last week's file without a murmur, and that is the exact rot this script exists to stop. So
    locate story.tex, then accept ONLY the directory holding it or its tables/ subdirectory. figures/,
    buildup/ and a path outside the paper are all refused."""
    if not os.path.isdir(p):
        raise SystemExit(f"FATAL: --texdir '{p}' is not a directory. Refusing to create it -- a wrong path "
                         "silently leaves the real paper compiling a stale table.")
    p = os.path.abspath(p)
    root = p if os.path.exists(os.path.join(p, "story.tex")) else os.path.dirname(p)
    if not os.path.exists(os.path.join(root, "story.tex")):
        raise SystemExit(f"FATAL: no story.tex in '{p}' or its parent. That does not look like the paper.")
    tables = os.path.join(root, "tables")
    if p not in (root, tables):
        raise SystemExit(f"FATAL: --texdir '{p}' is inside the paper but is not where generated LaTeX "
                         f"lives. Use '{tables}'.")
    return p


def main():
    ap = argparse.ArgumentParser(description="Emit tab:corruption (what repair buys downstream), gated.")
    ap.add_argument("--texdir", required=True)
    args = ap.parse_args()
    texdir = paper_dir(args.texdir)

    recs = collect()
    fails = gate(recs)
    if fails:
        raise SystemExit(f"\n*** GATE FAILED: {len(fails)} invariant(s) violated ***\n  " + "\n  ".join(fails)
                         + "\n\nNOT writing. The previous tab_corruption.tex is left on disk.")

    tex = emit(recs)
    if re.search(r"(?i)(?<![a-z])(nan|inf)(?![a-z])", tex):
        raise SystemExit("*** GATE FAILED (G7): rendered LaTeX contains nan/inf. NOT writing. ***")
    print("  [PASS] G7  rendered LaTeX carries no nan/inf")

    path = os.path.join(texdir, "tab_corruption.tex")
    with open(path, "w") as fh:
        fh.write(tex + "\n")
    print("\n" + tex + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
