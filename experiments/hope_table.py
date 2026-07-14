"""Emit tab:hope -- the hope, answered. Does repair buy anything on real data with an external truth?

THE HOPE. Every correction method in every field rests on one premise: that analysis run on the corrected
data is a better surrogate for the truth than analysis run on the raw data. This table tests it, on the five
real graphs that carry an external ground truth, against both probes:

    TOPOLOGY   k-NN Jaccard against the truth, at k = 5, 10, 20     (higher is better)
    GEOMETRY   Procrustes disparity of an MDS embedding             (lower is better)

WE REPORT THE BEST ALGORITHM, NOT THE MEDIAN -- and that is the opposite of tab:corruption, on purpose. The
claim here is a NEGATIVE one ("repair improves topology nowhere"), and a negative claim must be made in its
STRONGEST form or it is worthless. Reporting the median would leave a referee to ask whether some cleverer
method in the suite would have won. So we hand repair its best shot -- the best of every algorithm, chosen
WITH the ground truth, an oracle no practitioner could ever have -- and it still loses. If the best fails,
the median cannot save it.

THE CLAIM IS DERIVED, NOT ASSERTED. The caption does not carry a hardcoded sentence saying repair never
wins. It counts the wins and writes the sentence from the count. If a win ever appears, the caption says so
by itself, and nobody has to remember to go and edit it.

TWO DISCLOSURES THE TABLE WOULD BE DISHONEST WITHOUT, both forced by the gate:

  * pbmc3k at k <= 15 is TAUTOLOGICAL. That graph IS the 15-nearest-neighbour graph of its own truth, so
    k-NN recovery at k <= 15 is ~1.0 BY CONSTRUCTION and there is no headroom to win. The single positive
    topology lift in the whole study lives exactly there (+9e-06, i.e. +0.0009% relative) and is float
    noise on a cell that could not have gone the other way. G4 checks that the tautology is still present
    in the data -- if pbmc3k's observed recovery at low k ever stopped being ~1, the disclosure would be
    stale and the cell would need re-reading.

  * A "win" needs to be a WIN. We require a >TOL relative improvement. Without a threshold, +9e-06 counts
    as repair beating the raw data, and the paper's central negative result dies on a rounding artefact.

  inputs    analysis/summary_pure_real.csv     topology (k-NN lift), 5 graphs x 3 k x 16 algos
            analysis/summary_mds_sweep.csv     geometry, 4 of the graphs
            analysis/summary_mds_rna.csv       geometry, pbmc3k (it lives in its own file)
  output    <texdir>/tab_hope.tex

  usage     sage -python experiments/hope_table.py --texdir "<paper>/tables"

DOMR is the control, here as everywhere: by the decrease-only lemma its cover changes no shortest path, so
its lift is zero BY CONSTRUCTION. G2 checks it, and a nonzero reading is a bug rather than a result.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

TOPO = "analysis/summary_pure_real.csv"
GEOM = "analysis/summary_mds_sweep.csv"
GEOM_RNA = "analysis/summary_mds_rna.csv"

KS = (5, 10, 20)
TOL = 0.01          # a win must be a >1% RELATIVE improvement. Below that it is noise, not a repair.
CTRL_TOL = 1e-9     # the DOMR control is exact in theory; this allows for float drift only.

GRAPHS = [
    ("dimacs_ny_t", r"\code{dimacs\_ny\_t}", "road net (time)"),
    ("nmr_1d3z_atom", r"\code{nmr\_atom}", "protein (atom)"),
    ("nmr_1d3z_residue", r"\code{nmr\_res}", "protein (residue)"),
    ("pbmc3k_cosine_knn", r"\code{pbmc3k}", "single-cell"),
    ("ripe_atlas", r"\code{ripe\_atlas}", "internet latency"),
]

TEX = {"gmr_bestofk": r"\code{gmr\_bok}", "iomr_bestofk": r"\code{iomr\_bok}",
       "gmr_thr_naive": r"\code{gmr\_thr}", "iomr_thr_naive": r"\code{iomr\_thr}",
       "iomr_regiongrow": r"\code{iomr\_rgrow}"}


def a(name):
    return TEX.get(str(name), r"\code{%s}" % str(name).replace("_", r"\_"))


def sci(v):
    """LaTeX scientific notation. Python's %e emits '9e-06', which is not maths -- it is a Python repr that
    happens to survive TeX. Render it as $9 \\times 10^{-6}$."""
    if v == 0.0:
        return r"$0$"
    e = int(np.floor(np.log10(abs(v))))
    return r"$%+.0f \times 10^{%d}$" % (v / 10.0 ** e, e)


def load(path, what):
    if not os.path.exists(path):
        raise SystemExit(f"FATAL: {path} missing ({what}).")
    return pd.read_csv(path)


def topology():
    """Per (graph, k): the observed recovery, and the BEST lift any algorithm achieved.

    Best, not median: the claim is negative, so it must be made in its strongest form. Handing repair an
    oracle it could never have and watching it lose anyway is the only version of this result a referee
    cannot argue with."""
    d = load(TOPO, "topology")
    for c in ("k", "recovery_obs", "lift_med"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    ctrl = d[d.algo.eq("domr")]
    rep = d[d.algo.ne("domr")].dropna(subset=["lift_med"])
    out = {}
    for g, _, _ in GRAPHS:
        for k in KS:
            G = rep[rep.graph.eq(g) & rep.k.eq(k)]
            if G.empty:
                raise SystemExit(f"FATAL: no topology rows for {g} at k={k}. The claim is universal over "
                                 "these cells; refusing to render a table with a hole in it.")
            b = G.loc[G.lift_med.idxmax()]
            obs = float(G.recovery_obs.iloc[0])
            out[(g, k)] = dict(obs=obs, lift=float(b.lift_med), algo=str(b.algo), n=len(G),
                               rel=float(b.lift_med) / obs if obs else np.nan)
    return out, ctrl


def geometry():
    """Per graph: observed disparity, and the BEST repaired disparity. pbmc3k lives in its own CSV."""
    m = load(GEOM, "geometry")
    r = load(GEOM_RNA, "geometry (pbmc3k)")
    r = r.assign(graph="pbmc3k_cosine_knn")
    m = pd.concat([m, r], ignore_index=True)
    m = m[m.status.fillna("ok").eq("ok") & m.disp_smacof.notna()]
    out = {}
    for g, _, _ in GRAPHS:
        G = m[m.graph.eq(g)]
        if G.empty:
            raise SystemExit(f"FATAL: no geometry rows for {g}.")
        o = G[G.algo.eq("observed")]
        rep = G[~G.algo.isin(["observed", "domr"])]
        if o.empty or rep.empty:
            raise SystemExit(f"FATAL: {g} lacks an observed row or any repair row on the geometry axis.")
        od = float(o.disp_smacof.iloc[0])
        b = rep.loc[rep.disp_smacof.idxmin()]
        out[g] = dict(obs=od, best=float(b.disp_smacof), algo=str(b.algo), n=len(rep),
                      gain=(od - float(b.disp_smacof)) / od)
    return out


# ----------------------------------------------------------------------------
# The gate
# ----------------------------------------------------------------------------
def gate(T, ctrl, G):
    fails = []

    def chk(ok, name, obs):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<48} {obs}")
        if not ok:
            fails.append(name)

    print("GATE -- the negative claim is COUNTED, never asserted")

    # G1  every cell present. A universal claim ("nowhere") with a missing cell is not a universal claim.
    chk(len(T) == len(GRAPHS) * len(KS), "G1  every (graph, k) cell present",
        f"{len(T)} of {len(GRAPHS) * len(KS)}")

    # G2  THE CONTROL. DOMR's cover changes no shortest path (decrease-only lemma), so its k-NN lift is zero
    #     BY CONSTRUCTION. A nonzero reading is a bug in the chain, not a result.
    cl = pd.to_numeric(ctrl.lift_med, errors="coerce").dropna()
    worst = float(cl.abs().max()) if len(cl) else np.nan
    chk(len(cl) > 0 and worst <= CTRL_TOL, "G2  DOMR control has zero lift (Lemma)",
        f"{len(cl)} control rows, max |lift| = {worst:.1e}")

    # G4  THE TAUTOLOGY IS STILL THERE. pbmc3k IS the 15-NN graph of its own truth, so at k <= 15 its
    #     observed recovery is ~1 by construction and no repair can win. The single positive lift in the
    #     study sits exactly there. If that tautology ever stopped holding, the disclosure in the caption
    #     would be stale and the cell would have to be re-read -- so check it, do not assume it.
    lo = [T[("pbmc3k_cosine_knn", k)]["obs"] for k in KS if k <= 15]
    chk(all(v > 0.99 for v in lo), "G4  pbmc3k k<=15 is still tautological (obs ~ 1)",
        "obs = " + ", ".join(f"{v:.4f}" for v in lo))

    # G5  ranges
    bad = [k for k, v in T.items() if not (0 <= v["obs"] <= 1)]
    chk(not bad, "G5  k-NN recovery in [0,1]", "all in range" if not bad else str(bad))

    # --- THE CLAIM, COUNTED. Not asserted anywhere: the caption is written from these numbers.
    wins = {k: v for k, v in T.items() if v["rel"] > TOL}
    pos = {k: v for k, v in T.items() if v["lift"] > 0}
    gw = {g: v for g, v in G.items() if v["gain"] > TOL}
    print(f"\n  COUNTED -- topology: {len(wins)} of {len(T)} cells show a genuine win (>{100*TOL:.0f}% rel.)")
    for (g, k), v in pos.items():
        print(f"      the only positive lift at all: {g} k={k} ({v['algo']}) "
              f"{v['lift']:+.1e} = {100*v['rel']:+.4f}% relative"
              + ("   <-- BELOW the win threshold: noise" if v["rel"] <= TOL else "   <-- A GENUINE WIN"))
    if not pos:
        print("      no positive lift anywhere, at any k, by any algorithm")
    print(f"  COUNTED -- geometry: {len(gw)} of {len(G)} graphs improve "
          f"({', '.join(gw) if gw else 'none'})")
    return fails


# ----------------------------------------------------------------------------
# Emit
# ----------------------------------------------------------------------------
def emit(T, G):
    wins = {k: v for k, v in T.items() if v["rel"] > TOL}
    gw = {g: v for g, v in G.items() if v["gain"] > TOL}
    n_alg = max(v["n"] for v in T.values())
    pos = [(g, k, v) for (g, k), v in T.items() if v["lift"] > 0]

    # The headline sentence is WRITTEN FROM THE COUNT. If a win ever appears, the caption reports it by
    # itself -- nobody has to remember to come back and edit a hardcoded claim.
    if not wins:
        head = (r"\textbf{Repair improves topology nowhere: not on one graph, at one $k$, by one algorithm "
                r"--- not even the best one.}")
    else:
        head = (r"\textbf{Topology improves in $%d$ of the $%d$ cells} (%s)."
                % (len(wins), len(T), ", ".join(f"{g} $k{{=}}{k}$" for g, k in wins)))

    # The pbmc3k tautology is a disclosure the table would be DISHONEST without -- it stays, but terse.
    if pos:
        g, k, v = pos[0]
        caveat = (r" The one positive lift in the study (%s, \code{pbmc3k} at $k=%d$) is an artefact: that "
                  r"graph \emph{is} the $15$-nearest-neighbour graph of its own truth, so recovery at "
                  r"$k \le 15$ is $%.3f$ \emph{by construction}. Read it at $k=20$. "
                  % (sci(v["lift"]), k, v["obs"]))
    else:
        caveat = " "

    gwin = list(gw.items())
    geom = (r"\textbf{Geometry improves on $%d$ graph%s of $%d$}%s."
            % (len(gwin), "" if len(gwin) == 1 else "s", len(G),
               (r" --- \code{%s}, by $%+.1f\%%$" % (gwin[0][0].replace("_", r"\_"),
                                                    100 * gwin[0][1]["gain"]))
               if len(gwin) == 1 else ""))

    # SHORT -- see the note in corruption_table.py. Why best-and-not-median is an argument for the section
    # prose; the caption states only that it IS the best, which is what the reader needs to read the row.
    cap = (r"\caption{\textbf{The hope, answered.} The five real graphs carrying an external ground truth, "
           r"with no planted corruption --- the data as they actually arrive. Topology is $k$-NN Jaccard "
           r"against the truth ($\uparrow$), reported as the \textbf{lift} of the \emph{best} algorithm "
           r"over the observed graph; geometry is the Procrustes disparity of an MDS embedding "
           r"($\downarrow$), again the best. Best, not median: the claim is negative, so repair is handed an "
           r"oracle no practitioner could have (up to $%d$ algorithms per cell, chosen \emph{with} the "
           r"truth) and still loses. %s%s%s A win requires a $>\!%.0f\%%$ relative improvement. "
           r"\DOMR{} is the control, not a competitor. On \code{ripe\_atlas} only $5$ algorithms return "
           r"within the cap; on \code{nmr\_atom} the truth covers $343$ of the $430$ nodes.}"
           % (n_alg, head, caveat, geom, 100 * TOL))

    out = [r"% GENERATED by experiments/hope_table.py -- DO NOT EDIT, DO NOT TRANSCRIBE. Regenerate:",
           r"%   sage -python experiments/hope_table.py --texdir '<paper>/tables'",
           r"% Sources: analysis/summary_pure_real.csv, summary_mds_sweep.csv, summary_mds_rna.csv",
           r"\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}", cap, r"\label{tab:hope}",
           r"\begin{tabular}{@{}ll" + "rr" * len(KS) + r"@{\quad}rrlr@{}}", r"\toprule",
           r" & & " + " & ".join(r"\multicolumn{2}{c}{$k=%d$}" % k for k in KS)
           + r" & \multicolumn{4}{c}{geometry: Procrustes disparity ($\downarrow$)} \\",
           "".join(r"\cmidrule(lr){%d-%d}" % (3 + 2 * i, 4 + 2 * i) for i in range(len(KS)))
           + r"\cmidrule(l){%d-%d}" % (3 + 2 * len(KS), 6 + 2 * len(KS)),
           r"graph & what it is & " + " & ".join([r"obs. & best lift"] * len(KS))
           + r" & obs. & best & by & gain \\",
           r"\midrule"]

    for g, lab, what in GRAPHS:
        cells = [lab, what]
        for k in KS:
            v = T[(g, k)]
            cells += [r"$%.3f$" % v["obs"], r"$%+.4f$" % v["lift"]]
        gv = G[g]
        win = gv["gain"] > TOL
        cells += [r"$%.4f$" % gv["obs"], r"$%.4f$" % gv["best"], a(gv["algo"]),
                  (r"$\mathbf{%+.1f\%%}$" % (100 * gv["gain"])) if win
                  else r"$%+.1f\%%$" % (100 * gv["gain"])]
        out.append(" & ".join(cells) + r" \\")

    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
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
    ap = argparse.ArgumentParser(description="Emit tab:hope (does repair buy anything on real data?), gated.")
    ap.add_argument("--texdir", required=True)
    args = ap.parse_args()
    texdir = paper_dir(args.texdir)

    T, ctrl = topology()
    G = geometry()
    fails = gate(T, ctrl, G)
    if fails:
        raise SystemExit(f"\n*** GATE FAILED: {len(fails)} invariant(s) violated ***\n  " + "\n  ".join(fails)
                         + "\n\nNOT writing. The previous tab_hope.tex is left on disk.")

    tex = emit(T, G)
    if re.search(r"(?i)(?<![a-z])(nan|inf)(?![a-z])", tex):
        raise SystemExit("*** GATE FAILED: rendered LaTeX contains nan/inf. NOT writing. ***")
    print("  [PASS] G6  rendered LaTeX carries no nan/inf")

    path = os.path.join(texdir, "tab_hope.tex")
    with open(path, "w") as fh:
        fh.write(tex + "\n")
    print("\n" + tex + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
