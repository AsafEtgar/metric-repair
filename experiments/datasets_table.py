"""Emit tab:datasets -- what the real graphs ARE, and whether anything downstream actually needs a metric.

A ONE-COLUMN table (\\begin{table}, not table*), for Section 7, "The other datasets, honestly".

THE POINT OF THE TABLE. A dataset that exercises the algorithms is not the same as a dataset that motivates
them. This table separates the two, honestly, and it makes three admissions that a less careful paper would
bury:

  1. On most of these graphs, NO standard downstream tool needs the triangle inequality. Leiden, UMAP,
     modularity, diffusion gradients, matrix-factorisation latency prediction -- none of them do. Metric
     repair has nothing to offer there, and we say so.

  2. WE MANUFACTURED MOST OF THE NON-METRICITY ON THE CONNECTOME AND fMRI GRAPHS. Those are similarity
     graphs; turning a similarity into a distance requires a conversion, and the conversion decides the
     answer. The table shows the RANGE of |H|/m across the conversions we tried, on the SAME graph. On
     flycns_male it runs from 0.6% to 83.1% -- two orders of magnitude, from one editorial choice. A
     violated triangle inequality that appears or vanishes depending on how you divide is not a property
     of the data.

  3. The road network is metric BY CONSTRUCTION (|H| = 0). It is a planted base, not an application.

  input     analysis/summary_real.csv            n, m, |H|, non-metric fraction, per graph
            analysis/summary_pure_real.csv       which graphs carry an external ground truth
            analysis/summary_recovery.csv        (the planted road base carries one too)
  output    <texdir>/tab_datasets.tex

  usage     sage -python experiments/datasets_table.py --texdir "<paper>/tables"

!! ONE COLUMN OF THIS TABLE IS NOT A MEASUREMENT, AND THE CODE SAYS SO.

   n, m, |H|, |H|/m and "has an external ground truth" are all read or derived from the CSVs. The verdict
   -- does a standard tool in that field actually need the triangle inequality? -- is an EDITORIAL
   JUDGEMENT about the literature of six different fields. It cannot be computed, and pretending otherwise
   by burying it in a script would be worse than admitting it. It lives in VERDICT below, it is curated by
   hand, and it is the one thing here a referee must argue with us about rather than check.

   What the gate CAN do -- and does -- is keep the judgement honest about its scope: every graph in the data
   must have a verdict (G1), and every verdict must name a graph that exists (G2). So the curated list can
   never silently drift out of step with the campaign, which is the failure mode that turns an honest
   editorial column into a stale one.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

REAL = "analysis/summary_real.csv"
PURE = "analysis/summary_pure_real.csv"
REC = "analysis/summary_recovery.csv"

# ---------------------------------------------------------------------------
# THE CURATED COLUMN. Not measured. Not derivable. Argued.
#
#   base graph          -> (short label, what it is, needs a metric?, why)
#
# "needs a metric?" asks: does a STANDARD downstream tool in that field require the triangle inequality?
# Not "could one construct a pipeline that does" -- one always could. Does the field, in practice, run one.
# ---------------------------------------------------------------------------
VERDICT = {
    "nmr_1d3z_atom": (r"\code{nmr\_atom}", "protein, NOE bounds", "yes",
                      "distance geometry: the embedding IS the science"),
    "nmr_1d3z_residue": (r"\code{nmr\_res}", "protein, residue-coarsened", "yes",
                         "same, and the coarsening manufactures the violations"),
    "dimacs_ny_d": (r"\code{dimacs\_ny\_d}", "road net, distance", "null",
                    "metric by construction: a planted base, not an application"),
    "dimacs_ny_t": (r"\code{dimacs\_ny\_t}", "road net, travel time", "null",
                    "not real travel data: great-circle over a synthetic speed table"),
    "ripe_atlas": (r"\code{ripe\_atlas}", "internet latency", "no",
                   "the field has known of these violations for 25 years and tolerates, "
                   "discards, or globally shifts them --- it never sparsely reweights"),
    "pbmc3k_cosine_knn": (r"\code{pbmc3k}", "single-cell $k$-NN", "no",
                          "$k$-NN + geodesic + MDS is Isomap, which the field does not run"),
    "cassiopeia_barcode_knn": (r"\code{cassiopeia}", "lineage barcodes", "no",
                               "phylogeny needs \\emph{additivity}, strictly stronger than the "
                               "triangle inequality"),
    "bct_coactivation": (r"\code{bct\_coact}", "connectome (coactivation)", "no",
                         "Leiden, UMAP and modularity need no metric"),
    "flycns_male": (r"\code{flycns}", "connectome (fly CNS)", "no",
                    "same"),
    "fish1_ten": (r"\code{fish1\_ten}", "fMRI", "no",
                  "diffusion gradients need no metric"),
}
CONV = ("_lin", "_log", "_raw")     # the similarity->distance conversion variants


def base_of(g):
    """Strip the conversion suffix. bct_coactivation_lin -> bct_coactivation."""
    for s in CONV:
        if g.endswith(s):
            return g[: -len(s)]
    return g


def load(path, what):
    if not os.path.exists(path):
        raise SystemExit(f"FATAL: {path} missing ({what}).")
    return pd.read_csv(path)


def collect():
    d = load(REAL, "graph characterisation")
    need = ["graph", "n", "m", "H", "nonmetric_frac"]
    miss = [c for c in need if c not in d.columns]
    if miss:
        raise SystemExit(f"FATAL: {REAL} lacks {miss}.")
    d = d[need].drop_duplicates(subset=["graph"])
    for c in ("n", "m", "H", "nonmetric_frac"):
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # which graphs carry an EXTERNAL ground truth -- derived, not listed. A graph has one iff it appears in
    # an analysis that scores against a truth: the native real sweep, or the planted-base recovery array.
    truth = set(load(PURE, "topology").graph.unique())
    truth |= {base_of(re.sub(r"_(inflate|deflate|mixed)$", "", g))
              for g in load(REC, "planted recovery").graph.unique()}

    d["base"] = d.graph.map(base_of)
    recs = []
    for b, G in d.groupby("base", sort=False):
        if b not in VERDICT:
            raise SystemExit(f"FATAL (G1): graph '{b}' is in the data but has no verdict. The editorial "
                             "column must cover every graph the campaign ran, or the table is silently "
                             "incomplete. Add it to VERDICT.")
        label, what, need_metric, why = VERDICT[b]
        fr = G.nonmetric_frac.dropna()
        recs.append(dict(
            base=b, label=label, what=what, need=need_metric, why=why,
            n=int(G.n.iloc[0]), m=int(G.m.iloc[0]),
            frac_lo=float(fr.min()), frac_hi=float(fr.max()),
            n_conv=len(G),                        # how many similarity->distance conversions we ran
            H_lo=int(G.H.min()), H_hi=int(G.H.max()),
            truth=(b in truth) or any(g in truth for g in G.graph),
        ))
    return recs, d


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def gate(recs, d):
    fails = []

    def chk(ok, name, obs):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<50} {obs}")
        if not ok:
            fails.append(name)

    print("GATE -- the measured columns are checked; the EDITORIAL column is checked for SCOPE, not truth")

    # G1 is raised inside collect(): a graph in the data with no verdict stops the run.
    chk(True, "G1  every graph in the data has a verdict", f"{len(recs)} base graphs, all covered")

    # G2  the other direction. A verdict naming a graph that no longer exists is a stale opinion about a
    #     dataset we do not run -- and it would sit in the paper looking like a finding.
    have = {r["base"] for r in recs}
    orphan = set(VERDICT) - have
    chk(not orphan, "G2  no verdict names a graph absent from the data",
        "none" if not orphan else f"ORPHANED: {sorted(orphan)}")

    # G3  the non-metric fraction must BE |H|/m. If the stored column ever drifts from the stored H and m,
    #     one of the three is stale -- and it is the table's headline column that would be wrong.
    #
    #     The CSV stores the fraction ROUNDED TO SIX DECIMALS, so a raw equality check fails by up to 5e-7.
    #     The fix is NOT to loosen the tolerance to 1e-6: a genuinely stale value could hide inside that
    #     window. We check the stronger thing -- that the stored value is EXACTLY round(H/m, 6). A stale
    #     column would not survive that unless it happened to round identically, which is the same as being
    #     correct. (This gate caught the rounding on its first run, and refused to write. Good.)
    dd = d.dropna(subset=["nonmetric_frac", "H", "m"])
    err = float((dd.nonmetric_frac - (dd.H / dd.m).round(6)).abs().max())
    chk(err < 1e-12, "G3  nonmetric_frac == round(|H|/m, 6), exactly",
        f"max discrepancy {err:.1e} (the column is stored to 6 dp)")

    # G4  the conversion families must ACTUALLY have several conversions, or quoting a RANGE of |H|/m is a
    #     fiction. Anti-vacuity: if the _lin/_log/_raw variants ever vanished, the range would silently
    #     collapse to a point and the "we manufactured it" claim would evaporate without a word.
    fam = [r for r in recs if r["n_conv"] > 1]
    chk(len(fam) > 0, "G4  the conversion families have several conversions",
        ", ".join(f"{r['base']}: {r['n_conv']}" for r in fam))

    # G5  ranges
    bad = [r["base"] for r in recs if not (0 <= r["frac_lo"] <= r["frac_hi"] <= 1) or r["n"] <= 0 or r["m"] <= 0]
    chk(not bad, "G5  |H|/m in [0,1], n and m positive", "all in range" if not bad else str(bad))

    # --- COUNTED, not asserted: how many graphs carry a truth, and how many need a metric at all.
    nt = sum(1 for r in recs if r["truth"])
    ny = sum(1 for r in recs if r["need"] == "yes")
    print(f"\n  COUNTED -- {nt} of {len(recs)} base graphs carry an external ground truth")
    print(f"  COUNTED -- {ny} of {len(recs)} have a standard downstream tool that needs the metric")
    print("  REPORTED -- the conversion swing (the violations we manufactured ourselves):")
    for r in sorted(fam, key=lambda r: r["frac_hi"] - r["frac_lo"], reverse=True):
        print(f"      {r['base']:<22} |H|/m ranges {100*r['frac_lo']:5.1f}% -> {100*r['frac_hi']:5.1f}% "
              f"across {r['n_conv']} conversions  ({r['H_lo']} -> {r['H_hi']} heavy edges, same graph)")
    return fails


# ---------------------------------------------------------------------------
# Emit -- ONE COLUMN
# ---------------------------------------------------------------------------
def _n(x):
    return f"{int(x):,}".replace(",", "{,}")


def frac(r):
    """A single fraction, or the RANGE across the conversions -- which is the whole point on those rows."""
    if r["n_conv"] > 1 and abs(r["frac_hi"] - r["frac_lo"]) > 1e-9:
        return r"$%.1f$--$%.1f$" % (100 * r["frac_lo"], 100 * r["frac_hi"])
    return r"$%.1f$" % (100 * r["frac_lo"])


NEED = {"yes": r"\textbf{yes}", "no": "no", "null": r"n/a"}

ORDER = ["nmr_1d3z_atom", "nmr_1d3z_residue", "dimacs_ny_d", "dimacs_ny_t",
         "ripe_atlas", "pbmc3k_cosine_knn", "cassiopeia_barcode_knn",
         "bct_coactivation", "flycns_male", "fish1_ten"]


def emit(recs):
    by = {r["base"]: r for r in recs}
    rows = [by[b] for b in ORDER if b in by] + [r for r in recs if r["base"] not in ORDER]

    nt = sum(1 for r in rows if r["truth"])
    ny = sum(1 for r in rows if r["need"] == "yes")
    fam = [r for r in rows if r["n_conv"] > 1 and r["frac_hi"] - r["frac_lo"] > 1e-9]
    swing = max(fam, key=lambda r: r["frac_hi"] - r["frac_lo"]) if fam else None

    # SHORT -- see the note in corruption_table.py. The conversion-swing argument and the dimacs_ny_d
    # planted-base point are the section's; the caption keeps only what is needed to read the columns, plus
    # the one thing the table would be dishonest without: that the last column is a judgement, not a
    # measurement.
    cap = (r"\caption{\textbf{A dataset that exercises the algorithms is not a dataset that motivates "
           r"them.} The real collection. $|H|/m$ is the non-metric fraction; $%d$ of the $%d$ graphs carry "
           r"an external ground truth, and only those can say whether a repair moves a graph \emph{toward} "
           r"anything. The last column asks whether a \emph{standard} downstream tool in that field needs "
           r"the triangle inequality at all: for $%d$ of the $%d$ the answer is no. \textbf{That column is "
           r"the one judgement in this paper we cannot compute} --- it is a claim about the practice of six "
           r"literatures, argued in the text, not a measurement. On the similarity graphs the non-metricity "
           r"is manufactured by the distance conversion: on \code{%s}, $|H|/m$ swings from $%.1f\%%$ to "
           r"$%.1f\%%$ with the choice alone.}"
           % (nt, len(rows), sum(1 for r in rows if r["need"] != "yes"), len(rows),
              swing["base"].replace("_", r"\_") if swing else "--",
              100 * swing["frac_lo"] if swing else 0, 100 * swing["frac_hi"] if swing else 0))

    out = [r"% GENERATED by experiments/datasets_table.py -- DO NOT EDIT, DO NOT TRANSCRIBE. Regenerate:",
           r"%   sage -python experiments/datasets_table.py --texdir '<paper>/tables'",
           r"% ONE-COLUMN (table, not table*). Source: analysis/summary_real.csv.",
           r"% The 'metric?' column is an EDITORIAL JUDGEMENT, curated in VERDICT -- not a measurement.",
           r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}",
           cap, r"\label{tab:datasets}",
           r"\begin{tabular}{@{}lrrrcc@{}}", r"\toprule",
           r"graph & $n$ & $m$ & $|H|/m$ (\%) & truth & metric? \\",
           r"\midrule"]

    for i, r in enumerate(rows):
        if r["n_conv"] > 1 and (i == 0 or rows[i - 1]["n_conv"] <= 1):
            out.append(r"\midrule")
        out.append(r"%s & $%s$ & $%s$ & %s & %s & %s \\"
                   % (r["label"], _n(r["n"]), _n(r["m"]), frac(r),
                      r"$\checkmark$" if r["truth"] else r"\code{--}", NEED[r["need"]]))

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
    ap = argparse.ArgumentParser(description="Emit tab:datasets (one column), gated.")
    ap.add_argument("--texdir", required=True)
    args = ap.parse_args()
    texdir = paper_dir(args.texdir)

    recs, d = collect()
    fails = gate(recs, d)
    if fails:
        raise SystemExit(f"\n*** GATE FAILED: {len(fails)} invariant(s) violated ***\n  " + "\n  ".join(fails)
                         + "\n\nNOT writing. The previous tab_datasets.tex is left on disk.")

    tex = emit(recs)
    if re.search(r"(?i)(?<![a-z])(nan|inf)(?![a-z])", tex):
        raise SystemExit("*** GATE FAILED: rendered LaTeX contains nan/inf. NOT writing. ***")
    print("  [PASS] G6  rendered LaTeX carries no nan/inf")

    path = os.path.join(texdir, "tab_datasets.tex")
    with open(path, "w") as fh:
        fh.write(tex + "\n")
    print("\n" + tex + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
