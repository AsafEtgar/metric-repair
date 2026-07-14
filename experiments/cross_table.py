"""Emit tab:cross -- the SET x CORRECTION table, the one the paper turns on.

WHAT THE TABLE IS. A repair is two decisions: WHICH edges (the cover S) and WHAT weights on them (w').
The literature poses only the first and closes the second silently with one rule -- `restore` (build_F in
the code): reweight each cover edge to its shortest detour in G \\ S. This table varies BOTH, independently,
on a planted instance where the corrupted set B and the true weights w0 are both known:

    SET        x  CORRECTION     what it isolates
    -------------------------------------------------------------------------
    oracle B      oracle w0      the ceiling -- this IS the clean graph
    oracle B      restore        the RULE's own loss, on a perfect cover
    algo S        restore        what every implementation does today
    algo S        oracle w0      the SET's own loss, perfectly corrected

Read the oracle row across and the true-weights column down and the argument is finished: the perfect
cover under the canonical rule returns the observed graph unchanged, while covers of 40% and 20% precision
under the true weights reach the ceiling. Same information. Two rules. Everything, or nothing.

THIS SCRIPT DOES NOT RUN THE EXPERIMENT. `why_repair_fails.py` does, and writes the CSV. This reads that
CSV, pivots it, checks it, and renders it. Splitting the two means the table can be re-rendered without a
re-run -- and, more to the point, means this file is free to be paranoid about data it did not produce.

  input     analysis/why_repair_fails.csv    LONG: two rows per algo, one per correction rule
  output    <texdir>/tab_cross.tex           the whole table*, caption and \\label included

  usage     sage -python experiments/cross_table.py --texdir "<paper>/tables"

!! THE `corr` COLUMN IS A LANDMINE. `d.corr` is the pandas DataFrame METHOD (it computes correlations),
   NOT the column. Attribute access returns the bound method, every comparison against it is False, and the
   filter silently matches ZERO rows -- no exception, no warning. Always `d["corr"]`. This bit me while
   writing this file; it is the exact class of bug the gate exists for, and G3 below would have caught it
   (a pivot that yields no covers at all).

WHY THE PIVOT IS THE DANGEROUS PART. The CSV is long: each algorithm appears twice, distinguished only by
`corr`. Pivot them wrong -- match algo A's restore row against algo B's oracle row -- and every cell still
looks plausible. Nothing crashes. The table just quietly tells a different story. G4 is the check for that.

THE GATE HARDCODES NO EXPECTED VALUE. It never asserts `disp == 0.0192`; a check written against a number
someone typed cannot catch that number being wrong. It works two other ways:

  * PROVENANCE (G1) -- it REBUILDS the planted instance from its seeded spec in RGG_SPECS and requires the
    rebuilt |B| to equal the |S| on the CSV's oracle row. Two independent artifacts -- the code that
    generated the experiment, and the experiment's own output -- must agree. A stale CSV, a changed seed, a
    different spec: each surfaces here rather than as a quietly-wrong caption.

  * IDENTITIES THEORY GUARANTEES (G6, G7) -- a cover containing NONE of the corrupted edges cannot be
    improved by the true weights, because its edges already carry them; so its true-weights disparity must
    equal the observed disparity EXACTLY. That checks the whole with_true_weights() path, and it can fail.

Gate runs BEFORE the write. A failing gate leaves the previous .tex on disk: a paper should keep its old
numbers rather than gain untrustworthy new ones.
"""
import argparse
import os
import re
import sys

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_models import break_metric_graph, random_geometric_metric_graph   # noqa: E402
from mds_recovery import RGG_SPECS, seed_all                                 # noqa: E402

CSV = "analysis/why_repair_fails.csv"
GRAPH = "rgg_inflate"

OBSERVED = "observed (no repair)"
CEILING = "ORACLE set + ORACLE weights (= clean)"
ORACLE_RESTORE = "ORACLE set + build_F rule"
RESTORE, ORACLE = "build_F", "oracle"       # the two values of the `corr` column

# The identities below are EXACT in exact arithmetic. This tolerance is for the accumulated error of an
# all-pairs shortest-path pass plus an MDS -- not a licence for "roughly equal".
TOL = 1e-9
# "Reaches the ceiling" as the printed table can express it: equal to four decimal places.
CEIL_TOL = 5e-5

TEX = {"gmr_bestofk": r"\code{gmr\_bestofk}", "iomr_bestofk": r"\code{iomr\_bestofk}",
       "gmr_thr_naive": r"\code{gmr\_thr}", "iomr_thr_naive": r"\code{iomr\_thr}",
       "gmr_rand": r"\code{gmr\_rand}", "iomr_regiongrow": r"\code{iomr\_rgrow}"}


def a(name):
    return TEX.get(str(name), r"\code{%s}" % str(name).replace("_", r"\_"))


def rebuild(graph):
    """Regenerate the planted instance from its seeded spec; return n, m, |B| and the corruption knobs.

    This is the PROVENANCE reference, and it reproduces why_repair_fails.run() exactly -- same seed, same
    radius, same giant-component relabelling. Its |B| is therefore what the CSV's oracle row must say. If
    they disagree, the CSV came from a different experiment than the caption is about to describe, and no
    amount of care in the rendering can repair that."""
    spec = {s[0]: s for s in RGG_SPECS}.get(graph)
    if spec is None:
        raise SystemExit(f"FATAL: '{graph}' is not in RGG_SPECS. Known: {[s[0] for s in RGG_SPECS]}")
    _, n, deg, direction, frac, mag, seed = spec
    seed_all(seed)
    radius = float(np.sqrt(deg / (np.pi * max(n - 1, 1))))
    T = random_geometric_metric_graph(n, mode="radius", radius=radius)
    T = nx.convert_node_labels_to_integers(T.subgraph(max(nx.connected_components(T), key=len)).copy())
    C, corrupted = break_metric_graph(T, frac_q=frac, direction=direction, magnitude=mag)
    comp = max(nx.connected_components(C), key=len)
    mp = {u: i for i, u in enumerate(sorted(comp))}
    C = nx.relabel_nodes(C.subgraph(comp).copy(), mp)
    B = {(min(mp[u], mp[v]), max(mp[u], mp[v])) for u, v in corrupted if u in mp and v in mp}
    return dict(n=C.number_of_nodes(), m=C.number_of_edges(), B=len(B),
                frac=frac, mag=mag, direction=direction, seed=seed)


def load(path, graph):
    if not os.path.exists(path):
        raise SystemExit(f"FATAL: {path} missing. Run: sage -python experiments/why_repair_fails.py")
    d = pd.read_csv(path)
    for c in ("size", "prec", "rec", "disp", "knn"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d.graph.eq(graph)]
    if d.empty:
        raise SystemExit(f"FATAL: no rows for graph '{graph}' in {path}.")
    return d


def refs(d):
    """The three anchor rows. Without them the table has no scale to be read against."""
    out = {}
    for key, cond in (("observed", OBSERVED), ("ceiling", CEILING), ("oracle_restore", ORACLE_RESTORE)):
        row = d[d.cond.eq(cond)]
        if len(row) != 1:
            raise SystemExit(f"FATAL: expected exactly 1 '{cond}' row, found {len(row)}. The reference rows "
                             "are the table's scale; refusing to render without them.")
        out[key] = float(row.iloc[0].disp)
    out["oracle_size"] = int(d[d.cond.eq(CEILING)].iloc[0]["size"])
    out["gain"] = out["observed"] - out["ceiling"]      # the whole recoverable gap: the denominator
    return out


def pivot(d, R):
    """Long -> wide: the two correction rules become two columns, keyed on the algorithm.

    G4 lives here. Every algo must contribute exactly ONE restore row and ONE oracle row, and the two must
    agree on |S|, precision and recall -- those are properties of the SET, and both rows describe the SAME
    cover. If they disagree, the join matched the wrong rows, and that is the failure that renders a
    perfectly plausible, perfectly wrong table with nothing crashing."""
    algos = d[~d.setk.isin(["--", "oracle"])]
    rows, problems = [], []
    for algo, G in algos.groupby("setk", sort=False):
        r = G[G["corr"].eq(RESTORE)]                    # NOT G.corr -- see the module docstring
        o = G[G["corr"].eq(ORACLE)]
        if len(r) != 1 or len(o) != 1:
            problems.append(f"{algo}: {len(r)} restore row(s), {len(o)} oracle row(s) -- expected 1 and 1")
            continue
        r, o = r.iloc[0], o.iloc[0]
        for col in ("size", "prec", "rec"):
            if not np.isclose(float(r[col]), float(o[col]), rtol=0, atol=TOL):
                problems.append(f"{algo}: '{col}' differs across its two rows ({r[col]} vs {o[col]}) "
                                "-- they are not the same cover, so the pivot matched the wrong rows")
        rows.append(dict(algo=algo, size=int(r["size"]), prec=float(r.prec), rec=float(r.rec),
                         d_restore=float(r.disp), d_oracle=float(o.disp)))
    w = pd.DataFrame(rows)
    if not w.empty:
        # The fraction of the RECOVERABLE gap that this cover's true-weighted repair actually captures.
        # This is what makes the last column readable: 0.0159 means nothing; "20.7% of the gap" means
        # everything. Negative = the repair moved the embedding AWAY from the truth.
        w["captured"] = (R["observed"] - w.d_oracle) / R["gain"]
    return w, problems


# ----------------------------------------------------------------------------
# The gate
# ----------------------------------------------------------------------------
def gate(w, R, truth):
    fails = []

    def chk(ok, name, observed):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<50} {observed}")
        if not ok:
            fails.append(name)

    print("GATE -- provenance cross-checked against the harness; identities against theory")

    # G1  PROVENANCE. The one check that catches a STALE CSV -- the failure under which every other number
    #     is meaningless while every other check still passes.
    chk(truth["B"] == R["oracle_size"], "G1  rebuilt |B| == the CSV's oracle-row |S|",
        f"rebuild {truth['B']} vs csv {R['oracle_size']}")

    # G2  the gap must be positive, or "fraction of the recoverable gap" is a division by ~zero and every
    #     percentage in the caption is noise.
    chk(R["gain"] > TOL, "G2  the recoverable gap is positive",
        f"observed {R['observed']:.6f} - ceiling {R['ceiling']:.6f} = {R['gain']:.6f}")

    # G3  the pivot produced covers at all. Anti-vacuity: this is what fires if the `corr` landmine is ever
    #     stepped on again (attribute access -> bound method -> every filter matches nothing -> empty table).
    chk(not w.empty, "G3  the pivot produced covers", f"{len(w)} covers")

    # G6  THE NO-OP IDENTITY. A cover holding none of the corrupted edges cannot be helped by the true
    #     weights -- its edges already carry them -- so its true-weights disparity must EQUAL observed,
    #     exactly. This exercises the entire with_true_weights() path and fails if it ever touches an edge
    #     it should not.
    zero = w[w.rec.eq(0.0)]
    off = zero[~np.isclose(zero.d_oracle, R["observed"], rtol=0, atol=TOL)]
    chk(len(off) == 0, "G6  zero-overlap covers + true weights == observed",
        f"{len(zero)} such covers, {len(off)} violate it" + ("" if off.empty else f": {list(off.algo)}"))

    # G7  THE CEILING IS A CEILING. The clean graph is the target; nothing may embed better than it. If
    #     something does, the word "ceiling" in the caption is false and the paper must change. Fatal on
    #     purpose.
    best = float(min(w.d_restore.min(), w.d_oracle.min(), R["observed"]))
    chk(R["ceiling"] <= best + TOL, "G7  nothing beats the clean graph",
        f"ceiling {R['ceiling']:.6f} vs best cell {best:.6f}")

    # G8  ranges. Cheap, and every column rests on them.
    bad = w[(w.prec < 0) | (w.prec > 1) | (w.rec < 0) | (w.rec > 1) | (w["size"] <= 0)]
    chk(bad.empty, "G8  precision/recall in [0,1], |S| > 0",
        "all in range" if bad.empty else f"OUT OF RANGE: {list(bad.algo)}")

    # G9  THE CLAIM ITSELF -- surfaced, not assumed. The paper says the oracle set under the canonical rule
    #     returns the observed graph UNCHANGED. Both numbers are read from the CSV: this compares two
    #     measurements, it does not assert a constant. Fatal, because if the identity ever stops holding the
    #     finding has moved and Section 5 must be rewritten -- the table must NOT silently regenerate
    #     carrying a different story.
    gap = abs(R["oracle_restore"] - R["observed"])
    chk(gap <= TOL, "G9  oracle set + restore == observed (the paper's claim)",
        f"differ by {gap:.2e}")

    # ---- REPORT, not a gate: the equal-recall divergence.
    # The draft caption said "recall is a cliff, not a dial: 0.769 -> worse than nothing". Three covers sit
    # at recall 0.769 and they land 26 points of the gap apart. Quoting the harmful one alone is
    # cherry-picking. Print every tie loudly so the claim can never quietly drift back in.
    print("\n  REPORT -- covers sharing a recall, and what they actually capture:")
    for rec, G in w[w.rec > 0].groupby("rec"):
        if len(G) > 1:
            spread = 100 * (G.captured.max() - G.captured.min())
            print(f"      recall {rec:.3f}: " + ", ".join(f"{r.algo} {100*r.captured:+.1f}%"
                                                          for _, r in G.iterrows())
                  + f"   <-- {spread:.0f} points apart")
    print("      => recall does not determine the outcome. WHICH corrupted edges, not how many.")
    return fails


# ----------------------------------------------------------------------------
# Emit
# ----------------------------------------------------------------------------
def _n(x):
    return f"{int(x):,}".replace(",", "{,}")


def d4(v):
    return f"${v:.4f}$"


def emit(w, R, truth):
    # Ordered by RECALL, descending. Not cosmetic: recall is what the true-weights column actually measures
    # (restoring an uncorrupted edge is a no-op), so sorting by it is what makes the cliff visible.
    cov = w.sort_values(["rec", "prec"], ascending=False)
    keep = cov[cov.rec > 0]
    zero = cov[cov.rec.eq(0.0)]        # DERIVED, never a hand-picked list: the covers that found nothing

    reach = keep[np.isclose(keep.d_oracle, R["ceiling"], rtol=0, atol=CEIL_TOL)]
    miss = keep[~keep.index.isin(reach.index)]

    # Every number in the caption below is computed from the frame. Nothing is typed.
    #
    # The two precisions the caption quotes must come from the NON-TRIVIAL covers that reach the ceiling.
    # `domr` and the oracle row are at precision 1.000 -- they are the corrupted set itself, and a perfect
    # cover reaching the ceiling proves nothing. The claim is that *sloppy* covers reach it too, so quote
    # the sloppy ones. (My first draft quoted 100.0% here, which was the oracle sneaking into its own
    # evidence.)
    naive = reach[reach.prec < 1.0]
    p_hi, p_lo = 100 * float(naive.prec.max()), 100 * float(naive.prec.min())
    c_hi = 100 * float(naive.loc[naive.prec.idxmax()].captured)
    c_lo = 100 * float(naive.loc[naive.prec.idxmin()].captured)

    cliff_hi = float(reach.rec.min())                  # lowest recall that still reaches the ceiling
    cliff_lo = float(miss.rec.max())                   # highest recall that does not
    cliff_lo_cap = 100 * float(miss.loc[miss.rec.idxmax()].captured)
    band_lo, band_hi = 100 * float(miss.captured.min()), 100 * float(miss.captured.max())
    ties = [(r, G) for r, G in keep.groupby("rec") if len(G) > 1 and G.captured.max() - G.captured.min() > 0.01]
    tie_rec, tie_G = (ties[0] if ties else (float("nan"), None))
    tie_n = len(tie_G) if tie_G is not None else 0
    tie_spread = 100 * float(tie_G.captured.max() - tie_G.captured.min()) if tie_G is not None else 0.0

    # DOMR's cover IS the heavy set, and under inflation the heavy set IS the corrupted set -- so its row
    # and the oracle row coincide. That is a RESULT (Section 4), not a duplication, and the caption says so.
    dm = keep[keep.algo.eq("domr")]
    domr_is_oracle = (len(dm) == 1 and int(dm.iloc[0]["size"]) == R["oracle_size"]
                      and float(dm.iloc[0].prec) == 1.0 and float(dm.iloc[0].rec) == 1.0)

    gap = abs(R["oracle_restore"] - R["observed"])
    agree = ("to the last bit --- the two disparities are bit-identical" if gap == 0.0
             else r"to $%.0e$" % gap)

    # SHORT -- see the note in corruption_table.py. The recall-cliff argument, the precision-is-not-what-is-
    # measured point and the no-op identity are the SECTION's job, not the float's. Numbers stay derived.
    cap = (
        r"\caption{\textbf{Which choice kills recovery --- the set, or the correction?} A planted random "
        r"geometric graph: $n = %s$, $m = %s$, $%d\%%$ of edges inflated $%g\times$, planted corrupted set "
        r"$|B| = %d$. Each row is a cover $S$; the two disparity columns reweight \emph{that same} $S$ two "
        r"ways (lower is better), and \emph{captured} is the share of the recoverable gap --- observed minus "
        r"ceiling --- that the true weights close. Rows are ordered by recall. The perfect cover under the "
        r"canonical rule returns the observed disparity unchanged, %s, while covers of $%.1f\%%$ and "
        r"$%.1f\%%$ precision under the true weights close $%.1f\%%$ and $%.1f\%%$ of the gap. %s}"
        % (_n(truth["n"]), _n(truth["m"]), int(round(100 * truth["frac"])), truth["mag"], truth["B"],
           agree, p_hi, p_lo, c_hi, c_lo,
           (r"\code{domr}'s cover \emph{is} that oracle set --- the heavy set, computable with no ground "
            r"truth --- which is why the two rows coincide." if domr_is_oracle else "")))

    out = [r"% GENERATED by experiments/cross_table.py -- DO NOT EDIT, DO NOT TRANSCRIBE. Regenerate:",
           r"%   sage -python experiments/cross_table.py --texdir '<paper>/tables'",
           r"% Source: analysis/why_repair_fails.csv, gated against the seeded RGG_SPECS instance.",
           r"\begin{table*}[t]\centering\small", cap, r"\label{tab:cross}",
           r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
           r" & & & & \multicolumn{2}{c}{disparity after repair} & \\",
           r"\cmidrule(lr){5-6}",
           r"cover $S$ & $|S|$ & precision & recall & \restore{} & true weights & captured \\",
           r"\midrule",
           r"\multicolumn{4}{@{}l}{\emph{observed --- no repair}} & \multicolumn{2}{c}{%s} & $0\%%$ \\"
           % d4(R["observed"]),
           r"\multicolumn{4}{@{}l}{\emph{the clean graph --- the ceiling}} & "
           r"\multicolumn{2}{c}{$\mathbf{%.4f}$} & $\mathbf{100\%%}$ \\" % R["ceiling"],
           r"\midrule",
           r"oracle $B$ ($=H$) & $%d$ & $1.000$ & $1.000$ & %s & $\mathbf{%.4f}$ & $\mathbf{100\%%}$ \\"
           % (R["oracle_size"], d4(R["observed"]), R["ceiling"])]

    for _, r in keep.iterrows():
        hits = bool(np.isclose(r.d_oracle, R["ceiling"], rtol=0, atol=CEIL_TOL))
        # ALWAYS one decimal, bold or not. `pivot` captures 99.9%, not 100%: it misses one corrupted edge
        # (recall 0.995). A "%.0f" here rounds that to 100 and reintroduces, from a format string, exactly
        # the overclaim the abstract makes when it says a 19.6%-precision cover "recovers the clean graph
        # exactly". Bold marks reaching the ceiling to four decimals; it does not mean equality.
        out.append(r"%s & $%d$ & $%.3f$ & $%.3f$ & %s & %s & %s \\"
                   % (a(r.algo), r["size"], r.prec, r.rec, d4(r.d_restore),
                      (r"$\mathbf{%.4f}$" % r.d_oracle) if hits else d4(r.d_oracle),
                      (r"$\mathbf{%.1f\%%}$" % (100 * r.captured)) if hits
                      else r"$%.1f\%%$" % (100 * r.captured)))

    if len(zero):
        out.append(r"\midrule")
        out.append(r"every zero-overlap cover & $%d$--$%d$ & $0.000$ & $0.000$ & $%.3f$--$%.3f$ & %s & "
                   r"$0.0\%%$ \\"
                   % (zero["size"].min(), zero["size"].max(), zero.d_restore.min(), zero.d_restore.max(),
                      d4(R["observed"])))

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
    ap = argparse.ArgumentParser(description="Emit tab:cross (the SET x CORRECTION table), gated.")
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--graph", default=GRAPH, help="which planted instance (must be in RGG_SPECS)")
    ap.add_argument("--texdir", required=True, help="the paper's tables/ directory")
    args = ap.parse_args()

    texdir = paper_dir(args.texdir)
    d = load(args.csv, args.graph)
    truth = rebuild(args.graph)
    R = refs(d)
    w, problems = pivot(d, R)

    if problems:                          # G4 -- raised before the gate even prints
        raise SystemExit("*** GATE FAILED (G4): the long->wide pivot is not sound ***\n  "
                         + "\n  ".join(problems)
                         + "\n\nNOT writing. Each algorithm must contribute exactly one restore row and one "
                           "oracle row, and the two must describe the SAME cover.")

    fails = gate(w, R, truth)
    if fails:
        raise SystemExit(f"\n*** GATE FAILED: {len(fails)} invariant(s) violated ***\n  " + "\n  ".join(fails)
                         + "\n\nNOT writing. The previous tab_cross.tex is left on disk: a paper should keep "
                           "its old numbers rather than gain untrustworthy new ones.")

    tex = emit(w, R, truth)

    # G10  the final check, on the string that actually reaches the paper.
    if re.search(r"(?i)(?<![a-z])(nan|inf)(?![a-z])", tex):
        raise SystemExit("*** GATE FAILED (G10): rendered LaTeX contains nan/inf. NOT writing. ***")
    print("  [PASS] G10 rendered LaTeX carries no nan/inf")

    path = os.path.join(texdir, "tab_cross.tex")
    with open(path, "w") as fh:
        fh.write(tex + "\n")
    print("\n" + tex + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
