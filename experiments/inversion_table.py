"""Emit tab:invert -- the dense-vs-sparse inversion table -- and every number the SECTION 4 PROSE quotes.

WHY THIS SCRIPT EXISTS. This table was hand-computed, and a hand-computed table shipped a wrong number: the
draft printed `l1sep_iomr` sparse "returned" as 100.0% when it is 99.7% (1037/1040). Three rows are
status=="ok" but valid==0 -- l1_separation exiting at max_rounds without converging -- and the hand pass
filtered on `status` while the caption promised "returned AND verified". One row, in bold, in the paper.
So: nothing here is transcribed. The table AND the prose macros are generated, gated, and re-derived.

  outputs   analysis/tab_invert.tex          the whole table*, caption included
            analysis/inversion_macros.tex    \\newcommand per number the prose quotes
            analysis/summary_inversion.csv   tidy, one row per (family, algo)

  usage     sage -python experiments/inversion_table.py [--outdir analysis]

THE THREE TRAPS THIS FILE IS BUILT AROUND. Each has already produced a plausible, non-crashing, WRONG number
in this project:

  1. The sparse CSV is LONG-FORMAT (one row per (task, algo, knn_k)). 32,582 raw rows collapse to 16,640
     unique. Skip the dedup and gmr_bestofk reads as rho_H = 0.562 / 81.3% return instead of the correct
     0.332 / 62.6% -- and domr's "return rate" comes out at 215%, which is how you know.
  2. cpu == 0 is a KILLED-RUN SENTINEL, not a fast run. Filter on status=="ok" or the slowest algorithms
     report as the fastest and then vanish from every log-axis figure.
  3. valid == 0 with status == "ok" is a real state (the defect above). "Returned" means returned AND
     verified.

THE GATE FAILS CLOSED, AND HARDCODES NOTHING. It does not assert `ratio == 0.332`; a check written against a
number someone typed cannot catch that number being wrong. It imports the GRID DEFINITIONS -- harness
.all_tasks("large"), rgg_harness.all_tasks("large"), build_suite() minus the drop sets -- and checks the
collected CSV against them. That is two independent artifacts (the code that generated the campaign, and the
campaign itself) rather than the CSV against itself. Several invariants are deliberately paired with an
ANTI-VACUITY twin (the sentinel exists / the dedup removed something), because an invariant that cannot fail
is not an invariant, and this repo has already shipped two of those.

Gate runs BEFORE the write. A failing gate leaves the previous .tex on disk rather than overwriting the paper
with numbers it has just declared untrustworthy.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness                                                          # noqa: E402
import rgg_harness                                                      # noqa: E402

DENSE_CSV = "analysis/large/rows_with_ratio.csv"
SPARSE_CSV = "analysis/rgg_large/rgg_rows_with_ratio.csv"
FAMILIES = ("dense", "sparse")

# ----------------------------------------------------------------------------
# THE FOUR GROUPS. Read this before touching the table.
#
# The two CSVs are NOT two families. Each pools sweeps that answer different questions, and pooling their
# medians produced two claims that were false and looked fine:
#
#   * "pivot gives the best dense GMR cover (0.290)". It does not. That number is a median over exp1
#     (n = 1000-1500, p in {0.3, 0.5}) AND exp2b (n = 2000, p from 0.0046 -- which is not dense at all).
#     Split them: spc_gmr 0.225 beats pivot 0.457 on exp1, pivot 0.278 beats spc_gmr 0.418 on exp2b. Pooled,
#     pivot wins by 0.008 -- an artefact of the mixture, a fact about neither sweep.
#
#   * "l1sep_gmr finds the best sparse GMR cover, beating spc_gmr on 59.6% of tasks". It does not. It loses
#     to spc_gmr on 100% of the 420 INFLATE tasks (0.801 vs 0.256) and beats it on 100% of the 400 DEFLATE
#     tasks (0.124 vs 0.685). The "59.6%" was exactly (400 + 220) / 1040 -- the mixing ratio, not a
#     performance fact. THE CORRUPTION DIRECTION DOES NOT SHIFT THE RANKING, IT INVERTS IT.
#
# So the table reports four groups, never a pooled median. Within a group the survivorship confound largely
# vanishes on the sparse side (survivor |H|/m skew is exactly 1.00x within inflate and within deflate,
# because |H|/m is constant there); it persists on the dense side and is marked with a dagger.
#
# P2size IS DROPPED. It is a jitter sweep with NO corruption direction and |H|/m = 0.004 -- the graphs are
# already metric, so there is nothing to repair and every method scores ~0.01 by doing nearly nothing. It
# was 220 of the 1,040 "sparse" tasks and it dragged every median toward zero. A benchmark of repair
# algorithms on graphs that need no repair is not a benchmark.
# ----------------------------------------------------------------------------
DENSE_SWEEPS = {"dense-n": ("exp", {"exp1"}), "dense-p": ("exp", {"exp2b"})}
SPARSE_SWEEPS = {"inflate": ("sweep", {"S1", "S2", "S2k"}),
                 "deflate": ("sweep", {"P2df", "P2dm", "S1d"})}
DROPPED_SWEEPS = {"P2size"}          # jitter: no corruption, |H|/m = 0.004, nothing to repair

# (key, family, header label, sub-label)
ALL_GROUPS = [
    ("dense-n", "dense", r"$n$-sweep", r"$n\!=\!1000\text{--}1500$, $p \in \{0.3, 0.5\}$"),
    ("dense-p", "dense", r"$p$-sweep", r"$n\!=\!2000$, $p\!=\!0.005\text{--}0.16$"),
    ("inflate", "sparse", r"inflate", r"$n\!=\!1000\text{--}3000$"),
    ("deflate", "sparse", r"deflate", r"$n\!=\!1000\text{--}3000$"),
]
# Section 5 rests on the RGG: it is the only synthetic family with a planted corrupted set AND true weights,
# its ILP converges on 96% of the small grid (the dense grid manages 70%/36%), and its Euclidean weights make
# density and the weight model independent by construction. --groups rgg emits the two RGG columns only.
# --groups all keeps the dense pair alongside, for as long as the dense family is still in the paper.
GROUPS = [g for g in ALL_GROUPS if g[1] == "sparse"]
GKEYS = [g[0] for g in GROUPS]


def set_groups(which):
    """Called from main() before anything is built. GROUPS/GKEYS are module-level because emit_tex,
    emit_matched and the I11 dash gate all read them; rebinding here keeps them the single source."""
    global GROUPS, GKEYS
    GROUPS = ALL_GROUPS if which == "all" else [g for g in ALL_GROUPS if g[1] == "sparse"]
    GKEYS = [g[0] for g in GROUPS]
NUMCOLS = ("size", "valid", "cpu", "wall", "H", "E", "ratio_domr", "n", "knn_k")
KEYCOLS = ("status", "valid", "size", "H", "cpu", "wall", "ratio_domr")   # must agree within a dup group

# A median over survivors is only a median if there are survivors. iomr_bestofk returns on 1 of 720 dense
# instances -- its "median" rho_H is one number from one graph it happened to finish, and the surviving
# instances are exactly the easy ones, so the statistic is biased on top of being thin. Bold (i.e. make a
# CLAIM about) only the methods that return often enough for the median to carry a claim. Everything is
# still PRINTED; the threshold governs emphasis, not disclosure.
MARK_MIN_RET = 0.25

# A survivor set whose median |H|/m is >=1.5x or <=0.67x the family's own is a DIFFERENT population of
# graphs, and rho_H -- whose denominator IS |H| -- cannot be compared across that gap. The daggers in the
# table mark exactly these rows. The band is deliberately generous: it fires on gmr_bestofk (4.1x) and
# iomr_rgrow (0.04x), which are real, and not on sampling wobble.
SKEW_HI, SKEW_LO = 1.5, 0.67

TEX = {"gmr_bestofk": r"\code{gmr\_bestofk}", "iomr_bestofk": r"\code{iomr\_bestofk}",
       "gmr_thr_naive": r"\code{gmr\_thr}", "iomr_thr_naive": r"\code{iomr\_thr}",
       "iomr_regiongrow": r"\code{iomr\_rgrow}"}


def a(name):
    return TEX.get(str(name), r"\code{%s}" % str(name).replace("_", r"\_"))


def cell(v, fmt="%.3f"):
    """A missing cell is a DECISION (n_ok == 0), never a NaN that happened to format as one. See per_algo."""
    return r"\code{--}" if v is None or not np.isfinite(v) else "$" + (fmt % v) + "$"


def load(path, family):
    """Raw frame. No filtering, no dedup -- the gate needs to see the duplicates and the sentinels."""
    if not os.path.exists(path):
        raise SystemExit(f"FATAL: {family} CSV missing: {path}")
    d = pd.read_csv(path, low_memory=False)
    for c in NUMCOLS:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def grid_truth(family):
    """(n_tasks, algos) DERIVED FROM THE HARNESS GRID -- not from the CSV, and not from a literal.

    This is the whole point of the gate: the campaign's own definition is the reference, so a short collect,
    an orphan task file, or a suite edit surfaces as a disagreement instead of as a quietly-wrong median."""
    if family == "dense":
        n_tasks = len(harness.all_tasks("large"))
        algos = {e[0] for e in harness.build_suite(0)} - harness.DROP_LARGE
    else:
        n_tasks = len(rgg_harness.all_tasks("large"))
        algos = {e[0] for e in rgg_harness.build_suite_rgg(0, drop_ilp=True)}
    return n_tasks, algos


def dedup(raw):
    """Collapse the long-format (task, algo) x knn_k rows -- and VERIFY the duplicates agree before doing it.

    If the long rows ever disagree on status/valid/size/..., then keep="first" picks arbitrarily and every
    median downstream becomes order-dependent. Measured today: 0 disagreeing groups. That is what makes this
    check meaningful rather than decorative."""
    bad = 0
    if raw.duplicated(subset=["task", "algo"]).any():
        g = raw.groupby(["task", "algo"], sort=False)
        for c in KEYCOLS:
            if c in raw.columns:
                bad += int((g[c].nunique(dropna=False) > 1).sum())
    uni = raw.drop_duplicates(subset=["task", "algo"], keep="first").copy()
    return uni, dict(n_raw=len(raw), n_unique=len(uni), n_dropped=len(raw) - len(uni), n_inconsistent=bad)


def returning(df):
    """THE filter. Traps 2 and 3: a run counts only if it returned AND its cover verified."""
    return df[df.status.eq("ok") & df.valid.eq(1)]


def fam_hm(uni):
    """The group's own median |H|/m -- the reference every algorithm's survivor |H|/m is read against."""
    t = uni.drop_duplicates(subset=["task"])
    return float((t.H / t.E).median())


def split_groups(uni_by_fam):
    """Cut the two CSVs into the four groups, and ACCOUNT FOR EVERY TASK.

    The gate is the arithmetic: the groups plus the deliberately-dropped jitter sweep must reconstruct each
    CSV exactly. A sweep that belongs to no group would otherwise vanish from the table without a word --
    which is precisely how 220 metric graphs ended up inside a benchmark of repair algorithms."""
    out, sizes = {}, {}
    want = {g[1] for g in GROUPS}          # only the families the selected groups need
    for fam, spec in (("dense", DENSE_SWEEPS), ("sparse", SPARSE_SWEEPS)):
        if fam not in want:
            continue
        d = uni_by_fam[fam]
        seen = set()
        for key, (col, vals) in spec.items():
            sub = d[d[col].astype(str).isin(vals)]
            if sub.empty:
                raise SystemExit(f"FATAL: group '{key}' selects no rows ({col} in {sorted(vals)}). The "
                                 "table would silently print an empty column.")
            out[key] = sub
            sizes[key] = sub.task.nunique()
            seen |= set(sub.task.unique())
        dropped = d[~d.task.isin(seen)]
        dsw = sorted(set(dropped.sweep.dropna().astype(str))) if "sweep" in dropped.columns else []
        dexp = sorted(set(dropped.exp.dropna().astype(str))) if "exp" in dropped.columns else []
        n_drop = dropped.task.nunique()
        # every dropped task must belong to a sweep we MEANT to drop
        stray = set(dsw) - DROPPED_SWEEPS
        if stray:
            raise SystemExit(f"FATAL: {fam}: sweeps {sorted(stray)} ({n_drop} tasks) belong to no group and "
                             "are not on the drop list. Refusing to let a sweep disappear from the table.")
        if n_drop:
            print(f"  {fam}: DROPPED {n_drop} tasks (sweeps {dsw or dexp}) -- jitter, no corruption "
                  f"direction, |H|/m = {float((dropped.drop_duplicates('task').H / dropped.drop_duplicates('task').E).median()):.3f}: "
                  "the graphs are already metric, so there is nothing to repair")
        # anti-vacuity: the pieces must reconstruct the whole
        total = sum(sizes[k] for k in spec) + n_drop
        if total != d.task.nunique():
            raise SystemExit(f"FATAL: {fam}: groups ({total - n_drop}) + dropped ({n_drop}) = {total}, but "
                             f"the CSV holds {d.task.nunique()} tasks. Some task is in two groups or none.")
    return out, sizes


def per_algo(uni, n_tasks):
    r"""One row per algo. n_ok == 0 branches EXPLICITLY to NaN -- never .median() on an empty series, so a
    missing cell is something the code decided, not an artifact that happened to render as a dash.

    TWO COLUMNS BEYOND rho_H, AND THEY ARE NOT DECORATION.

    `sm_med` = |S|/m, the fraction of the graph the repair rewrites. rho_H and |S|/m are the SAME per-task
    comparison -- they differ by |H|/m, which is a constant of the task, so within any one graph they rank
    the algorithms identically. But rho_H hides how INVASIVE a repair is, and invasiveness is what metric
    repair claims to minimise: a cover at rho_H = 0.33 on a graph that is 41% broken still rewrites 13.5% of
    the edges. |S|/m is also the axis the rest of the paper turns on -- what a cover recovers tracks |S|/m,
    not rho_H.

    `hm_med` = the median |H|/m OF THE INSTANCES THIS ALGORITHM SURVIVED. This is the confound column, and it
    exists because the medians are over survivors and survivorship is correlated with |H|/m -- which is
    rho_H's own denominator. gmr_bestofk returns only on the badly broken graphs (survivor |H|/m = 0.411
    against a family median of 0.100), so a 4x larger denominator hands it a 4x smaller rho_H for free; it
    reads as the best cover in the suite and is not. iomr_regiongrow is the mirror image: gated at |H| <= 200
    it survives only near-metric graphs (0.004), so a tiny denominator punishes a rho_H of 2.659 while it in
    fact edits 1.1% of the edges -- the sparsest repair here. Printing rho_H without this column invites both
    misreadings, and the first one already reached the draft."""
    ok = returning(uni)
    rows = []
    for algo, G in uni.groupby("algo", sort=False):
        K = ok[ok.algo.eq(algo)]
        n_ok = len(K)
        n_ok_status = int(G.status.eq("ok").sum())
        # The variant is what the table blocks on, and what the best/worst comparison is scoped to. It comes
        # from the CSV, not from a name prefix -- `pivot` is GMR and `left_edge` is IOMR, and neither says so.
        vs = set(G.variant.dropna().astype(str))
        if len(vs) != 1:
            raise SystemExit(f"FATAL: algo '{algo}' carries {len(vs)} variants ({sorted(vs)}). The table "
                             "blocks on variant and compares bests within it; an ambiguous variant would "
                             "put a row in the wrong block and silently change which method is 'best'.")
        rows.append(dict(
            algo=algo, variant=vs.pop(), n_tasks=n_tasks, n_ok=n_ok, n_ok_status_only=n_ok_status,
            n_unverified=n_ok_status - n_ok,
            return_rate=(n_ok / n_tasks) if n_tasks else np.nan,
            ratio_domr_med=float(K.ratio_domr.median()) if n_ok else np.nan,
            ratio_domr_max=float(K.ratio_domr.max()) if n_ok else np.nan,
            sm_med=float((K["size"] / K.E).median()) if n_ok else np.nan,
            hm_med=float((K.H / K.E).median()) if n_ok else np.nan,
            wall_med=float(K.wall.median()) if n_ok else np.nan,
            n_timeout=int(G.status.eq("timeout").sum()),
            n_skipped=int(G.status.astype(str).str.startswith("skipped").sum()),
        ))
    return pd.DataFrame(rows).set_index("algo")


def classify(stats, uni):
    """(cover_algos, bound_algos), derived from the DATA and then cross-checked against the CODE.

    An algo is a BOUND row -- an LP value, not a cover -- iff `size` is NaN on every returning row in BOTH
    families. That is the ground truth: harness._lp returns `cover = None` for every LP except the GMR/rsp
    one ("the naive GMR LP and the IOMR LP are LOWER BOUNDS only"), and harness.py:260 then records
    size = valid = None.

    DO NOT cross-check this against build_suite()'s verify_key. That field is NOT a reliable bound-row
    signal: `gmr_lp_naive` is declared verify_key="gmr" yet returns no cover, while its exact twin
    `iomr_lp_naive` is declared "none". The label is dead code -- the `cover is None` branch short-circuits
    before any verifier runs -- but a gate keyed on it fires spuriously. (Found by this gate, on its first
    run. Left as a comment rather than a fix: harness.py is frozen mid-campaign.)

    So the two checks below use signals that ARE load-bearing, and each can genuinely fail:
      (a) every bound row is an LP-value row BY NAME. If a real heuristic -- pivot, spc, l1sep -- ever stops
          producing covers everywhere, it would otherwise be silently reclassified as "a bound" and vanish
          from the table. This catches a broken algorithm masquerading as a design decision.
      (b) every verify_key=="none" row that ran IS in the bound set. This catches the other direction: an
          LP-bound row that started returning covers, which would mean it belongs in the table."""
    algos = sorted(set(stats["dense"].index) | set(stats["sparse"].index))
    bound = set()
    for al in algos:
        sized = False
        for f in FAMILIES:
            K = returning(uni[f])
            K = K[K.algo.eq(al)]
            if len(K) and K["size"].notna().any():
                sized = True
        if not sized:
            bound.add(al)

    not_lp = {x for x in bound if "_lp_" not in x}
    if not_lp:
        raise SystemExit(f"FATAL: {sorted(not_lp)} produced no cover in EITHER family, but "
                         "these are not LP-value rows. A cover algorithm that returns nothing anywhere is a "
                         "broken algorithm, not a bound. Refusing to drop it from the table silently.")
    none_key = {e[0] for e in harness.build_suite(0) if e[2] == "none"} & set(algos)
    if not none_key <= bound:
        raise SystemExit(f"FATAL: {sorted(none_key - bound)} are declared verify_key=='none' (a pure lower "
                         "bound) yet returned a cover. If they now produce covers they belong in the table.")
    return [x for x in algos if x not in bound], sorted(bound)


def _order(stats, cover):
    """By dense |S|/m -- then the algos that never return dense, by sparse |S|/m.

    THE AXIS OF THE PAPER IS |S|/m, NOT rho_H. rho_H = |S|/|H| divides by the DOMR optimum, and |H| IS the
    GMR optimum (OPT/|H| = 1.000 median, Spearman 0.9999 on the small grid) but is NOT the IOMR optimum
    (1.250 median, drifting to 1.35 on the more broken graphs). So the same symbol would mean "approximation
    ratio" in the GMR block and "ratio to an unrelated quantity" in the IOMR block. m is neutral: one
    denominator, one meaning, in every block and every later section -- and it is the axis Sections 7 and 8
    already run on (captured tracks |S|/m at r = 0.945).

    It also sees what rho_H hid. In the dense IOMR block rho_H compresses every method into 4.1-4.4; |S|/m
    shows that all of them but l1sep rewrite 83-92% of the graph. `left_edge` edits 92% of the edges. That is
    not a repair, it is the graph -- and rho_H reported it as a middling 4.135.

    CLEANUP TODO: consider dropping the rho_H column entirely. It is kept for now because the DOMR row
    anchors it (rho_H = 1 by construction) and Section 5.1 licenses |H| as the GMR optimum, but nothing in
    the paper's argument needs it.
    """
    def key(al):
        d = stats["dense"].sm_med.get(al, np.nan)
        s = stats["sparse"].sm_med.get(al, np.nan)
        if np.isfinite(d):
            return (0, d)
        return (1, s if np.isfinite(s) else 1e9)
    return sorted(cover, key=key)


# ----------------------------------------------------------------------------
# The gate
# ----------------------------------------------------------------------------
def gate(raw, uni, stats, truth, dstats, cover, bound):
    fails, warns = [], []

    def chk(ok, name, observed):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<48} {observed}")
        if not ok:
            fails.append(name)

    print("GATE -- every expected value imported from the harness grid, none typed in")
    for f in FAMILIES:
        n_tasks, algos = truth[f]
        R, U, S, D = raw[f], uni[f], stats[f], dstats[f]
        n_csv_tasks = U.task.nunique()
        csv_algos = set(U.algo.unique())

        # I1  the campaign is complete -- the grid code says how many tasks there should be
        chk(n_csv_tasks == n_tasks, f"I1  {f}: task count == grid",
            f"csv {n_csv_tasks} vs grid {n_tasks}")
        # I2  the suite is the suite -- an algo renamed, added, or dropped surfaces here
        chk(csv_algos == algos, f"I2  {f}: algo set == suite",
            f"{len(csv_algos)} algos" + ("" if csv_algos == algos else f"  DIFF {csv_algos ^ algos}"))
        # I3  rectangular -- a partial task file makes the return-rate denominator a fiction
        chk(len(U) == n_tasks * len(algos), f"I3  {f}: grid rectangular",
            f"{len(U)} == {n_tasks} x {len(algos)} = {n_tasks * len(algos)}")
        # I5  the dedup is WELL-DEFINED: duplicated rows must agree, or keep="first" is arbitrary
        chk(D["n_inconsistent"] == 0, f"I5  {f}: dup groups self-consistent",
            f"{D['n_inconsistent']} disagreeing groups")
        # I6a trap 2 -- a killed run must never be labelled ok
        n_ok0 = int((R.status.eq("ok") & R.cpu.eq(0)).sum())
        chk(n_ok0 == 0, f"I6a {f}: no ok row has cpu == 0", f"{n_ok0} such rows")
        # I6b ANTI-VACUITY for I6a. If a future collect writes NaN instead of the 0 sentinel, I6a would pass
        #     trivially and prove nothing. Asserting the sentinel EXISTS is what makes I6a able to fail.
        n_sent = int(R.cpu.eq(0).sum())
        chk(n_sent > 0, f"I6b {f}: the cpu==0 sentinel exists", f"{n_sent} killed rows carry it")
        # I7  H > 0 everywhere. harness has a legitimate (status=ok, cpu=0) path for an ALREADY-METRIC graph
        #     -- the one case where cpu==0 is NOT a sentinel. No such task is in these grids, so I6a is
        #     unambiguous. If one ever enters, I7 fires and NAMES the reason rather than I6a firing mysteriously.
        n_h0 = int((U.H <= 0).sum())
        chk(n_h0 == 0, f"I7  {f}: H > 0 on every task", f"{n_h0} tasks with H <= 0")
        # I8  a return rate above 1 is arithmetically impossible -- and is the loudest possible signature of a
        #     skipped dedup (on the RAW sparse CSV, domr scores 215%)
        bad_rate = S[(S.return_rate < 0) | (S.return_rate > 1)]
        chk(len(bad_rate) == 0, f"I8  {f}: return rate in [0,1]",
            "all in range" if not len(bad_rate) else f"OUT OF RANGE: {list(bad_rate.index)}")
        # I12 rgg_analyze BLANKS the metric columns on valid==0, so an ok-but-unverified row carries NaN
        #     wall/ratio. The valid==1 filter must remove them -- not .median()'s skipna, silently.
        K = returning(U)
        n_nan = int(K.ratio_domr.isna().sum() + K.wall.isna().sum())
        chk(n_nan == 0, f"I12 {f}: no NaN survives the filter", f"{n_nan} NaN cells among returning rows")
        # I14 the last two table columns rest on this
        n_w0 = int((K.wall <= 0).sum())
        chk(n_w0 == 0, f"I14 {f}: wall > 0 on every returning row", f"{n_w0} rows with wall <= 0")

    # I4  the dedup DID something (sparse) and had nothing to do (dense). The anti-vacuity half of trap 1: if
    #     the RGG CSV is ever re-emitted wide, a no-op dedup would otherwise pass in silence.
    chk(dstats["sparse"]["n_dropped"] > 0, "I4a sparse: the dedup removed rows",
        f"{dstats['sparse']['n_raw']} -> {dstats['sparse']['n_unique']} "
        f"({dstats['sparse']['n_dropped']} dropped)")
    chk(dstats["dense"]["n_dropped"] == 0, "I4b dense: no duplicates to remove",
        f"{dstats['dense']['n_raw']} rows, {dstats['dense']['n_dropped']} dropped")
    # I9  the two grids must run the SAME suite, or the side-by-side compares two different studies and half
    #     the dashes are structural rather than scientific
    same = set(stats["dense"].index) == set(stats["sparse"].index)
    chk(same, "I9  dense and sparse run the same suite",
        f"{len(stats['dense'].index)} vs {len(stats['sparse'].index)} algos")

    # I13 NOT fatal -- a loud diff. This is what SURFACES the 100.0% -> 99.7% correction instead of silently
    #     changing a published number.
    print("\n  I13 status-only vs returned-AND-verified (the defect this script exists to catch):")
    any_unver = False
    for f in FAMILIES:
        S = stats[f]
        for al, r in S[S.n_unverified > 0].iterrows():
            any_unver = True
            print(f"      {f:<7}{al:<18} ok={int(r.n_ok_status_only):<6} ok&valid={int(r.n_ok):<6} "
                  f"-> {100 * r.n_ok_status_only / r.n_tasks:.1f}% becomes {100 * r.return_rate:.1f}%")
            warns.append(f"{f}/{al}: {int(r.n_unverified)} unverified")
    if not any_unver:
        print("      none -- every ok row verified")

    print(f"\n  cover rows (derived): {len(cover)}    bound rows (derived, no cover anywhere): {bound}")
    return fails, warns


# ----------------------------------------------------------------------------
# Emit
# ----------------------------------------------------------------------------
def _n(x):
    return f"{int(x):,}".replace(",", "{,}")


def emit_tex(gstats, gsizes, cover, bound, gunis):
    """The four-group table. There is no pooled column, on purpose -- see the GROUPS note at the top.

    Two disclosures survive from the pooled version, both still load-bearing:
      * BOLD IS SCOPED TO (variant x group). GMR and IOMR are different problems and a cross-variant winner
        would measure the constraint, not the algorithm. And a cross-GROUP winner would measure the mixture.
      * THE DAGGER. Within inflate and within deflate the survivorship confound is gone (survivor |H|/m skew
        is exactly 1.00x -- |H|/m is constant inside a direction). On the two DENSE sweeps it persists:
        l1sep_gmr survives the cleaner half of the n-sweep (0.53x), l1sep_iomr of the p-sweep (0.36x). So the
        dagger rides on the |S|/m cell itself rather than costing a whole column per group.
    """
    ghm = {k: fam_hm(gunis[k]) for k in GKEYS}

    # SHORT, and ONE COLUMN. The table is narrow -- an algorithm name and two (ret., |S|/m) pairs -- so it has
    # no business spanning the page. Everything the caption used to argue (why there is no pooled column, what
    # each sweep is, why the dagger exists) is the SECTION's job; what stays is only what the reader needs to
    # read a row.
    cap = (r"\caption{\textbf{The corruption decides.} $|S|/m$ --- the share of the graph the repair rewrites "
           r"--- per direction, over runs that returned \emph{and} verified; \emph{ret.} is how often a "
           r"verified cover came back at all (%s tasks). \DOMR{}'s row \emph{is} $|H|/m$. Bold: best and two "
           r"worst \emph{within} each variant and direction. $\dagger$: survived a markedly different slice of "
           r"its group, so not comparable with the unmarked rows. A dash is no cover. \textbf{There is no "
           r"pooled column: an average over the two directions reports the mixing ratio, not the methods.}}"
           % " + ".join(_n(gsizes[k]) for k in GKEYS))

    ncol = 1 + 2 * len(GKEYS)
    colspec = "@{}l" + "".join("rr" if i == 0 else "@{\\quad}rr" for i in range(len(GKEYS))) + "@{}"
    # BOTH header rows are DERIVED from GROUPS. They used to be hardcoded for the four-group layout, and
    # under --groups rgg that emitted a 9-column header over a 5-column tabular: the paper did not compile.
    # A header that does not follow the columns it labels is not a header.
    FAMLAB = {"dense": r"dense $\Gamma(n,p)$", "sparse": r"sparse geometric (RGG)"}
    fam_head, fam_rules, c = [], [], 2
    for fam in dict.fromkeys(g[1] for g in GROUPS):                # families, in GROUPS order, deduped
        k = sum(1 for g in GROUPS if g[1] == fam)                  # how many groups this family carries
        fam_head.append(r"\multicolumn{%d}{c}{%s}" % (2 * k, FAMLAB[fam]))
        fam_rules.append(r"\cmidrule(lr){%d-%d}" % (c, c + 2 * k - 1)); c += 2 * k
    head1, head2, rules, c = [], [], [], 2
    for key, famname, lab, sub in GROUPS:
        head1.append(r"\multicolumn{2}{c}{%s}" % lab)
        head2 += [r"ret.", r"$|S|/m$"]
        rules.append(r"\cmidrule(lr){%d-%d}" % (c, c + 1)); c += 2
    # ONE COLUMN when the group set fits (the RGG pair does; the four-group layout does not). The tabular is
    # wrapped in \resizebox so it shrinks to the column rather than overflowing it -- the numbers stay
    # readable at this width, and a table this narrow has no business spanning the page.
    onecol = len(GKEYS) <= 2
    env = "table" if onecol else "table*"
    width = r"\columnwidth" if onecol else r"\textwidth"
    out = [r"% GENERATED by experiments/inversion_table.py -- DO NOT EDIT, DO NOT TRANSCRIBE. Regenerate.",
           r"\begin{%s}[t]\centering\footnotesize" % env, cap, r"\label{tab:invert}",
           r"\resizebox{%s}{!}{%%" % width,
           r"\begin{tabular}{%s}" % colspec, r"\toprule"]
    # The family header row exists to separate dense from sparse. With ONE family it separates nothing --
    # it is a banner over the whole table saying what the caption already said. Drop it, and the rule under
    # it, rather than spend two rows of a one-column float on a tautology.
    if len(set(g[1] for g in GROUPS)) > 1:
        out += [" & ".join([""] + fam_head) + r" \\", " ".join(fam_rules)]
    out += [" & ".join([""] + head1) + r" \\",
            " ".join(rules),
            " & ".join(["algorithm"] + head2) + r" \\",
            r"\midrule"]
    # NO sub-label row. It ran the full width of the table* and overflowed the page; the group definitions
    # belong in the caption, where they cost nothing.

    def marks(key, algos):
        """Best and two worst -- within ONE variant and ONE group, over the comparable rows only."""
        S_ = gstats[key]
        e = S_.loc[[x for x in algos if x in S_.index]]
        e = e[e.sm_med.notna() & (e.return_rate >= MARK_MIN_RET) & (e.index != "domr")]
        if not len(e):
            return set()
        skew = e.hm_med / ghm[key]
        e = e[(skew < SKEW_HI) & (skew > SKEW_LO)]
        if not len(e):
            return set()
        return {e.sm_med.idxmin()} | set(e.sm_med.nlargest(2).index)

    # order: by the n-sweep's |S|/m, then whatever group does have a number
    def okey(al):
        for k in GKEYS:
            v = gstats[k].sm_med.get(al, np.nan)
            if np.isfinite(v):
                return (GKEYS.index(k), v)
        return (99, 0.0)
    order = sorted(cover, key=okey)

    var = {}
    for al in cover:
        for k in GKEYS:
            if al in gstats[k].index and pd.notna(gstats[k].variant.get(al)):
                var[al] = str(gstats[k].variant[al]); break
    blocks = [(v, [al for al in order if var.get(al) == v]) for v in ("DOMR", "GMR", "IOMR")]
    placed = {al for _, b in blocks for al in b}
    if placed != set(cover):
        raise SystemExit(f"FATAL: {sorted(set(cover) - placed)} fall outside DOMR/GMR/IOMR. A row that lands "
                         "in no block silently vanishes from the table.")
    mk = {(k, v): marks(k, als) for k in GKEYS for v, als in blocks}

    LABEL = {"DOMR": r"\DOMR{}", "GMR": r"\GMR{}", "IOMR": r"\IOMR{}"}
    for bi, (v, als) in enumerate(blocks):
        if not als:
            continue
        out.append(r"\midrule" if bi else r"\addlinespace[1pt]")
        out.append(r"\multicolumn{%d}{@{}l}{\textbf{%s}} \\[1pt]" % (ncol, LABEL[v]))
        for al in als:
            cells = [r"\quad " + a(al)]
            for k in GKEYS:
                if al not in gstats[k].index:
                    cells += [r"\code{--}", r"\code{--}"]
                    continue
                r = gstats[k].loc[al]
                cells.append(r"$%.1f\%%$" % (100 * r.return_rate))
                sm = cell(r.sm_med)
                if al in mk[(k, v)]:
                    sm = r"$\mathbf{%.3f}$" % r.sm_med
                elif np.isfinite(r.hm_med) and ghm[k] > 0 and al != "domr":
                    skew = r.hm_med / ghm[k]
                    if skew >= SKEW_HI or skew <= SKEW_LO:
                        sm = r"$%.3f^{\dagger}$" % r.sm_med
                cells.append(sm)
            out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{%s}" % env]
    return "\n".join(out)


def emit_macros(stats, uni, truth, cover, bound):
    """Every number the PROSE quotes. Generating the table but leaving the paragraph above it hand-typed
    would satisfy the no-transcription rule by half."""
    D, S = stats["dense"], stats["sparse"]
    nd, dalgos = truth["dense"]
    ns, _ = truth["sparse"]

    # gmr_bestofk's dense returns all sit at small n. Derive the next ladder step ABOVE its last success and
    # count the tasks at or beyond it, so "and on none of the N with n >= X" is computed, not remembered.
    okd = returning(uni["dense"])
    bok = okd[okd.algo.eq("gmr_bestofk")]
    ladder = sorted(uni["dense"].n.dropna().unique())
    if len(bok):
        nmax = float(bok.n.max())
        above = [v for v in ladder if v > nmax]
        thresh = float(above[0]) if above else np.inf
    else:
        nmax, thresh = np.nan, (float(ladder[0]) if ladder else np.inf)
    big = uni["dense"][uni["dense"].n >= thresh]
    n_big_tasks = int(big.task.nunique())
    n_big_bok = int(len(returning(big)[returning(big).algo.eq("gmr_bestofk")]))

    M = [r"% GENERATED by experiments/inversion_table.py -- DO NOT EDIT. Every number Section 4 quotes.",
         r"\newcommand{\invDenseTasks}{%s}" % _n(nd),
         r"\newcommand{\invSparseTasks}{%s}" % _n(ns),
         r"\newcommand{\invLargeTasks}{%s}" % _n(nd + ns),
         r"\newcommand{\invNAlgos}{%d}" % len(dalgos),
         r"\newcommand{\invNCover}{%d}" % len(cover),
         r"\newcommand{\invNBound}{%d}" % len(bound),
         r"\newcommand{\invRegionHMax}{%d}" % harness.REGION_H_MAX,
         r"\newcommand{\invBokSparseMed}{%.3f}" % S.ratio_domr_med["gmr_bestofk"],
         r"\newcommand{\invBokSparseRet}{%.1f}" % (100 * S.return_rate["gmr_bestofk"]),
         r"\newcommand{\invBokDenseOk}{%d}" % D.n_ok["gmr_bestofk"],
         r"\newcommand{\invBokDenseRet}{%.1f}" % (100 * D.return_rate["gmr_bestofk"]),
         r"\newcommand{\invBokDenseMaxN}{%s}" % (_n(nmax) if np.isfinite(nmax) else "--"),
         r"\newcommand{\invBokBigNThresh}{%s}" % (_n(thresh) if np.isfinite(thresh) else "--"),
         r"\newcommand{\invBokBigNTasks}{%s}" % _n(n_big_tasks),
         r"\newcommand{\invBokBigNOk}{%d}" % n_big_bok,
         r"\newcommand{\invIbokDenseOk}{%d}" % D.n_ok["iomr_bestofk"],
         r"\newcommand{\invIbokSparseMed}{%.3f}" % S.ratio_domr_med["iomr_bestofk"],
         r"\newcommand{\invPivotDenseMed}{%.3f}" % D.ratio_domr_med["pivot"],
         r"\newcommand{\invLeDenseMed}{%.3f}" % D.ratio_domr_med["left_edge"],
         r"\newcommand{\invPivotSparseMed}{%.3f}" % S.ratio_domr_med["pivot"],
         r"\newcommand{\invLeSparseMed}{%.3f}" % S.ratio_domr_med["left_edge"],
         r"\newcommand{\invPivotSparseMax}{%.1f}" % S.ratio_domr_max["pivot"],
         r"\newcommand{\invPivotSparseMaxInt}{%d}" % int(S.ratio_domr_max["pivot"]),
         r"\newcommand{\invLeSparseMax}{%.1f}" % S.ratio_domr_max["left_edge"],
         r"\newcommand{\invLoneDenseRet}{%.1f}" % (100 * D.return_rate["l1sep_gmr"]),
         r"\newcommand{\invLoneDenseMed}{%.3f}" % D.ratio_domr_med["l1sep_gmr"],
         r"\newcommand{\invLoneSparseRet}{%.1f}" % (100 * S.return_rate["l1sep_gmr"]),
         r"\newcommand{\invLoneSparseMed}{%.3f}" % S.ratio_domr_med["l1sep_gmr"],
         r"\newcommand{\invLoneiDenseRet}{%.1f}" % (100 * D.return_rate["l1sep_iomr"]),
         r"\newcommand{\invLoneiSparseRet}{%.1f}" % (100 * S.return_rate["l1sep_iomr"]),
         r"\newcommand{\invRgrowDenseRet}{%.1f}" % (100 * D.return_rate["iomr_regiongrow"]),
         ]
    return "\n".join(M)


def loglog(xs, ys):
    """Slope AND R^2 of a log-log least-squares fit -- the same procedure as cost_law.loglog_slope, which
    reports the slope alone. We also return R^2 and the point count, because for `pivot` vs m the slope is
    -0.02 at R^2 = 0.09: that is not a shallow power law, it is NO TREND, and a paper that prints the
    exponent without the fit quality has dressed a flat line up as a measurement. The point count matters for
    the same reason -- l1sep's m-fit survives on 3 of 5 points, the other two having been killed by the
    L1_GUARD_S time guard."""
    xy = [(x, y) for x, y in zip(xs, ys) if x and x > 0 and y and y > 0 and np.isfinite(y)]
    if len(xy) < 2:
        return float("nan"), float("nan"), len(xy)
    lx = np.log([p[0] for p in xy])
    ly = np.log([p[1] for p in xy])
    b, a = np.polyfit(lx, ly, 1)
    pred = a + b * lx
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(b), r2, len(xy)


def emit_costlaw(path):
    """The three cost laws -- as RATIOS first, exponents second.

    The ratios are direct measurements and need no fit, no R^2, and no apology: over the m-sweep, `pivot`
    goes from 0.203 s to 0.205 s while m grows 5.3x. That is the claim the paper actually wants to make ("it
    does not notice how many edges you hand it"), it is stronger than an exponent, and it cannot be attacked
    on fit quality. The exponents are emitted alongside, each with its R^2 and its surviving point count, so
    the prose can be honest about which of the three is a law and which is a flat line."""
    if not os.path.exists(path):
        raise SystemExit(f"FATAL: cost-law CSV missing: {path}")
    d = pd.read_csv(path)
    for c in ("knob", "n", "m", "H"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    AX = {"N": "n", "M": "m", "H": "H"}
    ALGOS = ("pivot", "gmr_bestofk", "l1sep_gmr")
    NAME = {"pivot": "Pivot", "gmr_bestofk": "Bok", "l1sep_gmr": "Lone"}

    M = [r"% --- the three cost laws, from analysis/cost_law.csv (same log-log fit as cost_law.py) ---"]
    for al in ALGOS:
        for S, xk in AX.items():
            g = d[d.sweep.eq(S)]
            b, r2, k = loglog(g[xk].tolist(), g[f"cpu_{al}"].tolist())
            tag = NAME[al] + S
            M += [r"\newcommand{\cl%sExp}{%.2f}" % (tag, b),
                  r"\newcommand{\cl%sRsq}{%.2f}" % (tag, r2),
                  r"\newcommand{\cl%sPts}{%d}" % (tag, k)]
        # the ratio form: how far the knob moved, and how far the clock moved with it, over the runs that
        # SURVIVED. This is what the prose quotes.
        for S, xk in AX.items():
            g = d[d.sweep.eq(S)].dropna(subset=[f"cpu_{al}"])
            g = g[g[f"cpu_{al}"] > 0]
            tag = NAME[al] + S
            if len(g) < 2:
                continue
            lo, hi = g.iloc[0], g.iloc[-1]
            M += [r"\newcommand{\cl%sKnobLo}{%s}" % (tag, _n(lo[xk])),
                  r"\newcommand{\cl%sKnobHi}{%s}" % (tag, _n(hi[xk])),
                  r"\newcommand{\cl%sKnobFac}{%.1f}" % (tag, hi[xk] / lo[xk]),
                  r"\newcommand{\cl%sCpuLo}{%.2f}" % (tag, lo[f"cpu_{al}"]),
                  r"\newcommand{\cl%sCpuHi}{%.2f}" % (tag, hi[f"cpu_{al}"]),
                  r"\newcommand{\cl%sCpuFac}{%.1f}" % (tag, hi[f"cpu_{al}"] / lo[f"cpu_{al}"])]
        # how many points the time guard killed -- the honest caveat on the l1sep m-exponent
        for S, xk in AX.items():
            g = d[d.sweep.eq(S)]
            M.append(r"\newcommand{\cl%sDead}{%d}" % (NAME[al] + S, int(g[f"cpu_{al}"].isna().sum())))
    return "\n".join(M)


def emit_domr_edit(path):
    """DOMR's precision and recall against the PLANTED edit set -- split by corruption direction, because
    the split is the point.

    The draft claimed "on planted data DOMR's cover IS the corrupted set: precision 1.000, recall 1.000",
    unqualified. That is true under INFLATION and false under deflation, where precision is ~0. The reason is
    structural, not statistical: a deflated edge is a shortcut, so it is shorter than its detour, so it is not
    heavy, so a decrease-only cover cannot contain it. Emitting both directions forces the prose to say so --
    and the asymmetry is the same one that drives Question 2, so conceding it strengthens the paper.

    Measured on the clean planted sweeps (break_type == "reweight"), where the corrupted set is known exactly.
    Medians AND means: exact-1.000 is a median statement, and the mean recall is 0.999, not 1.000."""
    if not os.path.exists(path):
        raise SystemExit(f"FATAL: planted-edit CSV missing: {path}")
    d = pd.read_csv(path, low_memory=False)
    d = d.drop_duplicates(subset=["task", "algo"], keep="first")            # trap 1, again
    d = d[d.algo.eq("domr") & d.status.eq("ok") & d.break_type.eq("reweight")]
    for c in ("edit_precision", "edit_recall"):
        d[c] = pd.to_numeric(d[c], errors="coerce")

    M = [r"% --- DOMR vs the planted edit set, by corruption direction (analysis/rgg/) ---"]
    for direction, tag in (("inflate", "Inf"), ("deflate", "Def")):
        g = d[d.direction.eq(direction)].dropna(subset=["edit_precision", "edit_recall"])
        if not len(g):
            raise SystemExit(f"FATAL: no domr/{direction}/reweight rows. The direction split is the claim; "
                             "refusing to emit a macro for a direction with no data.")
        M += [r"\newcommand{\domr%sPrec}{%.3f}" % (tag, g.edit_precision.median()),
              r"\newcommand{\domr%sRec}{%.3f}" % (tag, g.edit_recall.median()),
              r"\newcommand{\domr%sPrecMean}{%.3f}" % (tag, g.edit_precision.mean()),
              r"\newcommand{\domr%sRecMean}{%.3f}" % (tag, g.edit_recall.mean()),
              r"\newcommand{\domr%sTasks}{%s}" % (tag, _n(len(g))),
              r"\newcommand{\domr%sPerfect}{%.1f}" % (tag, 100.0 * float((g.edit_precision == 1.0).mean()))]
    return "\n".join(M)


def emit_matched(gunis):
    r"""MATCHED head-to-head, WITHIN each group. The pooled version of this function was wrong.

    It reported that l1sep_gmr beats spc_gmr on 59.6% of sparse tasks, and read that as a performance fact.
    It is not one. l1sep_gmr loses to spc_gmr on 100% of the 420 INFLATE tasks and beats it on 100% of the
    400 DEFLATE tasks (and 100% of the 220 jitter tasks, now dropped). 620/1040 = 59.6% -- the number was the
    MIXING RATIO. A head-to-head across a mixture of regimes measures the mixture.

    So every comparison here is scoped to one group. The direction does not shift the ranking; it inverts it,
    and the gate below makes that impossible to hide: any pair whose winner FLIPS between two groups is
    printed by name, and the count is emitted, so no prose can claim a universal winner that does not exist.
    """
    def h2h(k, A, B):
        d = returning(gunis[k]).copy()
        d["sm"] = d["size"] / d.E
        p = d[d.algo.isin([A, B])].pivot_table(index="task", columns="algo", values=["ratio_domr", "sm"])
        if ("sm", A) not in p.columns or ("sm", B) not in p.columns:
            return 0, np.nan, np.nan, np.nan
        sh = p.dropna()
        if not len(sh):
            return 0, np.nan, np.nan, np.nan
        ws = float((sh[("sm", A)] < sh[("sm", B)]).mean())
        wr = float((sh[("ratio_domr", A)] < sh[("ratio_domr", B)]).mean())
        if abs(wr - ws) > 1e-9:
            raise SystemExit(f"FATAL: {A} beats {B} on {ws:.4f} of shared {k} tasks by |S|/m but {wr:.4f} by "
                             "rho_H. These MUST agree -- the two differ by |H|/m, a constant of the task.")
        return len(sh), float(sh[("sm", A)].median()), float(sh[("sm", B)].median()), ws

    MIN_SHARED = 30
    L1 = {"GMR": "l1sep_gmr", "IOMR": "l1sep_iomr"}
    M = [r"% --- MATCHED head-to-head, PER GROUP. A pooled head-to-head measures the mixture: l1sep_gmr's",
         r"% --- famous 59.6% over spc_gmr was exactly (deflate + jitter) / all -- the mixing ratio. Never",
         r"% --- quote a cross-group win rate."]
    flips, wins = [], {}
    for k in GKEYS:
        st = stats_for_variant(gunis[k])
        for v, L in L1.items():
            for b in [x for x in st if st[x] == v and x != L]:
                n, ma, mb, w = h2h(k, L, b)
                if n >= MIN_SHARED:
                    wins[(k, L, b)] = (w, n, ma, mb)

    # THE GATE: does any pair's winner FLIP between groups? That is the paper's finding, and a generator
    # that emitted only per-group win rates would let the prose average them back into a lie.
    pairs = {(L, b) for (_, L, b) in wins}
    for L, b in sorted(pairs):
        ks = [k for k in GKEYS if (k, L, b) in wins]
        ws = [wins[(k, L, b)][0] for k in ks]
        if ws and max(ws) > 0.5 and min(ws) < 0.5:
            flips.append((L, b, ks, ws))
    print(f"\n  MATCHED, per group -- pairs whose WINNER FLIPS between groups: {len(flips)}")
    for L, b, ks, ws in flips:
        detail = ", ".join(f"{k} {100*w:.0f}%" for k, w in zip(ks, ws))
        print(f"    !! {L} vs {b}: {detail}   <-- no universal winner exists")
    if not flips:
        print("    (none -- every pair has the same winner in every group)")

    M += [r"\newcommand{\mtchNFlips}{%d}" % len(flips)]
    for i, (L, b, ks, ws) in enumerate(flips[:4]):
        lo, hi = int(np.argmin(ws)), int(np.argmax(ws))
        M += [r"\newcommand{\mtchFlip%sA}{%s}" % (chr(65 + i), a(L)),
              r"\newcommand{\mtchFlip%sB}{%s}" % (chr(65 + i), a(b)),
              r"\newcommand{\mtchFlip%sLoGrp}{%s}" % (chr(65 + i), ks[lo]),
              r"\newcommand{\mtchFlip%sLoWin}{%.1f}" % (chr(65 + i), 100 * ws[lo]),
              r"\newcommand{\mtchFlip%sHiGrp}{%s}" % (chr(65 + i), ks[hi]),
              r"\newcommand{\mtchFlip%sHiWin}{%.1f}" % (chr(65 + i), 100 * ws[hi])]

    # the headline pair, by name: l1sep_gmr vs spc_gmr, inflate vs deflate
    for k, tag in (("inflate", "Inf"), ("deflate", "Def")):
        n, ml, ms, w = h2h(k, "l1sep_gmr", "spc_gmr")
        if n:
            M += [r"\newcommand{\lone%sSm}{%.3f}" % (tag, ml),
                  r"\newcommand{\spc%sSm}{%.3f}" % (tag, ms),
                  r"\newcommand{\lone%sWin}{%.1f}" % (tag, 100 * w),
                  r"\newcommand{\lone%sN}{%s}" % (tag, _n(n))]
            print(f"    {k:<8}: l1sep_gmr {ml:.3f} vs spc_gmr {ms:.3f} -- l1sep wins {100*w:.1f}% of {n}")
    return "\n".join(M)


def stats_for_variant(uni):
    """algo -> variant, straight from the CSV."""
    out = {}
    for algo, G in uni.groupby("algo", sort=False):
        vs = set(G.variant.dropna().astype(str))
        if len(vs) == 1:
            out[algo] = vs.pop()
    return out


# The eight delivered arrays. This IS the campaign: the abstract's "N algorithms over M instances" is the
# sum over these, and nothing else.
ARRAYS = [
    ("small (geometric)", "rows_with_ratio.csv"),
    ("large (dense grid)", "large/rows_with_ratio.csv"),
    ("rgg", "rgg/rgg_rows_with_ratio.csv"),
    ("rgg_mixed", "rgg_mixed/rgg_rows_with_ratio.csv"),
    ("rgg_large", "rgg_large/rgg_rows_with_ratio.csv"),
    ("rgg_largemix", "rgg_largemix/rgg_rows_with_ratio.csv"),
    ("rgg_realrec", "rgg_realrec/rgg_rows_with_ratio.csv"),
    ("real", "real_rows_with_ratio.csv"),
]


def emit_campaign(root="analysis"):
    r"""The campaign totals the ABSTRACT quotes -- derived, never typed.

    This exists because the abstract said "sixteen algorithms over 11,965 instances" while Section 5 said
    "twenty-one algorithms over 11,965 instances". Both numbers were hand-typed; one of them was wrong. The
    11,965 is the sum over the EIGHT DELIVERED ARRAYS, and those arrays were run by the FULL 21-algorithm
    suite. Sixteen is a different quantity entirely -- the sub-suite the two LARGE grids compare, after
    DROP_LARGE removes the two ILPs and the three rsp-oracle methods. Conflating them understates the study
    and contradicts its own Section 5.

    So: \invNAlgosFull is the full suite, \invNAlgos (emitted elsewhere) is the large-grid sub-suite, and the
    two are now impossible to confuse because neither is typed.

    THE GATE IS THE ARITHMETIC. The sub-suite must be a strict subset of the full suite, and the arrays must
    total what the row count says they total. A missing array shows up as a short total rather than silently
    shrinking the campaign.
    """
    tasks = rows = 0
    seen = []
    for name, fn in ARRAYS:
        p = os.path.join(root, fn)
        if not os.path.exists(p):
            raise SystemExit(f"FATAL: campaign array '{name}' missing at {p}. The abstract's instance count "
                             "is the SUM over the eight arrays; refusing to emit a total with a hole in it.")
        d = pd.read_csv(p, low_memory=False)
        if name == "real":
            # The real array has no `task` column: one task is one (graph, mode, seed).
            t = len(d[["graph", "mode", "seed"]].drop_duplicates())
            r = len(d)
        else:
            t = d.task.nunique()
            # trap 1: the rgg arrays are LONG (one row per knn_k). Collapse before counting rows.
            r = len(d.drop_duplicates(subset=["task", "algo"], keep="first"))
        tasks += t
        rows += r
        seen.append((name, t, r))

    n_full = pd.read_csv(os.path.join(root, "rows_with_ratio.csv"), low_memory=False).algo.nunique()
    n_real = pd.read_csv(os.path.join(root, "real_rows_with_ratio.csv"), low_memory=False).graph.nunique()

    # The gate. Each of these can genuinely fail.
    sub = {e[0] for e in harness.build_suite(0)} - harness.DROP_LARGE
    full = {e[0] for e in harness.build_suite(0)}
    if not sub < full:
        raise SystemExit("FATAL: the large-grid sub-suite is not a STRICT subset of the full suite. The "
                         "abstract distinguishes the two counts; if they coincide the distinction is a lie.")
    if n_full != len(full):
        raise SystemExit(f"FATAL: the small array ran {n_full} algorithms but harness.build_suite defines "
                         f"{len(full)}. The campaign and the code that generated it disagree.")
    if len(seen) != 8 or tasks <= 0 or rows <= 0:
        raise SystemExit(f"FATAL: expected 8 arrays with a positive total; got {len(seen)}, "
                         f"{tasks} tasks, {rows} rows.")

    print("  campaign totals (the abstract's numbers, derived):")
    for name, t, r in seen:
        print(f"    {name:<20} {t:>6} tasks  {r:>7} (task, algo) rows")
    print(f"    {'TOTAL':<20} {tasks:>6} tasks  {rows:>7} rows   "
          f"[full suite {len(full)}, large-grid sub-suite {len(sub)}, {n_real} real graphs]")

    return "\n".join([
        r"% --- the campaign, summed over the EIGHT delivered arrays (what the ABSTRACT quotes) ---",
        r"% \invNAlgosFull = the WHOLE suite (what ran the campaign). \invNAlgos = the large-grid sub-suite",
        r"% that Section 5 compares head to head. They are DIFFERENT NUMBERS. Do not swap them.",
        r"\newcommand{\invNAlgosFull}{%d}" % len(full),
        r"\newcommand{\invNTasks}{%s}" % _n(tasks),
        r"\newcommand{\invNRows}{%s}" % _n(rows),
        r"\newcommand{\invNArrays}{%d}" % len(seen),
        r"\newcommand{\invNRealGraphs}{%d}" % n_real,
    ])


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
    ap = argparse.ArgumentParser(description="Emit tab:invert + the Section 4 prose macros, gated.")
    ap.add_argument("--outdir", default="analysis")
    ap.add_argument("--dense", default=DENSE_CSV)
    ap.add_argument("--sparse", default=SPARSE_CSV)
    ap.add_argument("--costlaw", default="analysis/cost_law.csv")
    ap.add_argument("--edit", default="analysis/rgg/rgg_rows_with_ratio.csv")
    ap.add_argument("--groups", default="rgg", choices=["rgg", "all"],
                    help="rgg = the two RGG columns only (Section 5 rests on the RGG); "
                         "all = keep the dense pair alongside")
    # Write the two generated .tex STRAIGHT into the paper directory, wherever that lives. The alternative --
    # emit into analysis/ and copy them next to the paper by hand -- is a step that rots the moment the paper
    # moves, and it rots SILENTLY: LaTeX happily compiles a stale table. Point this at the directory holding
    # story.tex and the numbers cannot drift from the campaign.
    ap.add_argument("--texdir", default=None,
                    help="the paper's tables/ directory. tab_invert.tex and inversion_macros.tex "
                         "are written here as well as to --outdir.")
    args = ap.parse_args()
    set_groups(args.groups)

    raw = {"dense": load(args.dense, "dense"), "sparse": load(args.sparse, "sparse")}
    truth = {f: grid_truth(f) for f in FAMILIES}
    uni, dstats = {}, {}
    for f in FAMILIES:
        uni[f], dstats[f] = dedup(raw[f])
    stats = {f: per_algo(uni[f], truth[f][0]) for f in FAMILIES}
    cover, bound = classify(stats, uni)

    # THE FOUR GROUPS. The pooled `stats` above stay only to feed the gate (which checks the CSV against the
    # harness grid, and the grid is per-CSV) and the deprecated pooled macros. The TABLE is built from these.
    print("\nSPLITTING INTO GROUPS (see the GROUPS note at the top of this file)")
    gunis, gsizes = split_groups(uni)
    gstats = {k: per_algo(gunis[k], gsizes[k]) for k in GKEYS}
    for k in GKEYS:
        print(f"  {k:<9} {gsizes[k]:>4} tasks   |H|/m = {fam_hm(gunis[k]):.3f}")

    fails, warns = gate(raw, uni, stats, truth, dstats, cover, bound)
    if fails:
        raise SystemExit(f"\n*** GATE FAILED: {len(fails)} invariant(s) violated ***\n  "
                         + "\n  ".join(fails)
                         + "\n\nNOT writing. The previous .tex is left on disk: a paper should keep its old "
                           "numbers rather than gain untrustworthy new ones.")

    # I15  THE IDENTITY. |S|/m == rho_H * (|H|/m), exactly, row by row. This is not a tautology of the
    #      table: rho_H comes from the CSV's own `ratio_domr` column, computed by the analyzer, while |S|/m
    #      and |H|/m are recomputed here from `size`, `H` and `E`. If the analyzer ever divided by the wrong
    #      quantity -- or if `E` is not the edge count the ratio was built against -- these disagree. It has
    #      to be checked before either new column is printed, because both columns rest on it.
    for f in FAMILIES:
        K = returning(uni[f]).dropna(subset=["size", "H", "E", "ratio_domr"])
        K = K[K.E.gt(0) & K.H.gt(0)]
        if not len(K):
            raise SystemExit(f"FATAL (I15): no rows to check the |S|/m identity on in the {f} family. "
                             "A check that cannot run has not passed.")
        lhs = K["size"] / K.E
        rhs = K.ratio_domr * (K.H / K.E)
        worst = float((lhs - rhs).abs().max())
        print(f"  [{'PASS' if worst < 1e-9 else 'FAIL'}] I15 |S|/m == rho_H*(|H|/m) [{f:<6}] "
              f"max |delta| = {worst:.2e} over {len(K)} rows")
        if worst >= 1e-9:
            raise SystemExit(f"*** GATE FAILED (I15): the {f} family's rho_H is not |S|/|H| as computed "
                             f"from size/H/E (max delta {worst:.2e}). The two new columns would be lying. ***")

    tex = emit_tex(gstats, gsizes, cover, bound, gunis)
    macros = "\n".join([emit_macros(stats, uni, truth, cover, bound),
                        emit_costlaw(args.costlaw),
                        emit_domr_edit(args.edit),
                        emit_matched(gunis),
                        emit_campaign()])

    # I11  the LAST check, run on the string that actually reaches the paper: no NaN/inf ever, and a dash
    #      appears IFF an algorithm returned no verified cover -- never for any other reason.
    if re.search(r"(?i)(?<![a-z])(nan|inf)(?![a-z])", tex) or \
       re.search(r"(?i)(?<![a-z])(nan|inf)(?![a-z])", macros):
        raise SystemExit("*** GATE FAILED (I11): rendered LaTeX contains nan/inf. NOT writing. ***")
    # One dash per (algorithm, GROUP) cell where the method returned no verified cover -- and never for any
    # other reason. The count is derived from gstats, so it re-derives itself if the group set ever changes;
    # it is not a constant that would have to be remembered.
    n_dash = tex.count(r"\code{--}")
    n_expect = 0
    empty = []
    for k in GKEYS:
        for al in cover:
            if al not in gstats[k].index:
                n_expect += 2                      # the whole cell pair is absent
                empty.append((k, al, "absent"))
            elif int(gstats[k].loc[al].n_ok) == 0:
                n_expect += 1                      # ret. prints 0.0%; only |S|/m dashes
                empty.append((k, al, "no cover"))
    if n_dash != n_expect:
        raise SystemExit(f"*** GATE FAILED (I11): {n_dash} dashes rendered, {n_expect} expected over "
                         f"{len(empty)} empty (group, algo) cells: {empty}. A dash must mean 'returned "
                         f"nothing' and nothing else. NOT writing. ***")
    print(f"  [PASS] I11 rendered LaTeX clean                   {n_dash} dashes over "
          f"{len(empty)} empty cells, all accounted for")
    for k, al, why in empty:
        print(f"         {k:<9} {al:<18} {why}")

    os.makedirs(args.outdir, exist_ok=True)
    tidy = pd.concat([gstats[k].assign(group=k) for k in GKEYS]).reset_index()
    tidy["kind"] = np.where(tidy.algo.isin(bound), "bound", "cover")
    tidy.to_csv(os.path.join(args.outdir, "summary_inversion.csv"), index=False)

    dests = [args.outdir]
    if args.texdir:
        dests.append(paper_dir(args.texdir))

    for d in dests:
        for name, body in (("tab_invert.tex", tex), ("inversion_macros.tex", macros)):
            with open(os.path.join(d, name), "w") as fh:
                fh.write(body + "\n")

    print("\n" + tex + "\n")
    print(macros + "\n")
    print(f"wrote {args.outdir}/tab_invert.tex, {args.outdir}/inversion_macros.tex, "
          f"{args.outdir}/summary_inversion.csv")
    if args.texdir:
        print(f"wrote {args.texdir}/tab_invert.tex, {args.texdir}/inversion_macros.tex  <- the paper")
    if warns:
        print(f"\nNOTE: {len(warns)} algo(s) had ok-but-unverified runs. The table reports returned-AND-"
              f"verified, which is what its caption promises. See I13 above.")


if __name__ == "__main__":
    main()
