"""gmr_vs_iomr.py -- the GMR-vs-IOMR approximation-ratio figure (Workstream 2).

The question this figure answers: on the SAME instance, does restricting to increase-only repair (IOMR) cost
you more than general repair (GMR)? That comparison is only meaningful where BOTH reference optima are exact,
so every panel conditions on `gmr_ref_kind == iomr_ref_kind == "exact"` -- 3,085 of the 5,000 RGG tasks. That
conditioning is itself a selection: IOMR's ILP is the binding constraint, so the surviving tasks are the EASY
ones. The panels say so, in a note, on the figure.

Three things here are easy to get wrong, and all three were verified against the data:

  * `rgg_rows_with_ratio.csv` is LONG-FORMAT -- Part-2 (kNN downstream) emits one row per knn_k in {5,10,20,50},
    so a task/algo pair appears up to 4 times. Without drop_duplicates(["task","algo"]) every quantile is
    silently mis-weighted (175,218 rows -> 90,000).
  * DOMR STAYS IN. `rgg_analyze.py:44` -- `GMR_REF_VARIANTS = {"GMR","DOMR"}` -- so a DOMR row's `ratio` shares
    the GMR denominator, and `plot_common.FAMILY` files DOMR inside the GMR family. Its median ratio (1.0909)
    beats every other box in the figure. Dropping it would make the panel assert "the polynomial GMR frontier
    is ~1.39", which is false.
  * `iomr_regiongrow` is unpaired BY DESIGN (`harness.py:210`: "region growing is an IOMR light-edge
    construction") and is the only ragged sample -- 2,411 of 3,085, the rest `status="skipped_H"`. A pair-keyed
    loop would either KeyError or quietly drop an IOMR algorithm from a GMR-vs-IOMR figure.
  * EVERY DATASET COMPUTES ITS OWN PAIRED DELTAS. `pair_deltas` is pure and is called once per frame; nothing
    from rgg_full's deltas may reach an rgg_mixed or real row. The datasets genuinely disagree -- on rgg_mixed
    IOMR beats GMR on 100% of matched tasks under pivot/left_edge (Δ = -20.84), while on rgg_full it loses
    (Δ = +0.69) -- so a shared delta is not a rounding error, it inverts the claim. The `real` frame collapses
    seeds to a per-graph median before pivoting, so its delta is a delta OF MEDIANS: the CSV says so in
    `pair_delta_kind`.

Colour encodes the MR FAMILY, not the algorithm: `plot_common.style_map` cycles 7 hues over 16 algorithms, so
`#0072B2` would be shared by gmr_bestofk, iomr_regiongrow and spc_iomr -- destroying the one contrast the
figure exists to show.

Run:  sage -python experiments/gmr_vs_iomr.py
Out:  analysis/figs/rgg/fig_gmr_vs_iomr_ratio.{pdf,png}
      analysis/summary_gmr_vs_iomr.csv
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")                        # non-interactive: save files, no display
import matplotlib.pyplot as plt              # noqa: E402
import numpy as np                           # noqa: E402
import pandas as pd                          # noqa: E402

from plot_common import _COLORS, family_of, ylab, figure_legend, note, save   # noqa: E402

# Two hues drawn from the Okabe-Ito order in plot_common -- one per MR family. NOT style_map (see docstring).
FAM_COLOR = {"gmr": _COLORS[0], "iomr": _COLORS[1]}          # #0072B2 blue / #D55E00 vermillion
FAM_LABEL = {"gmr": "GMR family (GMR, DOMR) — any-direction edits",
             "iomr": "IOMR family — increase-only edits"}
DIM = "0.45"                                                 # the exact (constant-1.0) algorithms

# The 7 GMR<->IOMR pairs. pivot/left_edge are each other's counterpart (harness.py): pivot is the GMR
# light-edge heuristic, left_edge the IOMR one.
PAIRS = [("ilp",             "gmr_ilp",        "iomr_ilp"),
         ("bestofk",         "gmr_bestofk",    "iomr_bestofk"),
         ("rand",            "gmr_rand",       "iomr_rand"),
         ("thr_naive",       "gmr_thr_naive",  "iomr_thr_naive"),
         ("l1sep",           "l1sep_gmr",      "l1sep_iomr"),
         ("spc",             "spc_gmr",        "spc_iomr"),
         ("pivot /\nleft_edge", "pivot",       "left_edge")]

# Unpaired by construction. DOMR has no IOMR counterpart (decrease-only lives in the GMR family);
# iomr_regiongrow has no GMR counterpart.
UNPAIRED = [("domr", "domr"), ("region\ngrow", "iomr_regiongrow")]

EXACT_ALGOS = {"gmr_ilp", "iomr_ilp"}        # the constant 1.0 -- they DEFINE the reference
LP_NAIVE = {"gmr_lp_naive", "iomr_lp_naive"}  # the reference's own LP relaxation, not a competing heuristic
FAMILY_VARIANTS = ["GMR", "IOMR", "DOMR"]

WHIS = (5, 95)                               # whiskers at the 5th/95th pct; fliers hidden (tails are in the CSV)


# ---------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------
def load_rgg(path, dedup=True, verify=False):
    """Load an RGG rows_with_ratio.csv and cut it down to the both-exact, valid, heuristic rows."""
    df = pd.read_csv(path)
    n_raw = len(df)
    if dedup:
        df = df.drop_duplicates(["task", "algo"])            # long-format: one row per knn_k in {5,10,20,50}
    n_dedup = len(df)

    both = df[(df["gmr_ref_kind"] == "exact") & (df["iomr_ref_kind"] == "exact")]
    n_tasks = both["task"].nunique()

    # Both references are exact => every row's OWN reference (GMR rows take gmr_ref, IOMR rows iomr_ref) is exact.
    assert (both["ref_kind"] == "exact").all(), "both-exact filter leaked a non-exact per-row reference"

    ok = both[(both["status"] == "ok") & (both["valid"] == 1) & both["ratio"].notna()]
    ok = ok[~ok["algo"].isin(LP_NAIVE)]
    hh = ok[ok["variant"].isin(FAMILY_VARIANTS)].copy()      # KEEP DOMR
    hh["family"] = hh["variant"].map(family_of)

    print(f"  {os.path.basename(os.path.dirname(path)) or path}: "
          f"{n_raw} raw -> {n_dedup} deduped -> {n_tasks} both-exact tasks -> {len(hh)} plotted rows")
    if verify:
        assert n_raw == 175218, f"expected 175218 raw rows, got {n_raw}"
        assert n_dedup == 90000, f"expected 90000 deduped rows, got {n_dedup}"
        assert n_tasks == 3085, f"expected 3085 both-exact tasks, got {n_tasks}"
        for a in sorted(EXACT_ALGOS):
            med = hh.loc[hh["algo"] == a, "ratio"].median()
            # The ILPs ARE the reference on the both-exact set. If this is not exactly 1.0 the reference logic
            # has been misapplied and every ratio in the figure is wrong -- stop rather than plot a lie.
            assert med == 1.0, f"{a} median ratio is {med!r}, not exactly 1.0 -- reference logic misapplied"
        print(f"    verified: 175218 -> 90000 dedup; 3085 both-exact; ref_kind all exact; "
              f"gmr_ilp/iomr_ilp medians exactly 1.00")
    return hh, n_tasks


def load_real(path):
    """Real graphs: no `task` column and no long-format kNN blow-up; several seeds per (graph, algo) under the
    randomized modes, so collapse to the per-graph median ratio."""
    df = pd.read_csv(path)
    both = df[(df["gmr_ref_kind"] == "exact") & (df["iomr_ref_kind"] == "exact")]
    assert (both["ref_kind"] == "exact").all(), "real: both-exact filter leaked a non-exact reference"
    ok = both[(both["status"] == "ok") & (both["valid"] == 1) & both["ratio"].notna()]
    ok = ok[~ok["algo"].isin(LP_NAIVE)]
    hh = ok[ok["variant"].isin(FAMILY_VARIANTS)]
    per = hh.groupby(["graph", "algo", "variant"])["ratio"].median().reset_index()
    per["family"] = per["variant"].map(family_of)
    print(f"  real: {both['graph'].nunique()} both-exact graphs -> {per['graph'].nunique()} plotted")
    return per


def _series(hh, algo):
    return hh.loc[hh["algo"] == algo, "ratio"].dropna().to_numpy(dtype=float)


# ---------------------------------------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------------------------------------
def _box(ax, data, pos, fam, width, dim=False):
    """One box. `dim` de-emphasises an algorithm drawn only for completeness (the exact ones, pinned at 1.0),
    following the plots.py:93-111 idiom -- de-emphasise and NAME it, never silently drop it."""
    if len(data) == 0:
        return
    color = DIM if dim else FAM_COLOR[fam]
    bp = ax.boxplot([data], positions=[pos], widths=width, whis=WHIS, showfliers=False,
                    patch_artist=True, showmeans=True, manage_ticks=False,
                    medianprops=dict(color="black", lw=1.3 if not dim else 0.9),
                    meanprops=dict(marker="d", ms=4, mfc="white", mec="black", mew=0.7),
                    boxprops=dict(facecolor=color, edgecolor=color, alpha=0.35 if dim else 0.75, lw=0.9),
                    whiskerprops=dict(color=color, lw=0.9, alpha=0.5 if dim else 1.0),
                    capprops=dict(color=color, lw=0.9, alpha=0.5 if dim else 1.0))
    if dim:
        # A boxplot of a CONSTANT collapses to a hairline that vanishes onto the axhline at 1.0 on a log axis.
        # Draw an explicit marker so the correctness check is visible rather than merely true.
        ax.plot([pos], [np.median(data)], marker="D", ms=6, mfc="white", mec=DIM, mew=1.4, zorder=5)
    return bp


def panel_grouped(ax, hh, n_tasks, title, sel_note=True):
    """Panel 1: grouped box of the per-task ratio, GMR box beside IOMR box within each pair. Log y."""
    xt, xl, pos = [], [], 0.0
    off, w = 0.19, 0.32
    for label, ga, ia in PAIRS:
        dim = ga in EXACT_ALGOS
        _box(ax, _series(hh, ga), pos - off, "gmr", w, dim=dim)
        _box(ax, _series(hh, ia), pos + off, "iomr", w, dim=dim)
        xt.append(pos); xl.append(label); pos += 1.0

    div = pos - 0.5
    ax.axvline(div, color="0.75", lw=0.9, ls=(0, (4, 3)), zorder=0)
    ax.text(div + 0.04, 0.985, "unpaired\nby construction", transform=ax.get_xaxis_transform(),
            va="top", ha="left", fontsize=6.5, color="0.4")

    for label, algo in UNPAIRED:
        d = _series(hh, algo)
        fam = family_of(hh.loc[hh["algo"] == algo, "variant"].iloc[0]) if len(d) else "gmr"
        _box(ax, d, pos, fam, w * 1.25)
        if len(d) < n_tasks:                                 # iomr_regiongrow: 2411/3085, 674 skipped_H
            # axes-fraction y: this runs BEFORE set_yscale("log"), so a data-coord y would be misplaced.
            n_skip = n_tasks - len(d)
            ax.text(pos, 0.02, f"n={len(d)}/{n_tasks}\n({n_skip} skipped_H)",
                    transform=ax.get_xaxis_transform(), va="bottom", ha="center", fontsize=6, color="0.35")
        xt.append(pos); xl.append(label); pos += 1.0

    ax.set_yscale("log")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.55, zorder=1)
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=8)
    ax.set_xlim(-0.75, pos - 0.35)
    ax.set_ylabel(ylab("approx. ratio (|S| / OPT)", "down"))
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", which="both", alpha=0.18, lw=0.5)

    note(ax, "gmr_ilp / iomr_ilp: exactly 1.0 on all "
             f"{n_tasks:,} both-exact tasks by construction —\nthey DEFINE the reference (drawn dimmed ◇, "
             "not omitted; a constant is invisible on log-y).", loc="upper left")
    if sel_note:
        # BELOW the axis, not inside it: an in-axes note at "lower right" lands squarely on the DOMR box --
        # the one box this figure most needs to show.
        med_d = np.median(_series(hh, "domr")) if len(_series(hh, "domr")) else float("nan")
        mean_d = float(np.mean(_series(hh, "domr"))) if len(_series(hh, "domr")) else float("nan")
        ax.text(0.5, -0.185,
                "SELECTION BIAS: conditioning on both ILPs converging conditions on instance easiness — IOMR's ILP is the "
                "binding constraint (1,700 RGG\ntasks have an exact GMR reference but only an IOMR LP bound), so these "
                "ratios are measured on the EASY tasks. Boxes: median + IQR,\nwhiskers 5–95th pct, ◇ = mean, fliers hidden "
                f"(tails in summary_gmr_vs_iomr.csv). DOMR is median-dominant but heavy-tailed: {med_d:.2f} vs {mean_d:.2f}.",
                transform=ax.transAxes, va="top", ha="center", fontsize=7.5, color="0.2",
                bbox=dict(boxstyle="round,pad=0.4", fc="#fbf7ef", ec="0.7", alpha=0.95))


def pair_deltas(hh):
    """PURE. The matched paired delta ratio_IOMR - ratio_GMR, per pair, FOR THE FRAME IT IS GIVEN.

    This is deliberately separated from the drawing (`panel_delta`) and from the summary (`pair_summary`).
    An earlier version coupled the three, so the deltas computed once for rgg_full were then written into the
    rgg_mixed and real rows of the summary CSV as though they had been measured there. They had not: several
    were SIGN-INVERTED against the truth. Compute this ONCE PER DATASET, never reuse across datasets.

    The pairing itself is only sound because the both-exact filter yields MATCHED instances: same task, same
    graph, same corruption, both references exact. The index column is `task` -- for the real frame the caller
    renames `graph` -> `task`, so a "task" there is a GRAPH and each cell is already a median over seeds
    (`load_real`). That makes the real delta a delta OF MEDIANS, and it is labelled as such in the CSV
    (`pair_delta_kind`) rather than passed off as a per-task delta.
    """
    piv = hh.pivot_table(index="task", columns="algo", values="ratio")
    out = {}
    for label, ga, ia in PAIRS:
        if ga in piv.columns and ia in piv.columns:
            out[label] = (piv[ia] - piv[ga]).dropna().to_numpy(dtype=float)
        else:
            out[label] = np.empty(0, dtype=float)              # a pair absent from this dataset -> honest empty
    return out


def pair_summary(deltas, kind):
    """The per-pair statistics of ONE dataset's deltas. `kind` records what a single observation IS:
    `per_task` (rgg) or `per_graph_median_over_seeds` (real) -- a delta of medians, not a median of deltas."""
    rows = []
    for label, ga, ia in PAIRS:
        d = deltas[label]
        rows.append((label, ga, ia,
                     float(np.median(d)) if len(d) else np.nan,
                     float((d < 0).mean()) if len(d) else np.nan,
                     len(d), kind))
    return pd.DataFrame(rows, columns=["pair", "gmr_algo", "iomr_algo",
                                       "pair_delta_median", "pair_iomr_better_frac", "pair_n",
                                       "pair_delta_kind"])


def panel_delta(ax, deltas, n_tasks):
    """Panel 2: draws the paired deltas of `deltas` (from `pair_deltas`). Drawing only -- no statistics are
    computed here, so nothing computed here can leak into another dataset's CSV rows."""
    xt, xl, pos = [], [], 0.0
    for label, ga, ia in PAIRS:
        d = deltas[label]
        dim = ga in EXACT_ALGOS
        color = DIM if dim else "#4d4d4d"
        ax.boxplot([d], positions=[pos], widths=0.45, whis=WHIS, showfliers=False, patch_artist=True,
                   manage_ticks=False,
                   medianprops=dict(color="black", lw=1.3),
                   boxprops=dict(facecolor=color, edgecolor=color, alpha=0.3 if dim else 0.6, lw=0.9),
                   whiskerprops=dict(color=color, lw=0.9), capprops=dict(color=color, lw=0.9))
        if dim:
            ax.plot([pos], [0.0], marker="D", ms=6, mfc="white", mec=DIM, mew=1.4, zorder=5)
        frac = float((d < 0).mean()) if len(d) else np.nan
        ax.text(pos, 0.985, f"{100 * frac:.0f}%", transform=ax.get_xaxis_transform(),
                va="top", ha="center", fontsize=7,
                color=FAM_COLOR["iomr"] if frac > 0.5 else FAM_COLOR["gmr"])
        xt.append(pos); xl.append(label); pos += 1.0

    # pivot/left_edge deltas reach ±40 while l1sep's sit inside ±0.1: symlog is the only scale that shows both.
    # linscale=1.8 buys the linear |Δ|<1 band -- where every pair except pivot/left_edge lives -- extra room.
    ax.set_yscale("symlog", linthresh=1.0, linscale=1.8)
    ax.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.55)
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=8)
    ax.set_xlim(-0.7, pos - 0.3)
    ax.set_ylabel(ylab("paired Δ = ratio$_{IOMR}$ − ratio$_{GMR}$\n(same task)", "down"))
    ax.set_title("Panel 2 — paired per-task cost of increase-only repair "
                 f"(matched instances, n={n_tasks:,}); % = tasks where IOMR wins", fontsize=10)
    ax.grid(axis="y", which="both", alpha=0.18, lw=0.5)
    note(ax, "Δ < 0: the increase-only restriction is FREE on that task (IOMR's heuristic beat GMR's).\n"
             "ilp: Δ ≡ 0 by construction (dimmed ◇). Symlog y (linear inside ±1). Unpaired algorithms\n"
             "(domr, iomr_regiongrow) are excluded here — they have no counterpart to subtract.",
         loc="lower left")


def panel_real(ax, per):
    """Real graphs: a paired dot-strip -- one dot per (graph, algorithm), the two members of a pair joined, so
    a downward line = the increase-only restriction was free on that graph."""
    xt, xl, pos = [], [], 0.0
    rng = np.random.default_rng(0)
    for label, ga, ia in PAIRS:
        g = per[per["algo"] == ga].set_index("graph")["ratio"]
        i = per[per["algo"] == ia].set_index("graph")["ratio"]
        common = sorted(set(g.index) & set(i.index))
        for gr in common:
            j = rng.uniform(-0.05, 0.05)
            ax.plot([pos - 0.16 + j, pos + 0.16 + j], [g[gr], i[gr]], color="0.6", lw=0.6, alpha=0.7, zorder=1)
            ax.plot([pos - 0.16 + j], [g[gr]], marker="o", ms=4, color=FAM_COLOR["gmr"], alpha=0.85, zorder=2,
                    mec="white", mew=0.4)
            ax.plot([pos + 0.16 + j], [i[gr]], marker="o", ms=4, color=FAM_COLOR["iomr"], alpha=0.85, zorder=2,
                    mec="white", mew=0.4)
        xt.append(pos); xl.append(label.replace("\n", " ")); pos += 1.0
    ax.set_yscale("log")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.55)
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=7, rotation=30, ha="right")
    ax.set_xlim(-0.7, pos - 0.3)
    ax.set_ylabel(ylab("approx. ratio (|S| / OPT)", "down"), fontsize=8)
    ax.set_title(f"Real graphs — {per['graph'].nunique()} both-exact graphs (paired)", fontsize=9)
    ax.grid(axis="y", which="both", alpha=0.18, lw=0.5)


# ---------------------------------------------------------------------------------------------------------
# summary CSV
# ---------------------------------------------------------------------------------------------------------
def summarize(hh, n_tasks, dataset, delta):
    """One row per algorithm. Carries BOTH `ratio` (|S|/OPT, what the figure plots) and `ratio_domr` (|S|/|H|),
    plus the per-task median quotient against DOMR -- that quotient is the provenance the paper's decrease-only
    premium needs, and it must come from a matched per-task comparison, not from a quotient of two medians.

    `delta` MUST be `pair_summary(pair_deltas(hh), ...)` -- the deltas of THIS `hh`, never another dataset's.
    Every column emitted here is therefore a measurement of `dataset` and of nothing else."""
    piv = hh.pivot_table(index="task", columns="algo", values="ratio") if "task" in hh else None
    out = []
    order = [a for _l, g, i in PAIRS for a in (g, i)] + [a for _l, a in UNPAIRED]
    dmap = {r.gmr_algo: r for r in delta.itertuples()} if delta is not None else {}
    dmap.update({r.iomr_algo: r for r in delta.itertuples()} if delta is not None else {})
    for algo in order:
        s = hh[hh["algo"] == algo]
        if s.empty:
            continue
        r = s["ratio"].dropna()
        rec = dict(dataset=dataset, algo=algo, variant=s["variant"].iloc[0],
                   family=s["family"].iloc[0],
                   paired=("yes" if algo in dmap else "no"),
                   pair=(dmap[algo].pair.replace("\n", " ") if algo in dmap else ""),
                   n=len(r), n_tasks_total=n_tasks,
                   ratio_min=r.min(), ratio_q25=r.quantile(.25), ratio_median=r.median(),
                   ratio_mean=r.mean(), ratio_q75=r.quantile(.75), ratio_max=r.max(), ratio_std=r.std())
        if "ratio_domr" in s:
            rd = s["ratio_domr"].dropna()
            rec["ratio_domr_median"] = rd.median() if len(rd) else np.nan
            rec["ratio_domr_mean"] = rd.mean() if len(rd) else np.nan
        if piv is not None and "domr" in piv and algo != "domr":
            q = (piv[algo] / piv["domr"]).replace([np.inf, -np.inf], np.nan).dropna()
            rec["median_ratio_vs_domr"] = q.median() if len(q) else np.nan   # per-task, matched
        if algo in dmap:
            rec["pair_delta_median"] = dmap[algo].pair_delta_median
            rec["pair_iomr_better_frac"] = dmap[algo].pair_iomr_better_frac
            rec["pair_n"] = dmap[algo].pair_n
            rec["pair_delta_kind"] = dmap[algo].pair_delta_kind      # per_task | per_graph_median_over_seeds
        out.append(rec)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgg", default="analysis/rgg/rgg_rows_with_ratio.csv")
    ap.add_argument("--mixed", default="analysis/rgg_mixed/rgg_rows_with_ratio.csv")
    ap.add_argument("--geo", default="analysis/rows_with_ratio.csv")
    ap.add_argument("--real", default="analysis/real_rows_with_ratio.csv")
    ap.add_argument("--outdir", default="analysis/figs/rgg")
    ap.add_argument("--csv", default="analysis/summary_gmr_vs_iomr.csv")
    a = ap.parse_args()

    print("loading:")
    hh, n_tasks = load_rgg(a.rgg, verify=True)               # verify=True -> the WS2 assertions, hard-stop

    hm, nm = (None, 0)
    if os.path.exists(a.mixed):
        hm, nm = load_rgg(a.mixed)
    per = load_real(a.real) if os.path.exists(a.real) else None

    # geo_small: DEGENERATE under this filter -- the both-exact set selects the smallest instances, on which 8
    # of the 17 algorithms sit at median ratio 1.0 exactly. A boxplot of that says nothing. Footnote it.
    geo_txt = ""
    if os.path.exists(a.geo):
        g = pd.read_csv(a.geo).drop_duplicates(["task", "algo"])
        gb = g[(g["gmr_ref_kind"] == "exact") & (g["iomr_ref_kind"] == "exact")]
        gok = gb[(gb["status"] == "ok") & (gb["valid"] == 1) & gb["ratio"].notna()]
        gok = gok[~gok["algo"].isin(LP_NAIVE) & gok["variant"].isin(FAMILY_VARIANTS)]
        med = gok.groupby("algo")["ratio"].median()
        flat = int((med <= 1.0).sum())
        geo_txt = (f"geo_small is degenerate under the both-exact filter ({gb['task'].nunique()} tasks, exp2a/2b): "
                   f"{flat} of {len(med)} algorithms sit at median ratio 1.00 — not plotted.")
        print(f"  geo_small: {geo_txt}")

    # ---- figure ----
    fig = plt.figure(figsize=(13.0, 14.2))
    # top=0.945 pulls the axes up under the suptitle; the matplotlib default (~0.88) leaves ~1.6in of dead band
    # that bbox_inches="tight" faithfully preserves, because the suptitle anchors the top of the bbox.
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 0.85], hspace=0.52, wspace=0.22,
                          top=0.925, bottom=0.075)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])

    panel_grouped(ax1, hh, n_tasks,
                  f"Panel 1 — approximation ratio by MR family, RGG suite "
                  f"(both references exact: {n_tasks:,} of 5,000 tasks)")

    # One delta computation PER DATASET. Panel 2 draws rgg_full's; the mixed/real deltas are never drawn (their
    # panels are the half-width secondaries) but they ARE measured, from their own frames, for the CSV.
    d_full = pair_deltas(hh)
    panel_delta(ax2, d_full, n_tasks)
    delta_full = pair_summary(d_full, "per_task")

    if hm is not None:
        panel_grouped(ax3, hm, nm, f"rgg_mixed (secondary) — {nm} both-exact tasks", sel_note=False)
        ax3.set_xticklabels([t.get_text() for t in ax3.get_xticklabels()], fontsize=6.5, rotation=30, ha="right")
        ax3.set_ylabel(ylab("approx. ratio (|S| / OPT)", "down"), fontsize=8)
        ax3.set_title(f"rgg_mixed (secondary) — {nm} both-exact tasks", fontsize=9)
        for t in ax3.texts[:]:                               # the two long notes do not fit a half-width panel
            t.remove()
    if per is not None:
        panel_real(ax4, per)

    # family legend: proxy patches (boxplots register no handles of their own)
    for fam in ("gmr", "iomr"):
        ax1.fill_between([], [], [], color=FAM_COLOR[fam], alpha=0.75, label=FAM_LABEL[fam])
    ax1.plot([], [], marker="D", ms=6, mfc="white", mec=DIM, mew=1.4, ls="none",
             label="exact ILP — constant 1.0 (defines OPT)")
    ax1.plot([], [], marker="d", ms=5, mfc="white", mec="black", mew=0.7, ls="none", label="mean")
    leg = figure_legend(fig, title="MR family", ncol=1)

    fig.suptitle("GMR vs IOMR — what does the increase-only restriction cost?\n"
                 "RGG suite | both references exact (gmr_ref_kind = iomr_ref_kind = \"exact\") | "
                 "DOMR is drawn in the GMR family (its covers are valid GMR covers, rgg_analyze.py:44)",
                 fontsize=11.5, y=0.988)
    if geo_txt:
        fig.text(0.5, 0.012, geo_txt, ha="center", fontsize=7, color="0.35")
    save(fig, a.outdir, "fig_gmr_vs_iomr_ratio", legend=leg)

    # ---- csv ----
    frames = [summarize(hh, n_tasks, "rgg_full", delta_full)]
    if hm is not None:
        frames.append(summarize(hm, nm, "rgg_mixed", pair_summary(pair_deltas(hm), "per_task")))
    if per is not None:
        # `real` collapses seeds to a per-graph median BEFORE pivoting, so its "paired delta" is a delta of
        # medians over the 11 both-exact graphs -- weaker than the rgg per-task delta, and labelled so.
        per2 = per.rename(columns={"graph": "task"})
        frames.append(summarize(per2, per["graph"].nunique(), "real",
                                pair_summary(pair_deltas(per2), "per_graph_median_over_seeds")))
    out = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
    out.to_csv(a.csv, index=False)
    print(f"\n  wrote {a.csv}  ({len(out)} rows)")
    print("\nrgg_full medians (the figure's Panel 1):")
    m = out[out["dataset"] == "rgg_full"][["algo", "family", "n", "ratio_median", "ratio_mean",
                                           "median_ratio_vs_domr"]].sort_values("ratio_median")
    print(m.to_string(index=False))

    # Print the paired statistics PER DATASET. They differ -- sharply, and in sign -- and the CSV must show that.
    print("\npaired delta (ratio_IOMR - ratio_GMR) BY DATASET — computed independently from each frame:")
    pc = out[out["paired"] == "yes"][["dataset", "pair", "algo", "pair_delta_median",
                                      "pair_iomr_better_frac", "pair_n", "pair_delta_kind"]]
    pc = pc.drop_duplicates(["dataset", "pair"]).sort_values(["pair", "dataset"])
    print(pc.drop(columns=["algo"]).to_string(index=False))


if __name__ == "__main__":
    main()
