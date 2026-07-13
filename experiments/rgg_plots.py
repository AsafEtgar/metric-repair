"""rgg_plots.py -- the RGG figures from the summaries (rgg_analyze.py output).

Runs LOCALLY; needs matplotlib + pandas. On this Mac use sage's Python (it bundles both):

    sage -python experiments/rgg_plots.py --edit analysis/summary_edit.csv \
        --knn analysis/summary_knn.csv --outdir analysis/figs/rgg

Figures are written split by MR axis into per-family subfolders of --outdir:
    <outdir>/gmr/   GMR (general metric repair): variants {GMR, DOMR}
    <outdir>/iomr/  IOMR (increase-only metric repair): variant {IOMR}
GMR and IOMR are different problems (different covers, different optima), so they never share a panel.

Styling is shared with plots.py via plot_common: every algorithm keeps one colour+marker+linestyle across all
figures, and ONE legend sits outside the axes (the per-panel legends used to bury the data -- S6 alone had 27
entries). The two-factor S6 sweep (magnitude x frac_q) is collapsed to one line per algorithm (median over
frac_q) so every panel reads the same way. Median lines only -- with up to ~10 algorithms per panel, IQR bands
overlap into mush; the spread is in the summary CSVs.

  Part 1 -- regime characterization (one panel per swept knob):
    fig_edit_recall     recall of the cover vs the planted broken edges   (up: found the corruption?)
    fig_edit_precision  precision of the cover vs the planted broken edges (up: cut ONLY corruption?)
    fig_ratio_domr      cover size / |H|  (down: lower is a tighter bound on the GMR approximation ratio)

  Part 2 -- kNN recovery (jitter breaks):
    fig_knn_jaccard     jaccard_TF (repaired) vs the damage baseline jaccard_TC   (up; panels: sweep x k)
    fig_knn_lift        lift = jaccard_TF - jaccard_TC   (up; >0 == repair helps the neighborhood graph)
    fig_triplet         triplet-ordering accuracy, repaired (F) vs corrupted (C)  (up; panels: sweep)

  Cost -- what the repair actually costs to run:
    fig_runtime_vs_n        CPU vs n on the size ladders, log-log       (down)
    fig_memory_vs_n         peak RSS vs n on the size ladders           (down)
    fig_cost_vs_corruption  CPU + peak RSS vs corruption severity       (down)

Each figure is PDF + PNG. Missing parts/families are skipped with a note rather than crashing.

COST FIGURES -- two things to know.

1. WHERE MEMORY COMES FROM. rgg_analyze summarises `cpu`/`wall` but NOT `peak_mb` (it sits in NUM, never in
   DIST_EDIT), so summary_edit.csv has no memory column at all. Memory therefore has to be aggregated here,
   from the per-task rows (--rows, default rgg_rows_with_ratio.csv beside --edit).

2. THE LONG-FORMAT FOOTGUN. Those rows are LONG: Part 2 emits one row per knn_k, so a Part-2 task repeats its
   cover-level columns (cpu, peak_mb, size, ...) once per k -- 4x in this grid. Median them as they lie and
   every Part-2 task is silently counted 4:1 against a Part-1 task. load_rows() collapses to one row per
   (task, algo) FIRST -- the same `groupby(["task","algo"]).first()` rgg_analyze.aggregate_edit does -- and
   blanks invalid-cover rows the same way, so a point here rests on exactly the row set behind the summary's
   cpu_med. Timeouts are kept, their cost censored at the cap (this pipeline counts a failed run's cost
   rather than hiding it; see the survivorship note in rgg_analyze).

WHAT THE COST FIGURES CAN AND CANNOT SAY. The size ladders (S1/S1d) only ever ran at ONE corruption setting
(mag=3, frac=0.1). The extreme values -- frac up to 0.30, magnitude up to 10 -- exist ONLY at n=1000 (P2df,
P2dm). So "runtime at n=3000 under the worst corruption" is not in the data and is not plotted. The ladders
answer "how does cost grow with n (at a fixed, mild corruption)"; the corruption sweeps answer "does more
corruption cost more (at a fixed, small n)". Panel titles name the fixed knob so neither is misread as the
worst case.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")                       # non-interactive: save files, no display
import matplotlib.pyplot as plt             # noqa: E402
import numpy as np                          # noqa: E402
import pandas as pd                          # noqa: E402

from plot_common import (FAMILY, FAMILIES, FAM_TITLE, style_map, ylab, band,      # noqa: E402
                         figure_legend, note, save)

# stable panel order; label each panel by the knob it sweeps
# SWEEP_X / SWEEP_TITLE now live in experiments/sweeps.py, shared with rgg_analyze. They were duplicated,
# and the copy in rgg_analyze fell behind: it lacked S1d, S1m, S3m, S4m, P2df, P2dm, P2mf, P2mm and every
# RR_* sweep, so add_x() left x=NaN and those panels vanished from the figures without an error.
from sweeps import SWEEP_X, SWEEP_TITLE          # noqa: E402
ORDER = ["S1", "S1d", "S1m", "S2", "S2k", "S3", "S3d", "S3m", "S4i", "S4d", "S4m", "S5a", "S5b", "S5c", "S6",
         "POCsize_inflate", "POCsize_jitter", "P2size", "P2df", "P2dm", "P2mf", "P2mm", "P2s", "P2j", "P2n",
         "RR_inflate", "RR_deflate", "RR_mixed"]

# Cost figures. The size ladders sweep n at a FIXED corruption; the corruption sweeps sweep severity at a
# FIXED (small) n. Kept as ordered lists so a grid that ran only some of them plots only those (see _cost_sweeps).
SIZE_LADDERS = ["S1", "S1d", "S1m", "P2size"]        # x = n
CORRUPT_SWEEPS = ["P2df", "P2dm", "P2mf", "P2mm"]    # x = corruption severity (frac_q / magnitude)


def _family(df, fam):
    return df[df["variant"].isin(FAMILY[fam])] if len(df) else df


def load_rows(path):
    """Per-task rows -> ONE row per (task, algo), invalid covers blanked. Mirrors rgg_analyze.aggregate_edit.

    The rows CSV is long: Part 2 writes one row per knn_k, so a Part-2 task's cover-level columns (cpu, wall,
    peak_mb, size) repeat once per k. Aggregating without collapsing first weights those tasks 4:1 -- silently,
    and in the direction that flatters nothing in particular, which is why it is easy to miss. Collapse on
    (task, algo) exactly as the analyzer does, then blank the metrics of covers that do not verify, so a cost
    median here rests on the same rows as the summary's cpu_med."""
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    n_raw = len(df)
    if "knn_k" in df:
        df = df.sort_values("knn_k")
    one = df.groupby(["task", "algo"], as_index=False).first()
    if "valid" in one:
        inval = one["valid"] == 0
        if inval.any():
            one.loc[inval, [c for c in ("cpu", "wall", "size") if c in one]] = np.nan

    # ZERO IS A MISSING-VALUE SENTINEL HERE, NOT A MEASUREMENT. When a run is killed at the timeout the harness
    # never gets its accounting back: it writes cpu=0.0 and peak_mb=0.0, while `wall` correctly holds the full
    # cap (~1800 s). So a timed-out run looks INSTANTANEOUS and MEMORYLESS in exactly the cells where it was
    # neither. Median those zeros as they lie and the ranking inverts -- gmr_bestofk, the most expensive
    # algorithm in the grid, times out on the whole upper half of the inflate ladder and comes out at 0 s / 0 MB,
    # i.e. plotted as the cheapest. (This is also why summary_edit's cpu_med cannot carry the runtime figure: at
    # n=3000 on S1 it reports cpu_med=0 for gmr_bestofk, and even a MINORITY of timeouts drags the median down --
    # iomr_bestofk at n=1800 summarises to 861 s against 1231 s over the runs that actually finished.)
    #
    # Blank them. cpu and peak_mb are therefore reported over runs that COMPLETED, and the figures say so; where
    # an algorithm stopped completing is drawn explicitly (fig_runtime_vs_n's DNF markers) rather than left as a
    # line that just stops.
    for col in ("cpu", "peak_mb"):
        if col in one:
            zero = one[col] == 0
            if zero.any():
                by = one[zero].groupby("status").size().to_dict()
                print(f"rows: blanking {int(zero.sum())} {col}==0 sentinels (killed run, never reported) "
                      f"-- by status: {by}")
                one.loc[zero, col] = np.nan
    print(f"rows: {n_raw} long -> {len(one)} unique (task, algo) over {one['task'].nunique()} tasks"
          f"; cpu on {one['cpu'].notna().mean():.1%}, peak_mb on {one['peak_mb'].notna().mean():.1%}")
    return one


def _dnf(rows, sweep, xcol="n", thresh=0.5):
    """Where each algorithm STOPPED FINISHING: the smallest x at which the median run timed out.

    Without this a timed-out algorithm's curve simply ends, which reads as "no data" rather than "this is the
    expensive one". Returns {algo: (x, cap)} with cap the wall-clock timeout it ran into."""
    s = rows[rows["sweep"] == sweep]
    if s.empty or "status" not in s:
        return {}, None
    cap = s.loc[s["status"] == "timeout", "wall"].median()
    g = s.groupby([xcol, "algo"])["status"].apply(lambda z: (z == "timeout").mean()).reset_index(name="rate")
    out = {}
    for algo, gg in g[g["rate"] >= thresh].groupby("algo"):
        out[algo] = float(gg[xcol].min())
    return out, (float(cap) if np.isfinite(cap) else None)


def _agg(rows, xcol, ycol, min_pts=1):
    """median + IQR of a per-task quantity per (x, algo), in the column names band() expects.

    min_pts drops any point resting on fewer than that many samples -- the same guard main() applies to the
    summaries via --min-usable, and for the same reason. It matters most here: on the inflate ladder the
    bestofk algorithms time out on everything except the easiest few draws, so an unguarded median over the 1-2
    survivors slopes DOWNWARD with n. Memory does not fall as the graph grows; that line is pure survivorship."""
    d = rows.dropna(subset=[xcol, ycol])
    if d.empty:
        return pd.DataFrame()
    g = d.groupby([xcol, "algo"])[ycol]
    # NOT `n=` for the count: on the size ladders xcol IS "n", and reset_index() would then try to insert a
    # second column called n and die. n_pts is the sample count behind each median.
    out = g.agg(med="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75),
                n_pts="size").reset_index()
    out = out[out["n_pts"] >= min_pts]
    return out.rename(columns={xcol: "x", "med": f"{ycol}_med", "q25": f"{ycol}_q25", "q75": f"{ycol}_q75"})


def _share_y(axs, head=1.0):
    """Put a row of panels on ONE y-scale (they exist to be compared), with `head` x of headroom on top for an
    annotation. A multiplicative factor is right on both scales here: the log axes want a decade, and the linear
    memory axis just wants a sliver."""
    lims = [ax.get_ylim() for ax in axs]
    lo, hi = min(l[0] for l in lims), max(l[1] for l in lims)
    for ax in axs:
        ax.set_ylim(lo, hi * head)


def _cost_sweeps(rows, wanted):
    """The sweeps from `wanted` this grid actually has rows for -- rgg/full and rgg_largemix run different
    sweeps than rgg_large, so every cost figure guards on presence and skips cleanly instead of crashing."""
    if rows is None or rows.empty:
        return []
    have = set(rows["sweep"].dropna().unique())
    return [s for s in wanted if s in have]


def _fixed(rows, sweep):
    """Name the knobs a sweep held FIXED (read off the data, never hardcoded -- a different grid may pin them
    elsewhere). This is what stops the size ladder being read as a worst case: it ran at mag=3, frac=0.1 only."""
    s = rows[rows["sweep"] == sweep]
    bits = []
    for col, lab in (("n", "n"), ("magnitude", "mag"), ("frac_q", "frac"), ("deg", "deg")):
        if col in s:
            v = s[col].dropna().unique()
            if len(v) == 1:
                bits.append(f"{lab}={v[0]:g}")
    return ", ".join(bits)


def _ref_curve(ax, sub, xcol, ycol, label, color="0.35"):
    """A single reference line (median across algos) for a per-task quantity (the damage baseline)."""
    r = sub.dropna(subset=[xcol, ycol]).groupby(xcol)[ycol].median().reset_index().sort_values(xcol)
    if r.empty:
        return False
    ax.plot(r[xcol], r[ycol], color=color, lw=1.6, ls=(0, (5, 2)), marker="X", ms=5, label=label, zorder=1)
    return True


def _grid(n, ncols=3):
    ncols = min(ncols, max(1, n))
    return int(np.ceil(n / ncols)), ncols


def _finish(fig, outdir, name):
    """Lay out, attach the shared legend, and save. The legend is deliberately NOT passed to save():
    plot_common.save() forwards an explicit bbox_extra_artists list to savefig(bbox_inches="tight"), and an
    explicit list REPLACES matplotlib's default artist set -- the set that carries the figure suptitle. So a
    passed-in legend silently clips the suptitle off every PNG/PDF (each figure lost its "Part 1 -- ... --
    GMR/IOMR" title). Attaching the legend to the figure here and letting save()'s default tight bbox collect
    BOTH keeps title + legend. Same convention as plots.py (see its fig1 note)."""
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    figure_legend(fig)
    save(fig, outdir, name)


def _lines(ax, sub, ycol, sty):
    """One median line per algorithm. Any secondary series (S6's frac_q) is collapsed by the median per x, so
    every panel is one clean line per algorithm -- and the shared legend stays ~10 entries instead of ~30."""
    drew = False
    cols = [f"{ycol}_med", f"{ycol}_q25", f"{ycol}_q75"]
    for algo, gg in sub.groupby("algo"):
        agg = gg.groupby("x", as_index=False)[cols].median()
        drew |= band(ax, agg, "x", f"{ycol}_med", f"{ycol}_q25", f"{ycol}_q75", algo, sty[algo], iqr=False)
    return drew


def _present(df, sweeps):
    have = set(df["sweep"].dropna().unique())
    return [s for s in ORDER if s in have and s in sweeps]


def fig_part1(edit, sty, outdir, metric, name, title, better, ylim=None, hline=None):
    e = edit[edit["part"] == "p1"]
    sweeps = _present(e, set(e["sweep"].unique()))
    if not sweeps:
        print(f"    skip {name} (no Part-1 sweeps)"); return
    nrows, ncols = _grid(len(sweeps))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)
    for idx, sweep in enumerate(sweeps):
        ax = axes[idx // ncols][idx % ncols]
        _lines(ax, e[e["sweep"] == sweep], metric, sty)
        if hline is not None:
            ax.axhline(hline, color="k", lw=0.7, ls="--", alpha=0.5)
        ax.set_title(SWEEP_TITLE.get(sweep, sweep))
        ax.set_xlabel(SWEEP_X.get(sweep, "x"))
        ax.set_ylabel(ylab(metric.replace("_", " "), better))
        if ylim:
            ax.set_ylim(*ylim)
    for j in range(len(sweeps), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(title, fontsize=12)
    _finish(fig, outdir, name)


def fig_knn_grid(knn, sty, outdir, ycol, name, title, better, baseline=None, hline=None):
    """Panels = sweep (rows) x knn_k (cols). One median line per algo; optional per-task baseline curve."""
    if knn.empty:
        print(f"    skip {name} (no Part-2 kNN rows)"); return
    sweeps = _present(knn, set(knn["sweep"].unique()))
    ks = sorted(knn["knn_k"].dropna().unique())
    if not sweeps or not ks:
        print(f"    skip {name} (no kNN sweeps/k)"); return
    nrows, ncols = len(sweeps), len(ks)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows), squeeze=False)
    for i, sweep in enumerate(sweeps):
        for j, K in enumerate(ks):
            ax = axes[i][j]
            sub = knn[(knn["sweep"] == sweep) & (knn["knn_k"] == K)]
            _lines(ax, sub, ycol, sty)
            if baseline:
                _ref_curve(ax, sub, "x", f"{baseline}_med", baseline.replace("_", "-"))
            if hline is not None:
                ax.axhline(hline, color="k", lw=0.7, ls="--", alpha=0.5)
            ax.set_title(f"{SWEEP_TITLE.get(sweep, sweep)}, k={int(K)}")
            ax.set_xlabel(SWEEP_X.get(sweep, "x"))
            if j == 0:
                ax.set_ylabel(ylab(ycol.replace("_", " "), better))
    fig.suptitle(title, fontsize=12)
    _finish(fig, outdir, name)


def fig_triplet(knn, sty, outdir):
    """Triplet accuracy is k-independent (replicated across knn_k) -> one panel per sweep at the smallest k."""
    if knn.empty:
        print("    skip fig_triplet (no Part-2 kNN rows)"); return
    kmin = knn["knn_k"].dropna().min()
    t = knn[knn["knn_k"] == kmin]
    sweeps = _present(t, set(t["sweep"].unique()))
    if not sweeps:
        print("    skip fig_triplet (no sweeps)"); return
    nrows, ncols = _grid(len(sweeps))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)
    for idx, sweep in enumerate(sweeps):
        ax = axes[idx // ncols][idx % ncols]
        sub = t[t["sweep"] == sweep]
        _lines(ax, sub, "triplet_acc_F", sty)
        _ref_curve(ax, sub, "x", "triplet_acc_C_med", "corrupted (C)")
        ax.axhline(0.5, color="grey", lw=0.7, ls=":", alpha=0.6)
        ax.set_title(SWEEP_TITLE.get(sweep, sweep))
        ax.set_xlabel(SWEEP_X.get(sweep, "x"))
        ax.set_ylabel(ylab("triplet accuracy", "up"))
        ax.set_ylim(0.45, 1.02)
    for j in range(len(sweeps), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Part 2 -- triplet-ordering accuracy: repaired (F) vs corrupted (C)", fontsize=12)
    _finish(fig, outdir, "fig_triplet")


def fig_runtime_vs_n(edit, rows, sty, fam, outdir, min_pts=3):
    """CPU vs n on the size ladders, log-log -- the RGG analogue of plots.fig2_runtime_vs_n.

    Computed from the PER-TASK ROWS, not from summary_edit's cpu_med, and that is deliberate: a killed run
    reports cpu=0, so cpu_med is dragged toward zero by every timeout and collapses to exactly 0 once they take
    the majority -- it calls the slowest algorithm in the grid instantaneous (see load_rows). Median here is over
    the runs that FINISHED; where an algorithm stops finishing, an open marker on the timeout cap says so, so the
    expensive algorithms end at the ceiling instead of quietly disappearing.

    One panel per ladder: inflate and deflate are different cost regimes, and averaging them would hide that. The
    title names the corruption each ladder held FIXED -- it is mild, and these curves are not a worst case."""
    ladders = _cost_sweeps(rows, SIZE_LADDERS) if rows is not None else \
        [s for s in SIZE_LADDERS if (edit["sweep"] == s).any()]
    if not ladders or rows is None:
        print("    skip fig_runtime_vs_n (no size-ladder sweeps / no per-task rows)"); return
    ok = rows[rows["status"] == "ok"] if "status" in rows else rows
    fig, axes = plt.subplots(1, len(ladders), figsize=(5.6 * len(ladders), 4.3), squeeze=False)
    for i, sweep in enumerate(ladders):
        ax = axes[0][i]
        agg = _agg(ok[ok["sweep"] == sweep], "n", "cpu", min_pts=min_pts)
        for algo, gg in (agg.groupby("algo") if not agg.empty else []):
            band(ax, gg, "x", "cpu_med", "cpu_q25", "cpu_q75", algo, sty[algo], logx=True, logy=True, iqr=False)
        dnf, cap = _dnf(rows, sweep)
        if cap:
            ax.axhline(cap, color="k", lw=0.8, ls=":", alpha=0.6)
            ax.text(0.99, cap, f" timeout cap ({cap:.0f}s wall) ", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=7, color="0.3")
            for algo, xn in dnf.items():
                ax.plot([xn], [cap], marker=sty[algo]["marker"], ms=8, mfc="none", mew=1.6,
                        color=sty[algo]["color"], ls="none", zorder=4, clip_on=False)
        ax.set_xscale("log"); ax.set_yscale("log")
        fixed = _fixed(rows, sweep)
        ax.set_title(f"{SWEEP_TITLE.get(sweep, sweep)}" + (f"\ncorruption held fixed: {fixed}" if fixed else ""),
                     fontsize=10)
        ax.set_xlabel("n")
        if i == 0:
            ax.set_ylabel(ylab("CPU seconds", "down"))
    # One y-scale across the ladders -- the whole point is to compare inflate against deflate -- plus headroom
    # above the cap so the note does not sit on the DNF markers.
    _share_y(axes[0], head=6.0)
    note(axes[0][0], "median over runs that FINISHED (a killed run reports no CPU).\n"
                     "Hollow marker on the cap = the n where that algorithm's\nmedian run stopped finishing.",
         loc="upper left")
    # THE POINT OF PUTTING THE TWO LADDERS SIDE BY SIDE. Deflate breaks FAR more edges than inflate, and is FAR
    # cheaper to repair. So the size of the broken-edge set does not set the cost -- the algorithms' cost laws
    # do. Numbers read off the data so the claim cannot drift away from the figure.
    if "S1" in ladders and "S1d" in ladders:
        j = ladders.index("S1d")
        h, c = {}, {}
        for s in ("S1", "S1d"):
            top = rows.loc[rows["sweep"] == s, "n"].max()
            at_top = rows[(rows["sweep"] == s) & (rows["n"] == top)]
            h[s] = at_top["H"].median()
            # WALL, over ALL rows -- not the completed-runs CPU used for the curves. Comparing the two ladders
            # on completed runs only would invert the answer: inflate is precisely where the expensive
            # algorithms time out and drop out of the median, so dropping them makes the HARD ladder look CHEAP.
            # wall is the one timing a killed run still reports (it sits at the cap), so it counts a timeout as
            # the ~1800 s it actually burned instead of deleting it.
            c[s] = at_top["wall"].median()
        if all(np.isfinite(v) for v in list(h.values()) + list(c.values())) and c["S1d"] > 0:
            note(axes[0][j], f"deflate breaks {h['S1d'] / h['S1']:.1f}x MORE edges than inflate\n"
                             f"(|H| {h['S1d']:.0f} vs {h['S1']:.0f} at n={int(top)}) yet costs "
                             f"{c['S1'] / c['S1d']:.1f}x LESS wall time\n(median {c['S1']:.0f}s vs {c['S1d']:.0f}s, "
                             f"timeouts counted at the cap).\n|H| is not the cost driver; the cost laws are.",
                 loc="lower right")
    fig.suptitle(f"Cost -- CPU time vs n (log-log) -- {FAM_TITLE[fam]}", fontsize=12)
    _finish(fig, outdir, "fig_runtime_vs_n")


def fig_memory_vs_n(rows, sty, fam, outdir, min_pts=3):
    """Peak RSS vs n on the size ladders. Aggregated HERE because the summary carries no memory column.

    y is LINEAR on purpose. The story is a 2.6x gap, not an exponent: pivot (and left_edge) complete the graph
    to Theta(n^2) before they cover it, so they carry ~4.5M edges at n=3000 where every combinatorial algorithm
    carries the ~17k edges the RGG actually has. A log y-axis would fold that gap flat."""
    sweeps = _cost_sweeps(rows, SIZE_LADDERS)
    if not sweeps:
        print("    skip fig_memory_vs_n (no per-task rows / no size ladder)"); return
    # "Completed" has to mean completed. The handful of timeouts that DID report a peak before being killed are
    # a biased sample of one or two easy draws -- keeping them draws a memory curve that falls as n grows.
    ok = rows[rows["status"] == "ok"] if "status" in rows else rows
    fig, axes = plt.subplots(1, len(sweeps), figsize=(5.6 * len(sweeps), 4.3), squeeze=False)
    drew = False
    for i, sweep in enumerate(sweeps):
        ax = axes[0][i]
        agg = _agg(ok[ok["sweep"] == sweep], "n", "peak_mb", min_pts=min_pts)
        for algo, gg in (agg.groupby("algo") if not agg.empty else []):
            drew |= band(ax, gg, "x", "peak_mb_med", "peak_mb_q25", "peak_mb_q75", algo, sty[algo], logx=True)
        ax.set_xscale("log")
        fixed = _fixed(rows, sweep)
        ax.set_title(f"{SWEEP_TITLE.get(sweep, sweep)}" + (f"\ncorruption held fixed: {fixed}" if fixed else ""),
                     fontsize=10)
        ax.set_xlabel("n")
        if i == 0:
            ax.set_ylabel(ylab("peak RSS (MB)", "down"))
    if not drew:
        print("    skip fig_memory_vs_n (no peak_mb)"); plt.close(fig); return
    _share_y(axes[0], head=1.18)        # one scale across the ladders; headroom for the note
    # pivot is GMR and left_edge is IOMR -- they never share a panel, so name the completing algorithm of THIS
    # family rather than both (the reader would look for a line that is not there).
    who = {"gmr": "pivot", "iomr": "left_edge"}[fam]
    completer = [a for a in (who,) if (rows["algo"] == a).any()]
    if completer:
        note(axes[0][0], f"{who} completes the graph to $\\Theta(n^2)$ before\ncovering it: ~4.5M edges at "
                         "n=3000, vs ~17k in\nthe RGG itself. That is the whole gap.", loc="upper left")
    note(axes[0][-1] if len(sweeps) > 1 else axes[0][0],
         "peak RSS is only recorded for runs that COMPLETED\n(a killed run never reports one)", loc="lower right")
    fig.suptitle(f"Cost -- peak memory vs n (completed runs) -- {FAM_TITLE[fam]}", fontsize=12)
    _finish(fig, outdir, "fig_memory_vs_n")


def fig_cost_vs_corruption(rows, sty, fam, outdir, min_pts=3):
    """Cost vs corruption SEVERITY -- the question the size ladder cannot answer.

    The ladders are pinned at mag=3, frac=0.1; the extreme settings (frac 0.30, magnitude 10) were only ever
    run at n=1000. So severity gets its own figure rather than a fabricated point on the ladder. Rows = CPU and
    peak RSS on a shared x, so "more corruption costs more time" and "...costs more memory" are read off the
    same column."""
    sweeps = _cost_sweeps(rows, CORRUPT_SWEEPS)
    if not sweeps:
        print("    skip fig_cost_vs_corruption (no corruption sweeps)"); return
    metrics = [("cpu", "CPU seconds"), ("peak_mb", "peak RSS (MB)")]
    fig, axes = plt.subplots(len(metrics), len(sweeps), figsize=(5.4 * len(sweeps), 3.7 * len(metrics)),
                             squeeze=False)
    for j, sweep in enumerate(sweeps):
        sub = rows[rows["sweep"] == sweep]
        xcol = SWEEP_X.get(sweep, "x")
        if xcol not in sub:
            xcol = "x"
        for i, (ycol, lab) in enumerate(metrics):
            ax = axes[i][j]
            # Completed runs only, for BOTH metrics: a killed run reports neither its CPU nor its peak (both
            # come back as the 0 sentinel). Same convention as fig_runtime_vs_n / fig_memory_vs_n.
            src = sub[sub["status"] == "ok"] if "status" in sub else sub
            agg = _agg(src, xcol, ycol, min_pts=min_pts)
            for algo, gg in (agg.groupby("algo") if not agg.empty else []):
                band(ax, gg, "x", f"{ycol}_med", f"{ycol}_q25", f"{ycol}_q75", algo, sty[algo], logy=(ycol == "cpu"))
            if ycol == "cpu":
                ax.set_yscale("log")
            fixed = _fixed(rows, sweep)
            if i == 0:
                ax.set_title(f"{SWEEP_TITLE.get(sweep, sweep)}" + (f"\nheld fixed: {fixed}" if fixed else ""),
                             fontsize=10)
            ax.set_xlabel(xcol)
            if j == 0:
                ax.set_ylabel(ylab(lab, "down"))
    # Share the y-scale ACROSS each row: the two corruption axes are meant to be compared ("which knob costs
    # more?"), and independently autoscaled panels quietly defeat that. Headroom on the top row keeps the note
    # off the data.
    for i, (ycol, _) in enumerate(metrics):
        _share_y(axes[i], head=(4.0 if ycol == "cpu" else 1.15))
    note(axes[0][0], "the extreme corruption values live ONLY at this n;\nthe size ladder never left mag=3, frac=0.1",
         loc="upper left")
    fig.suptitle(f"Cost -- CPU and peak memory vs corruption severity -- {FAM_TITLE[fam]}", fontsize=12)
    _finish(fig, outdir, "fig_cost_vs_corruption")


def make_family(edit, knn, rows, sty, fam, outdir, min_pts=3):
    e, k = _family(edit, fam), _family(knn, fam)
    r = _family(rows, fam) if rows is not None else None
    if e.empty and k.empty:
        print(f"[{fam}] no rows -- skip"); return
    print(f"[{fam}] {FAM_TITLE[fam]} -> {outdir}")
    # Part 1 -- regime characterization
    fig_part1(e, sty, outdir, "edit_recall", "fig_edit_recall",
              f"Part 1 -- edit recall vs planted broken edges -- {FAM_TITLE[fam]}", "up", ylim=(0, 1.02))
    fig_part1(e, sty, outdir, "edit_precision", "fig_edit_precision",
              f"Part 1 -- edit precision vs planted broken edges -- {FAM_TITLE[fam]}", "up", ylim=(0, 1.02))
    fig_part1(e, sty, outdir, "ratio_domr", "fig_ratio_domr",
              f"Part 1 -- cover size / |H| -- {FAM_TITLE[fam]}", "down", hline=1.0)
    # Part 2 -- kNN recovery
    fig_knn_grid(k, sty, outdir, "jaccard_TF", "fig_knn_jaccard",
                 f"Part 2 -- kNN Jaccard vs true graph: repaired (lines) vs damage jaccard_TC (dashed) "
                 f"-- {FAM_TITLE[fam]}", "up", baseline="jaccard_TC")
    fig_knn_grid(k, sty, outdir, "lift", "fig_knn_lift",
                 f"Part 2 -- kNN lift = jaccard_TF - jaccard_TC -- {FAM_TITLE[fam]}", "up", hline=0.0)
    fig_triplet(k, sty, outdir)
    # Cost. Runtime comes from the summary (cpu_med is already there); memory MUST come from the per-task rows
    # -- rgg_analyze never aggregates peak_mb -- so these three are skipped cleanly when --rows is absent.
    if not e.empty:
        fig_runtime_vs_n(e, r, sty, fam, outdir, min_pts)
    if r is not None and not r.empty:
        fig_memory_vs_n(r, sty, fam, outdir, min_pts)
        fig_cost_vs_corruption(r, sty, fam, outdir, min_pts)
    else:
        print("    skip fig_memory_vs_n / fig_cost_vs_corruption (no --rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edit", required=True, help="summary_edit.csv from rgg_analyze.py")
    ap.add_argument("--knn", required=True, help="summary_knn.csv from rgg_analyze.py")
    ap.add_argument("--rows", default=None,
                    help="per-task rows (rgg_rows_with_ratio.csv). Defaults to that file beside --edit. Carries "
                         "peak_mb, which the summaries do NOT -- without it the memory figures are skipped.")
    ap.add_argument("--outdir", default="analysis/figs/rgg",
                    help="root for the figures; per-family gmr/ and iomr/ subfolders are created under it")
    ap.add_argument("--min-usable", type=int, default=3,
                    help="drop any group whose median rests on fewer than this many usable samples")
    a = ap.parse_args()

    edit = pd.read_csv(a.edit)
    knn = pd.read_csv(a.knn)
    rows_path = a.rows or os.path.join(os.path.dirname(os.path.abspath(a.edit)), "rgg_rows_with_ratio.csv")
    rows = load_rows(rows_path)
    if rows is None:
        print(f"  no per-task rows at {rows_path} -- memory/corruption-cost figures will be skipped")

    # A median over 2 surviving seeds is not a measurement. n_usable is the count that actually backs each
    # *_med (completed AND the cover verifies); groups below the threshold are removed rather than drawn as
    # if they were solid. Silence here would be the survivorship bias this plot exists to avoid.
    for nm, d in (("edit", edit), ("knn", knn)):
        if "n_usable" in d:
            thin = d["n_usable"] < a.min_usable
            if thin.any():
                print(f"  {nm}: dropping {int(thin.sum())}/{len(d)} groups with n_usable < {a.min_usable}")
                d.drop(d.index[thin], inplace=True)

    # REALREC: the five real base graphs share a sweep id (RR_inflate/RR_deflate/RR_mixed). Plotting them in
    # one panel would median a road network together with a scRNA cosine-kNN graph. Facet by base instead --
    # rgg_analyze.attach_base() puts the column there precisely so this is possible.
    bases = sorted(edit["base"].dropna().unique()) if "base" in edit else []
    # The OFAT sweeps (S1, S1d, P2*, ...) carry base=NaN and belong at the top level; the realrec sweeps
    # carry a base and are faceted into per-base subfolders. These are ADDITIVE, not either/or: the old
    # `[...] if bases else [(None, ...)]` dropped EVERY base-less sweep the moment a single RR_* row was
    # present in the same summary (e.g. a hand-merged edit CSV), silently losing S1/S1d/P2* -- exactly the
    # survivorship the rest of this pipeline is built to avoid. Emit the top-level set whenever any base-less
    # row exists (or when there are no bases at all), then a set per base.
    base_e = edit[edit["base"].isna()] if "base" in edit else edit
    base_k = knn[knn["base"].isna()] if "base" in knn else knn

    def _by_base(df, b):
        """Facet the per-task rows the same way as the summaries, so the cost figures follow the split too."""
        if df is None:
            return None
        if "base" not in df:
            return df if b is None else df.iloc[0:0]
        return df[df["base"].isna()] if b is None else df[df["base"] == b]

    facets = []
    if not bases or not base_e.empty or not base_k.empty:
        facets.append((None, base_e, base_k, _by_base(rows, None)))
    facets += [(b, edit[edit["base"] == b], knn[knn["base"] == b], _by_base(rows, b)) for b in bases]
    if bases:
        print(f"faceting by base graph: {bases}")

    sty = style_map(set(edit["algo"].dropna().unique()) | set(knn["algo"].dropna().unique())
                    | (set(rows["algo"].dropna().unique()) if rows is not None else set()))
    print(f"loaded {len(edit)} edit groups, {len(knn)} kNN groups; "
          f"sweeps={sorted(set(edit['sweep'].dropna().unique()))}")
    for base, e, k, r in facets:
        root = a.outdir if base is None else os.path.join(a.outdir, base)
        for fam in FAMILIES:
            sub = os.path.join(root, fam)
            os.makedirs(sub, exist_ok=True)
            make_family(e, k, r, sty, fam, sub, a.min_usable)
        print(f"figures -> {root}/{{gmr,iomr}}")


if __name__ == "__main__":
    main()
