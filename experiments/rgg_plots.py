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

Each figure is PDF + PNG. Missing parts/families are skipped with a note rather than crashing.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")                       # non-interactive: save files, no display
import matplotlib.pyplot as plt             # noqa: E402
import numpy as np                          # noqa: E402
import pandas as pd                          # noqa: E402

from plot_common import (FAMILY, FAMILIES, FAM_TITLE, style_map, ylab, band,      # noqa: E402
                         figure_legend, save)

# stable panel order; label each panel by the knob it sweeps
# SWEEP_X / SWEEP_TITLE now live in experiments/sweeps.py, shared with rgg_analyze. They were duplicated,
# and the copy in rgg_analyze fell behind: it lacked S1d, S1m, S3m, S4m, P2df, P2dm, P2mf, P2mm and every
# RR_* sweep, so add_x() left x=NaN and those panels vanished from the figures without an error.
from sweeps import SWEEP_X, SWEEP_TITLE          # noqa: E402
ORDER = ["S1", "S1d", "S1m", "S2", "S2k", "S3", "S3d", "S3m", "S4i", "S4d", "S4m", "S5a", "S5b", "S5c", "S6",
         "POCsize_inflate", "POCsize_jitter", "P2size", "P2df", "P2dm", "P2mf", "P2mm", "P2s", "P2j", "P2n",
         "RR_inflate", "RR_deflate", "RR_mixed"]


def _family(df, fam):
    return df[df["variant"].isin(FAMILY[fam])] if len(df) else df


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


def make_family(edit, knn, sty, fam, outdir):
    e, k = _family(edit, fam), _family(knn, fam)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edit", required=True, help="summary_edit.csv from rgg_analyze.py")
    ap.add_argument("--knn", required=True, help="summary_knn.csv from rgg_analyze.py")
    ap.add_argument("--outdir", default="analysis/figs/rgg",
                    help="root for the figures; per-family gmr/ and iomr/ subfolders are created under it")
    ap.add_argument("--min-usable", type=int, default=3,
                    help="drop any group whose median rests on fewer than this many usable samples")
    a = ap.parse_args()

    edit = pd.read_csv(a.edit)
    knn = pd.read_csv(a.knn)

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
    facets = []
    if not bases or not base_e.empty or not base_k.empty:
        facets.append((None, base_e, base_k))
    facets += [(b, edit[edit["base"] == b], knn[knn["base"] == b]) for b in bases]
    if bases:
        print(f"faceting by base graph: {bases}")

    sty = style_map(set(edit["algo"].dropna().unique()) | set(knn["algo"].dropna().unique()))
    print(f"loaded {len(edit)} edit groups, {len(knn)} kNN groups; "
          f"sweeps={sorted(set(edit['sweep'].dropna().unique()))}")
    for base, e, k in facets:
        root = a.outdir if base is None else os.path.join(a.outdir, base)
        for fam in FAMILIES:
            sub = os.path.join(root, fam)
            os.makedirs(sub, exist_ok=True)
            make_family(e, k, sty, fam, sub)
        print(f"figures -> {root}/{{gmr,iomr}}")


if __name__ == "__main__":
    main()
