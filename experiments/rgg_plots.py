"""rgg_plots.py -- the RGG figures from the summaries (rgg_analyze.py output).

Runs LOCALLY; needs matplotlib + pandas. On this Mac use sage's Python (it bundles both):

    sage -python experiments/rgg_plots.py --edit analysis/summary_edit.csv \
        --knn analysis/summary_knn.csv --outdir analysis/figs/rgg

Figures are written split by MR axis into per-family subfolders of --outdir:
    <outdir>/gmr/   GMR (general metric repair): variants {GMR, DOMR}
    <outdir>/iomr/  IOMR (increase-only metric repair): variant {IOMR}
GMR and IOMR are different problems (different covers, different optima), so they never share a panel.

Each family folder gets two figure families (median line + IQR band, one line per algorithm, stable colors),
with every metric axis annotated higher/lower-is-better:

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

FAMILY = {"gmr": {"GMR", "DOMR"}, "iomr": {"IOMR"}}     # MR axis -> variants drawn in that family's folder
FAMILIES = ["gmr", "iomr"]
FAM_TITLE = {"gmr": "GMR (general MR)", "iomr": "IOMR (increase-only MR)"}

# stable panel order; label each panel by the knob it sweeps
SWEEP_X = {
    "S1": "n", "POCsize_inflate": "n", "POCsize_jitter": "n",
    "S2": "deg", "S2k": "k", "S3": "magnitude", "S3d": "magnitude", "S6": "magnitude",
    "S4i": "frac_q", "S4d": "frac_q", "S5a": "n_jitter", "P2n": "n_jitter",
    "S5b": "jitter", "P2j": "jitter", "S5c": "subset_s", "P2s": "subset_s",
}
SWEEP_TITLE = {
    "S1": "S1 size", "S2": "S2 density (radius)", "S2k": "S2' density (knn)",
    "S3": "S3 inflate mag.", "S3d": "S3' deflate mag.", "S4i": "S4 inflate frac",
    "S4d": "S4' deflate frac", "S5a": "S5a jitter count", "S5b": "S5b jitter mag.",
    "S5c": "S5c jitter subset", "S6": "S6 mag x frac",
    "POCsize_inflate": "POC size (inflate)", "POCsize_jitter": "POC size (jitter)",
    "P2s": "P2 subset", "P2j": "P2 jitter mag.", "P2n": "P2 jitter count",
}
ORDER = ["S1", "S2", "S2k", "S3", "S3d", "S4i", "S4d", "S5a", "S5b", "S5c", "S6",
         "POCsize_inflate", "POCsize_jitter", "P2s", "P2j", "P2n"]


def _ylab(base, better=None):
    tag = {"up": "↑ higher is better", "down": "↓ lower is better"}.get(better)
    return f"{base}\n({tag})" if tag else base


def _color_map(*dfs):
    algos = sorted(set().union(*[set(d["algo"].dropna().unique()) for d in dfs if len(d)]))
    cmap = plt.get_cmap("tab20")
    return {a: cmap(i % 20) for i, a in enumerate(algos)}


def _family(df, fam):
    return df[df["variant"].isin(FAMILY[fam])] if len(df) else df


def _band(ax, sub, xcol, ycol, lo, hi, label, color, logx=False, logy=False):
    """One algo's median line + IQR band vs xcol. Returns True if anything non-NaN was drawn."""
    sub = sub.sort_values(xcol)
    x = sub[xcol].to_numpy(dtype=float)
    y = sub[ycol].to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if logx:
        m &= (x > 0)
    if logy:
        m &= (y > 0)
    if not m.any():
        return False
    ax.plot(x[m], y[m], marker="o", ms=3, lw=1.3, color=color, label=label)
    if lo in sub and hi in sub:
        ylo, yhi = sub[lo].to_numpy(dtype=float), sub[hi].to_numpy(dtype=float)
        b = m & np.isfinite(ylo) & np.isfinite(yhi)
        if b.any():
            ax.fill_between(x[b], ylo[b], yhi[b], color=color, alpha=0.15, lw=0)
    return True


def _ref_curve(ax, sub, xcol, ycol, label, color="k"):
    """A single reference line (median across algos) for a per-task quantity (damage baseline)."""
    r = sub.dropna(subset=[xcol, ycol]).groupby(xcol)[ycol].median().reset_index().sort_values(xcol)
    if r.empty:
        return False
    ax.plot(r[xcol], r[ycol], color=color, lw=1.4, ls="--", marker="s", ms=3, label=label)
    return True


def _grid(n, ncols=3):
    ncols = min(ncols, max(1, n))
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def _lines(ax, sub, ycol, colors):
    """Draw one banded line per (algo, series). series (e.g. S6 frac_q) is appended to the label."""
    drew = False
    for (algo, series), gg in sub.groupby(["algo", "series"], dropna=False):
        label = algo if pd.isna(series) else f"{algo} q={series:g}"
        drew |= _band(ax, gg, "x", f"{ycol}_med", f"{ycol}_q25", f"{ycol}_q75", label, colors[algo])
    return drew


def _present(df, sweeps):
    have = set(df["sweep"].dropna().unique())
    return [s for s in ORDER if s in have and s in sweeps]


def _save(fig, outdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    wrote {name}.pdf / .png")


def fig_part1(edit, colors, outdir, metric, name, title, better, ylim=None, hline=None):
    e = edit[edit["part"] == "p1"]
    sweeps = _present(e, set(e["sweep"].unique()))
    if not sweeps:
        print(f"    skip {name} (no Part-1 sweeps)"); return
    nrows, ncols = _grid(len(sweeps))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)
    for idx, sweep in enumerate(sweeps):
        ax = axes[idx // ncols][idx % ncols]
        drew = _lines(ax, e[e["sweep"] == sweep], metric, colors)
        if hline is not None:
            ax.axhline(hline, color="k", lw=0.7, ls="--", alpha=0.5)
        ax.set_title(SWEEP_TITLE.get(sweep, sweep))
        ax.set_xlabel(SWEEP_X.get(sweep, "x"))
        ax.set_ylabel(_ylab(metric.replace("_", " "), better))
        if ylim:
            ax.set_ylim(*ylim)
        if drew:
            ax.legend(fontsize=6, ncol=2)
    for j in range(len(sweeps), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, outdir, name)


def fig_knn_grid(knn, colors, outdir, ycol, name, title, better, baseline=None, hline=None):
    """Panels = sweep (rows) x knn_k (cols). One banded line per algo; optional per-task baseline curve."""
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
            drew = _lines(ax, sub, ycol, colors)
            if baseline:
                drew |= _ref_curve(ax, sub, "x", f"{baseline}_med", baseline.replace("_", "-"))
            if hline is not None:
                ax.axhline(hline, color="k", lw=0.7, ls="--", alpha=0.5)
            ax.set_title(f"{SWEEP_TITLE.get(sweep, sweep)}, k={int(K)}")
            ax.set_xlabel(SWEEP_X.get(sweep, "x"))
            if j == 0:
                ax.set_ylabel(_ylab(ycol.replace("_", " "), better))
            if drew:
                ax.legend(fontsize=6, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, outdir, name)


def fig_triplet(knn, colors, outdir):
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
        drew = _lines(ax, sub, "triplet_acc_F", colors)
        drew |= _ref_curve(ax, sub, "x", "triplet_acc_C_med", "corrupted (C)")
        ax.axhline(0.5, color="grey", lw=0.7, ls=":", alpha=0.6)
        ax.set_title(SWEEP_TITLE.get(sweep, sweep))
        ax.set_xlabel(SWEEP_X.get(sweep, "x"))
        ax.set_ylabel(_ylab("triplet accuracy", "up"))
        ax.set_ylim(0.45, 1.02)
        if drew:
            ax.legend(fontsize=6, ncol=2)
    for j in range(len(sweeps), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Part 2 -- triplet-ordering accuracy: repaired (F) vs corrupted (C)")
    fig.tight_layout()
    _save(fig, outdir, "fig_triplet")


def make_family(edit, knn, colors, fam, outdir):
    e, k = _family(edit, fam), _family(knn, fam)
    if e.empty and k.empty:
        print(f"[{fam}] no rows -- skip"); return
    print(f"[{fam}] {FAM_TITLE[fam]} -> {outdir}")
    # Part 1 -- regime characterization
    fig_part1(e, colors, outdir, "edit_recall", "fig_edit_recall",
              f"Part 1 -- edit recall vs planted broken edges -- {FAM_TITLE[fam]}", "up", ylim=(0, 1.02))
    fig_part1(e, colors, outdir, "edit_precision", "fig_edit_precision",
              f"Part 1 -- edit precision vs planted broken edges -- {FAM_TITLE[fam]}", "up", ylim=(0, 1.02))
    fig_part1(e, colors, outdir, "ratio_domr", "fig_ratio_domr",
              f"Part 1 -- cover size / |H| -- {FAM_TITLE[fam]}", "down", hline=1.0)
    # Part 2 -- kNN recovery
    fig_knn_grid(k, colors, outdir, "jaccard_TF", "fig_knn_jaccard",
                 f"Part 2 -- kNN Jaccard vs true graph: repaired (lines) vs damage jaccard_TC (dashed) "
                 f"-- {FAM_TITLE[fam]}", "up", baseline="jaccard_TC")
    fig_knn_grid(k, colors, outdir, "lift", "fig_knn_lift",
                 f"Part 2 -- kNN lift = jaccard_TF - jaccard_TC -- {FAM_TITLE[fam]}", "up", hline=0.0)
    fig_triplet(k, colors, outdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edit", required=True, help="summary_edit.csv from rgg_analyze.py")
    ap.add_argument("--knn", required=True, help="summary_knn.csv from rgg_analyze.py")
    ap.add_argument("--outdir", default="analysis/figs/rgg",
                    help="root for the figures; per-family gmr/ and iomr/ subfolders are created under it")
    a = ap.parse_args()

    edit = pd.read_csv(a.edit)
    knn = pd.read_csv(a.knn)
    colors = _color_map(edit, knn)
    print(f"loaded {len(edit)} edit groups, {len(knn)} kNN groups; "
          f"sweeps={sorted(set(edit['sweep'].dropna().unique()))}")
    for fam in FAMILIES:
        sub = os.path.join(a.outdir, fam)
        os.makedirs(sub, exist_ok=True)
        make_family(edit, knn, colors, fam, sub)
    print(f"figures -> {a.outdir}/{{gmr,iomr}}")


if __name__ == "__main__":
    main()
