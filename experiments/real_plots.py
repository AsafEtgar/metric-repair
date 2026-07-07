"""real_plots.py -- figures from the real-dataset summary (real_analyze.py output).

Runs LOCALLY; needs matplotlib + pandas (use sage's Python on this Mac).

    sage -python experiments/real_plots.py --summary analysis/summary_real.csv --outdir analysis/figs/real

Real data is a SET of heterogeneous graphs (not a sweep), so the per-family figures are heatmaps
(algorithm x graph). Split by MR axis into <outdir>/{gmr,iomr}/ as elsewhere; graph-level characterization
figures go at the <outdir> root.

  <outdir>/{gmr,iomr}/fig_ratio_domr   size / |H|  (down; always available -- no ILP needed)
  <outdir>/{gmr,iomr}/fig_ratio        size / OPT-or-bound (down; a column is red-labelled when its ref is
                                       only the LP lower bound, i.e. the ILP is absent/timed-out)
  <outdir>/{gmr,iomr}/fig_runtime      CPU seconds (down, log color)
  <outdir>/fig_characterize            non-metric fraction + |H| per graph
  <outdir>/fig_inversion               non-metric fraction across the 4 similarity conversions (R3b)

Runs against the heuristics array alone (ratios then vs the LP bound). PDF + PNG each.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt             # noqa: E402
from matplotlib.colors import LogNorm        # noqa: E402
import numpy as np                          # noqa: E402
import pandas as pd                          # noqa: E402

FAMILIES = ["gmr", "iomr"]
FAM_TITLE = {"gmr": "GMR (general MR)", "iomr": "IOMR (increase-only MR)"}
BOUND_ALGOS = {"gmr_lp_naive", "iomr_lp_naive"}     # references, not covers -> omit from cover heatmaps
GRAPH_ORDER = ["dimacs_ny_d", "dimacs_ny_t", "ripe_atlas", "pbmc3k_cosine_knn", "cassiopeia_barcode_knn",
               "nmr_1d3z_residue", "nmr_1d3z_atom",
               "bct_coactivation", "bct_coactivation_lin", "bct_coactivation_log", "bct_coactivation_raw",
               "flycns_male", "flycns_male_lin", "flycns_male_log", "flycns_male_raw",
               "fish1_ten", "fish1_ten_lin", "fish1_ten_log", "fish1_ten_raw"]
ALGO_ORDER = {
    "gmr": ["domr", "gmr_ilp", "gmr_thr_naive", "gmr_bestofk", "gmr_rand", "l1sep_gmr", "spc_gmr", "pivot"],
    "iomr": ["iomr_ilp", "iomr_thr_naive", "iomr_bestofk", "iomr_rand", "iomr_regiongrow", "l1sep_iomr",
             "spc_iomr", "left_edge"],
}
INV_BASES = ["bct_coactivation", "flycns_male", "fish1_ten"]
INV_CONV = [("", "1/s"), ("_lin", "lin"), ("_log", "log"), ("_raw", "raw")]


def _order(items, ref):
    return [x for x in ref if x in set(items)] + [x for x in sorted(set(items)) if x not in ref]


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    wrote {os.path.join(outdir, name)}.png")


def _heatmap(summ, fam, ycol, name, title, outroot, log=False, fmt="{:.1f}"):
    sub = summ[(summ["family"] == fam) & (~summ["algo"].isin(BOUND_ALGOS))]
    if sub.empty:
        print(f"    skip {fam}/{name} (no rows)"); return
    graphs = _order(sub["graph"].unique(), GRAPH_ORDER)
    algos = _order(sub["algo"].unique(), ALGO_ORDER[fam])
    M = np.full((len(algos), len(graphs)), np.nan)
    for i, a in enumerate(algos):
        for j, g in enumerate(graphs):
            r = sub[(sub["algo"] == a) & (sub["graph"] == g)]
            if len(r):
                M[i, j] = r[ycol].iloc[0]
    fig, ax = plt.subplots(figsize=(0.55 * len(graphs) + 3, 0.5 * len(algos) + 2))
    finite = M[np.isfinite(M)]
    norm = LogNorm(vmin=max(finite.min(), 1e-3), vmax=finite.max()) if (log and finite.size) else None
    im = ax.imshow(M, aspect="auto", cmap="viridis_r", norm=norm)
    ax.set_xticks(range(len(graphs))); ax.set_xticklabels(graphs, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(algos))); ax.set_yticklabels(algos, fontsize=8)
    # red x-labels where the reference is only the LP bound (ILP absent/timed-out)
    rk = summ[summ["family"] == fam].groupby("graph")["ref_kind"].first()
    for t, g in zip(ax.get_xticklabels(), graphs):
        if rk.get(g) == "lower_bound":
            t.set_color("tab:red")
    for i in range(len(algos)):
        for j in range(len(graphs)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if (norm is None and M[i, j] > np.nanmean(finite)) else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label=title.split("—")[-1].strip())
    ax.set_title(f"{title} — {FAM_TITLE[fam]}", fontsize=10)
    fig.tight_layout()
    _save(fig, os.path.join(outroot, fam), name)


def fig_characterize(summ, outroot):
    """Per-graph non-metric fraction + |H| (one row per graph; take any algo's meta)."""
    per = summ.groupby("graph").agg(nonmetric_frac=("nonmetric_frac", "first"), H=("H", "first")).reset_index()
    graphs = _order(per["graph"].unique(), GRAPH_ORDER)
    per = per.set_index("graph").loc[graphs].reset_index()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(0.5 * len(graphs) + 3, 6), sharex=True)
    a1.bar(range(len(graphs)), per["nonmetric_frac"], color="tab:purple")
    a1.set_ylabel("non-metric fraction"); a1.set_title("Real graphs — non-metricity & |H|")
    a2.bar(range(len(graphs)), per["H"], color="tab:gray")
    a2.set_ylabel("|H| = |DOMR|"); a2.set_yscale("log")
    a2.set_xticks(range(len(graphs))); a2.set_xticklabels(graphs, rotation=60, ha="right", fontsize=7)
    fig.tight_layout()
    _save(fig, outroot, "fig_characterize")


def fig_inversion(summ, outroot):
    """Non-metric fraction across the 4 similarity conversions for each inverted base (the R3b finding)."""
    per = summ.groupby("graph")["nonmetric_frac"].first()
    present = [b for b in INV_BASES if any((b + s) in per.index for s, _ in INV_CONV)]
    if not present:
        print("    skip fig_inversion (no inversion variants)"); return
    fig, axes = plt.subplots(1, len(present), figsize=(4 * len(present), 3.6), squeeze=False)
    for k, base in enumerate(present):
        ax = axes[0][k]
        labels, vals = [], []
        for suf, lab in INV_CONV:
            g = base + suf
            if g in per.index:
                labels.append(lab); vals.append(per[g])
        ax.bar(labels, vals, color=["tab:blue", "tab:green", "tab:orange", "tab:red"][:len(vals)])
        ax.set_title(base); ax.set_ylabel("non-metric fraction")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Similarity-inversion sensitivity (R3b): non-metricity is largely a 1/s artifact")
    fig.tight_layout()
    _save(fig, outroot, "fig_inversion")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="summary_real.csv from real_analyze.py")
    ap.add_argument("--outdir", default="analysis/figs/real")
    a = ap.parse_args()
    summ = pd.read_csv(a.summary)
    print(f"loaded {len(summ)} groups, {summ['graph'].nunique()} graphs")
    for fam in FAMILIES:
        _heatmap(summ, fam, "ratio_domr_med", "fig_ratio_domr",
                 "cover size / |H|  (↓ lower is better)", a.outdir, log=True)
        _heatmap(summ, fam, "ratio_med", "fig_ratio",
                 "size / OPT-or-bound  (↓ lower is better; red = LP-bound ref)", a.outdir, log=True)
        _heatmap(summ, fam, "cpu_med", "fig_runtime",
                 "CPU seconds  (↓ lower is better)", a.outdir, log=True, fmt="{:.2g}")
    fig_characterize(summ, a.outdir)
    fig_inversion(summ, a.outdir)
    print(f"figures -> {a.outdir}/{{gmr,iomr}} + root")


if __name__ == "__main__":
    main()
