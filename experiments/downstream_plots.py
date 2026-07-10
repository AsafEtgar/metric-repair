"""Figures for the pure real-data downstream experiment (downstream_recovery.py).

    sage -python experiments/downstream_plots.py --summary analysis/summary_downstream.csv --outdir analysis/figs/downstream

Reads summary_downstream.csv (from downstream_analyze.py) and writes, per the experimental design:

  fig_recovery_lift   per graph, kNN-recovery lift vs the external true metric, as a function of k, one line
                      per repair variant. DOMR sits flat at zero by decrease-only invariance -- it is drawn as
                      the reference, so the figure shows the self-check and the GMR/IOMR signal at once.
  fig_spearman_delta  per graph, the change in global rank fidelity (Spearman vs the true metric) per variant.

Each variant's line is the median over that variant's algorithms of the per-algorithm median (summary rows are
already per (graph, variant, algo, k)); the band is the inter-algorithm IQR where more than one algorithm
exists. Standalone matplotlib -- imports nothing from the running campaign.
"""
import argparse
import os

import numpy as np
import pandas as pd

VARIANT_ORDER = ["DOMR", "GMR", "IOMR"]
VARIANT_COLOR = {"DOMR": "#888888", "GMR": "#0072B2", "IOMR": "#D55E00"}
VARIANT_MARK = {"DOMR": "o", "GMR": "s", "IOMR": "^"}
GRAPH_TITLE = {
    "ripe_atlas": "ripe_atlas  (latency $\\to$ geography)",
    "nmr_1d3z_atom": "nmr_1d3z_atom  (NOE $\\to$ 3D structure)",
    "nmr_1d3z_residue": "nmr_1d3z_residue  (NOE $\\to$ 3D structure)",
}


def _variant_lines(sub, ycol):
    """For one graph: {variant: (ks, median, q25, q75)} aggregating over the variant's algorithms per k."""
    out = {}
    for v in VARIANT_ORDER:
        vv = sub[sub["variant"] == v]
        if vv.empty:
            continue
        g = vv.groupby("k")[ycol]
        ks = np.array(sorted(vv["k"].unique()))
        med = g.median().reindex(ks).values
        q25 = g.quantile(.25).reindex(ks).values
        q75 = g.quantile(.75).reindex(ks).values
        out[v] = (ks, med, q25, q75)
    return out


def fig_recovery_lift(df, outdir):
    import matplotlib.pyplot as plt
    graphs = [g for g in GRAPH_TITLE if g in set(df["graph"])] or sorted(df["graph"].unique())
    fig, axes = plt.subplots(1, len(graphs), figsize=(4.2 * len(graphs), 3.6), squeeze=False)
    for ax, graph in zip(axes[0], graphs):
        sub = df[df["graph"] == graph]
        lines = _variant_lines(sub, "lift_med")
        for v, (ks, med, q25, q75) in lines.items():
            ax.plot(ks, med, marker=VARIANT_MARK[v], color=VARIANT_COLOR[v], label=v, lw=1.8, ms=5)
            if np.isfinite(q25).any() and (q75 - q25 > 0).any():
                ax.fill_between(ks, q25, q75, color=VARIANT_COLOR[v], alpha=0.15, lw=0)
        ax.axhline(0, color="black", lw=0.7, ls=":")
        ax.set_title(GRAPH_TITLE.get(graph, graph), fontsize=9)
        ax.set_xlabel("$k$ (neighborhood size)")
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("kNN-recovery lift vs true metric\n(repaired $-$ observed; $\\uparrow$ better)")
    axes[0][0].legend(title="variant", fontsize=8, frameon=False)
    fig.suptitle("Does repair recover the true metric?  DOMR is flat at 0 by construction (self-check).",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig_recovery_lift.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    wrote fig_recovery_lift.pdf / .png")


def fig_spearman_delta(df, outdir):
    import matplotlib.pyplot as plt
    graphs = [g for g in GRAPH_TITLE if g in set(df["graph"])] or sorted(df["graph"].unique())
    # one delta per (graph, variant): median over algorithms and over k (k-independent for a global corr,
    # but the summary carries a row per k, so collapse them)
    rows = []
    for graph in graphs:
        for v in VARIANT_ORDER:
            vv = df[(df["graph"] == graph) & (df["variant"] == v)]
            if not vv.empty:
                rows.append((graph, v, vv["delta_spearman_med"].median()))
    if not rows:
        print("    skip fig_spearman_delta (no rows)"); return
    piv = pd.DataFrame(rows, columns=["graph", "variant", "delta"]).pivot(
        index="graph", columns="variant", values="delta").reindex(columns=VARIANT_ORDER)
    fig, ax = plt.subplots(figsize=(1.6 * len(graphs) + 2, 3.4))
    x = np.arange(len(piv)); w = 0.25
    for i, v in enumerate([v for v in VARIANT_ORDER if v in piv.columns]):
        ax.bar(x + (i - 1) * w, piv[v].values, w, label=v, color=VARIANT_COLOR[v])
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(piv.index, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("$\\Delta$ Spearman vs true metric\n(repaired $-$ observed; $\\uparrow$ better)")
    ax.set_title("Global rank fidelity gained by repair", fontsize=9)
    ax.legend(title="variant", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig_spearman_delta.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    wrote fig_spearman_delta.pdf / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="analysis/summary_downstream.csv")
    ap.add_argument("--outdir", default="analysis/figs/downstream")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    df = pd.read_csv(a.summary)
    for c in df.columns:
        if c not in ("graph", "gt_kind", "variant", "algo"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"loaded {len(df)} rows, {df['graph'].nunique()} graphs -> {a.outdir}")
    fig_recovery_lift(df, a.outdir)
    fig_spearman_delta(df, a.outdir)
    print(f"figures -> {a.outdir}")


if __name__ == "__main__":
    main()
