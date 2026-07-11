"""Figures for the MDS-recovery experiment (mds_recovery.py).

    sage -python experiments/mds_plots.py --data analysis/summary_mds.csv \
         --emb analysis/mds_embeddings.npz --outdir analysis/figs/mds

Reads summary_mds.csv (scalars) and mds_embeddings.npz (the procrustes-aligned point clouds) and writes:

  fig_mds_maps      the money figure -- per 2-D graph, three scatter panels (true / observed / repaired),
                    every point coloured by its TRUE first coordinate. A faithful embedding preserves the
                    colour gradient; a distorted one scrambles it. ripe: latency map vs repaired vs geography.
  fig_mds_residual  Procrustes disparity to the true configuration, observed vs each repair variant, per
                    graph (lower = closer to truth). DOMR sits exactly on observed (Lemma 6.1 self-check).
  fig_mds_negeig    negative-eigenvalue mass (non-Euclidean-ness of the distances) observed vs repaired --
                    the theory hook: repair removes triangle violations, so the distances become more nearly
                    Euclidean-embeddable and the mass shrinks.

Standalone matplotlib -- imports nothing from the running campaign.
"""
import argparse
import os

import numpy as np
import pandas as pd

VAR_ORDER = ["observed", "DOMR", "GMR", "IOMR"]
VAR_COLOR = {"observed": "#000000", "DOMR": "#888888", "GMR": "#0072B2", "IOMR": "#D55E00"}
GRAPH_TITLE = {
    "ripe_atlas": "ripe_atlas (latency $\\to$ geography)",
    "nmr_1d3z_atom": "nmr_1d3z_atom (NOE $\\to$ 3D)",
    "nmr_1d3z_residue": "nmr_1d3z_residue (NOE $\\to$ 3D)",
    "rgg_inflate": "RGG, inflate break",
    "rgg_deflate": "RGG, deflate break",
    "rgg_mixed": "RGG, mixed break",
}


def _tag(graph, corruption):
    return f"{graph}::{corruption}"


def _disp(df, graph, variant, algo):
    r = df[(df["graph"] == graph) & (df["variant"] == variant) & (df["mds_algo"] == algo)]
    return float(r["procrustes_disp"].iloc[0]) if len(r) and pd.notna(r["procrustes_disp"].iloc[0]) else np.nan


def fig_mds_maps(df, emb, outdir, algo="smacof"):
    """One row per 2-D graph; columns = true, observed, GMR-repaired; points coloured by true x-coordinate."""
    import matplotlib.pyplot as plt
    # 2-D graphs that actually have an observed embedding stored
    graphs = []
    for g in df["graph"].unique():
        d = int(df[df["graph"] == g]["dim"].iloc[0])
        corr = df[df["graph"] == g]["corruption"].iloc[0]
        if d == 2 and f"emb::{_tag(g, corr)}::observed::{algo}" in emb:
            graphs.append((g, corr))
    if not graphs:
        print("    skip fig_mds_maps (no 2-D embeddings)"); return
    cols = [("__true__", "true configuration"), ("observed", "observed (no repair)"), ("GMR", "GMR-repaired")]
    fig, axes = plt.subplots(len(graphs), 3, figsize=(9.6, 3.15 * len(graphs)), squeeze=False)
    for row, (g, corr) in enumerate(graphs):
        tag = _tag(g, corr)
        color = emb[f"color::{tag}"]
        for col, (label, ctitle) in enumerate(cols):
            ax = axes[row][col]
            if label == "__true__":
                P = emb[f"true::{tag}"]
                sub = ""
            else:
                key = f"emb::{tag}::{label}::{algo}"
                if key not in emb:
                    ax.set_axis_off(); continue
                P = emb[key]
                sub = f"\ndisparity {_disp(df, g, label, algo):.3f}"
            ax.scatter(P[:, 0], P[:, 1], c=color, cmap="Spectral", s=14, edgecolor="white", linewidth=0.2)
            ax.set_title((ctitle + sub) if row == 0 or sub else ctitle + sub, fontsize=8.5)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal", "datalim")
        axes[row][0].set_ylabel(GRAPH_TITLE.get(g, g), fontsize=9)
    fig.suptitle(f"MDS embeddings ({algo}); colour = true x-coordinate. Repair should restore the true "
                 "gradient.\nDisparity = Procrustes distance to truth (lower is better).", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, outdir, "fig_mds_maps")


def _grouped_bars(df, outdir, ycol, name, ylabel, title, algo="smacof", dedupe_algo=False):
    import matplotlib.pyplot as plt
    d = df[df["mds_algo"] == algo] if not dedupe_algo else df.drop_duplicates(["graph", "variant"])
    graphs = [g for g in GRAPH_TITLE if g in set(d["graph"])] or sorted(d["graph"].unique())
    variants = [v for v in VAR_ORDER if v in set(d["variant"])]
    piv = (d.pivot_table(index="graph", columns="variant", values=ycol, aggfunc="first")
           .reindex(index=graphs, columns=variants))
    fig, ax = plt.subplots(figsize=(1.5 * len(graphs) + 2.5, 3.8))
    x = np.arange(len(graphs)); w = 0.8 / max(len(variants), 1)
    for i, v in enumerate(variants):
        ax.bar(x + (i - (len(variants) - 1) / 2) * w, piv[v].values, w,
               label=v, color=VAR_COLOR[v], edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([GRAPH_TITLE.get(g, g) for g in graphs], rotation=18, ha="right", fontsize=7.5)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9)
    ax.legend(title="variant", fontsize=8, frameon=False, ncol=len(variants))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, outdir, name)


def fig_mds_residual(df, outdir, algo="smacof"):
    _grouped_bars(df, outdir, "procrustes_disp", "fig_mds_residual",
                  "Procrustes disparity to truth\n($\\downarrow$ closer to true layout)",
                  f"How close is the embedding to the true layout?  ({algo}; DOMR $=$ observed self-check)",
                  algo=algo)


def fig_mds_negeig(df, outdir):
    _grouped_bars(df, outdir, "neg_mass", "fig_mds_negeig",
                  "negative-eigenvalue mass\n($\\downarrow$ more Euclidean-embeddable)",
                  "Non-Euclidean-ness of the distances (repair should shrink it; DOMR $=$ observed)",
                  dedupe_algo=True)


def _save(fig, outdir, name):
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {name}.pdf / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="analysis/summary_mds.csv")
    ap.add_argument("--emb", default="analysis/mds_embeddings.npz")
    ap.add_argument("--outdir", default="analysis/figs/mds")
    ap.add_argument("--algo", default="smacof", choices=["classical", "smacof"])
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    df = pd.read_csv(a.data)
    for c in ("procrustes_disp", "stress", "neg_mass", "dim", "n"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    emb = np.load(a.emb, allow_pickle=True) if os.path.exists(a.emb) else {}
    print(f"loaded {len(df)} rows, {df['graph'].nunique()} graphs -> {a.outdir}")
    fig_mds_maps(df, emb, a.outdir, algo=a.algo)
    fig_mds_residual(df, a.outdir, algo=a.algo)
    fig_mds_negeig(df, a.outdir)
    print(f"figures -> {a.outdir}")


if __name__ == "__main__":
    main()
