"""Figures for the MDS-recovery experiment (mds_recovery.py + mds_sweep.py).

    sage -python experiments/mds_plots.py --data analysis/summary_mds.csv \
         --emb analysis/mds_embeddings.npz --outdir analysis/figs/mds

Reads summary_mds.csv (scalars) + mds_embeddings.npz (the procrustes-aligned point clouds), and -- when they
exist -- summary_mds_sweep.csv + mds_sweep_embeddings.npz (the same, per ALGORITHM). Writes:

  fig_mds_maps      the money figure -- per graph, three scatter panels (true / observed / repaired), every
                    point coloured by its TRUE first coordinate. A faithful embedding preserves the colour
                    gradient; a distorted one scrambles it. ripe: latency map vs repaired vs geography.
  fig_mds_residual  Procrustes disparity to the true configuration, observed vs each repair variant, per
                    graph (lower = closer to truth). DOMR sits exactly on observed (Lemma 6.1 self-check).
  fig_mds_negeig    negative-eigenvalue mass (non-Euclidean-ness of the distances) observed vs repaired --
                    the theory hook: repair removes triangle violations, so the distances become more nearly
                    Euclidean-embeddable and the mass shrinks.
  fig_mds_grid_*    per (graph, corruption) and per FAMILY: true / observed / one panel per algorithm, best
                    to worst. The per-algorithm view of the same question -- does cover quality decide
                    whether geometry comes back?
  fig_mds_sweep_rgg disparity vs cover size on the RGG controls (the real-graph twin lives in mds_sweep.py).
  table_mds_sweep   the headline per-algorithm comparison, CSV + LaTeX.

Every figure states its FULL PROVENANCE in the suptitle -- graph, corruption, frac, magnitude, n/core,
degree, seed, MDS algorithm, embedding dimension. A figure that cannot say which instance it came from
cannot be checked.

Standalone matplotlib -- imports nothing from the running campaign.
"""
import argparse
import os

import numpy as np
import pandas as pd

VAR_ORDER = ["observed", "DOMR", "GMR", "IOMR"]
VAR_COLOR = {"observed": "#000000", "DOMR": "#888888", "GMR": "#0072B2", "IOMR": "#D55E00"}
# The repo's family convention (plot_common.py:20): DOMR is filed INSIDE the GMR family -- a DOMR cover is a
# valid GMR cover. So the GMR grid draws domr, and its panel must come out pixel-identical to `observed`
# (Lemma 6.1: reweighting the heavy set to its detour leaves every shortest path unchanged, so D_F == D_G).
# That identity is the grid's built-in correctness check, which is why domr is drawn rather than dropped.
FAMILY = {"gmr": {"GMR", "DOMR"}, "iomr": {"IOMR"}}
FAMILIES = ["gmr", "iomr"]
FAM_TITLE = {"gmr": "GMR (general MR)", "iomr": "IOMR (increase-only MR)"}
GRAPH_TITLE = {
    "ripe_atlas": "ripe_atlas (latency $\\to$ geography)",
    "nmr_1d3z_atom": "nmr_1d3z_atom (NOE $\\to$ 3D)",
    "nmr_1d3z_residue": "nmr_1d3z_residue (NOE $\\to$ 3D)",
    "rgg_inflate": "RGG $n{=}300$, inflate break",
    "rgg_deflate": "RGG $n{=}300$, deflate break",
    "rgg_mixed": "RGG $n{=}300$, mixed break",
    "rgg_inflate_n1000": "RGG $n{=}1000$, inflate break",
    "rgg_deflate_n1000": "RGG $n{=}1000$, deflate break",
    "rgg_mixed_n1000": "RGG $n{=}1000$, mixed break",
    "dimacs_ny_d": "dimacs_ny_d (road distance $\\to$ geography)",
    "dimacs_ny_t": "dimacs_ny_t (travel time $\\to$ geography)",
    "dimacs_ny_d_inflate": "NY roads, planted inflate break",
    "dimacs_ny_d_deflate": "NY roads, planted deflate break",
    "dimacs_ny_d_mixed": "NY roads, planted mixed break",
}

# Graphs whose true configuration is a PROTEIN and therefore has a backbone: the polyline through the
# residue centroids, in sequence order, is the depth cue that makes a 3-D fold legible from a still image.
NMR_GRAPHS = ("nmr_1d3z_atom", "nmr_1d3z_residue")
ELEV, AZIM = 18.0, -62.0        # one shared viewpoint; the panels are Procrustes co-aligned, so this is legit


def _tag(graph, corruption):
    return f"{graph}::{corruption}"


def _disp(df, graph, variant, algo):
    r = df[(df["graph"] == graph) & (df["variant"] == variant) & (df["mds_algo"] == algo)]
    return float(r["procrustes_disp"].iloc[0]) if len(r) and pd.notna(r["procrustes_disp"].iloc[0]) else np.nan


def _get(row, col, default=""):
    """A column that may be absent (an old CSV) or NaN."""
    if row is None or col not in row or pd.isna(row[col]):
        return default
    return row[col]


def _fmt(v, nd=3):
    if v == "" or v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "?"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(f)) if f == int(f) else f"{f:.{nd}g}"


def _provenance(row, mds_algo, dim):
    """The one line that lets a reader reproduce the panel: which graph, broken how, at what size, from which
    seed, embedded by which MDS into how many dimensions. `row` is any row of that (graph, corruption)."""
    g = _get(row, "graph", "?")
    corr = _get(row, "corruption", "?")
    n, n_used = _get(row, "n"), _get(row, "n_used")
    core = f"{_fmt(n)} (core {_fmt(n_used)})" if str(n_used) != "" and str(n_used) != str(n) else _fmt(n)
    bits = [f"corruption={corr}"]
    for label, col in (("frac", "frac"), ("magnitude", "magnitude")):
        v = _get(row, col)
        if str(v) != "":
            bits.append(f"{label}={_fmt(v)}")
    bits.append(f"n={core}")
    for label, col in (("avg_deg", "avg_degree"), ("seed", "seed"), ("radius", "radius")):
        v = _get(row, col)
        if str(v) != "":
            bits.append(f"{label}={_fmt(v, 4)}")
    return f"{g}  |  {', '.join(bits)}  |  MDS={mds_algo}, dim={dim}"


def _dim_of(emb, tag, fallback=2):
    """Take the embedding dimension from the ARRAY SHAPE, not from a column: summary_mds_sweep.csv has no
    `dim` column at all, and the true configuration is the one object guaranteed to be present."""
    key = f"true::{tag}"
    if key in emb:
        return int(np.asarray(emb[key]).shape[1])
    return fallback


# ----------------------------------------------------------------------------
# The NMR backbone -- re-derived, never assumed
# ----------------------------------------------------------------------------
_FOLD_CACHE = {}


def fold_labels(graph):
    """(residue_id_per_row, replayed_true_configuration) for an NMR graph.

    The npz stores the point clouds with NO node labels, so the backbone polyline needs the labels back. They
    are deterministically re-derivable with zero repair computation by REPLAYING the pipeline that produced
    the rows -- load_graph -> true_distances -> finite_core -> classical_mds -- and reading off the surviving
    labels. The replayed configuration is returned alongside them so the caller can verify the row ORDER, not
    merely the row COUNT.

    Do NOT shortcut this with `sorted(node_set)`. `true_distances` returns the ground-truth block in the
    TRUTH file's node order, and `gt_ix` is NOT monotone, so the stored row order is a permutation of the
    graph's lexicographic order, not that order itself. Only the replay reproduces it."""
    if graph in _FOLD_CACHE:
        return _FOLD_CACHE[graph]
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (here, os.path.dirname(here)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from downstream_recovery import load_graph, true_distances, apsp      # noqa: PLC0415
    from mds_recovery import finite_core, classical_mds, PURE_REAL_DIM    # noqa: PLC0415

    nodes, _idx, edges = load_graph(graph)
    gt_ix, Dtrue = true_distances(graph, nodes)
    core = finite_core(apsp(edges, len(nodes))[np.ix_(gt_ix, gt_ix)])
    labels = [str(nodes[gt_ix[i]]) for i in core]
    res = np.array([int(lab.split(":")[0]) for lab in labels], dtype=int)   # '23:HA*' -> 23; '23' -> 23
    # Replay the TRUTH the same way mds_recovery/mds_sweep built it: classical MDS of the true distance block
    # on the core, centered and Frobenius-normalized (mds_recovery.py:271-273, mds_sweep._store_true).
    cfg, _neg = classical_mds(Dtrue[np.ix_(core, core)], PURE_REAL_DIM[graph])
    cfg = cfg - cfg.mean(0)
    cfg = cfg / (np.linalg.norm(cfg) or 1.0)
    _FOLD_CACHE[graph] = (res, cfg)
    return _FOLD_CACHE[graph]


def _align_signs(A, B):
    """Flip each column of A to the sign that agrees with B. The eigenvectors classical MDS returns are only
    defined up to a per-axis sign, so a sign flip is not evidence of anything; a PERMUTATION is. Flipping
    first isolates the question we actually care about -- is row i of the replay row i of the npz? -- and
    leaves a permutation exactly as detectable as before."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    for j in range(min(A.shape[1], B.shape[1])):
        if np.abs(A[:, j] + B[:, j]).max() < np.abs(A[:, j] - B[:, j]).max():
            A[:, j] = -A[:, j]
    return A


BACKBONE_TOL = 1e-6


def _fold_info(graph, emb, tag):
    """(residue_ids, colour_vector) for a panel. For an NMR graph the colour is the RESIDUE INDEX -- the
    sequence position, which is independent information. The stored `color::` is true_cfg[:, 0], i.e. the
    x-axis itself; colouring a scatter by its own abscissa says nothing a reader cannot already see.

    VERIFIES THE ROW ORDER, not just the row count. Checking `len(res) == rows` would pass on any PERMUTATION
    of the right labels -- and a permuted backbone draws a confident, wrong protein, which is precisely the
    failure this guard exists to prevent. So we replay the whole truth pipeline and compare the reconstructed
    configuration to the stored `true::` array elementwise (max|delta| = 0.0 today on both NMR graphs). A
    mismatch raises rather than drawing a plausible lie."""
    stored = np.asarray(emb[f"true::{tag}"])
    if graph not in NMR_GRAPHS:
        return None, np.asarray(emb[f"color::{tag}"])
    res, cfg = fold_labels(graph)
    if len(res) != stored.shape[0]:
        raise RuntimeError(
            f"backbone re-derivation MISMATCH for {graph}: replayed core has {len(res)} nodes but the stored "
            f"embedding has {stored.shape[0]} rows. The polyline would connect the wrong atoms. Refusing to "
            f"draw. (Did load_graph / true_distances / finite_core change?)")
    delta = float(np.abs(_align_signs(cfg.copy(), stored) - stored).max())
    if not np.isfinite(delta) or delta > BACKBONE_TOL:
        raise RuntimeError(
            f"backbone ROW-ORDER MISMATCH for {graph}: the replayed true configuration differs from the "
            f"stored true::{tag} by max|delta| = {delta:.3e} > {BACKBONE_TOL:.0e}. Same length, different "
            f"rows -- the polyline would connect the wrong atoms in the wrong order. Refusing to draw.")
    print(f"    backbone check {graph}: core {len(res)} rows, replayed truth matches stored "
          f"true::{tag} to max|delta| = {delta:.2e}")
    return res, res.astype(float)


def _colour_note(res):
    """What the colour MEANS in a panel -- and it is NOT the same for every panel. The NMR rows are coloured
    by residue index and carry the backbone polyline (`_fold_info`); every other graph is coloured by the
    stored `color::`, i.e. the true first coordinate. A figure whose caption asserts one colour key while its
    panels draw another is the exact defect this one-liner exists to close, so every figure that mixes the
    two must print it PER ROW, never once in the suptitle."""
    return ("colour = residue index (N $\\to$ C), grey line = backbone through the residue centroids"
            if res is not None else "colour = true first coordinate")


def _backbone(P, res):
    """The polyline through the residue centroids, in SEQUENCE order. On the residue graph there is one point
    per residue, so a centroid is that point; on the atom graph it averages the residue's atoms -- which is
    exactly the backbone trace a structural biologist reads a fold from."""
    P = np.asarray(P, float)
    return np.array([P[res == r].mean(0) for r in sorted(set(res.tolist()))])


# ----------------------------------------------------------------------------
# Point clouds -- 2-D and (nmr) 3-D
# ----------------------------------------------------------------------------
ZOOM3D = 1.45          # mpl draws a 3-D cube well inside its axes; without this the cloud is a stamp


def shared_limits(clouds, dim):
    """ONE axis box for every panel of a figure, as the union over every cloud that figure will draw.

    Panels within a figure are all Procrustes-aligned into the SAME frame, so letting matplotlib autoscale
    each one to its own cloud is not a convenience -- it is a lie. Measured on the stored embeddings, panels
    inside a single grid differ in extent by up to 3.6x (rgg_mixed_n1000/l1sep_gmr; rgg_inflate_n1000/gmr_ilp
    is 3.3x, ripe/observed 2.4x). Autoscaled, a cloud 3.6x more distended renders at the same apparent size as
    a compact one, and the damage the figure exists to show becomes invisible.

    Returned as (mid, half) with `half` per-axis. The box is the union, so no point is ever clipped. Limits
    are shared WITHIN a figure only -- never across datasets, whose frames are unrelated."""
    d = 3 if dim >= 3 else 2
    A = [np.asarray(P, float)[:, :d] for P in clouds if P is not None and len(np.asarray(P))]
    if not A:
        return None
    lo = np.min([P.min(0) for P in A], axis=0)
    hi = np.max([P.max(0) for P in A], axis=0)
    mid, half = (lo + hi) / 2.0, np.maximum((hi - lo) / 2.0, 1e-12)
    if d == 3:                       # a 3-D box must be a CUBE, or the fold's proportions are fiction
        half = np.full(3, float(half.max()))
    return mid, half * 1.04          # a hair of margin so no point sits on the frame


def _equal3d(ax, P, lim=None):
    """Equal aspect in 3-D. `lim` is the figure-wide box from `shared_limits`; without one we fall back to
    this panel's own extent, which is only safe when a single cloud is drawn. `zoom` scales the drawn cube
    only, so no point is ever clipped."""
    if lim is None:
        P = np.asarray(P, float)
        lo, hi = P.min(0), P.max(0)
        mid = (lo + hi) / 2.0
        half = np.full(3, float((hi - lo).max()) / 2.0 or 1.0)
    else:
        mid, half = lim
    ax.set_xlim(mid[0] - half[0], mid[0] + half[0])
    ax.set_ylim(mid[1] - half[1], mid[1] + half[1])
    ax.set_zlim(mid[2] - half[2], mid[2] + half[2])
    try:
        ax.set_box_aspect((1, 1, 1), zoom=ZOOM3D)
    except TypeError:                                  # mpl < 3.3 has no `zoom`
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:                             # very old mpl
        pass


def _scatter3d(ax, P, color, res=None, cmap="viridis", s=12, lim=None):
    """A 3-D point cloud plus, when the residue ids are known, the backbone polyline. The polyline is not
    decoration: a still image of a 3-D scatter is viewpoint-ambiguous, and the chain is what lets the eye
    order the points in depth and see that a fold is (or is not) still a fold."""
    P = np.asarray(P, float)
    if res is not None:
        bb = _backbone(P, res)
        ax.plot(bb[:, 0], bb[:, 1], bb[:, 2], color="#555555", lw=1.0, alpha=0.85, zorder=1)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=color, cmap=cmap, s=s,
               edgecolor="white", linewidth=0.15, depthshade=False, zorder=2)
    _equal3d(ax, P, lim=lim)
    ax.view_init(elev=ELEV, azim=AZIM)               # ONE angle for every panel: they are co-aligned
    # No cube, no panes, no ticks. The axes are meaningless here (a Procrustes-standardized frame has no
    # units), and at ZOOM3D the cube's edges spill outside the panel and collide with the variant frame.
    ax.set_axis_off()


def _scatter(ax, P, color, dim, res=None, cmap=None, s=14, lim=None):
    """Draw a point cloud. dim >= 3 (nmr) is drawn in 3-D on an Axes3D -- it used to be silently flattened to
    P[:, 0], P[:, 1], discarding an axis that carries ~21-23% of the structure and making an intact fold
    indistinguishable from a wrecked one.

    `lim` is the figure-wide box from `shared_limits`. Pass it. Without it each panel autoscales to its own
    cloud, which silently rescales a wrecked embedding to look like an intact one."""
    P = np.asarray(P)
    if dim >= 3 and P.shape[1] >= 3 and hasattr(ax, "zaxis"):
        _scatter3d(ax, P, color, res=res, cmap=cmap or "viridis", s=s - 2, lim=lim)
        return
    ax.scatter(P[:, 0], P[:, 1], c=color, cmap=cmap or "Spectral", s=s, edgecolor="white", linewidth=0.2)
    ax.set_xticks([]); ax.set_yticks([])
    if lim is None:
        ax.set_aspect("equal", "datalim")            # single-cloud fallback only
        return
    mid, half = lim
    ax.set_xlim(mid[0] - half[0], mid[0] + half[0])
    ax.set_ylim(mid[1] - half[1], mid[1] + half[1])
    # "box" (not "datalim"): the LIMITS are fixed and shared, and matplotlib shrinks the axes box to make the
    # data scale equal. "datalim" would do the opposite -- widen the limits back out, per panel.
    ax.set_aspect("equal", "box")


SHARED_AXES_NOTE = ("All panels share one axis box (the union over the panels drawn); no panel is rescaled "
                    "to fit, so a distended embedding LOOKS distended.")


def _proj_note(dim):
    return "" if dim <= 2 else f"  ({dim}-D, drawn in 3-D; shared viewpoint elev={ELEV:.0f}, azim={AZIM:.0f})"


def _aspect(P):
    """height/width of a point cloud's first two axes -- used to size a grid cell to the data it will hold."""
    P = np.asarray(P, float)
    dx = float(np.ptp(P[:, 0])) or 1.0
    dy = float(np.ptp(P[:, 1])) or 1.0
    return dy / dx


# ----------------------------------------------------------------------------
# fig_mds_maps / fig_mds_residual / fig_mds_negeig  (per-VARIANT, from mds_recovery)
# ----------------------------------------------------------------------------
def _row_annot(fig, axr, dim, label, foot):
    """The row's graph name (left) and its colour key + provenance (underneath). MUST be called AFTER the
    layout is final, because on a 3-D row the two strings are drawn in FIGURE coordinates read off the axes'
    final positions.

    Why not simply set_ylabel / set_xlabel, as the 2-D rows do?  Because `_scatter3d` ends with
    `ax.set_axis_off()`, which clears `axison`; an Axes3D then draws NEITHER axis artist, and the ylabel and
    xlabel hang off exactly those artists. The labels are set and never rendered -- silently. That deleted the
    graph name and the provenance line from precisely the rows the 3-D change touched, and a figure that
    cannot say which instance it came from cannot be checked."""
    if dim < 3:
        axr[0].set_ylabel(label, fontsize=9)
        axr[1].set_xlabel(foot, fontsize=5.5)
        return
    p0, p1 = axr[0].get_position(), axr[1].get_position()
    fig.text(p0.x0 - 0.010, 0.5 * (p0.y0 + p0.y1), label, rotation=90,
             ha="center", va="center", fontsize=9)
    fig.text(0.5 * (p1.x0 + p1.x1), p1.y0 - 0.004, foot, ha="center", va="top", fontsize=5.5)


def fig_mds_maps(df, emb, outdir, algo="smacof"):
    """One row per graph; columns = true, observed, GMR-repaired.

    The colour key is PER ROW and must be printed per row: the NMR rows are 3-D, coloured by residue index
    and carrying the backbone; the rest are 2-D, coloured by the true first coordinate. The suptitle used to
    assert one key for all six rows, which was false for two of them."""
    import matplotlib.pyplot as plt
    graphs = []
    for g in df["graph"].unique():
        corr = df[df["graph"] == g]["corruption"].iloc[0]
        if f"emb::{_tag(g, corr)}::observed::{algo}" in emb:
            graphs.append((g, corr))
    if not graphs:
        print("    skip fig_mds_maps (no embeddings)"); return
    cols = [("__true__", "true configuration"), ("observed", "observed (no repair)"), ("GMR", "GMR-repaired")]
    fig = plt.figure(figsize=(9.6, 3.3 * len(graphs)))
    meta, any3d = [], False
    for row, (g, corr) in enumerate(graphs):
        tag = _tag(g, corr)
        dim = _dim_of(emb, tag)
        res, color = _fold_info(g, emb, tag)
        rec = df[df["graph"] == g].iloc[0]
        # ONE box per ROW, i.e. per dataset. Never one box across rows: the rows are different graphs whose
        # frames are unrelated, and forcing a common scale on them would be a different lie.
        lim = shared_limits([emb[f"true::{tag}"]]
                            + [emb[f"emb::{tag}::{c}::{algo}"] for c, _ in cols
                               if f"emb::{tag}::{c}::{algo}" in emb], dim)
        # Rows can differ in dimension (nmr is 3-D, rgg/ripe 2-D), so each axes is created with ITS OWN
        # projection -- a figure-wide subplot_kw would force one mode on both and flatten the fold again.
        axr = []
        for col, (label, ctitle) in enumerate(cols):
            ax = fig.add_subplot(len(graphs), 3, row * 3 + col + 1,
                                 projection="3d" if dim >= 3 else None)
            axr.append(ax)
            if label == "__true__":
                P, sub = emb[f"true::{tag}"], ""
            else:
                key = f"emb::{tag}::{label}::{algo}"
                if key not in emb:
                    ax.set_axis_off(); continue
                P = emb[key]
                sub = f"\ndisparity {_disp(df, g, label, algo):.3f}"
            _scatter(ax, P, color, dim, res=res, lim=lim)
            ax.set_title(ctitle + sub, fontsize=8.5)
        any3d = any3d or dim >= 3
        foot = f"{_colour_note(res)}  |  {_provenance(rec, algo, dim)}{_proj_note(dim)}"
        meta.append((axr, dim, GRAPH_TITLE.get(g, g), foot))
    fig.suptitle("MDS embeddings: true configuration vs observed vs GMR-repaired.\n"
                 "Disparity = Procrustes distance to truth (lower is better).\n"
                 "The COLOUR KEY DIFFERS BY ROW (the NMR rows are 3-D and coloured by residue index, not by "
                 "the x-coordinate): it is stated, with the provenance, under each row.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if any3d:
        # An axis-off Axes3D fills its whole cell, so tight_layout leaves nowhere to put the row's provenance
        # line. Buy the strip back explicitly, then place the annotations against the FINAL positions.
        fig.subplots_adjust(hspace=0.34)
    for axr, dim, label, foot in meta:
        _row_annot(fig, axr, dim, label, foot)
    _save(fig, outdir, "fig_mds_maps")


def _grouped_bars(df, outdir, ycol, name, ylabel, title, algo="smacof", dedupe_algo=False):
    import matplotlib.pyplot as plt
    d = df[df["mds_algo"] == algo] if not dedupe_algo else df.drop_duplicates(["graph", "variant"])
    graphs = [g for g in GRAPH_TITLE if g in set(d["graph"])] or sorted(d["graph"].unique())
    variants = [v for v in VAR_ORDER if v in set(d["variant"])]
    piv = (d.pivot_table(index="graph", columns="variant", values=ycol, aggfunc="first")
           .reindex(index=graphs, columns=variants))
    fig, ax = plt.subplots(figsize=(1.5 * len(graphs) + 2.5, 4.2))
    x = np.arange(len(graphs)); w = 0.8 / max(len(variants), 1)
    for i, v in enumerate(variants):
        ax.bar(x + (i - (len(variants) - 1) / 2) * w, piv[v].values, w,
               label=v, color=VAR_COLOR[v], edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([GRAPH_TITLE.get(g, g) for g in graphs], rotation=18, ha="right", fontsize=7.5)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9)
    ax.legend(title="variant", fontsize=8, frameon=False, ncol=len(variants))
    ax.grid(axis="y", alpha=0.25)
    # provenance for a multi-graph bar chart = one line per graph, compactly
    lines = []
    for g in graphs:
        rec = df[df["graph"] == g].iloc[0]
        lines.append(_provenance(rec, "-" if dedupe_algo else algo, int(_get(rec, "dim", 2) or 2)))
    fig.text(0.0, -0.02 - 0.035 * len(lines), "\n".join(lines), fontsize=5.2, va="top", family="monospace")
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


# ----------------------------------------------------------------------------
# fig_mds_grid -- per (graph, corruption, family): true / observed / every algorithm
# ----------------------------------------------------------------------------
def _family_of(variant):
    return "gmr" if variant in FAMILY["gmr"] else "iomr"


def _null_reason(rec):
    """Why does this panel carry no point cloud?  Say the TRUE reason, which is not always the same reason.

    The sweep records exactly one failure mode -- `timeout` -- and not one row in summary_mds_sweep.csv has
    ever reported a convergence failure. So the old blanket label "never converged" was a fabrication: an
    algorithm that hit the wall-clock cap was still converging when we killed it, and an algorithm with NO ROW
    AT ALL for this graph (rec is None) was never even started here. Those are three different facts:

        no row       -> not run on this graph (e.g. iomr_regiongrow, which only ever ran on nmr_* and
                        dimacs_ny_t; it is absent, not failed).
        timeout      -> the run was cut off at the cap. The cap is read off the row's own `wall`, never guessed.
        anything else -> report the recorded status verbatim rather than interpret it."""
    if rec is None:
        return "not run on this graph"
    status = str(_get(rec, "status", "not measured"))
    if status == "timeout":
        try:
            return f"timeout at {float(_get(rec, 'wall', np.nan)):.0f}s cap"
        except (TypeError, ValueError):
            return "timeout"
    if status in ("ok", ""):
        return "cover found, no embedding stored"
    return status


def fig_mds_grid(sw, semb, outdir, algo="smacof"):
    """Per (graph, corruption) and family: panel 1 = true, panel 2 = observed, then ONE PANEL PER ALGORITHM of
    that family, ordered best -> worst by disp_smacof. Algorithms that did not land get a labelled EMPTY panel
    -- never a silent omission."""
    import matplotlib.pyplot as plt
    made = 0
    for (g, corr), grp in sw.groupby(["graph", "corruption"], sort=False):
        tag = _tag(g, corr)
        if f"true::{tag}" not in semb:
            print(f"    skip fig_mds_grid {g} (no stored embeddings)"); continue
        dim = _dim_of(semb, tag)
        res, color = _fold_info(g, semb, tag)         # nmr -> residue index (+ backbone); else true x-coord
        obs = grp[grp["variant"] == "observed"]
        for fam in FAMILIES:
            sub = grp[grp["variant"].isin(FAMILY[fam])].copy()
            if sub.empty:
                continue
            ok = sub[sub["disp_smacof"].notna()].sort_values("disp_smacof")
            bad = sub[sub["disp_smacof"].isna()]
            panels = [("__true__", None), ("__observed__", obs.iloc[0] if len(obs) else None)]
            panels += [(r["algo"], r) for _, r in ok.iterrows()]
            panels += [(r["algo"], r) for _, r in bad.iterrows()]

            ncol = min(5, len(panels))
            nrow = int(np.ceil(len(panels) / ncol))
            # ONE box for the whole grid: the union over the true configuration and EVERY cover drawn in it.
            # Autoscaling per panel let a cover whose embedding is 3.6x more distended render at the same
            # apparent size as a compact one -- the grid's whole job is to let the eye see that damage.
            clouds = [semb[f"true::{tag}"]]
            for _lab, _rec in panels:
                _ka = "observed" if _lab == "__observed__" else _lab
                if _ka != "__true__" and f"emb::{tag}::{_ka}::{algo}" in semb:
                    clouds.append(semb[f"emb::{tag}::{_ka}::{algo}"])
            lim = shared_limits(clouds, dim)
            # Size the CELL to the SHARED box's aspect (not the true config's -- the shared box is what every
            # panel is now drawn in). With aspect="equal" matplotlib shrinks each axes BOX to that ratio, and
            # any mismatch between cell and box is dead white space. (A fixed cell height left a void half the
            # figure tall on ripe.) Title band reserved in INCHES -- as a fraction it over-reserves on a
            # 1-row grid and under-reserves on a 4-row one.
            CELL_W, TITLE_IN, FOOT_IN, PANEL_TITLE_IN = 2.35, 0.62, 0.30, 0.55
            # A 3-D panel is drawn in a square box (set_box_aspect), so the data aspect does not apply.
            asp = 1.0 if (dim >= 3 or lim is None) else float(
                np.clip(float(lim[1][1]) / float(lim[1][0]), 0.45, 1.45))
            cell_h = CELL_W * asp + PANEL_TITLE_IN
            h = cell_h * nrow + TITLE_IN + FOOT_IN
            skw = {"projection": "3d"} if dim >= 3 else {}
            fig, axes = plt.subplots(nrow, ncol, figsize=(CELL_W * ncol, h), squeeze=False,
                                     subplot_kw=skw)
            for ax in axes.ravel():
                ax.set_axis_off()
            for i, (label, rec) in enumerate(panels):
                ax = axes[i // ncol][i % ncol]
                ax.set_axis_on()
                if label == "__true__":
                    _scatter(ax, semb[f"true::{tag}"], color, dim, res=res, lim=lim)
                    ax.set_title("true configuration", fontsize=8, color="#222")
                    _frame(ax, "#222222")
                    continue
                key_algo = "observed" if label == "__observed__" else label
                key = f"emb::{tag}::{key_algo}::{algo}"
                variant = "observed" if label == "__observed__" else str(rec["variant"])
                if key not in semb:
                    # Two DIFFERENT reasons a panel has no point cloud, and conflating them would be a lie:
                    #   source=cluster -> the cover exists, on the cluster, and we have its DISPARITY; what we
                    #                     lack is the local embedding. Report the number, say where it is from.
                    #   otherwise      -> the algorithm was tried here and did not land (status says how).
                    if str(_get(rec, "source", "local")) == "cluster":
                        d = _get(rec, "disp_smacof", np.nan)
                        _msg(ax, f"{key_algo}\ncluster-only cover\ndisp {_fmt(d, 4)}\n(no local embedding)",
                             "#555")
                        ax.set_title(key_algo, fontsize=8, color="#555")
                        _frame(ax, "#888888", ls="--")
                    else:
                        _msg(ax, f"{key_algo}\nNO COVER\n({_null_reason(rec)})", "#B00")
                        ax.set_title(key_algo, fontsize=8, color="#B00")
                        _frame(ax, "#BB0000", ls=":")
                    _blank_ticks(ax); _apply_lim(ax, lim, dim)
                    continue
                _scatter(ax, semb[key], color, dim, res=res, lim=lim)
                d = _get(rec, "disp_smacof", np.nan)
                r = _get(rec, "ratio_domr", np.nan)
                extra = ""
                if key_algo == "domr":
                    extra = "\n$=$ observed (Lemma 6.1)"      # the grid's built-in self-check
                ttl = (f"{key_algo}\ndisp {_fmt(d, 4)}   $|S|/|H|$ {_fmt(r, 3)}{extra}"
                       if key_algo != "observed" else
                       f"observed (no repair)\ndisp {_fmt(d, 4)}")
                ax.set_title(ttl, fontsize=7.5)
                _frame(ax, VAR_COLOR.get(variant, "#555555"))
            rec0 = grp.iloc[0]
            cnote = _colour_note(res)
            fig.suptitle(f"{FAM_TITLE[fam]} -- repaired MDS maps, best to worst\n"
                         f"{_provenance(rec0, algo, dim)}{_proj_note(dim)}\n"
                         f"{cnote}; disp = Procrustes distance to truth "
                         "($\\downarrow$ better); frame colour = variant",
                         fontsize=8)
            note = _absent_note(sw, g, fam, sub)
            note = (note + "  " if note else "") + SHARED_AXES_NOTE
            if note:
                fig.text(0.5, 0.008, note, ha="center", va="bottom", fontsize=6.2, color="#444", wrap=True)
            fig.tight_layout(rect=(0, (FOOT_IN / h) if note else 0, 1, 1 - TITLE_IN / h))
            if nrow > 1:
                # tight_layout does not reserve enough for a 2-3 line panel title, so a lower row's title
                # lands on the bottom edge of the panel above it. Buy the gap back explicitly. (3-D is worse:
                # an axis-off Axes3D occupies its ENTIRE cell, leaving nowhere at all for the title.)
                fig.subplots_adjust(hspace=0.24 if dim >= 3 else 0.34)
            _save(fig, outdir, f"fig_mds_grid_{g}_{corr}_{fam}")
            made += 1
    print(f"    fig_mds_grid: {made} figure(s)")


def _absent_note(sw, graph, fam, sub):
    """Why is this grid thin?  Two honest, DIFFERENT reasons, and the figure must not conflate them:

      cluster-only : the cover was computed on the cluster and its disparity is in the CSV, but the cover file
                     is not on this machine, so there is no point cloud to draw.
      absent       : the algorithm has NO ROW for this graph -- it was never run here. That is all a missing
                     row can tell us.

    The roster is the union of algorithms over every graph in the sweep, so "absent" means only "ran
    elsewhere, not here". It does NOT mean the algorithm failed: iomr_regiongrow, for instance, has rows on
    nmr_1d3z_{atom,residue} and dimacs_ny_t and nowhere else, so on every other grid it is simply not run.
    Calling that "never converged" -- as this note once did -- asserted a failure that never happened.

    An algorithm that WAS run and did not land keeps its row (status=timeout) and therefore gets its own
    labelled NO COVER panel; it is not silently folded into this footnote. Every claim here is read off the
    data, never asserted."""
    roster = {a for a in sw[sw["variant"].isin(FAMILY[fam])]["algo"].unique()}
    here = set(sub["algo"].unique())
    absent = sorted(roster - here)
    cluster = sorted(sub[sub.get("source", "local").astype(str) == "cluster"]["algo"].unique()) \
        if "source" in sub.columns else []
    timedout = sorted(sub[sub["status"].astype(str) == "timeout"]["algo"].unique()) \
        if "status" in sub.columns else []
    bits = []
    if cluster:
        bits.append("cover exists on the cluster but not locally (disparity known, embedding not drawable): "
                    + ", ".join(cluster))
    if absent:
        bits.append("not run on this graph (no row in the sweep; these ran on other graphs, so the roster "
                    "lists them): " + ", ".join(absent))
    if timedout:
        bits.append("run but returned no cover before the wall-clock cap (drawn as labelled NO COVER "
                    "panels): " + ", ".join(timedout))
    return ("Absent panels -- " + "  |  ".join(bits)) if bits else ""


def _frame(ax, color, ls="-"):
    """Colour the panel border by variant. An Axes3D has no usable spines, so there we draw the border as a
    rectangle in axes coordinates -- same visual contract, different artist."""
    if hasattr(ax, "zaxis"):
        # The rectangle goes on the FIGURE, not the axes: an Axes3D draw() sorts every patch it owns by
        # artist.do_3d_projection(), which a plain Rectangle does not have (AttributeError at savefig).
        # Anchoring it to ax.transAxes still pins it to the panel, and the transform is resolved at draw
        # time -- so it survives tight_layout moving the axes.
        from matplotlib.patches import Rectangle                          # noqa: PLC0415
        r = Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor=color,
                      linewidth=1.8, linestyle=ls, zorder=10, clip_on=False)
        ax.get_figure().add_artist(r)
        return
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(color); s.set_linewidth(1.8); s.set_linestyle(ls)


def _apply_lim(ax, lim, dim):
    """Give a panel that draws NO cloud (NO COVER, cluster-only) the same box as the panels that do. Without
    this the empty axes keeps its full cell while the real panels shrink to the shared aspect, and its title
    rides up into the row above."""
    if lim is None:
        return
    mid, half = lim
    if dim >= 3 and hasattr(ax, "zaxis"):
        _equal3d(ax, None, lim=lim)
        return
    ax.set_xlim(mid[0] - half[0], mid[0] + half[0])
    ax.set_ylim(mid[1] - half[1], mid[1] + half[1])
    ax.set_aspect("equal", "box")


def _blank_ticks(ax):
    if hasattr(ax, "zaxis"):
        ax.set_axis_off()          # a null panel must not draw an empty 3-D cube around its "NO COVER" text
        return
    ax.set_xticks([]); ax.set_yticks([])


def _msg(ax, txt, color, fontsize=7):
    """Centred text on an empty panel. Axes3D.text() means text(x, y, z, s) -- it does NOT accept a 2-D
    transform -- so a null panel on a 3-D grid needs text2D or it raises."""
    fn = ax.text2D if hasattr(ax, "text2D") else ax.text
    fn(0.5, 0.5, txt, ha="center", va="center", fontsize=fontsize, color=color, transform=ax.transAxes)


# ----------------------------------------------------------------------------
# fig_mds_sweep_rgg + the headline table
# ----------------------------------------------------------------------------
def fig_mds_sweep_rgg(sw, outdir):
    """The RGG twin of mds_sweep.fig_sweep_real: disparity vs cover size, per algorithm, per broken RGG. These
    30 rows have been sitting in summary_mds_sweep.csv unplotted -- fig_sweep_real filters to the real graphs."""
    import matplotlib.pyplot as plt
    rgg = sw[sw["dataset"] == "rgg"]
    d = rgg[rgg["disp_smacof"].notna()]
    if d.empty:
        print("    skip fig_mds_sweep_rgg (no RGG rows)"); return
    # The panel roster comes from the UNFILTERED rows: an instance on which every algorithm timed out must
    # still get a panel saying so, not vanish from the grid.
    graphs = [g for g in GRAPH_TITLE if g in set(rgg["graph"])]
    # Six RGG instances now (n=300 control + n=1000), so the old single row would be ~26 inches wide.
    ncol = min(3, len(graphs))
    nrow = int(np.ceil(len(graphs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 4.0 * nrow), squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(graphs):]:
        ax.set_axis_off()
    for ax, g in zip(flat, graphs):
        allsub = rgg[rgg["graph"] == g]
        sub = d[d["graph"] == g]
        obs = sub[sub["variant"] == "observed"]
        if len(obs):
            ax.axhline(float(obs["disp_smacof"].iloc[0]), color="black", lw=0.9, ls=":",
                       label="observed / DOMR")
        for _, r in sub.iterrows():
            if r["variant"] == "observed":
                continue
            x, y = float(r["ratio_domr"]), float(r["disp_smacof"])
            ax.scatter(x, y, s=44, color=VAR_COLOR.get(r["variant"], "#555"),
                       edgecolor="white", linewidth=0.5, zorder=3)
            ax.annotate(r["algo"], (x, y), fontsize=6, xytext=(3, 2), textcoords="offset points", alpha=0.85)
        # A scatter has no slot for an algorithm with no disparity, so a NaN row would simply not be drawn --
        # and iomr_ilp times out on exactly the two n=1000 panels where the reader most wants it. Dropping it
        # silently is the one thing this figure may not do: name it, and say why it is not there.
        miss = allsub[allsub["disp_smacof"].isna()]
        if len(miss):
            names = ", ".join(f"{r['algo']} ({_null_reason(r)})" for _, r in miss.iterrows())
            ax.text(0.02, 0.97, "no cover returned:\n" + names, transform=ax.transAxes,
                    ha="left", va="top", fontsize=6, color="#B00000",
                    bbox={"facecolor": "white", "edgecolor": "#BB0000", "linestyle": ":",
                          "linewidth": 0.8, "boxstyle": "round,pad=0.25", "alpha": 0.9})
        # n is PER PANEL now (the n=1000 specs sit beside the n=300 control), so it cannot live in the
        # suptitle: one shared "n=300" over a grid containing n=1000 would be a plain misstatement.
        r0 = allsub.iloc[0]
        n, nu = _fmt(_get(r0, "n")), _fmt(_get(r0, "n_used"))
        ttl = GRAPH_TITLE.get(g, g)                       # already names n; add the core only if it BIT
        if nu != n:
            ttl += f"\ncore {nu} of {n} (giant component)"
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("cover size  $|S|/|H|$"); ax.grid(alpha=0.25)
    for i in range(nrow):
        axes[i][0].set_ylabel("Procrustes disparity (SMACOF)\n($\\downarrow$ closer to truth)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=VAR_COLOR[v], label=v)
               for v in ("GMR", "IOMR", "DOMR")]
    handles.append(plt.Line2D([], [], color="black", ls=":", label="observed"))
    flat[len(graphs) - 1].legend(handles=handles, fontsize=7, frameon=False, loc="best")
    rec0 = sw[sw["dataset"] == "rgg"].iloc[0]
    fig.suptitle("Broken-RGG controls: does editing less preserve geometry better?\n"
                 f"avg_deg={_fmt(_get(rec0, 'avg_degree'))}, frac={_fmt(_get(rec0, 'frac'))}, "
                 f"magnitude={_fmt(_get(rec0, 'magnitude'))}, seed={_fmt(_get(rec0, 'seed'))}  |  "
                 "MDS=SMACOF, dim=2  |  n per panel", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.10 / nrow))
    _save(fig, outdir, "fig_mds_sweep_rgg")


def table_mds_sweep(sw, outdir):
    """The headline per-algorithm comparison mds_sweep.py's docstring promises and which did not exist:
    disparity / neg-mass / |S|/|H| per algorithm per graph, as CSV and as LaTeX."""
    cols = ["dataset", "graph", "corruption", "algo", "variant", "cover_size", "ratio_domr",
            "disp_classical", "disp_smacof", "neg_mass", "status", "source"]
    have = [c for c in cols if c in sw.columns]
    t = sw[have].copy()
    t = t.sort_values(["dataset", "graph", "disp_smacof"], na_position="last")
    csv_path = os.path.join(outdir, "table_mds_sweep.csv")
    os.makedirs(outdir, exist_ok=True)
    t.to_csv(csv_path, index=False)

    lat = t.copy()
    for c in ("disp_classical", "disp_smacof", "neg_mass", "ratio_domr"):
        if c in lat:
            lat[c] = lat[c].map(lambda v: "--" if pd.isna(v) else f"{float(v):.4f}")
    if "cover_size" in lat:
        lat["cover_size"] = lat["cover_size"].map(lambda v: "--" if pd.isna(v) else f"{int(float(v))}")
    keep = [c for c in ("graph", "algo", "variant", "cover_size", "ratio_domr", "disp_smacof", "neg_mass")
            if c in lat]
    body = "\n".join(" & ".join(str(r[c]).replace("_", r"\_") for c in keep) + r" \\"
                     for _, r in lat.iterrows())
    tex = ("% auto-generated by experiments/mds_plots.py -- do not edit by hand\n"
           "\\begin{tabular}{" + "l" * len(keep) + "}\n\\toprule\n"
           + " & ".join(c.replace("_", r"\_") for c in keep) + r" \\" + "\n\\midrule\n"
           + body + "\n\\bottomrule\n\\end{tabular}\n")
    tex_path = os.path.join(outdir, "table_mds_sweep.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"    wrote table_mds_sweep.csv / .tex ({len(t)} rows)")


# ----------------------------------------------------------------------------
# fig_nmr_fold_3d -- the ubiquitin fold, in the 3-D it was always stored in
# ----------------------------------------------------------------------------
# The panels that carry the story, in order. NOT a best-to-worst ranking: the point is precisely that the
# exact repairs are INDISTINGUISHABLE from observed, and that the two cheap heuristics are not.
FOLD_PANELS = ["observed", "domr", "gmr_ilp", "iomr_ilp", "pivot", "left_edge"]
_M_CACHE = {}


def edge_count(graph):
    """|E| of a real graph -- needed to say what FRACTION of the edge set a cover edits. |S|/|H| (ratio_domr)
    cannot answer that: |H| is the heavy set, not the graph."""
    if graph not in _M_CACHE:
        import sys
        here = os.path.dirname(os.path.abspath(__file__))
        for p in (here, os.path.dirname(here)):
            if p not in sys.path:
                sys.path.insert(0, p)
        from downstream_recovery import load_graph                        # noqa: PLC0415
        _M_CACHE[graph] = len(load_graph(graph)[2])
    return _M_CACHE[graph]


def fig_nmr_fold_3d(sw, semb, outdir, algo="smacof"):
    """The NMR fold in 3-D -- the figure the 2-D projection could not draw.

    The embeddings were ALWAYS 3-D (PURE_REAL_DIM[nmr] = 3); `_scatter` merely sliced off the third axis,
    which carries ~21-23% of the structure. In 2-D an intact fold and a wrecked one look alike, so the figure
    could not support any claim about either.

    What it shows, and what it does NOT: the good covers edit 14-16 of the graph's ~1,357 edges, and moving
    ~1% of the edges cannot move a 340-point Procrustes disparity -- so `observed`, `domr`, `gmr_ilp` and
    `iomr_ilp` come out VISUALLY IDENTICAL. That is the finding, not a bug, and this is emphatically NOT a
    disparity-improvement figure. The claim it does support: exact repair leaves the fold INTACT, while
    `pivot` and `left_edge` -- which cover ~74% and ~79% of the entire edge set -- destroy it."""
    import matplotlib.pyplot as plt
    made = 0
    for g in NMR_GRAPHS:
        grp = sw[sw["graph"] == g]
        tag = _tag(g, "none")
        if grp.empty or f"true::{tag}" not in semb:
            print(f"    skip fig_nmr_fold_3d {g} (no sweep rows / embeddings)"); continue
        res, color = _fold_info(g, semb, tag)        # ASSERTS the re-derived core against the stored rows
        dim = _dim_of(semb, tag)
        if dim < 3:
            print(f"    skip fig_nmr_fold_3d {g} (stored dim={dim}, not 3-D)"); continue
        m = edge_count(g)

        # Every story panel keeps its slot whether or not it landed -- a missing algorithm becomes a labelled
        # null panel below, never a silent gap in the narrative.
        panels = [("__true__", None)]
        for a in FOLD_PANELS:
            r = grp[grp["algo"] == a]
            panels.append((a, r.iloc[0] if len(r) else None))
        # ONE cube for every panel. A per-panel cube rescales a wrecked fold back to the size of an intact
        # one -- and "is the fold still a fold" is the only question this figure asks.
        _cl = [semb[f"true::{tag}"]]
        for _a, _r in panels:
            if _a != "__true__" and f"emb::{tag}::{_a}::{algo}" in semb:
                _cl.append(semb[f"emb::{tag}::{_a}::{algo}"])
        lim = shared_limits(_cl, dim)
        ncol, nrow = 4, 2
        fig = plt.figure(figsize=(2.7 * ncol, 2.9 * nrow + 1.0))
        for i, (label, rec) in enumerate(panels):
            ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
            if label == "__true__":
                P = semb[f"true::{tag}"]
                _scatter3d(ax, P, color, res=res, lim=lim)
                ax.set_title("true structure (NMR)\nthe ubiquitin fold", fontsize=8, color="#222")
                _frame(ax, "#222222")
                continue
            key = f"emb::{tag}::{label}::{algo}"
            if key not in semb:
                _msg(ax, f"{label}\nNO COVER\n({_null_reason(rec)})", "#B00")
                ax.set_title(label, fontsize=8, color="#B00")
                _blank_ticks(ax); _frame(ax, "#BB0000", ls=":"); _apply_lim(ax, lim, dim)
                continue
            P = semb[key]
            _scatter3d(ax, P, color, res=res, lim=lim)
            d = _get(rec, "disp_smacof", np.nan)
            cs = _get(rec, "cover_size", np.nan)
            pct = f"{100.0 * float(cs) / m:.0f}% of $|E|$" if str(cs) != "" and np.isfinite(float(cs)) else "?"
            if label == "observed":
                ttl = f"observed (no repair)\ndisp {_fmt(d, 4)}   $|S|=0$"
            else:
                extra = "\n$=$ observed (Lemma 6.1)" if label == "domr" else ""
                ttl = f"{label}\ndisp {_fmt(d, 4)}   $|S|={_fmt(cs)}$ ({pct}){extra}"
            wrecker = label in ("pivot", "left_edge")
            ax.set_title(ttl, fontsize=7.5, color="#B00" if wrecker else "#222")
            _frame(ax, "#BB0000" if wrecker else VAR_COLOR.get(str(_get(rec, "variant", "")), "#555555"))

        rec0 = grp.iloc[0]
        # the colour bar goes in the leftover cell -- residue index is the figure's only legend
        cax = fig.add_axes([0.86, 0.14, 0.012, 0.30])
        sm = plt.cm.ScalarMappable(cmap="viridis",
                                   norm=plt.Normalize(vmin=float(res.min()), vmax=float(res.max())))
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label("residue index (N $\\to$ C)", fontsize=7)
        cb.ax.tick_params(labelsize=6)
        # The headline is COMPUTED, not asserted: "destroys the fold" is earned on residue (left_edge is 3.1x
        # the observed disparity) but would be an overstatement on atom (pivot is 1.4x). So state the measured
        # multiple and let it carry the claim, per graph.
        obs_d = _get(grp[grp["algo"] == "observed"].iloc[0], "disp_smacof", np.nan) \
            if len(grp[grp["algo"] == "observed"]) else np.nan
        wreck = grp[grp["algo"].isin(["pivot", "left_edge"])].sort_values("disp_smacof")
        head = ""
        if len(wreck) and np.isfinite(float(obs_d) or np.nan):
            w = wreck.iloc[-1]
            head = (f" -- the fold survives exact repair; {w['algo']} ({100.0 * float(w['cover_size']) / m:.0f}"
                    f"% of $|E|$) drives the disparity to {float(w['disp_smacof']) / float(obs_d):.1f}"
                    f"$\\times$ observed")
        fig.suptitle(
            f"{GRAPH_TITLE.get(g, g)}{head}\n"
            f"{_provenance(rec0, algo, dim)}{_proj_note(dim)}, $|E|={m}$\n"
            "colour = residue index; grey line = backbone through the residue centroids.  "
            "observed / domr / gmr_ilp / iomr_ilp are visually IDENTICAL -- they edit 1-5% of the edges, so "
            "they cannot move the disparity.  This is NOT a disparity-improvement figure.",
            fontsize=8)
        fig.tight_layout(rect=(0, 0, 0.85, 0.94))
        fig.subplots_adjust(wspace=0.02, hspace=0.12, top=0.86)
        _save(fig, outdir, f"fig_nmr_fold_3d_{g}")
        made += 1
    print(f"    fig_nmr_fold_3d: {made} figure(s)")


# ----------------------------------------------------------------------------
# fig_dimacs_distance_vs_time -- repair fixes broken metrics, not faithful reports of a different geometry
# ----------------------------------------------------------------------------
def fig_dimacs_distance_vs_time(sw, semb, outdir, algo="smacof", repair="gmr_ilp"):
    """NY road network, truth = geography for both graphs. Road DISTANCE embeds close to the map; travel TIME
    does not -- and repair barely moves it.

    The paper's thesis on real data, stated as a null and left as one: `dimacs_ny_t` carries only 17 heavy
    edges out of 6,017 (0.28% non-metric), so it is not a broken metric at all. Its gap to geography is
    highway physics -- travel time is a FAITHFUL report of a different geometry -- and metric repair has
    nothing to fix. Do not dress this up as a recovery."""
    import matplotlib.pyplot as plt
    need = ["dimacs_ny_d::none", "dimacs_ny_t::none"]
    if any(f"true::{t}" not in semb for t in need):
        print("    skip fig_dimacs_distance_vs_time (no dimacs embeddings)"); return

    def _row(graph, a):
        r = sw[(sw["graph"] == graph) & (sw["algo"] == a)]
        return r.iloc[0] if len(r) else None

    m = edge_count("dimacs_ny_t")
    hd = _row("dimacs_ny_t", "domr")
    H_t = _get(hd, "cover_size", np.nan) if hd is not None else np.nan
    panels = [
        ("__true__", "dimacs_ny_t", None,
         "true geography\n(DIMACS lat/lon)"),
        ("observed", "dimacs_ny_d", _row("dimacs_ny_d", "observed"),
         "road DISTANCE, observed\nexactly metric: $|H|=0$"),
        ("observed", "dimacs_ny_t", _row("dimacs_ny_t", "observed"),
         f"travel TIME, observed\n$|H|={_fmt(H_t)}$ of {m} edges "
         f"({100.0 * float(H_t) / m:.2f}% non-metric)"),
        (repair, "dimacs_ny_t", _row("dimacs_ny_t", repair),
         f"travel TIME, repaired ({repair})\nrepair has almost nothing to fix"),
    ]
    color = np.asarray(semb["color::dimacs_ny_t::none"])
    # All four panels are Procrustes-aligned to the SAME truth (geography), so one shared box is not merely
    # safe here, it is required: the claim is that travel time embeds WORSE, and a per-panel autoscale would
    # rescale the worse embedding until it looked just as good.
    _cl = []
    for _lab, _g, _r, _t in panels:
        _k = f"true::{_tag(_g, 'none')}" if _lab == "__true__" else f"emb::{_tag(_g, 'none')}::{_lab}::{algo}"
        if _k in semb:
            _cl.append(semb[_k])
    lim = shared_limits(_cl, 2)
    fig, axes = plt.subplots(1, 4, figsize=(3.1 * 4, 3.9))
    for ax, (lab, graph, rec, ttl) in zip(axes, panels):
        tag = _tag(graph, "none")
        if lab == "__true__":
            P, sub = semb[f"true::{tag}"], ""
        else:
            key = f"emb::{tag}::{lab}::{algo}"
            if key not in semb:
                _msg(ax, f"{lab}\nNO COVER\n({_get(rec, 'status', 'not measured')})", "#B00")
                ax.set_title(ttl, fontsize=8, color="#B00"); _blank_ticks(ax); _frame(ax, "#BB0000", ls=":")
                _apply_lim(ax, lim, 2)
                continue
            P = semb[key]
            sub = f"\ndisparity to geography {_fmt(_get(rec, 'disp_smacof', np.nan), 4)}"
        _scatter(ax, P, color, 2, s=3, lim=lim)
        ax.set_title(ttl + sub, fontsize=8)
        _frame(ax, "#222222" if lab == "__true__" else VAR_COLOR.get(str(_get(rec, "variant", "")), "#555"))
    d_obs = _get(_row("dimacs_ny_d", "observed"), "disp_smacof", np.nan)
    t_obs = _get(_row("dimacs_ny_t", "observed"), "disp_smacof", np.nan)
    fold = float(t_obs) / float(d_obs) if np.isfinite(float(d_obs)) and float(d_obs) else float("nan")
    # Disclose the BEST cover in the whole roster, not just the exact one we chose to draw. Showing gmr_ilp
    # alone and calling the result a null would invite exactly the objection that we picked the repair that
    # flatters the claim. The best cover does better -- and still does not reach the road-distance panel.
    rep = sw[(sw["graph"] == "dimacs_ny_t") & (sw["variant"] != "observed") & sw["disp_smacof"].notna()]
    best = rep.sort_values("disp_smacof").iloc[0] if len(rep) else None
    bnote = ""
    if best is not None:
        bnote = (f"  Best cover in the roster ({best['algo']}, $|S|$={_fmt(best['cover_size'])}) reaches only "
                 f"{_fmt(best['disp_smacof'], 4)} -- still {float(best['disp_smacof']) / float(d_obs):.1f}"
                 f"$\\times$ the road-distance panel.")
    fig.suptitle(
        "NY road network (DIMACS), 5000 nodes, 6017 edges -- truth = geography for BOTH graphs\n"
        f"road distance embeds to {_fmt(d_obs, 4)}; travel time to {_fmt(t_obs, 4)} -- "
        f"{fold:.1f}$\\times$ worse. Travel time is not a BROKEN metric, it is a FAITHFUL report of a "
        "different geometry, and repair cannot close that gap.\n" + bnote, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    _save(fig, outdir, "fig_dimacs_distance_vs_time")


def check_instances(sw, semb):
    """Which instances the plotter is READY to draw and the data does not contain.

    GRAPH_TITLE is the roster of instances this module knows how to draw. When the sweep produced only some
    of them, the figures come out looking complete -- there is simply one fewer file in the output directory,
    and a directory listing does not raise. That is how a missing arm of an experiment turns into a paper
    claim about the arm that DID run. So we diff the roster against the data and say the gap out loud, once,
    every run. This checks nothing about mds_plots.py; it checks that the CSV/npz it was handed are whole."""
    have = set(sw["graph"].unique()) if len(sw) else set()
    missing = [g for g in GRAPH_TITLE if g not in have]
    print("  instance roster (GRAPH_TITLE vs the sweep data)")
    print(f"    {len(GRAPH_TITLE) - len(missing)}/{len(GRAPH_TITLE)} drawable")
    if missing:
        print("    *** NO DATA (the plotter is ready; the sweep did not produce these): "
              + ", ".join(missing))
        print("    *** Any claim about these instances is unsupported. Do NOT infer one from a sibling that "
              "did run (a null deflate arm says nothing about the inflate arm).")
    return missing


def check_domr_identity(semb, algo="smacof", tol=1e-9):
    """The grid's built-in correctness check, run as a CHECK and not merely asserted in a panel title.

    Lemma 6.1: reweighting the heavy set to its detour leaves every shortest path unchanged, so D_F == D_G,
    so DOMR's embedding must be the observed embedding EXACTLY -- not merely close. A drifting DOMR panel is
    a pipeline bug (a permuted node order, a stale cover), and it is exactly the kind of bug that produces a
    plausible figure. So we measure it and say the number out loud."""
    tags = sorted({f"{k.split('::')[1]}::{k.split('::')[2]}" for k in semb.files if k.startswith("true::")})
    worst, checked = 0.0, 0
    print("  DOMR self-check (must be 0: D_F == D_G by Lemma 6.1)")
    for t in tags:
        o, d = f"emb::{t}::observed::{algo}", f"emb::{t}::domr::{algo}"
        if o not in semb.files or d not in semb.files:
            continue
        gap = float(np.abs(np.asarray(semb[o]) - np.asarray(semb[d])).max())
        worst = max(worst, gap); checked += 1
        if gap >= tol:
            print(f"    {t:32s} max|domr-observed| = {gap:.3e}   <-- MISMATCH, PIPELINE BUG")
    print(f"    {checked} tag(s) checked, max |gap| = {worst:.2e}  "
          f"({'OK' if worst < tol else 'CHECK PIPELINE'})")


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
    ap.add_argument("--sweep-data", default="analysis/summary_mds_sweep.csv")
    ap.add_argument("--sweep-emb", default="analysis/mds_sweep_embeddings.npz")
    ap.add_argument("--outdir", default="analysis/figs/mds")
    ap.add_argument("--algo", default="smacof", choices=["classical", "smacof"])
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    df = pd.read_csv(a.data)
    for c in ("procrustes_disp", "stress", "neg_mass", "dim", "n", "n_used"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    emb = np.load(a.emb, allow_pickle=True) if os.path.exists(a.emb) else {}
    print(f"loaded {len(df)} rows, {df['graph'].nunique()} graphs -> {a.outdir}")
    fig_mds_maps(df, emb, a.outdir, algo=a.algo)
    fig_mds_residual(df, a.outdir, algo=a.algo)
    fig_mds_negeig(df, a.outdir)

    # The per-algorithm grid needs BOTH the sweep CSV and the sweep npz. Warn and carry on if either is
    # missing (same idiom as the other guards) -- but say so, never skip silently.
    if not os.path.exists(a.sweep_data):
        print(f"warn: {a.sweep_data} absent -- skipping fig_mds_grid / fig_mds_sweep_rgg / table_mds_sweep."
              "  Run mds_sweep.py first."); return
    sw = pd.read_csv(a.sweep_data)
    for c in ("cover_size", "ratio_domr", "disp_classical", "disp_smacof", "neg_mass",
              "n", "n_used", "avg_degree", "frac", "magnitude", "seed", "radius"):
        if c in sw.columns:
            sw[c] = pd.to_numeric(sw[c], errors="coerce")
    print(f"loaded {len(sw)} sweep rows, {sw['graph'].nunique()} graphs")
    check_instances(sw, None)
    fig_mds_sweep_rgg(sw, a.outdir)
    table_mds_sweep(sw, a.outdir)
    if not os.path.exists(a.sweep_emb):
        print(f"warn: {a.sweep_emb} absent -- skipping fig_mds_grid.  Re-run mds_sweep.py to write it.")
    else:
        semb = np.load(a.sweep_emb, allow_pickle=True)
        check_domr_identity(semb, algo=a.algo)
        fig_mds_grid(sw, semb, a.outdir, algo=a.algo)
        fig_nmr_fold_3d(sw, semb, a.outdir, algo=a.algo)
        fig_dimacs_distance_vs_time(sw, semb, a.outdir, algo=a.algo)
    print(f"figures -> {a.outdir}")


if __name__ == "__main__":
    main()
