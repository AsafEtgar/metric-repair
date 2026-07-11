"""mds_sweep.py -- per-ALGORITHM MDS geometry: Procrustes disparity for every repair algorithm, not
just one representative cover per variant (mds_recovery.py's default). Answers the sharper question:
does the geometry recovery depend on WHICH algorithm -- i.e. on cover quality/size?

  real graphs (ripe/nmr): every saved campaign cover -> disparity + cover size |S|/|H| (cheap: just a
                          build_F + MDS per cover). This is the headline comparison (a compact table),
                          and the disparity-vs-cover-size scatter is the appendix view.
  RGG controls:           recompute each algorithm's cover -> disparity (numbers only, no per-graph
                          figures, per the plan).

Reuses mds_recovery.py's MDS + scoring helpers verbatim, so it cannot diverge from the main analysis.
Self-contained otherwise; never imported by a running task.

    sage -python experiments/mds_sweep.py --outdir analysis --plot
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                       # experiments/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # repo root
from mds_recovery import (classical_mds, smacof, _procrustes_disp, finite_core,       # noqa: E402
                          _nx_from_edges, _norm_cover, _bestofk, PURE_REAL_DIM, RGG_SPECS)
from downstream_recovery import (DOWNSTREAM_GRAPHS, load_graph, load_cover, _covers_dir,  # noqa: E402
                                 apsp, build_F_distances, true_distances, variant_of)
from graph_models import seed_all, random_geometric_metric_graph, break_metric_graph  # noqa: E402
from metric_repair import (domr_alg, shortest_path_cover, l1_separation,              # noqa: E402
                           pivot_heuristic, left_edge_heuristic)

# RGG recompute set: algorithm -> (cover function, variant). One representative per repair family, GMR
# and IOMR both. ILP is omitted (NP-hard, unreliable convergence at n=300); the covering-LP bestofk is
# the near-optimal stand-in.
SWEEP_ALGOS = {
    "domr":         (lambda G: domr_alg(G),                           "DOMR"),
    "gmr_bestofk":  (lambda G: _bestofk(G, iomr=False),               "GMR"),
    "iomr_bestofk": (lambda G: _bestofk(G, iomr=True),                "IOMR"),
    "spc_gmr":      (lambda G: shortest_path_cover(G, general=True),  "GMR"),
    "spc_iomr":     (lambda G: shortest_path_cover(G, general=False), "IOMR"),
    "l1sep_gmr":    (lambda G: l1_separation(G, general=True),        "GMR"),
    "l1sep_iomr":   (lambda G: l1_separation(G, general=False),       "IOMR"),
    "pivot":        (lambda G: pivot_heuristic(G),                    "GMR"),
    "left_edge":    (lambda G: left_edge_heuristic(G),                "IOMR"),
}
SWEEP_FIELDS = ["dataset", "graph", "corruption", "algo", "variant", "cover_size", "ratio_domr",
                "disp_classical", "disp_smacof", "neg_mass"]

VAR_COLOR = {"observed": "#000000", "DOMR": "#888888", "GMR": "#0072B2", "IOMR": "#D55E00"}
REAL_TITLE = {"ripe_atlas": "ripe_atlas", "nmr_1d3z_atom": "nmr_1d3z_atom",
              "nmr_1d3z_residue": "nmr_1d3z_residue"}


def _score(D, true_cfg, dim):
    """Classical + SMACOF Procrustes disparity to true_cfg, plus neg_mass (a property of D)."""
    Yc, neg = classical_mds(D, dim)
    Ys, _ = smacof(D, dim, init=Yc)
    dc, _, _ = _procrustes_disp(true_cfg, Yc)
    ds, _, _ = _procrustes_disp(true_cfg, Ys)
    return dc, ds, neg


def _row(dataset, graph, corruption, algo, variant, size, H, dc, ds, neg):
    return {"dataset": dataset, "graph": graph, "corruption": corruption, "algo": algo,
            "variant": variant, "cover_size": size, "ratio_domr": round(size / max(H, 1), 4),
            "disp_classical": round(dc, 6) if np.isfinite(dc) else "",
            "disp_smacof": round(ds, 6) if np.isfinite(ds) else "",
            "neg_mass": round(neg, 6) if np.isfinite(neg) else ""}


def _iter_saved(graph, idx, covers_root):
    """One representative saved cover per algorithm (first sorted file; randomized algos at one seed)."""
    cdir = _covers_dir(covers_root, graph) if covers_root else None
    if not cdir:
        return
    seen = set()
    for cf in sorted(glob.glob(os.path.join(cdir, "*.txt"))):
        algo = os.path.basename(cf)[:-4].split("__")[0]
        if algo in seen:
            continue
        seen.add(algo)
        yield algo, load_cover(cf, idx)


def run_real_sweep(graph, covers_root):
    """Every saved cover of a real graph -> disparity per algorithm, scored against the external truth."""
    print(f"[sweep:real] {graph}", flush=True)
    nodes, idx, edges = load_graph(graph)
    n = len(nodes)
    gt_ix, Dtrue = true_distances(graph, nodes)
    dim = PURE_REAL_DIM[graph]
    Dobs_gt = apsp(edges, n)[np.ix_(gt_ix, gt_ix)]
    core = finite_core(Dobs_gt)
    true_cfg, _ = classical_mds(Dtrue[np.ix_(core, core)], dim)

    def restrict(D):
        return D[np.ix_(gt_ix, gt_ix)][np.ix_(core, core)]

    H = max(len(_norm_cover(domr_alg(_nx_from_edges(edges, n)))), 1)
    rows = [_row("pure_real", graph, "none", "observed", "observed", 0, H,
                 *_score(Dobs_gt[np.ix_(core, core)], true_cfg, dim))]
    for algo, cover in _iter_saved(graph, idx, covers_root):
        try:
            dc, ds, neg = _score(restrict(build_F_distances(edges, cover, n)), true_cfg, dim)
        except Exception as e:                                        # noqa: BLE001
            print(f"    [skip {algo}] {type(e).__name__}: {e}", flush=True)
            continue
        rows.append(_row("pure_real", graph, "none", algo, variant_of(algo), len(cover), H, dc, ds, neg))
    return rows


def run_rgg_sweep(name, n, deg, direction, frac, mag, seed):
    """Recompute each algorithm's cover on a broken RGG -> disparity per algorithm (numbers only)."""
    print(f"[sweep:rgg] {name}", flush=True)
    seed_all(seed)
    radius = float(np.sqrt(deg / (np.pi * max(n - 1, 1))))
    T = random_geometric_metric_graph(n, mode="radius", radius=radius)
    T = nx.convert_node_labels_to_integers(T.subgraph(max(nx.connected_components(T), key=len)).copy())
    C, _ = break_metric_graph(T, frac_q=frac, direction=direction, magnitude=mag)
    C = nx.convert_node_labels_to_integers(C.subgraph(max(nx.connected_components(C), key=len)).copy())
    nc = C.number_of_nodes()
    pos = np.array([C.nodes[i]["pos"] for i in range(nc)], dtype=float)
    edges = [(u, v, C[u][v]["weight"]) for u, v in C.edges()]
    Dobs = apsp(edges, nc)
    core = finite_core(Dobs)
    true_cfg = pos[core]

    def restrict(D):
        return D[np.ix_(core, core)]

    H = max(len(_norm_cover(domr_alg(C))), 1)
    rows = [_row("rgg", name, direction, "observed", "observed", 0, H, *_score(restrict(Dobs), true_cfg, 2))]
    for algo, (fn, variant) in SWEEP_ALGOS.items():
        try:
            cover = _norm_cover(fn(C))
            dc, ds, neg = _score(restrict(build_F_distances(edges, cover, nc)), true_cfg, 2)
        except Exception as e:                                        # noqa: BLE001
            print(f"    [skip {algo}] {type(e).__name__}: {e}", flush=True)
            continue
        rows.append(_row("rgg", name, direction, algo, variant, len(cover), H, dc, ds, neg))
    return rows


def fig_sweep_real(rows, outdir):
    """Appendix scatter: per real graph, SMACOF disparity vs cover size |S|/|H|, one point per
    algorithm, coloured by variant. Lower-left = edits little AND keeps geometry."""
    import matplotlib.pyplot as plt
    graphs = [g for g in REAL_TITLE if any(r["graph"] == g for r in rows)]
    if not graphs:
        print("    skip fig_sweep_real (no real rows)"); return
    fig, axes = plt.subplots(1, len(graphs), figsize=(4.3 * len(graphs), 3.8), squeeze=False)
    for ax, g in zip(axes[0], graphs):
        sub = [r for r in rows if r["graph"] == g and r["disp_smacof"] != ""]
        obs = next((float(r["disp_smacof"]) for r in sub if r["variant"] == "observed"), None)
        if obs is not None:
            ax.axhline(obs, color="black", lw=0.8, ls=":", label="observed / DOMR")
        for r in sub:
            if r["variant"] == "observed":
                continue
            x, y = float(r["ratio_domr"]), float(r["disp_smacof"])
            ax.scatter(x, y, s=42, color=VAR_COLOR.get(r["variant"], "#555"),
                       edgecolor="white", linewidth=0.5, zorder=3)
            ax.annotate(r["algo"], (x, y), fontsize=6, xytext=(3, 2),
                        textcoords="offset points", alpha=0.8)
        ax.set_title(REAL_TITLE.get(g, g), fontsize=9)
        ax.set_xlabel("cover size  $|S|/|H|$"); ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("Procrustes disparity (SMACOF)\n($\\downarrow$ closer to truth)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=VAR_COLOR[v], label=v) for v in ("GMR", "IOMR")]
    handles.append(plt.Line2D([], [], color="black", ls=":", label="observed / DOMR"))
    axes[0][-1].legend(handles=handles, fontsize=7, frameon=False, loc="best")
    fig.suptitle("Does editing less preserve geometry better?  Disparity vs cover size, per algorithm.",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig_mds_sweep_real.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    wrote fig_mds_sweep_real.pdf / .png")


def _report(rows):
    """Per-graph algorithm ranking by SMACOF disparity (best first), to eyeball the sweep."""
    print("\n=== per-graph disparity ranking (SMACOF; observed baseline first) ===")
    for g in dict.fromkeys(r["graph"] for r in rows):
        sub = [r for r in rows if r["graph"] == g and r["disp_smacof"] != ""]
        obs = next((r for r in sub if r["variant"] == "observed"), None)
        ranked = sorted((r for r in sub if r["variant"] != "observed"), key=lambda r: float(r["disp_smacof"]))
        base = f"obs={float(obs['disp_smacof']):.4f}" if obs else "obs=?"
        print(f"  {g:18s} {base}")
        for r in ranked:
            tag = "  <-- beats observed" if obs and float(r["disp_smacof"]) < float(obs["disp_smacof"]) else ""
            print(f"      {r['algo']:16s} {r['variant']:5s} |S|/|H|={r['ratio_domr']:>7} "
                  f"disp={float(r['disp_smacof']):.4f}{tag}")


def _default_covers_root():
    for cand in ("results_real_covers", os.path.join("results_real", "results_real_covers")):
        if os.path.isdir(cand):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="analysis")
    ap.add_argument("--covers-root", default=None)
    ap.add_argument("--only", choices=["real", "rgg", "all"], default="all")
    ap.add_argument("--plot", action="store_true", help="also write fig_mds_sweep_real (needs matplotlib)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    covers_root = a.covers_root or _default_covers_root()
    print(f"covers_root = {covers_root or '(none)'}")

    rows = []
    if a.only in ("real", "all"):
        for g in sorted(DOWNSTREAM_GRAPHS, key=lambda x: 0 if x.startswith("nmr") else 1):
            try:
                rows += run_real_sweep(g, covers_root)
            except FileNotFoundError as e:
                print(f"  [skip {g}] {e}")
    if a.only in ("rgg", "all"):
        for spec in RGG_SPECS:
            rows += run_rgg_sweep(*spec)

    path = os.path.join(a.outdir, "summary_mds_sweep.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SWEEP_FIELDS)
        w.writeheader()
        w.writerows(rows)
    _report(rows)
    if a.plot:
        fig_sweep_real(rows, a.outdir)
    print(f"\nwrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
