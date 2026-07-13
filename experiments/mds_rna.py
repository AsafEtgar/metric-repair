"""mds_rna.py -- MDS geometry recovery on the single-cell RNA graph (pbmc3k_cosine_knn), against the REAL
expression-space ground truth. A pure processing step: it reuses the covers the campaign already computed.

    sage -python experiments/mds_rna.py --outdir analysis

WHY THIS IS A GENUINE GROUND TRUTH, NOT A SELF-REFERENTIAL ONE. `build_pbmc3k` (build_real_graphs.py:325)
log-CPM normalizes the 10x counts, takes the top-2000 highly-variable genes, projects to 50 PCs, row-normalizes
to `Pn`, and then sets each k-NN edge weight to `1 - (Pn @ Pn.T)[i, j]`. So the ambient matrix

        d*  =  1 - Pn @ Pn.T          (all 2700 x 2700 pairs, cosine distance in PCA-50 space)

is not a different space we are smuggling in: it is EXACTLY the space the edge weights already live in. The
graph retains only 15 neighbours per cell; its shortest paths extend those 15 to all pairs. Asking whether the
graph's shortest-path metric recovers d* is therefore the same question `ripe_atlas` asks of geography and
`nmr_1d3z` asks of the PDB structure -- an external truth the graph is trying to approximate. This makes the
RNA graph a THIRD real modality for the pure-real recovery experiment.

WHAT IT SHOWS (measured, and it is not what we predicted). The graph is natively almost metric -- |H| = 66 of
31,639 edges (0.21%) -- so we expected repair to do nothing here, as it does on `nmr`. It does not do nothing.
EVERY one of the fourteen covers beats the observed graph, and the SMALLEST cover helps most: the exact GMR
optimum edits 32 edges (0.1% of the graph) and improves the Procrustes disparity from 0.148771 to 0.148139.
The gain is small in absolute terms, but it is unanimous and it is ordered -- this is the first real graph in
the study on which repair consistently improves the geometry (on `ripe_atlas` repair actively HURTS; on `nmr`
it is a wash).

And the anti-correlation with cover size is violent at the other end: `left_edge` covers 26,830 edges (84.8% of
the graph) and TRIPLES the disparity, 0.149 -> 0.493. Cover size and geometric fidelity pull against each other.

DOMR is the control: by Lemma 6.1 it leaves every shortest-path distance unchanged, so its disparity must equal
the observed one exactly. It does, to 0.00e+00, on real single-cell data.

`pivot` is ABSENT, not omitted: the campaign ran it on this graph (|S| = 23,599) but did not persist its cover,
and no cover file exists under results_real_covers/pbmc3k_cosine_knn/. The figure says so.

COLOUR. Louvain communities of the graph (networkx; no new dependency). A VISUAL AID only -- derived from the
same data, never claimed as external cell-type labels and never used as a truth.

THE ORDERING TRAP. `load_graph` sorts node labels as STRINGS, so '10' sorts before '2'. The graph's node labels
are the ROW INDICES of the expression matrix, so the truth must be permuted into the loader's order with
int(label) -- not assumed to line up. Getting this wrong yields a plausible, stable, and completely meaningless
disparity. It is asserted below, not commented.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx                                                          # noqa: E402
from networkx.algorithms.community import louvain_communities                  # noqa: E402

from downstream_recovery import load_graph, apsp, build_F_distances            # noqa: E402
from mds_recovery import (classical_mds, smacof, _procrustes_disp,             # noqa: E402
                          finite_core, _nx_from_edges, _norm_cover)

GRAPH = "pbmc3k_cosine_knn"
DIM = 2
SEED = 1000
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VARIANT_OF = {
    "domr": "DOMR",
    "gmr_ilp": "GMR", "gmr_bestofk": "GMR", "gmr_rand": "GMR", "gmr_thr_naive": "GMR",
    "l1sep_gmr": "GMR", "spc_gmr": "GMR", "pivot": "GMR",
    "iomr_ilp": "IOMR", "iomr_bestofk": "IOMR", "iomr_rand": "IOMR", "iomr_thr_naive": "IOMR",
    "iomr_regiongrow": "IOMR", "l1sep_iomr": "IOMR", "spc_iomr": "IOMR", "left_edge": "IOMR",
}

FIELDS = ["graph", "truth", "algo", "variant", "cover_size", "ratio_domr", "cover_frac_of_m",
          "disp_classical", "disp_smacof", "neg_mass", "n", "n_used", "m", "H", "dim", "cover_file"]


def ambient_truth(n_cells):
    """d* = 1 - Pn @ Pn.T, reproducing build_pbmc3k's space EXACTLY (log-CPM -> 2000 HVG -> 50 PCs -> cosine)."""
    from scipy.io import mmread
    d = os.path.join(REPO, "data", "raw", "scrna_pbmc3k", "filtered_gene_bc_matrices", "hg19")
    X = mmread(os.path.join(d, "matrix.mtx")).tocsc().T.toarray().astype(float)      # cells x genes
    lib = X.sum(1, keepdims=True)
    lib[lib == 0] = 1
    X = np.log1p(X / lib * 1e4)
    var = X.var(0)
    hv = np.argsort(var)[-2000:]
    Xh = X[:, hv]
    Xh -= Xh.mean(0)
    U, S, _Vt = np.linalg.svd(Xh, full_matrices=False)
    P = U[:, :50] * S[:50]
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    assert Pn.shape[0] == n_cells, f"expression matrix has {Pn.shape[0]} cells, graph has {n_cells}"
    Dstar = 1.0 - (Pn @ Pn.T)
    np.fill_diagonal(Dstar, 0.0)
    return np.maximum(Dstar, 0.0)


def saved_covers(covers_root):
    """{algo: (cover_as_index_pairs, filename)} from the campaign's own covers for the NATIVE graph."""
    cdir = os.path.join(covers_root, GRAPH)
    out = {}
    for path in sorted(glob.glob(os.path.join(cdir, "*.txt"))):
        algo = os.path.basename(path)[:-4].split("__")[0]
        if algo in out:
            continue                                    # first seed file per algo, deterministically
        out[algo] = path
    return out


def load_cover(path, idx):
    """Cover files store ORIGINAL node labels, one 'u v' pair per line. Map into the loader's index space."""
    cov = set()
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            u, v = parts
            if u not in idx or v not in idx:
                continue
            iu, iv = idx[u], idx[v]
            cov.add((iu, iv) if iu <= iv else (iv, iu))
    return cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="analysis")
    ap.add_argument("--covers-root", default=None)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    covers_root = a.covers_root
    if covers_root is None:
        for cand in ("results_real_covers", os.path.join("results_real", "results_real_covers")):
            if os.path.isdir(os.path.join(cand, GRAPH)):
                covers_root = cand
                break
    if covers_root is None:
        sys.exit("no saved covers found for %s -- looked for results_real_covers/ and "
                 "results_real/results_real_covers/" % GRAPH)

    np.random.seed(SEED)

    # ---- the graph, in ONE node order for the whole file ----------------------------------------------
    nodes, idx, edges = load_graph(GRAPH)
    n, m = len(nodes), len(edges)
    print(f"[{GRAPH}] n={n}  m={m}  avg_deg={2.0 * m / n:.1f}  covers_root={covers_root}", flush=True)

    # ---- THE ORDERING TRAP. load_graph sorts labels as STRINGS ('10' < '2'). The graph's node labels are
    #      the expression matrix's ROW INDICES, so permute the truth into the loader's order. Assert it.
    cell_of_row = np.array([int(lbl) for lbl in nodes])
    assert sorted(cell_of_row.tolist()) == list(range(n)), \
        "node labels are not a permutation of 0..n-1 -- the truth cannot be aligned"
    assert not np.array_equal(cell_of_row, np.arange(n)), \
        "expected a non-identity permutation (lexicographic vs numeric); check load_graph"
    print(f"  node order: lexicographic, NOT numeric (e.g. {nodes[:4]}) -> truth permuted by int(label)",
          flush=True)

    Dstar_cells = ambient_truth(n)
    Dstar = Dstar_cells[np.ix_(cell_of_row, cell_of_row)]        # into the loader's index space

    # sanity: every graph edge weight must equal d* on that pair, since the build sets w = 1 - sim
    err = max(abs(w - Dstar[iu, iv]) for iu, iv, w in edges[:2000])
    print(f"  truth check: max |edge weight - d*| over 2000 edges = {err:.2e}  "
          f"{'OK -- the graph lives in the truth space' if err < 1e-6 else '*** MISALIGNED ***'}", flush=True)
    assert err < 1e-6, "edge weights do not match d*: the truth is in the wrong space or the wrong order"

    # ---- the true configuration, and the observed graph ------------------------------------------------
    D_obs = apsp(edges, n)
    core = finite_core(D_obs)
    n_used = len(core)

    def restrict(D):
        return D[np.ix_(core, core)]

    true_cfg, _ = classical_mds(restrict(Dstar), DIM)
    print(f"  finite core {n_used}/{n};  truth = classical MDS of d* (expression space), dim={DIM}", flush=True)

    G = _nx_from_edges(edges, n)
    comms = louvain_communities(G, seed=SEED)
    community_of = np.full(n, -1, dtype=int)
    for ci, cset in enumerate(comms):
        for node in cset:
            community_of[node] = ci
    print(f"  Louvain: {len(comms)} communities (colour only, never a truth claim)", flush=True)

    H_set = _norm_cover({(u, v) for (u, v, w) in edges if w > D_obs[u, v] + 1e-9})
    H = max(len(H_set), 1)
    print(f"  |H| = {H} of {m} edges ({H / m:.2%}) -- the graph is natively almost metric\n", flush=True)

    store, rows = {}, []
    tag = f"{GRAPH}::none"
    t = true_cfg - true_cfg.mean(axis=0)
    store[f"true::{tag}"] = t / (np.linalg.norm(t) or 1.0)
    store[f"color::{tag}"] = community_of[core].astype(float)

    def score_and_record(algo, variant, D, size, cover_file):
        Yc, neg = classical_mds(restrict(D), DIM)
        Ys, _ = smacof(restrict(D), DIM, init=Yc)
        dc, _, Ac = _procrustes_disp(true_cfg, Yc)
        ds, _, As = _procrustes_disp(true_cfg, Ys)
        store[f"emb::{tag}::{algo}::classical"] = Ac
        store[f"emb::{tag}::{algo}::smacof"] = As
        rows.append({
            "graph": GRAPH, "truth": "expression space (1 - cosine, PCA-50)", "algo": algo, "variant": variant,
            "cover_size": "" if size is None else int(size),
            "ratio_domr": "" if size is None else round(size / H, 4),
            "cover_frac_of_m": "" if size is None else round(size / m, 4),
            "disp_classical": round(float(dc), 6), "disp_smacof": round(float(ds), 6),
            "neg_mass": round(float(neg), 6), "n": n, "n_used": n_used, "m": m, "H": H, "dim": DIM,
            "cover_file": cover_file,
        })
        return ds

    obs = score_and_record("observed", "observed", D_obs, 0, "")
    print(f"  {'observed':16s} (no repair)          disparity {obs:.6f}   <- the floor the graph starts at",
          flush=True)

    for algo, path in sorted(saved_covers(covers_root).items()):
        cover = load_cover(path, idx)
        if not cover:
            print(f"  [{algo:16s}] empty cover -- skipped", flush=True)
            continue
        D_rep = build_F_distances(edges, cover, n)
        ds = score_and_record(algo, VARIANT_OF.get(algo, "?"), D_rep, len(cover), os.path.basename(path))
        delta = ds - obs
        flag = "  <- WRECKS THE MAP" if delta > 0.05 else ("  <- beats observed" if delta < -1e-4 else "")
        print(f"  {algo:16s} |S|={len(cover):6d} ({len(cover)/m:6.1%} of m)  disparity {ds:.6f} "
              f"({delta:+.6f}){flag}", flush=True)

    csv_path = os.path.join(a.outdir, "summary_mds_rna.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    npz_path = os.path.join(a.outdir, "mds_rna_embeddings.npz")
    np.savez_compressed(npz_path, **store)
    print(f"\nwrote {csv_path} ({len(rows)} rows), {npz_path} ({len(store)} arrays)", flush=True)

    # DOMR self-check (Lemma 6.1): a decrease-only cover leaves every shortest-path distance unchanged, so its
    # embedding must equal the observed one. A nonzero gap is a pipeline error, not a result.
    o = [r for r in rows if r["algo"] == "observed"]
    d = [r for r in rows if r["algo"] == "domr"]
    if o and d:
        gap = abs(d[0]["disp_smacof"] - o[0]["disp_smacof"])
        print(f"DOMR self-check: |domr - observed| = {gap:.2e}  "
              f"{'OK' if gap < 1e-9 else '*** NONZERO -- PIPELINE ERROR ***'}", flush=True)

    if a.plot:
        plot_grid(rows, store, tag, a.outdir, m)


def plot_grid(rows, store, tag, outdir, m):
    """The map: true expression-space layout, the observed graph, then every cover, coloured by community."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    drawn = [r for r in rows if f"emb::{tag}::{r['algo']}::smacof" in store]
    obs = [r for r in drawn if r["algo"] == "observed"][0]
    rest = sorted([r for r in drawn if r["algo"] != "observed"], key=lambda r: r["disp_smacof"])
    panels = [("__true__", None)] + [("observed", obs)] + [(r["algo"], r) for r in rest]

    color = store[f"color::{tag}"]
    ncols = 4
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.2 * nrows), squeeze=False)

    # ONE axis box for every panel, as the union over all of them. Every panel is Procrustes-aligned into the
    # same expression-space frame, so a per-panel autoscale would zoom a wrecked embedding (left_edge triples
    # the disparity) back to the size of an intact one and hide the very thing the grid is drawn to show.
    clouds = [store[f"true::{tag}"]]
    for algo, _r in panels:
        if algo != "__true__":
            clouds.append(store[f"emb::{tag}::{algo}::smacof"])
    lo = np.min([np.asarray(P, float)[:, :2].min(0) for P in clouds], axis=0)
    hi = np.max([np.asarray(P, float)[:, :2].max(0) for P in clouds], axis=0)
    mid, half = (lo + hi) / 2.0, np.maximum((hi - lo) / 2.0, 1e-12) * 1.04

    for ax, (algo, r) in zip([a for row in axes for a in row], panels):
        P = store[f"true::{tag}"] if algo == "__true__" else store[f"emb::{tag}::{algo}::smacof"]
        ax.scatter(P[:, 0], P[:, 1], c=color, cmap="tab10", s=5, linewidth=0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(mid[0] - half[0], mid[0] + half[0])
        ax.set_ylim(mid[1] - half[1], mid[1] + half[1])
        ax.set_aspect("equal", "box")
        if algo == "__true__":
            ax.set_title("TRUE  (expression space)\nthe target", fontsize=9, fontweight="bold")
            for s in ax.spines.values():
                s.set_color("black"); s.set_linewidth(1.8)
        else:
            frac = r["cover_frac_of_m"]
            delta = r["disp_smacof"] - obs["disp_smacof"]
            sub = "" if algo == "observed" else f"\n|S|={r['cover_size']} ({frac:.1%} of m)   {delta:+.4f}"
            ax.set_title(f"{algo}\ndisp {r['disp_smacof']:.4f}{sub}", fontsize=8)
            if delta > 0.05:
                for s in ax.spines.values():
                    s.set_color("#D55E00"); s.set_linewidth(1.8)
            elif algo != "observed" and delta < -1e-4:
                for s in ax.spines.values():
                    s.set_color("#009E73"); s.set_linewidth(1.4)

    for ax in [a for row in axes for a in row][len(panels):]:
        ax.set_axis_off()

    fig.suptitle(
        "pbmc3k_cosine_knn (2,700 cells, 31,639 edges)  |  truth = ambient cosine distance in PCA-50 "
        "expression space -- the space the edge weights are themselves drawn from\n"
        "colour = Louvain community of the graph (a visual aid, NOT a cell-type label). "
        "Green frame: beats the observed graph. Orange: worsens it.\n"
        "Every one of the 14 covers beats `observed`, and the SMALLEST (gmr_ilp, 32 edges = 0.1% of m) beats it "
        "most -- but the gain is 0.4%, so the repaired panels are visually identical to `observed`.\n"
        "The visible effect is at the other end: left_edge covers 84.8% of the graph and triples the disparity "
        "(0.149 -> 0.493), collapsing the inter-cluster geometry -- the communities survive, their spatial "
        "relations do not.\n"
        "DOMR equals observed exactly (Lemma 6.1). pivot has NO saved cover for this graph (the campaign ran "
        "it, |S|=23,599, but never persisted it), so it is absent rather than omitted.",
        fontsize=8, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    for ext in ("pdf", "png"):
        path = os.path.join(outdir, f"fig_mds_rna.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.join(outdir, 'fig_mds_rna')}.{{pdf,png}}  ({len(panels)} panels)", flush=True)


if __name__ == "__main__":
    main()
