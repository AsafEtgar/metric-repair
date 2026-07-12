"""mds_recovery.py -- geometry of repair: does metric repair pull a graph's distances back onto a
low-dimensional configuration that matches the TRUE one? (Advisor's MDS suggestion, done both ways.)

This is a third lens on the same Tier-2 question the kNN-lift / Spearman experiments ask, but GEOMETRIC
rather than local or ordinal. We embed a graph's shortest-path distance matrix into R^d with MDS and compare
the point cloud to a known true layout. It COMPLEMENTS downstream_recovery.py (it reuses that module's
distance builders verbatim -- same `restore`/APSP code, so it cannot diverge from the recovery experiment)
and adds nothing to any task's import path; run it off the cluster's critical path like cost_law.py.

Two settings, per the design's two Tier-2 axes (see experimental_design.tex):

  PURE REAL   ripe_atlas / nmr -- no planted corruption; the true layout is EXTERNAL (geography, 3-D protein
              structure). "Option 1": embed the observed distances D_G and the repaired distances D_F, and
              compare BOTH to the true configuration. The headline is ripe: latency MDS is a distorted map,
              repaired MDS should sit closer to real geography.
  BROKEN RGG  "Option 2" / the controlled companion: a random geometric graph is metric BY CONSTRUCTION with
              a KNOWN 2-D layout (its generating points). Break it (inflate or deflate), repair it, and ask
              whether the repaired embedding returns to the true points. Same question as ripe, but with
              ground truth we planted, so it is the clean control for the real-data claim.

Two MDS algorithms, both from the advisor's message (we will likely keep just the second):
  classical   Torgerson/PCoA: square, double-center B = -1/2 J D^2 J, eigendecompose, project the top-d
              eigenvectors. Deterministic, and it hands us the NEGATIVE-EIGENVALUE MASS for free -- a scalar
              measure of how far the distances are from Euclidean-embeddable, which repair should shrink.
  smacof      metric stress-majorization (Guttman transform), initialized from the classical solution so it
              is deterministic (no random restarts). A compact NumPy implementation -- identical objective to
              sklearn.manifold.MDS(dissimilarity="precomputed"), which is a one-line swap if that env has it.

Scoring (both a figure and a number, per the design):
  procrustes  disparity of the MDS embedding vs the true configuration after optimal translation/scale/
              rotation (scipy.spatial.procrustes; lower = closer to truth). The recovery metric.
  neg_mass    negative-eigenvalue mass of the distance matrix (classical route). The non-Euclidean-ness
              diagnostic; a property of the distances, not the embedding.
  stress      normalized stress-1 of the embedding (both algorithms).

DOMR is the self-check with teeth here too: reweighting the heavy set to its detour leaves every shortest
path unchanged (Lemma 6.1), so D_F == D_G exactly and DOMR's embedding, disparity and neg_mass EQUAL the
observed ones. A DOMR bar that does not sit on the observed bar is a pipeline bug.

    sage -python experiments/mds_recovery.py --outdir analysis                 # everything
    sage -python experiments/mds_recovery.py --outdir analysis --only rgg      # just the broken-RGG control
    sage -python experiments/mds_plots.py --data analysis/summary_mds.csv     # then the figures
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import procrustes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Reuse the downstream experiment's distance machinery verbatim -- the A1-correct sparse APSP and the
# `restore` F-construction -- so this analysis and the kNN/Spearman recovery numbers rest on identical code.
from downstream_recovery import (                                                  # noqa: E402
    DOWNSTREAM_GRAPHS, load_graph, load_cover, _covers_dir, apsp, build_F_distances, true_distances)
from graph_models import seed_all, random_geometric_metric_graph, break_metric_graph   # noqa: E402
from metric_repair import domr_alg, covering_lp_cover                              # noqa: E402

# For the real graphs we PREFER a cover the campaign already saved (results_real_covers/), exactly as
# downstream_recovery does -- so this figure rests on the same covers as the recovery numbers and costs only
# a build_F + MDS. One representative algorithm per variant, in preference order; the first whose cover file
# is present wins. DOMR is always recomputed instead (one APSP -- cheap and exact, and it guarantees the
# D_F == D_G self-check). If no saved cover is found we fall back to recomputing bestofk, but only when the
# graph is small enough (RECOMPUTE_EDGE_CAP) -- ripe_atlas is near-complete (~442k edges) and its covering LP
# is a cluster job, not a laptop one.
# Ordered smallest/most-optimal first (exact ILP, then near-optimal bestofk, ...) so "GMR"/"IOMR" is the
# fairest cover available -- the least it can edit. On ripe every cover is still huge (95% non-metric), which
# is the point, not an artifact.
GMR_PREFS = ["gmr_ilp", "gmr_bestofk", "gmr_rand", "l1sep_gmr", "gmr_thr_naive", "spc_gmr", "pivot"]
IOMR_PREFS = ["iomr_ilp", "iomr_bestofk", "iomr_rand", "l1sep_iomr", "iomr_regiongrow", "spc_iomr", "left_edge"]
RECOMPUTE_EDGE_CAP = 50_000

# Embedding dimension of the true layout, per dataset. ripe geography and the RGG points are 2-D; an NMR
# structure is 3-D.
PURE_REAL_DIM = {"ripe_atlas": 2, "nmr_1d3z_atom": 3, "nmr_1d3z_residue": 3,
                 "dimacs_ny_d": 2, "dimacs_ny_t": 2}

# Broken-RGG control instances: (name, n, avg_degree, direction, frac_broken, magnitude, seed). A couple of
# corruption directions at a legible size -- illustrative showcases (same generator as realrec), not the 900
# realrec seeds. inflate = heavy edge off the shortest paths; deflate = shortcut on them (the recoverable one).
#
# The n=1000 triple is the SAME instance family at 3.3x the nodes, kept alongside the n=300 control rather
# than replacing it: at n=300 a map panel is legible but a reader can fairly ask whether the geometry story
# is a small-sample effect. Same degree, same frac/magnitude, same seed -- only n moves. The exact solvers
# are the risk (the campaign drops both ILPs above n=500); mds_sweep gives them a raised cap and emits a
# labelled non-converged row rather than a silent drop if they still do not land.
RGG_SPECS = [
    ("rgg_inflate", 300, 10, "inflate", 0.15, 3.0, 1000),
    ("rgg_deflate", 300, 10, "deflate", 0.15, 3.0, 1000),
    ("rgg_mixed",   300, 10, "mixed",   0.15, 3.0, 1000),
    ("rgg_inflate_n1000", 1000, 10, "inflate", 0.15, 3.0, 1000),
    ("rgg_deflate_n1000", 1000, 10, "deflate", 0.15, 3.0, 1000),
    ("rgg_mixed_n1000",   1000, 10, "mixed",   0.15, 3.0, 1000),
]

# One representative cover per repair variant, computed inline. GMR/IOMR use best-of-k DETERMINISTIC (threshold)
# rounding of the covering LP -- byte-for-byte the campaign's `*_bestofk` (harness._cov, harness.py:214-215,
# 222-223) -- so this figure's representative cover is the same construction the rest of the paper measures.
# The greedy shortest-path cover over-covers (it adds every alternative-path edge) and `restore` then reweights
# those too, so it distorts geometry and is a misleadingly pessimistic representative. DOMR is exactly the heavy
# set (the invariant self-check).
#
# `rounding` MUST be "deterministic": `best_of_k` is read ONLY in the deterministic branch
# (metric_repair.py:1391-1397; its docstring at :1366 says "deterministic only"). This function previously
# passed rounding="randomized", so `best_of_k=12` was silently ignored and every `*_bestofk` MDS row was really
# a single-shot RANDOMIZED rounding at seed 0 -- i.e. the campaign's `*_rand`, mislabelled. Measured on
# rgg_deflate: the old (randomized) path gave |S|=253, the true deterministic best-of-12 gives |S|=249.
# oracle="naive" is mandatory here: the rsp oracle needs integer weights and SKIPS float-weighted graphs
# (RGG points, latency), which would silently drop the cover.
BEST_OF_K = 12
LP_SEED = 0                      # seed for _optimal_face_vertices' secondary-objective sampling
ROUNDING = "deterministic"
ORACLE = "naive"


def _bestofk(G, iomr):
    return covering_lp_cover(G, solve="separation", rounding=ROUNDING, oracle=ORACLE,
                             iomr=iomr, seed=LP_SEED, best_of_k=BEST_OF_K)[0]


COVER_FNS = {
    "GMR":  lambda G: _bestofk(G, iomr=False),
    "IOMR": lambda G: _bestofk(G, iomr=True),
    "DOMR": lambda G: domr_alg(G),
}
VARIANTS = ("GMR", "IOMR", "DOMR")

# Provenance columns. A panel is identified by its GRAPH parameters (n/avg_degree/frac/magnitude/seed/radius)
# AND by the COVER that produced it (cover_algo/cover_file plus the cover-side knobs rounding/oracle/best_of_k/
# lp_seed). Those knobs are exactly the values the `_bestofk` bug misreported, so they are recorded, not assumed.
PROV_FIELDS = ["avg_degree", "frac", "magnitude", "seed", "radius",
               "cover_algo", "cover_file", "rounding", "oracle", "best_of_k", "lp_seed"]
FIELDS = ["dataset", "graph", "corruption", "variant", "family", "mds_algo",
          "n", "n_used", "dim", "procrustes_disp", "stress", "neg_mass"] + PROV_FIELDS

# The cover-side knobs of a locally recomputed *_bestofk cover (a saved campaign cover overrides these).
RECOMPUTED_PROV = {"rounding": ROUNDING, "oracle": ORACLE, "best_of_k": BEST_OF_K, "lp_seed": LP_SEED}
DOMR_PROV = {"cover_algo": "domr", "cover_file": "(recomputed)", "rounding": "", "oracle": "",
             "best_of_k": "", "lp_seed": ""}
OBSERVED_PROV = {"cover_algo": "", "cover_file": "", "rounding": "", "oracle": "",
                 "best_of_k": "", "lp_seed": ""}


# ----------------------------------------------------------------------------
# MDS -- classical (eigendecomposition) and metric SMACOF (stress majorization)
# ----------------------------------------------------------------------------
def _pairwise(X):
    d = X[:, None, :] - X[None, :, :]
    return np.sqrt((d * d).sum(-1))


def _stress1(D, X):
    """Normalized stress-1: sqrt( sum (||xi-xj|| - dij)^2 / sum dij^2 ) over finite pairs (lower = better)."""
    Dx = _pairwise(X)
    iu = np.triu_indices(D.shape[0], 1)
    a, b = Dx[iu], np.asarray(D, float)[iu]
    m = np.isfinite(b)
    a, b = a[m], b[m]
    denom = float(np.sqrt((b * b).sum()))
    return float(np.sqrt(((a - b) ** 2).sum()) / denom) if denom > 0 else float("nan")


def classical_mds(D, dim):
    """Torgerson: B = -1/2 J D^2 J, eigendecompose, top-`dim` eigenvectors scaled by sqrt(eigenvalue).
    Returns (Y[n,dim], neg_mass) where neg_mass = sum|negative eigenvalues| / sum|all eigenvalues| in [0,1) --
    0 iff the distances embed exactly in Euclidean space, growing as they become more non-Euclidean."""
    D = np.asarray(D, float)
    n = D.shape[0]
    J = np.eye(n) - 1.0 / n                       # centering I - (1/n) 11^T
    B = -0.5 * J @ (D ** 2) @ J
    B = (B + B.T) / 2                             # symmetrize away round-off
    w, V = np.linalg.eigh(B)                      # ascending
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    L = np.clip(w[:dim], 0.0, None)
    Y = V[:, :dim] * np.sqrt(L)
    tot = float(np.abs(w).sum())
    neg_mass = float(-w[w < 0].sum() / tot) if tot > 0 else float("nan")
    return Y, neg_mass


def smacof(D, dim, init, n_iter=300, tol=1e-7):
    """Metric MDS by SMACOF (Guttman transform), started from the classical embedding so it is deterministic.
    Identical objective to sklearn.manifold.MDS(metric=True, dissimilarity='precomputed', init=...)."""
    D = np.asarray(D, float)
    n = D.shape[0]
    Z = np.asarray(init, float).copy()
    prev = _stress1(D, Z)
    for _ in range(n_iter):
        Dz = _pairwise(Z)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(Dz > 1e-12, D / Dz, 0.0)
        Bz = -ratio
        np.fill_diagonal(Bz, 0.0)
        Bz[np.diag_indices(n)] = -Bz.sum(axis=1)  # Guttman B(Z): off-diag -d/dz, diag = -rowsum
        Z = (Bz @ Z) / n                          # X = n^{-1} B(Z) Z  (centered Z -> V^+ = (1/n) J)
        s = _stress1(D, Z)
        if abs(prev - s) < tol:
            break
        prev = s
    return Z, _stress1(D, Z)


def _procrustes_disp(true_cfg, Y):
    """Disparity of Y against the true configuration after optimal translate/scale/rotate. Returns
    (disparity, aligned_true, aligned_Y) with both configs in the same standardized frame for plotting."""
    try:
        mtx1, mtx2, disp = procrustes(np.asarray(true_cfg, float), np.asarray(Y, float))
        return float(disp), mtx1, mtx2
    except ValueError:                            # a degenerate (near-zero-variance) embedding
        return float("nan"), None, None


# ----------------------------------------------------------------------------
# Distances -> the largest mutually-finite block (MDS needs a connected, finite matrix)
# ----------------------------------------------------------------------------
def finite_core(D):
    """Indices of the largest set of nodes with all-finite pairwise distances. Since D is an all-pairs
    shortest-path matrix, a finite entry means same component, so this is exactly the giant component."""
    fin = np.isfinite(D) & (D > 0)
    ncomp, lab = connected_components(csr_matrix(fin.astype(np.int8)), directed=False)
    if ncomp <= 1:
        return np.arange(D.shape[0])
    big = np.bincount(lab).argmax()
    return np.where(lab == big)[0]


def _nx_from_edges(edges, n):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for iu, iv, w in edges:
        G.add_edge(iu, iv, weight=float(w))
    return G


def _norm_cover(cover):
    """Whatever the repair returns (label pairs, position pairs, or (u,v,w) triples) -> set of (min,max)."""
    S = set()
    for e in cover:
        u, v = e[0], e[1]
        S.add((u, v) if u <= v else (v, u))
    return S


# ----------------------------------------------------------------------------
# One (dataset) -> rows + embeddings.  `store` collects arrays for the figures.
# ----------------------------------------------------------------------------
def _embed_and_score(tag, dataset, graph, corruption, family_by_variant,
                     D_by_variant, true_cfg, dim, store, n_total, graph_prov, prov_by_variant):
    """Run both MDS algorithms on the observed matrix and each repaired matrix, score vs true_cfg, and record.
    D_by_variant maps a label ('observed'|'GMR'|'IOMR'|'DOMR') -> distance matrix on the SAME node order as
    true_cfg. Returns a list of CSV row dicts.

    `n_total` is the TRUE node count of the graph (before the mutually-finite-core restriction); `n_used` is
    the core actually embedded. These used to be set to the same value, which hid the core restriction --
    the RGG specs ask for n=300 but the giant component of the broken graph is smaller.
    `graph_prov` carries the generator parameters; `prov_by_variant` maps a label -> its cover's provenance."""
    rows = []
    # neg_mass is a property of the distance matrix; compute once per variant via the classical route.
    embeds = {}
    for label, D in D_by_variant.items():
        Yc, neg = classical_mds(D, dim)
        Ys, _ = smacof(D, dim, init=Yc)
        embeds[label] = {"classical": Yc, "smacof": Ys, "neg_mass": neg}
    # store the true configuration (standardized like the embeddings) + a colour = its first axis
    tstd = (true_cfg - true_cfg.mean(0))
    tstd = tstd / (np.linalg.norm(tstd) or 1.0)
    store[f"true::{tag}"] = tstd
    store[f"color::{tag}"] = np.asarray(true_cfg, float)[:, 0]
    for label in D_by_variant:
        variant = "observed" if label == "observed" else label
        family = family_by_variant.get(label, "-")
        for algo in ("classical", "smacof"):
            Y = embeds[label][algo]
            disp, _, aligned = _procrustes_disp(true_cfg, Y)
            if aligned is not None:
                store[f"emb::{tag}::{label}::{algo}"] = aligned
            row = {
                "dataset": dataset, "graph": graph, "corruption": corruption,
                "variant": variant, "family": family, "mds_algo": algo,
                "n": n_total, "n_used": true_cfg.shape[0], "dim": dim,
                "procrustes_disp": round(disp, 6) if np.isfinite(disp) else "",
                "stress": round(_stress1(D_by_variant[label], Y), 6),
                "neg_mass": round(embeds[label]["neg_mass"], 6),
            }
            row.update({k: "" for k in PROV_FIELDS})
            row.update(graph_prov)
            row.update(prov_by_variant.get(label, OBSERVED_PROV))
            rows.append(row)
    return rows


def _covers_for(Gidx, n):
    """Compute one cover per variant on the index-space graph; drop a variant whose repair fails/empties.
    Returns (covers, prov) -- prov maps the variant to the provenance of the cover we actually built."""
    out, prov = {}, {}
    for v in VARIANTS:
        try:
            cov = _norm_cover(COVER_FNS[v](Gidx))
        except Exception as e:                                        # noqa: BLE001
            print(f"    [skip {v}] {type(e).__name__}: {e}", flush=True)
            continue
        out[v] = cov
        prov[v] = dict(DOMR_PROV) if v == "DOMR" else {
            "cover_algo": f"{v.lower()}_bestofk", "cover_file": "(recomputed)", **RECOMPUTED_PROV}
    return out, prov


def _saved_cover_file(graph, prefs, covers_root):
    """First saved cover file (results_real_covers/<graph>/<algo>__*.txt) matching a preferred algorithm."""
    cdir = _covers_dir(covers_root, graph) if covers_root else None
    if not cdir:
        return None, None
    for algo in prefs:
        hits = sorted(glob.glob(os.path.join(cdir, f"{algo}__*.txt")))
        if hits:
            return hits[0], algo
    return None, None


def _acquire_real_covers(graph, Gidx, idx, m_edges, covers_root):
    """One cover per variant for a real graph. DOMR is always recomputed (one APSP -- cheap, exact, keeps the
    self-check honest); GMR/IOMR come from a saved campaign cover when present, else a recomputed bestofk, but
    only if the graph is under RECOMPUTE_EDGE_CAP (ripe's covering LP is a cluster job).

    Returns (covers, prov). `prov` is PER VARIANT, not a scalar: GMR and IOMR resolve down DIFFERENT preference
    lists (GMR_PREFS / IOMR_PREFS), so on the same graph they routinely come from different algorithms and
    different files -- one shared 'cover_algo' would misattribute at least one of them."""
    out, prov = {}, {}
    try:
        out["DOMR"] = _norm_cover(domr_alg(Gidx))
        prov["DOMR"] = dict(DOMR_PROV)
    except Exception as e:                                            # noqa: BLE001
        print(f"    [skip DOMR] {type(e).__name__}: {e}", flush=True)
    for variant, prefs in (("GMR", GMR_PREFS), ("IOMR", IOMR_PREFS)):
        path, algo = _saved_cover_file(graph, prefs, covers_root)
        if path:
            out[variant] = load_cover(path, idx)
            # A saved campaign cover: its rounding/oracle/best_of_k/lp_seed are the CAMPAIGN's, not ours. The
            # file name is the honest record (e.g. gmr_bestofk__rand_s0.txt), so name it and leave the knobs
            # blank rather than assert our local defaults over a cover we did not compute.
            prov[variant] = {"cover_algo": algo, "cover_file": os.path.basename(path),
                             "rounding": "", "oracle": "", "best_of_k": "", "lp_seed": ""}
            print(f"    {variant}: saved cover [{algo}] {os.path.basename(path)}  |S|={len(out[variant])}",
                  flush=True)
        elif m_edges <= RECOMPUTE_EDGE_CAP:
            try:
                out[variant] = _norm_cover(COVER_FNS[variant](Gidx))
                prov[variant] = {"cover_algo": f"{variant.lower()}_bestofk", "cover_file": "(recomputed)",
                                 **RECOMPUTED_PROV}
                print(f"    {variant}: recomputed bestofk  |S|={len(out[variant])}", flush=True)
            except Exception as e:                                    # noqa: BLE001
                print(f"    [skip {variant}] {type(e).__name__}: {e}", flush=True)
        else:
            print(f"    [skip {variant}] no saved cover and m={m_edges} > {RECOMPUTE_EDGE_CAP}; "
                  f"pass --covers-root to reuse the campaign cover", flush=True)
    return out, prov


def run_pure_real(graph, store, covers_root=None):
    """ripe_atlas / nmr: observed vs repaired distances, both scored against the external true layout, which
    we take as the classical-MDS configuration of the true distance matrix (geography / 3-D structure)."""
    print(f"[pure_real] {graph}", flush=True)
    nodes, idx, edges = load_graph(graph)
    n = len(nodes)
    gt_ix, Dtrue = true_distances(graph, nodes)
    dim = PURE_REAL_DIM[graph]

    Gidx = _nx_from_edges(edges, n)
    covers, prov = _acquire_real_covers(graph, Gidx, idx, len(edges), covers_root)
    prov["observed"] = dict(OBSERVED_PROV)
    graph_prov = {"avg_degree": round(2.0 * len(edges) / max(n, 1), 3), "frac": "", "magnitude": "",
                  "seed": "", "radius": ""}

    Dobs = apsp(edges, n)
    D_gt = {"observed": Dobs[np.ix_(gt_ix, gt_ix)]}
    for v, cov in covers.items():
        DF = build_F_distances(edges, cov, n)
        D_gt[v] = DF[np.ix_(gt_ix, gt_ix)]

    # restrict to the mutually-finite core of the observed gt-block (F shares the observed connectivity), and
    # apply the same node subset to the true distances so every matrix and the true config share one order.
    core = finite_core(D_gt["observed"])
    true_cfg, _ = classical_mds(Dtrue[np.ix_(core, core)], dim)
    D_by = {lab: D[np.ix_(core, core)] for lab, D in D_gt.items()}
    fam = {"observed": "-", "GMR": "gmr", "IOMR": "iomr", "DOMR": "gmr"}
    tag = f"{graph}::none"
    if len(core) < D_gt["observed"].shape[0]:
        print(f"    core {len(core)}/{D_gt['observed'].shape[0]} gt nodes (rest not mutually reachable)",
              flush=True)
    return _embed_and_score(tag, "pure_real", graph, "none", fam, D_by, true_cfg, dim, store,
                            n_total=n, graph_prov=graph_prov, prov_by_variant=prov)


def run_rgg(name, n, deg, direction, frac, mag, seed, store):
    """A metric RGG with known 2-D points, broken then repaired: does the repaired embedding return to the
    true points? The controlled companion to ripe (ground truth we planted, not merely external)."""
    print(f"[rgg] {name}  n={n} deg={deg} dir={direction} frac={frac} mag={mag}", flush=True)
    seed_all(seed)
    radius = np.sqrt(deg / (np.pi * max(n - 1, 1)))
    T = random_geometric_metric_graph(n, mode="radius", radius=float(radius))
    T = nx.convert_node_labels_to_integers(T.subgraph(max(nx.connected_components(T), key=len)).copy())
    C, _corrupted = break_metric_graph(T, frac_q=frac, direction=direction, magnitude=mag)
    C = nx.convert_node_labels_to_integers(C.subgraph(max(nx.connected_components(C), key=len)).copy())
    nc = C.number_of_nodes()
    pos = np.array([C.nodes[i]["pos"] for i in range(nc)], dtype=float)   # TRUE 2-D layout

    edges = [(u, v, C[u][v]["weight"]) for u, v in C.edges()]
    covers, prov = _covers_for(C, nc)
    prov["observed"] = dict(OBSERVED_PROV)
    Dobs = apsp(edges, nc)
    D_all = {"observed": Dobs}
    for v, cov in covers.items():
        D_all[v] = build_F_distances(edges, cov, nc)

    core = finite_core(D_all["observed"])
    D_by = {lab: D[np.ix_(core, core)] for lab, D in D_all.items()}
    true_cfg = pos[core]
    fam = {"observed": "-", "GMR": "gmr", "IOMR": "iomr", "DOMR": "gmr"}
    tag = f"{name}::{direction}"
    # `n` is the SPEC node count; `n_used` (set inside _embed_and_score) is the finite core actually embedded.
    graph_prov = {"avg_degree": deg, "frac": frac, "magnitude": mag, "seed": seed,
                  "radius": round(float(radius), 6)}
    return _embed_and_score(tag, "rgg", name, direction, fam, D_by, true_cfg, dim=2, store=store,
                            n_total=n, graph_prov=graph_prov, prov_by_variant=prov)


def _default_covers_root():
    """Where downstream_recovery keeps the saved covers, if that tree exists next to us."""
    for cand in ("results_real_covers", os.path.join("results_real", "results_real_covers")):
        if os.path.isdir(cand):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="analysis")
    ap.add_argument("--only", choices=["real", "rgg", "all"], default="all")
    ap.add_argument("--covers-root", default=None,
                    help="results_real_covers/ dir; reuses saved campaign covers for the real graphs "
                         "(auto-detected if present). Without it, big graphs (ripe) emit observed+DOMR only.")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    covers_root = a.covers_root or _default_covers_root()
    print(f"covers_root = {covers_root or '(none -- recompute where small enough)'}")

    rows, store = [], {}
    if a.only in ("real", "all"):
        # smallest first, so a local run without covers finishes nmr before it reaches near-complete ripe
        for g in sorted(DOWNSTREAM_GRAPHS, key=lambda x: 0 if x.startswith("nmr") else 1):
            try:
                rows += run_pure_real(g, store, covers_root=covers_root)
            except FileNotFoundError as e:
                print(f"    [skip {g}] missing data: {e}", flush=True)
    if a.only in ("rgg", "all"):
        for spec in RGG_SPECS:
            rows += run_rgg(*spec, store=store)

    csv_path = os.path.join(a.outdir, "summary_mds.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    npz_path = os.path.join(a.outdir, "mds_embeddings.npz")
    np.savez_compressed(npz_path, **store)

    # DOMR self-check: its disparity must equal the observed disparity (D_F == D_G by Lemma 6.1).
    print("\nDOMR self-check (disparity should equal observed, per graph/algo):")
    by = {}
    for r in rows:
        if r["variant"] in ("observed", "DOMR") and r["procrustes_disp"] != "":
            by.setdefault((r["graph"], r["mds_algo"]), {})[r["variant"]] = float(r["procrustes_disp"])
    worst = 0.0
    for (g, al), d in sorted(by.items()):
        if "observed" in d and "DOMR" in d:
            gap = abs(d["observed"] - d["DOMR"])
            worst = max(worst, gap)
            flag = "" if gap < 1e-6 else "  <-- MISMATCH"
            print(f"    {g:20s} {al:9s} obs={d['observed']:.4f} domr={d['DOMR']:.4f} |gap|={gap:.2e}{flag}")
    print(f"  max |gap| = {worst:.2e}  ({'OK' if worst < 1e-6 else 'CHECK PIPELINE'})")

    print(f"\nwrote {csv_path} ({len(rows)} rows), {npz_path} ({len(store)} arrays)")
    print(f"next:  sage -python experiments/mds_plots.py --data {csv_path} --emb {npz_path}")


if __name__ == "__main__":
    main()
