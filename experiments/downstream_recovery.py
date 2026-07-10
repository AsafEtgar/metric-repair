"""downstream_recovery.py -- PURE real-data Tier-2 experiment: does metric repair recover the TRUE metric?

Complement to `realrec` (rgg_harness). realrec corrupts a graph whose true metric we KNOW (a real metric
base) and tests whether repair undoes a PLANTED injury. This module tests the harder thing: take a graph that
is non-metric for REAL reasons -- internet latency, an NOE contact graph -- with NO planted corruption, repair
it, and ask whether its shortest-path structure moves closer to an EXTERNAL ground truth (geography, 3-D
protein structure). There is no true graph T; there is a true DISTANCE, from outside the graph.

It is a post-processing pass over the covers the main real-data campaign already saved -- no repair re-run.
For each (graph, saved cover) we build the repaired graph F by the `restore` construction, then compare the
observed graph G and F against the ground-truth distance on two axes:

  kNN recovery   for each node, the Jaccard of its graph k-NN with its TRUE k-NN; lift = recovery(F) - recovery(G).
  rank fidelity  Spearman correlation of graph distance vs true distance over a pair sample; delta = corr(F) - corr(G).

DOMR is the control with teeth: its cover is exactly the heavy edges, and reweighting them to their detour
leaves every shortest-path distance unchanged, so lift and delta are EXACTLY 0 for DOMR. Any nonzero DOMR
number is a pipeline bug, which makes the experiment self-checking.

SELF-CONTAINED: imports nothing from harness.py / rgg_harness.py / metric_repair.py, so it cannot be affected
by, and cannot affect, the running campaign. Ground truth is read straight from data/processed/gt/.

    sage -python experiments/run_downstream_task.py --graph ripe_atlas
"""
import csv
import glob
import math
import os

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

# ----------------------------------------------------------------------------
# Datasets: which real graphs have an external, non-graph-derived ground truth.
# ----------------------------------------------------------------------------
#   coords  -> data/processed/gt/<graph>__coords.csv  (node,lat,lon); true dist = haversine
#   dmat    -> data/processed/gt/<graph>__truedist.npz (nodes, D);    true dist = D
DOWNSTREAM_GRAPHS = {
    "ripe_atlas":       "coords",   # internet latency, 95.3% non-metric; true = geography
    "nmr_1d3z_atom":    "dmat",     # NOE proton-contact graph;         true = 3-D structure
    "nmr_1d3z_residue": "dmat",
}
K_LIST = (5, 10, 20)
PAIR_SAMPLE = 20000                 # node pairs for the Spearman estimate on large graphs

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_DIR = os.path.join(REPO, "data", "processed", "gt")

# GMR verifies cover edges freely; IOMR constrains them (raise-only); DOMR's cover is the heavy set.
GMR_ALGOS = {"gmr_thr_naive", "gmr_bestofk", "gmr_rand", "gmr_ilp", "l1sep_gmr", "spc_gmr", "pivot"}
IOMR_ALGOS = {"iomr_thr_naive", "iomr_bestofk", "iomr_rand", "iomr_ilp", "iomr_regiongrow", "l1sep_iomr",
              "spc_iomr", "left_edge"}


def variant_of(algo):
    if algo == "domr":
        return "DOMR"
    return "GMR" if algo in GMR_ALGOS else ("IOMR" if algo in IOMR_ALGOS else "?")


# ----------------------------------------------------------------------------
# Graph + cover IO (labels may be strings, e.g. nmr "11:HA"; keep them as-is)
# ----------------------------------------------------------------------------
def load_graph(graph):
    """Return (nodes, idx, edges) with edges as (iu, iv, w) in contiguous index space."""
    path = os.path.join(REPO, "data", "processed", f"{graph}.csv")
    raw = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        # tolerate a header row or not; a numeric first cell means no header
        if header and _is_number(header[-1]):
            raw.append(header)
        for row in r:
            raw.append(row)
    node_set, es = set(), []
    for row in raw:
        u, v, w = row[0], row[1], float(row[2])
        node_set.add(u); node_set.add(v); es.append((u, v, w))
    nodes = sorted(node_set)
    idx = {u: i for i, u in enumerate(nodes)}
    edges = [(idx[u], idx[v], w) for (u, v, w) in es]
    return nodes, idx, edges


def load_cover(path, idx):
    """A saved cover file: 'u v' per line, ORIGINAL labels. Return a set of sorted index pairs."""
    S = set()
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 2:
                continue
            u, v = p[0], p[1]
            if u in idx and v in idx:      # a label could in principle be absent; skip defensively
                a, b = idx[u], idx[v]
                S.add((a, b) if a <= b else (b, a))
    return S


def _is_number(s):
    try:
        float(s); return True
    except (TypeError, ValueError):
        return False


# ----------------------------------------------------------------------------
# APSP (sparse CSR -> no dense-zero masking; this is the A1 bug done right)
# ----------------------------------------------------------------------------
def apsp(edges, n):
    if not edges:
        D = np.full((n, n), np.inf); np.fill_diagonal(D, 0.0); return D
    r, c, w = [], [], []
    for iu, iv, wt in edges:
        r += [iu, iv]; c += [iv, iu]; w += [wt, wt]
    A = csr_matrix((w, (r, c)), shape=(n, n))
    return shortest_path(A, method="D", directed=False)


def build_F_distances(edges, cover, n):
    """`restore`: reweight each cover edge to its detour distance in G\\S; bridges keep their weight.
    Returns the all-pairs distance matrix of the repaired graph F."""
    not_cov = [(iu, iv, w) for (iu, iv, w) in edges
               if ((iu, iv) if iu <= iv else (iv, iu)) not in cover]
    Ddet = apsp(not_cov, n)
    edgesF = []
    for (iu, iv, w) in edges:
        key = (iu, iv) if iu <= iv else (iv, iu)
        if key in cover and np.isfinite(Ddet[iu, iv]):
            edgesF.append((iu, iv, Ddet[iu, iv]))   # lower a heavy edge / raise a shortcut, both to the detour
        else:
            edgesF.append((iu, iv, w))               # non-cover edge, or a bridge with no detour
    return apsp(edgesF, n)


# ----------------------------------------------------------------------------
# Ground truth: an external true-distance matrix over the graph's nodes
# ----------------------------------------------------------------------------
def _haversine_matrix(latlon):
    """Great-circle distance (km) between every pair of (lat, lon) rows."""
    lat = np.radians(latlon[:, 0])[:, None]
    lon = np.radians(latlon[:, 1])[:, None]
    dlat = lat - lat.T
    dlon = lon - lon.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def true_distances(graph, nodes):
    """Return (gt_ix, Dtrue) where gt_ix are the graph-node indices that carry ground truth, and Dtrue is the
    true distance matrix over exactly those nodes (aligned to gt_ix order)."""
    kind = DOWNSTREAM_GRAPHS[graph]
    pos = {u: i for i, u in enumerate(nodes)}
    if kind == "coords":
        latlon, gt_ix = {}, []
        with open(os.path.join(GT_DIR, f"{graph}__coords.csv")) as f:
            r = csv.reader(f); next(r, None)
            for row in r:
                node, la, lo = row[0], float(row[1]), float(row[2])
                if node in pos:
                    latlon[node] = (la, lo)
        gt_nodes = [u for u in nodes if u in latlon]
        gt_ix = [pos[u] for u in gt_nodes]
        M = np.array([latlon[u] for u in gt_nodes], dtype=float)
        return np.array(gt_ix), _haversine_matrix(M)
    else:  # dmat
        z = np.load(os.path.join(GT_DIR, f"{graph}__truedist.npz"), allow_pickle=True)
        gt_node_labels = [str(x) for x in z["nodes"]]
        D = np.asarray(z["D"], dtype=float)
        keep = [(j, lab) for j, lab in enumerate(gt_node_labels) if lab in pos]
        gt_ix = np.array([pos[lab] for _, lab in keep])
        jj = np.array([j for j, _ in keep])
        return gt_ix, D[np.ix_(jj, jj)]


# ----------------------------------------------------------------------------
# Scoring: kNN recovery to truth, and rank fidelity
# ----------------------------------------------------------------------------
def _knn_sets(D, k):
    """For each row, the set of its k nearest columns (excluding self, finite distance)."""
    n = D.shape[0]
    out = []
    for i in range(n):
        d = D[i].copy(); d[i] = np.inf
        order = np.argsort(d, kind="stable")
        order = [j for j in order if np.isfinite(d[j])][:k]
        out.append(frozenset(order))
    return out


def knn_recovery(Dgraph_gt, Dtrue, k):
    """Mean over nodes of Jaccard(graph k-NN, true k-NN). Both matrices are over the SAME gt-node set/order."""
    g = _knn_sets(Dgraph_gt, k)
    t = _knn_sets(Dtrue, k)
    js = []
    for gg, tt in zip(g, t):
        if not gg and not tt:
            continue
        inter = len(gg & tt); union = len(gg | tt)
        js.append(inter / union if union else 1.0)
    return float(np.mean(js)) if js else float("nan")


def _spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = math.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / denom) if denom else float("nan")


def rank_fidelity(Dgraph_gt, Dtrue, seed):
    """Spearman correlation of graph distance vs true distance over a sample of finite pairs."""
    n = Dtrue.shape[0]
    iu = np.triu_indices(n, k=1)
    gt = Dtrue[iu]; gr = Dgraph_gt[iu]
    ok = np.isfinite(gt) & np.isfinite(gr)
    gt, gr = gt[ok], gr[ok]
    if len(gt) > PAIR_SAMPLE:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(gt), PAIR_SAMPLE, replace=False)
        gt, gr = gt[sel], gr[sel]
    return _spearman(gr, gt)


# ----------------------------------------------------------------------------
# One graph -> rows (one per cover file per k)
# ----------------------------------------------------------------------------
FIELDS = ["graph", "gt_kind", "algo", "variant", "mode", "seed", "n", "n_gt", "k",
          "recovery_obs", "recovery_rep", "lift",
          "spearman_obs", "spearman_rep", "delta_spearman", "n_covers_seen"]


def _covers_dir(covers_root, graph):
    for cand in (os.path.join(covers_root, graph),
                 os.path.join(covers_root, "results_real_covers", graph),
                 os.path.join("results_real", "results_real_covers", graph)):
        if os.path.isdir(cand):
            return cand
    return None


def run_one_graph(graph, covers_root="results_real_covers"):
    nodes, idx, edges = load_graph(graph)
    n = len(nodes)
    gt_ix, Dtrue = true_distances(graph, nodes)
    n_gt = len(gt_ix)

    # observed graph, scored once (repair-independent)
    Dobs = apsp(edges, n)
    Dobs_gt = Dobs[np.ix_(gt_ix, gt_ix)]
    rec_obs = {k: knn_recovery(Dobs_gt, Dtrue, k) for k in K_LIST}
    sp_obs = rank_fidelity(Dobs_gt, Dtrue, seed=0)

    cdir = _covers_dir(covers_root, graph)
    cover_files = sorted(glob.glob(os.path.join(cdir, "*.txt"))) if cdir else []
    rows = []
    for cf in cover_files:
        base = os.path.basename(cf)[:-4]           # "<algo>__<tag>"
        algo, _, tag = base.partition("__")
        mode = "rand" if tag.startswith("rand") or "_s" in tag else "det"
        seed = int(tag.split("_s")[-1]) if "_s" in tag else 0
        cover = load_cover(cf, idx)
        DF = build_F_distances(edges, cover, n)
        DF_gt = DF[np.ix_(gt_ix, gt_ix)]
        sp_rep = rank_fidelity(DF_gt, Dtrue, seed=0)
        for k in K_LIST:
            rec_rep = knn_recovery(DF_gt, Dtrue, k)
            rows.append({
                "graph": graph, "gt_kind": DOWNSTREAM_GRAPHS[graph],
                "algo": algo, "variant": variant_of(algo), "mode": mode, "seed": seed,
                "n": n, "n_gt": n_gt, "k": k,
                "recovery_obs": round(rec_obs[k], 6), "recovery_rep": round(rec_rep, 6),
                "lift": round(rec_rep - rec_obs[k], 6),
                "spearman_obs": round(sp_obs, 6), "spearman_rep": round(sp_rep, 6),
                "delta_spearman": round(sp_rep - sp_obs, 6),
                "n_covers_seen": len(cover_files),
            })
    return rows
