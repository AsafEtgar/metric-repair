"""
build_real_graphs.py -- turn every fetched real dataset (data/raw/*) into the weighted-graph
representation (networkx Graph with edge 'weight'), save each to data/processed/<name>.csv (edge list),
and print a tractability report: n, m, components/giant, weight range, cycle_length_bound, is_metric,
non-metric-edge fraction, and (small graphs only) the broken-cycle count.

Modeling choices are explicit and documented inline (subsampling the huge graphs, inverting
similarities to distances, symmetrizing directed connectomes, building kNN graphs). Run:

    sage -python build_real_graphs.py

Deps: numpy, scipy, pandas, networkx, pyarrow (all in the Sage env). Reuses datasets.save_edgelist and
metric_repair.broken_cycle_length_bound / broken_cycles.
"""
import os, gzip, re
from collections import defaultdict, deque, Counter

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path, connected_components
import networkx as nx

RAW = "data/raw"
OUT = "data/processed"
os.makedirs(OUT, exist_ok=True)
SEED = 12345
rng = np.random.default_rng(SEED)

DIMACS_N = 1200        # DIMACS road BFS-ball size (nodes); override with --dimacs-n
METRIC_MAX = 30000     # above this, the dense-ish APSP metric check is skipped (graph still built)

from datasets import save_edgelist, graph_from_matrix
from metric_repair import broken_cycle_length_bound, broken_cycles


# --------------------------------------------------------------------------- report
def _count_nonmetric(A, ii, jj, ww, n, chunk=2000):
    """Count edges (a,b,w) with a strictly shorter path than w. Computes shortest paths in row-chunks
    so peak memory is chunk*n, not n*n -- lets the metric check scale to large (sparse) road graphs."""
    ii = np.asarray(ii); jj = np.asarray(jj); ww = np.asarray(ww, float)
    order = np.argsort(ii, kind="stable")
    ii, jj, ww = ii[order], jj[order], ww[order]
    nonmetric = 0
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        Dc = shortest_path(A, method="D", directed=False, indices=list(range(start, stop)))
        lo = int(np.searchsorted(ii, start, "left")); hi = int(np.searchsorted(ii, stop, "left"))
        for a, b, w in zip(ii[lo:hi], jj[lo:hi], ww[lo:hi]):
            if Dc[a - start, b] < w - max(1e-9, 1e-9 * w):
                nonmetric += 1
    return nonmetric


def graph_report(G, name, note=""):
    """Fast, uniform stats. is_metric + non-metric fraction via scipy Dijkstra; broken-cycle count only
    on small graphs (else the enumeration is exponential)."""
    n, m = G.number_of_nodes(), G.number_of_edges()
    r = {"dataset": name, "n": n, "m": m, "note": note}
    if m == 0:
        r.update(components="-", giant="-", w_min=None, w_max=None, w_med=None,
                 cycle_bound=None, is_metric=None, nonmetric_frac=None, broken_cycles="-")
        return r
    nodes = list(G.nodes()); idx = {u: i for i, u in enumerate(nodes)}
    ii, jj, ww = [], [], []
    for u, v, w in G.edges(data="weight"):
        ii.append(idx[u]); jj.append(idx[v]); ww.append(float(w))
    ww = np.asarray(ww)
    A = sp.csr_matrix((np.r_[ww, ww], (np.r_[ii, jj], np.r_[jj, ii])), shape=(n, n))
    ncomp, lab = connected_components(A, directed=False)
    giant = int(np.bincount(lab).max())
    if n <= METRIC_MAX:
        nonmetric = _count_nonmetric(A, ii, jj, ww, n)
        is_metric = (nonmetric == 0); nmf = round(nonmetric / m, 4)
    else:
        is_metric = None; nmf = None                     # too large for the APSP metric check
    bound = broken_cycle_length_bound(G)
    # broken-cycle count only where enumeration is safe (small, sparse, small bound)
    bc = "skipped"
    if n <= 150 and bound is not None and bound <= 25 and m <= 4000:
        cnt = 0
        for _ in broken_cycles(G):
            cnt += 1
            if cnt > 200000:
                cnt = ">200000"; break
        bc = cnt
    r.update(components=ncomp, giant=giant,
             w_min=round(float(ww.min()), 4), w_max=round(float(ww.max()), 4),
             w_med=round(float(np.median(ww)), 4),
             cycle_bound=bound, is_metric=is_metric,
             nonmetric_frac=nmf, broken_cycles=bc)
    return r


def finish(G, name, note=""):
    save_edgelist(G, os.path.join(OUT, name + ".csv"))
    rep = graph_report(G, name, note)
    nmf = rep["nonmetric_frac"]
    print("  [ok] %-22s n=%d m=%d metric=%s nonmetric=%s bound=%s broken=%s" % (
        name, rep["n"], rep["m"], rep["is_metric"],
        ("%.1f%%" % (100 * nmf)) if nmf is not None else "n/a",
        rep["cycle_bound"], rep["broken_cycles"]), flush=True)
    return rep


def bfs_ball(adj, seed, target):
    """Grow a connected node set from seed via BFS until it reaches `target` nodes."""
    seen = {seed}; q = deque([seed])
    while q and len(seen) < target:
        x = q.popleft()
        for y in adj.get(x, ()):  # adj may be a dict of iterables
            if y not in seen:
                seen.add(y); q.append(y)
                if len(seen) >= target:
                    break
    return seen


def bfs_ball_arrays(pre, post, seed, target):
    """BFS ball over an undirected edge set given as parallel arrays (numpy)."""
    seen = np.zeros(0, dtype=pre.dtype)
    frontier = np.array([seed], dtype=pre.dtype)
    seenset = {int(seed)}
    while frontier.size and len(seenset) < target:
        mask = np.isin(pre, frontier) | np.isin(post, frontier)
        nbr = np.unique(np.concatenate([pre[mask], post[mask]]))
        new = [int(x) for x in nbr if int(x) not in seenset]
        if not new:
            break
        for x in new:
            seenset.add(x)
            if len(seenset) >= target:
                break
        frontier = np.array(new[:max(1, target - (len(seenset) - len(new)))], dtype=pre.dtype)
    return seenset


reports = []
def run(fn, *a):
    name = fn.__name__.replace("build_", "")
    print("==", name, flush=True)
    try:
        for rep in (fn(*a) or []):
            reports.append(rep)
    except Exception as e:
        import traceback; traceback.print_exc()
        print("  [FAIL] %s: %r" % (name, e), flush=True)


# =========================================================================== builders

def build_dimacs():
    """Road network: parse the DIMACS .gr, take a connected BFS ball (~1200 nodes) from a seed, induce
    the SAME node set on both -d (distance) and -t (travel time). Weight = integer edge length/time."""
    def parse(path):
        U, V, W = [], [], []
        with gzip.open(path, "rt") as f:
            for line in f:
                if line and line[0] == "a":
                    _, u, v, w = line.split()
                    U.append(int(u)); V.append(int(v)); W.append(int(w))
        return np.array(U), np.array(V), np.array(W)
    Ud, Vd, Wd = parse(os.path.join(RAW, "dimacs/USA-road-d.NY.gr.gz"))
    adj = defaultdict(set)
    for u, v in zip(Ud.tolist(), Vd.tolist()):
        adj[u].add(v); adj[v].add(u)
    ball = bfs_ball(adj, seed=1, target=DIMACS_N)
    out = []
    for tag, path in [("d", "USA-road-d.NY.gr.gz"), ("t", "USA-road-t.NY.gr.gz")]:
        U, V, W = (Ud, Vd, Wd) if tag == "d" else parse(os.path.join(RAW, "dimacs", path))
        G = nx.Graph()
        for u, v, w in zip(U.tolist(), V.tolist(), W.tolist()):
            if u in ball and v in ball:
                G.add_edge(u, v, weight=w)
        # largest connected component
        cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(cc).copy()
        out.append(finish(G, "dimacs_ny_%s" % tag, "BFS ball n=%d; weight=%s" % (
                          DIMACS_N, "meters" if tag == "d" else "travel-time")))
    return out


def build_ripe():
    """Internet latency: load the anchoring-mesh RTT edge list directly (already a distance graph)."""
    import pandas as pd
    e = pd.read_csv(os.path.join(RAW, "ripe_atlas/rtt_edgelist.csv"))
    G = nx.Graph()
    for u, v, w in zip(e.u, e.v, e.rtt_ms):
        if w > 0:
            G.add_edge(int(u), int(v), weight=float(w))
    return [finish(G, "ripe_atlas", "IPv4 ping mesh; weight=min RTT ms")]


def build_bct():
    """Brain coactivation network: similarity -> distance via d = 1/w on the observed (nonzero) edges."""
    from scipy.io import loadmat
    M = loadmat(os.path.join(RAW, "bct/Coactivation_matrix.mat"))["Coactivation_matrix"].astype(float)
    D = np.where(M > 0, 1.0 / np.maximum(M, 1e-12), np.nan)
    np.fill_diagonal(D, np.nan)
    G = graph_from_matrix(D)
    return [finish(G, "bct_coactivation", "638-node fMRI; d=1/similarity")]


def build_flycns():
    """Insect connectome: restrict the 151.9M-row weight table to Traced neurons, take a BFS ball
    (~1200 neurons) around the highest-degree seed, symmetrize (undirected synapse strength = sum of both
    directions), and invert to distance d = 1/strength."""
    import pyarrow.feather as pf, pyarrow.compute as pc
    ann = pf.read_table(os.path.join(RAW, "flycns/body-annotations-male-cns-v1.0-minconf-0.5.feather"),
                        columns=["bodyId", "status"])
    mask = pc.equal(ann.column("status"), "Traced")
    traced = pc.filter(ann.column("bodyId"), mask).to_numpy()
    tset = np.asarray(traced, dtype=np.int64)
    w = pf.read_table(os.path.join(RAW, "flycns/connectome-weights-male-cns-v1.0-minconf-0.5.feather"),
                      memory_map=True)
    pre = w.column("body_pre").to_numpy(); post = w.column("body_post").to_numpy()
    wt = w.column("weight").to_numpy()
    keep = np.isin(pre, tset) & np.isin(post, tset)
    pre, post, wt = pre[keep], post[keep], wt[keep]
    # highest-degree traced neuron as seed
    deg = Counter(pre.tolist()); deg.update(post.tolist())
    seed = deg.most_common(1)[0][0]
    ball = bfs_ball_arrays(pre, post, seed, target=1200)
    barr = np.fromiter(ball, dtype=np.int64)
    sub = np.isin(pre, barr) & np.isin(post, barr)
    pre2, post2, wt2 = pre[sub], post[sub], wt[sub]
    strength = defaultdict(int)
    for a, b, x in zip(pre2.tolist(), post2.tolist(), wt2.tolist()):
        if a != b:
            strength[(min(a, b), max(a, b))] += int(x)
    G = nx.Graph()
    for (a, b), s in strength.items():
        G.add_edge(a, b, weight=1.0 / s)
    cc = max(nx.connected_components(G), key=len); G = G.subgraph(cc).copy()
    return [finish(G, "flycns_male", "Traced BFS ball; d=1/(sum synapse counts both dirs)")]


def build_fish1_ten():
    """Vertebrate circuit (zebrafish TEN): parse the incoming/outgoing synapse adjacency lists (repeats =
    synapse count), build a directed weighted graph, symmetrize, BFS ball (~1000) from the TEN core,
    invert to distance d = 1/strength."""
    import pandas as pd
    base = os.path.join(RAW, "fish1/TEN_analysis")
    def parse_lists(fname, incoming):
        df = pd.read_csv(os.path.join(base, fname))
        pre, post = [], []
        col = df.columns[1]
        for q, lst in zip(df["query_neuron"], df[col]):
            if not isinstance(lst, str):
                continue
            partners = [int(x) for x in re.findall(r"\d+", lst)]
            for p in partners:
                if incoming:      # partner -> query
                    pre.append(p); post.append(int(q))
                else:             # query -> partner
                    pre.append(int(q)); post.append(p)
        return pre, post
    pi, po = parse_lists("incoming_synapses.csv", True)
    oi, oo = parse_lists("outgoing_synapses.csv", False)
    pre = np.array(pi + oi, dtype=np.int64); post = np.array(po + oo, dtype=np.int64)
    core = set()
    cats = pd.read_csv(os.path.join(base, "TEN_cats.csv"))
    for c in cats.columns:
        core.update(int(x) for x in cats[c].dropna().tolist())
    seed = next(iter(core))
    ball = bfs_ball_arrays(pre, post, seed, target=1000)
    barr = np.fromiter(ball, dtype=np.int64)
    sub = np.isin(pre, barr) & np.isin(post, barr)
    strength = defaultdict(int)
    for a, b in zip(pre[sub].tolist(), post[sub].tolist()):
        if a != b:
            strength[(min(a, b), max(a, b))] += 1
    G = nx.Graph()
    for (a, b), s in strength.items():
        G.add_edge(a, b, weight=1.0 / s)
    cc = max(nx.connected_components(G), key=len); G = G.subgraph(cc).copy()
    return [finish(G, "fish1_ten", "TEN core BFS ball; d=1/synapse count")]


def build_nmr():
    """NMR distance restraints (1D3Z): parse the XPLOR NOE section. Node = (resid, atom-group), edge
    weight = NOE upper bound = d + d_plus (tightest kept on duplicates). Build atom-level and
    residue-level (min upper bound between residues) graphs."""
    txt = open(os.path.join(RAW, "nmr/1D3Z.mr")).read()
    m = re.search(r"A\.\s*NOE RESTRAINTS(.*?)\nB\.", txt, re.S)
    noe = m.group(1)
    # each restraint: assign (sel1) (sel2)  d dminus dplus
    # split into 'assign' records
    records = re.split(r"(?=^\s*assign)", noe, flags=re.M)
    atomG = nx.Graph(); resG_min = defaultdict(lambda: float("inf"))
    for rec in records:
        rec = rec.strip()
        if not rec.lower().startswith("assign"):
            continue
        sels = re.findall(r"\(\s*(.*?)\s*\)", rec.replace("\n", " "), re.S)
        # numbers after the last ')'
        tail = rec[rec.rfind(")") + 1:]
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", tail)
        if len(sels) < 2 or len(nums) < 3:
            continue
        d, dm, dp = float(nums[0]), float(nums[1]), float(nums[2])
        upper = d + dp
        def group(sel):
            resids = re.findall(r"resid\s+(\d+)", sel)
            names = re.findall(r"name\s+(\S+)", sel)
            rid = int(resids[0]) if resids else None
            label = "%s:%s" % (rid, "|".join(sorted(set(names))))
            return rid, label
        r1, l1 = group(sels[0]); r2, l2 = group(sels[1])
        if l1 == l2 or r1 is None or r2 is None:
            continue
        if not atomG.has_edge(l1, l2) or upper < atomG[l1][l2]["weight"]:
            atomG.add_edge(l1, l2, weight=upper)
        if r1 != r2:
            key = (min(r1, r2), max(r1, r2))
            resG_min[key] = min(resG_min[key], upper)
    resG = nx.Graph()
    for (a, b), w in resG_min.items():
        resG.add_edge(a, b, weight=w)
    out = [finish(atomG, "nmr_1d3z_atom", "NOE upper bounds; node=(resid,atomgroup)")]
    out.append(finish(resG, "nmr_1d3z_residue", "NOE upper bounds; min over atom pairs; node=residue"))
    return out


def build_pbmc3k():
    """scRNA cosine kNN: log-CPM normalize, PCA (50 PCs), cosine-distance kNN (k=15) union-symmetrized.
    Weight = cosine distance (non-metric)."""
    from scipy.io import mmread
    d = os.path.join(RAW, "scrna_pbmc3k/filtered_gene_bc_matrices/hg19")
    X = mmread(os.path.join(d, "matrix.mtx")).tocsc().T.toarray().astype(float)   # cells x genes
    lib = X.sum(1, keepdims=True); lib[lib == 0] = 1
    X = np.log1p(X / lib * 1e4)
    var = X.var(0); hv = np.argsort(var)[-2000:]                                  # top-2000 HVG
    Xh = X[:, hv]; Xh -= Xh.mean(0)
    U, S, Vt = np.linalg.svd(Xh, full_matrices=False)
    P = U[:, :50] * S[:50]                                                        # PCA scores
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    sim = Pn @ Pn.T; np.fill_diagonal(sim, -np.inf)
    k = 15; G = nx.Graph()
    for i in range(P.shape[0]):
        nn = np.argpartition(-sim[i], k)[:k]
        for j in nn:
            dist = 1.0 - float(sim[i, j])
            if dist <= 0:
                dist = 1e-6
            if not G.has_edge(i, j) or dist < G[i][j]["weight"]:
                G.add_edge(int(i), int(j), weight=dist)
    return [finish(G, "pbmc3k_cosine_knn", "log-CPM->PCA50->cosine kNN k=15")]


def build_cassiopeia():
    """Single-cell lineage: barcode dissimilarity between cells (fraction of shared intBC sites with
    disagreeing alleles), kNN (k=15) on the 1000 best-covered cells. Weight = dissimilarity."""
    import pandas as pd
    df = pd.read_csv(os.path.join(RAW, "cassiopeia_spatial/spatial_allele_table.txt"), sep="\t")
    # collapse to one allele per (cell, intBC) = the highest-readCount call
    df = df.sort_values("readCount", ascending=False).drop_duplicates(["cellBC", "intBC"])
    piv = df.pivot(index="cellBC", columns="intBC", values="allele")
    cover = piv.notna().sum(1)
    cells = cover.sort_values(ascending=False).index[:1000]                       # best-covered 1000
    A = piv.loc[cells].to_numpy(dtype=object)
    n = len(cells); present = np.array([[x is not None and x == x for x in row] for row in A])
    G = nx.Graph()
    # pairwise dissimilarity (n=1000 -> 0.5M pairs, fine)
    dis = np.full((n, n), np.nan)
    for i in range(n):
        pi = present[i]
        for j in range(i + 1, n):
            shared = pi & present[j]
            ns = shared.sum()
            if ns >= 3:                                                          # need enough overlap
                diff = sum(1 for k in np.where(shared)[0] if A[i, k] != A[j, k])
                dis[i, j] = dis[j, i] = diff / ns
    k = 15
    for i in range(n):
        row = dis[i].copy(); row[i] = np.nan
        valid = np.where(np.isfinite(row))[0]
        if valid.size == 0:
            continue
        nn = valid[np.argsort(row[valid])[:k]]
        for j in nn:
            wgt = float(row[j]);  wgt = wgt if wgt > 0 else 1e-6
            if not G.has_edge(i, j) or wgt < G[i][j]["weight"]:
                G.add_edge(int(i), int(j), weight=wgt)
    return [finish(G, "cassiopeia_barcode_knn", "barcode dissimilarity kNN k=15 on 1000 cells")]


# =========================================================================== main
BUILDERS = {                       # name -> builder (a builder may emit several graphs)
    "dimacs": build_dimacs, "ripe": build_ripe, "bct": build_bct, "nmr": build_nmr,
    "pbmc3k": build_pbmc3k, "cassiopeia": build_cassiopeia,
    "fish1_ten": build_fish1_ten, "flycns": build_flycns,
}

if __name__ == "__main__":
    import argparse
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="",
                    help="comma-separated builders to run (default: all). choices: " + ", ".join(BUILDERS))
    ap.add_argument("--dimacs-n", type=int, default=DIMACS_N,
                    help="DIMACS road BFS-ball size in nodes; bigger = larger road subgraph "
                         "(default %d; NY has 264,346 nodes). The metric check is skipped above %d." % (
                             DIMACS_N, METRIC_MAX))
    a = ap.parse_args()
    DIMACS_N = a.dimacs_n                                 # module-scope: seen by build_dimacs

    sel = [s.strip() for s in a.only.split(",") if s.strip()] or list(BUILDERS)
    bad = [s for s in sel if s not in BUILDERS]
    if bad:
        ap.error("unknown builder(s) %s; choices: %s" % (bad, ", ".join(BUILDERS)))
    for name in sel:
        run(BUILDERS[name])

    cols = ["dataset", "n", "m", "components", "giant", "w_min", "w_max", "w_med",
            "cycle_bound", "is_metric", "nonmetric_frac", "broken_cycles", "note"]
    rep = pd.DataFrame(reports)
    rep = rep[[c for c in cols if c in rep.columns]]
    path = os.path.join(OUT, "REAL_GRAPHS_REPORT.csv")
    if a.only and os.path.exists(path):                   # partial run: update those rows, keep the rest
        old = pd.read_csv(path)
        old = old[~old["dataset"].isin(rep["dataset"])]
        full = pd.concat([old, rep], ignore_index=True)
    else:
        full = rep
    full.to_csv(path, index=False)
    print("\n================ REAL GRAPHS REPORT (this run) ================")
    print(rep.to_string(index=False))
    print("\nsaved edge lists + updated REAL_GRAPHS_REPORT.csv (%d rows) to %s" % (len(full), OUT))
