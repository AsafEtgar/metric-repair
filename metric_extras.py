"""
metric_extras.py  --  pure-Python port of metric_extras.sage

Auxiliary metric helpers that are NOT used by any repair algorithm in metric_repair.py: weight
setters, metric/coherence diagnostics, the triangle & cycle matrices, graph subdivision, cycle
counting, and a deprecated wrapper + back-compat aliases. Kept for interactive use and parity with
the older notebooks.

Depends on metric_repair.py (uses make_index_encoding, get_weights, _mvd_pivot_rec, ...).
"""

import numpy as np
import networkx as nx
from itertools import combinations

from metric_repair import (
    make_index_encoding, get_weights, all_pairs_distances, sorted_edges, edges_with_weights,
    domr_alg, _norm, _mvd_pivot_rec,
)


def get_edge_indices(G, D):
    """Given a graph G and an edge -> index encoding D, return the inverse dict index -> edge."""
    return {index: edge for edge, index in D.items()}


def set_weights(G, W):
    """Set edge weights in place from a list W ordered to match sorted_edges(G)."""
    assert G.number_of_edges() == len(W)
    for i, (u, v) in enumerate(sorted_edges(G)):
        G[u][v]["weight"] = W[i]


def update_weights(G, Wp, ind_enc):
    """Given a weighted graph G, additive corrections Wp and an index encoding ind_enc: [m] -> E,
    set the new weights w + Wp in place."""
    Dw = get_weights(G)
    for i in np.nonzero(Wp)[0]:
        u, v = ind_enc[i]
        G[u][v]["weight"] = Dw[_norm(u, v)] + Wp[i]


def check_symmetric(a, tol=1e-8):
    """True if matrix a is symmetric up to tolerance tol."""
    return np.all(np.abs(a - a.T) < tol)


def is_metric(G):
    """True if the weighted graph G already satisfies the metric: every edge weight equals the
    shortest-path distance between its endpoints."""
    apsp = all_pairs_distances(G)
    for u, v, w in sorted_edges(G, weight=True):
        if w != apsp[u][v]:
            return False
    return True


def cumulative_coherence(D, p):
    """Cumulative coherence (Babel function) mu_1(p) of the redundant dictionary D.

    Atoms (rows of D) have norm >= 1; by Cauchy-Schwarz the largest entry in each row of |D D^T| is
    the squared atom norm, so it is dropped before accumulating. Only the p+1 largest entries of each
    row are needed, so np.partition (O(N^2)) instead of a full sort."""
    S = np.abs(D @ D.T)
    N = S.shape[1]
    k = p + 1                                  # row max (the dropped atom norm) + the p we accumulate
    if k < N:
        topk = -np.partition(-S, k - 1, axis=1)[:, :k]   # k largest per row, unordered
    else:
        topk = S
    topk = -np.sort(-topk, axis=1)             # sort just those few candidates, descending
    return topk[:, 1:k].sum(axis=1).max()      # drop the per-row max, sum the next p, take the worst row


def is_triangle(E, i, j, k):
    """True if indices i, j, k form a triangle in the edge set E (E holds sorted (u, v) tuples)."""
    return (_norm(i, j) in E and _norm(i, k) in E and _norm(j, k) in E)


def iter_triangles(G):
    """Yield the triangles of G as sorted vertex triples (a, b, c), a < b < c.

    Enumerates by scanning each edge's common neighbours: O(sum_v deg(v)^2)."""
    nbrs = {v: set(G.neighbors(v)) for v in G.nodes()}
    for a in sorted(nbrs):
        Na = nbrs[a]
        for b in Na:
            if b <= a:
                continue
            for c in nbrs[b] & Na:              # common neighbour of a and b
                if c > b:
                    yield (a, b, c)


def metric_triangles_matrix(G):
    """Dense metric-testing matrix for the broken triangles of G (3 rows per triangle, each row a
    triangle inequality with one edge negated). Row ordering follows iter_triangles."""
    D = make_index_encoding(G)
    e = G.number_of_edges()
    rows = []
    for tri in iter_triangles(G):
        idx = [D[_norm(*pair)] for pair in combinations(tri, 2)]
        for neg in range(3):                    # which of the 3 edges carries the -1
            z = np.zeros(e)
            z[idx] = 1
            z[idx[neg]] = -1
            rows.append(z)
    if not rows:
        return np.zeros(e)
    return np.array(rows)


def count_simple_cycles(G, kmin=3, kmax=0):
    """Count the induced (chordless) cycles of G with length in [kmin, kmax].

    Sage counted these via ``subgraph_search_count(CycleGraph(k), induced=True) / |Aut(C_k)|``; an
    induced subgraph isomorphic to C_k is exactly a chordless cycle of length k, so this enumerates
    them directly with ``networkx.chordless_cycles`` (cleaner and avoids the automorphism division)."""
    if kmin < 3:
        kmin = 3
    if kmax > G.number_of_nodes() or kmax == 0:
        kmax = G.number_of_nodes()
    return sum(1 for C in nx.chordless_cycles(G) if kmin <= len(C) <= kmax)


def get_subdivided_graph(G):
    """Unweighted graph obtained by subdividing each edge e of the weighted graph G into w(e) unit
    edges (w(e) - 1 fresh intermediate vertices). networkx has no subdivide_edge, so the path is
    built explicitly; fresh vertex labels continue past max(G.nodes())."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    nxt = (max(G.nodes()) + 1) if G.number_of_nodes() else 0
    for u, v in sorted_edges(G):
        w = int(G[u][v]["weight"])
        if w <= 1:
            H.add_edge(u, v, weight=1)
            continue
        prev = u
        for _ in range(w - 1):
            H.add_edge(prev, nxt, weight=1)
            prev, nxt = nxt, nxt + 1
        H.add_edge(prev, v, weight=1)
    return H


def on_broken_cycle(e, e_heavy, APSP_D, w_heavy, w_light):
    """Four-point check: could the light edge e and heavy edge e_heavy lie on a common broken cycle?
    (Scalar reference for the vectorised get_truly_light_edges below.)"""
    i, j = e
    x, y = e_heavy
    return not ((w_light + APSP_D[i][x] + APSP_D[j][y] >= w_heavy)
                and (w_light + APSP_D[i][y] + APSP_D[j][x] >= w_heavy))


def get_truly_light_edges(G, Heavy_Edges, APSP_D):
    """Light edges that provably DO participate in some broken cycle, i.e. survive the four-point test
    against at least one heavy edge (the provably-light ones are dropped).

    Vectorized: the O(|light| * |heavy|) double loop becomes one broadcast over the (light, heavy)
    grid, after materialising the APSP dict-of-dicts into a dense matrix once. Heavy_Edges and the
    returned light edges are (u, v, w) triples.
    """
    Light_Edges = list(edges_with_weights(G) - set(Heavy_Edges))
    Heavy = list(Heavy_Edges)
    if not Light_Edges or not Heavy:           # no heavy edges => nothing is on a broken cycle
        return set()
    verts = list(APSP_D)
    vmap = {v: a for a, v in enumerate(verts)}
    Dmat = np.empty((len(verts), len(verts)))
    for u, row in APSP_D.items():
        a = vmap[u]
        for v, d in row.items():
            Dmat[a, vmap[v]] = d
    li = np.fromiter((vmap[e[0]] for e in Light_Edges), int, len(Light_Edges))
    lj = np.fromiter((vmap[e[1]] for e in Light_Edges), int, len(Light_Edges))
    wl = np.fromiter((e[-1] for e in Light_Edges), float, len(Light_Edges))
    hx = np.fromiter((vmap[e[0]] for e in Heavy), int, len(Heavy))
    hy = np.fromiter((vmap[e[1]] for e in Heavy), int, len(Heavy))
    wh = np.fromiter((e[-1] for e in Heavy), float, len(Heavy))
    # four-point condition broadcast over (light row, heavy col)
    cond1 = wl[:, None] + Dmat[np.ix_(li, hx)] + Dmat[np.ix_(lj, hy)] >= wh[None, :]
    cond2 = wl[:, None] + Dmat[np.ix_(li, hy)] + Dmat[np.ix_(lj, hx)] >= wh[None, :]
    on_broken = ~(cond1 & cond2)               # the four-point test, elementwise
    keep = on_broken.any(axis=1)               # survives against at least one heavy edge
    return {Light_Edges[t] for t in np.nonzero(keep)[0]}


def truly_light_heuristic(G):
    """Discard provably-light edges, then return the cycle dimension of the remaining subgraph
    (heavy + ambiguous edges) together with that subgraph H.

    A standalone preprocessing / complexity tool -- NOT used by any repair heuristic. It keeps the
    edges that could matter (heavy + light-on-a-broken-cycle) and reports the cycle dimension
    (E - V + 1) of what remains."""
    APSP_D = all_pairs_distances(G)
    Heavy_Edges = domr_alg(G, with_weights=1)
    Truly_Light = get_truly_light_edges(G, Heavy_Edges, APSP_D)
    H = nx.Graph()
    for u, v, w in (Heavy_Edges | Truly_Light):
        H.add_edge(u, v, weight=w)
    return H.number_of_edges() - H.number_of_nodes() + 1, H


# ----------------------------------------------------------------------------
# Back-compat aliases
# ----------------------------------------------------------------------------

com_coh = cumulative_coherence

metric_triangles_mtx = metric_triangles_matrix

get_edge_inedcies = get_edge_indices          # note: original spelling had a typo


def MVD_Pivot_Rec(ind, x, S, Kn=None):
    """Deprecated signature wrapper around _mvd_pivot_rec (the Kn argument is ignored)."""
    return _mvd_pivot_rec(ind, x, S)
