# ============================================================================
# metric_extras.sage
#
# Auxiliary metric helpers that are NOT used by any repair algorithm in
# metric_repair.sage: weight setters, metric/coherence diagnostics, the triangle
# & cycle matrices, graph subdivision, and a deprecated wrapper + back-compat
# aliases. Kept for interactive use and the older notebooks that reference them
# (e.g. is_metric).
#
# Depends on metric_repair.sage -- some helpers call make_index_encoding,
# get_weights or _mvd_pivot_rec from there. Load metric_repair.sage FIRST.
# ============================================================================

import numpy as np
from itertools import combinations

def get_edge_indices(G, D):
    """Given a graph G and an edge -> index encoding D, return the inverse dict  index -> edge."""
    return {index: edge for edge, index in D.items()}

def set_weights(G, W):
    """Set edge weights in place from a list W ordered to match G.edges(sort=True)."""
    assert G.size() == len(W)                      # BUGFIX: was len(w)
    for i, e in enumerate(G.edges(sort=True)):
        G.set_edge_label(e[0], e[1], W[i])

def update_weights(G, Wp, ind_enc):
    """Given a weighted graph G, additive corrections Wp and an index encoding ind_enc: [n] -> E,
    set the new weights w + Wp in place."""
    Dw = get_weights(G)
    for i in np.nonzero(Wp)[0]:
        e = ind_enc[i]
        G.set_edge_label(e[0], e[1], Dw[e] + Wp[i])

def check_symmetric(a, tol=1e-8):
    """True if matrix a is symmetric up to tolerance tol."""
    return np.all(np.abs(a - a.T) < tol)

def is_metric(G):
    """True if the weighted graph G already satisfies the metric: every edge weight equals the
    shortest-path distance between its endpoints."""
    apsp = G.distance_all_pairs(by_weight=True)
    for u, v, w in G.edges(sort=True):
        if w != apsp[u][v]:
            return False
    return True

def cumulative_coherence(D, p):
    """Cumulative coherence (Babel function) mu_1(p) of the redundant dictionary D.

    Atoms (rows of D) are not assumed normalized, but are assumed to have norm >= 1, which holds
    in our use case (atom norms are at least 3). By Cauchy-Schwarz the largest entry in each row of
    |D D^T| is the squared atom norm, so it is dropped before accumulating.

    Perf: only the p+1 largest entries of each row are needed, so we np.partition (O(N^2)) instead
    of fully sorting every row (O(N^2 log N)). Result is identical to the old full-sort version."""
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
    """True if indices i, j, k form a triangle in the edge set E."""
    return (tuple(sorted([i, j])) in E
            and tuple(sorted([i, k])) in E
            and tuple(sorted([j, k])) in E)

def iter_triangles(G):
    """Yield the triangles of G as sorted vertex triples (a, b, c), a < b < c.

    Enumerates by scanning each edge's common neighbours: O(sum_v deg(v)^2) rather than the old
    O(|E|^3) scan over all edge-index triples."""
    nbrs = {v: set(G.neighbors(v)) for v in G.vertices(sort=False)}
    for a in sorted(nbrs):
        Na = nbrs[a]
        for b in Na:
            if b <= a:
                continue
            for c in nbrs[b] & Na:              # common neighbour of a and b
                if c > b:
                    yield (a, b, c)

def metric_triangles_matrix(G):
    """Dense metric testing matrix for the broken triangles of G (3 rows per triangle, each row a
    triangle inequality with one edge negated).

    Rewritten to enumerate actual triangles via iter_triangles (was an O(|E|^3) scan), and to build
    the matrix in one np.array call instead of repeated np.vstack. Same set of constraint rows; row
    ordering follows iter_triangles (this matrix has no external callers)."""
    D = make_index_encoding(G)
    e = G.size()
    rows = []
    for tri in iter_triangles(G):
        idx = [D[tuple(sorted(pair))] for pair in combinations(tri, 2)]
        for neg in range(3):                    # which of the 3 edges carries the -1
            z = np.zeros(e)
            z[idx] = 1
            z[idx[neg]] = -1
            rows.append(z)
    if not rows:
        return np.zeros(e)
    return np.array(rows)

def count_simple_cycles(G, kmin=3, kmax=0):
    """Count simple cycles of G with length in [kmin, kmax] (defaults: 3 .. G.order())."""
    if kmin < 3:
        kmin = 3
    if kmax > G.order() or kmax == 0:
        kmax = G.order()
    S = 0
    for k in range(kmin, kmax + 1):
        H = graphs.CycleGraph(k)
        S += G.subgraph_search_count(H, induced=True) / H.automorphism_group(return_group=False, order=True)
    return S

def get_subdivided_graph(G):
    """Unweighted graph obtained by subdividing each edge e of the weighted graph G into w(e) edges."""
    H = G.copy()
    for e in G.edges(labels=False, sort=True):
        k = G.edge_label(e[0], e[1]) - 1
        H.set_edge_label(e[0], e[1], 1)
        H.subdivide_edge(e, k)
    return H

def on_broken_cycle(e, e_heavy, APSP_D, w_heavy, w_light):
    """Four-point check: could the light edge e and heavy edge e_heavy lie on a common broken cycle?"""
    i, j = e
    x, y = e_heavy
    return not ((w_light + APSP_D[i][x] + APSP_D[j][y] >= w_heavy)
                and (w_light + APSP_D[i][y] + APSP_D[j][x] >= w_heavy))

com_coh = cumulative_coherence

metric_triangles_mtx = metric_triangles_matrix

get_edge_inedcies = get_edge_indices          # note: original spelling had a typo

def MVD_Pivot_Rec(ind, x, S, Kn=None):
    """Deprecated signature wrapper around _mvd_pivot_rec (the Kn argument is ignored)."""
    return _mvd_pivot_rec(ind, x, S)
