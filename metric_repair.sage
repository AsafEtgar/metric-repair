# ============================================================================
# metric_repair.sage
#
# The metric-repair library: the repair algorithms and exactly the support they
# use (edge/weight encoding, cycle matrix, graph completion, verifier). Auxiliary
# helpers not used by any algorithm live in metric_extras.sage.
# Self-contained -- does NOT depend on graph_models.sage. Load on its own, or
# via the Packages_and_Functions.ipynb loader together with graph_models.sage.
# ============================================================================

import numpy as np
import networkx as nx
from itertools import chain, combinations
from scipy import sparse
from scipy.optimize import linprog
from sage.graphs.base.boost_graph import floyd_warshall_shortest_paths


# ----------------------------------------------------------------------------
# Edge encoding & weights
# ----------------------------------------------------------------------------

def make_index_encoding(G):
    """Encode the edges of G as indices, ignoring labels and sorting each edge (small, large).
    Returns a dict  D: E -> [n]."""
    return dict(zip(G.edges(sort=True, labels=False), range(G.num_edges())))


def get_weights(G):
    """Return the weight function w: E -> R as a dict keyed by (u, v) with u < v."""
    assert G.weighted()
    return {(u, v): w for u, v, w in G.edges(sort=True)}

def get_weights_vector(G, D):
    """Return the weight vector indexed by the edge encoding D: E -> [n]."""
    w = np.zeros(G.size())
    for edge, weight in get_weights(G).items():
        w[D[edge]] = weight
    return w


# ----------------------------------------------------------------------------
# Matrix <-> vertex-label boundary
#
# Matrix algorithms (MVD_Pivot, Gilbert_Jain_IOMR) work on a position-indexed
# adjacency matrix, but a graph's vertex LABELS are frequently NOT 0..n-1 --
# the generators build graphs from their edge list, so isolated vertices are
# dropped and the labels have gaps. The single rule for these algorithms:
# enter with graph_to_matrix, leave with positions_to_labels. Keeping both
# halves of the translation here (rather than re-derived inline) is what makes
# the position-as-label bug structurally impossible.
# ----------------------------------------------------------------------------

def graph_to_matrix(G):
    """Return (A, verts): G's weighted adjacency matrix as a float NumPy array indexed by POSITION,
    together with verts = G.vertices(sort=True) -- the position -> label map. Pass the cover that a
    matrix algorithm produces (in positions) back through positions_to_labels(., verts) before
    returning it, or downstream label-space code (verifier, reduce_solution) sees the wrong edges."""
    verts = G.vertices(sort=True)
    A = np.array(G.weighted_adjacency_matrix(vertices=verts), dtype=float)
    return A, verts

def positions_to_labels(S, verts):
    """Translate a set of matrix-position pairs (i, j) into vertex-LABEL pairs (verts[i], verts[j]),
    each normalized to (small, large). Inverse companion to graph_to_matrix."""
    out = set()
    for i, j in S:
        a, b = verts[i], verts[j]
        out.add((a, b) if a < b else (b, a))
    return out


# ----------------------------------------------------------------------------
# Metric utilities
# ----------------------------------------------------------------------------


def verifier(G, S, tol=1e-9):
    """Verify that S is a cover of G: returns 1 if heavying-up the edges of S yields a metric, else 0.

    Heavy every cover edge to M (the max weight) -- large enough to act like removal, since any path
    using a cover edge then costs >= M >= every direct weight and can never undercut one. A non-cover
    edge (u, v) is therefore unfixable iff a path avoiding the cover already undercuts it: D[u][v] < w.

    The comparison uses a float tolerance. On non-integer weights D[u][v] is a sum of many float edge
    weights, so a shortest path that merely TIES the direct edge differs from w by rounding noise; an
    exact != would misread that tie as a break and spuriously reject a valid cover -- and the error
    compounds on larger graphs with longer paths. (iomr_verifier already guards this the same way.)"""
    H = G.copy()
    M = max(G.edge_labels())
    for e in S:
        H.set_edge_label(e[0], e[1], M)
    D = H.distance_all_pairs(by_weight=True)
    for u, v, w in H.edges(sort=True):
        if (u, v) in S or (v, u) in S:
            continue
        if D[u][v] < float(w) - tol:
            return 0
    return 1

def iomr_verifier(G, S, tol=1e-9):
    """Verify that S is a valid INCREASE-ONLY (IOMR) cover of G: return 1 if the graph can be made
    metric by increasing ONLY the weights of edges in S, else 0.

    Like verifier(), but for the increase-only model. Remove the cover edges (in IOMR they may be
    raised arbitrarily, even to +inf) and require that EVERY edge -- cover and non-cover alike -- is
    unbroken by a path among the remaining (frozen) edges: d_{G-S}(u, v) >= w(u, v). Cover edges are
    checked too, because increase-only cannot lower a cover edge that a frozen detour undercuts
    (general repair can, which is why verifier() exempts them).

    Note: IOMR feasibility implies general feasibility, so this is strictly stronger than verifier()
    -- it never accepts a cover that verifier() rejects. A float tolerance avoids misreading a tie
    detour as a shortcut on non-integer weights."""
    H = G.copy()
    for a, b in S:
        if H.has_edge(a, b):
            H.delete_edge(a, b)
    D = H.distance_all_pairs(by_weight=True)
    for u, v, w in G.edges(sort=True):
        duv = D.get(u, {}).get(v, float('inf'))
        if duv < float(w) - tol:
            return 0
    return 1


# ----------------------------------------------------------------------------
# Cycle functions
# ----------------------------------------------------------------------------


def get_list_of_edges(cyc_vtx):
    """Given the vertex list of a cycle, return its edges as sorted (u, v) tuples."""
    k = len(cyc_vtx)
    return [tuple(sorted((cyc_vtx[i], cyc_vtx[(i + 1) % k]))) for i in range(k)]

def get_chordless_cycles(G):
    """Generator of all chordless cycles of G (as edge lists), delegating to networkx."""
    for C in nx.chordless_cycles(G.networkx_graph()):
        yield get_list_of_edges(C)


def induced_cycle_matrix(G):
    """Metric testing matrix Phi for the chordless cycles of G, as a sparse CSR matrix.

    Returns (Phi, count) where count is the number of chordless cycles. If count == 0 returns
    (zero-vector, 0). For each cycle (edge indices C_ind) we emit one row per edge: +1 on every
    cycle edge and -1 on the singled-out one.

    Perf: assembled from COO triplets in a single sparse build, instead of the old per-row
    np.vstack on dense vectors (which was O(rows * |E|) memory and quadratic in reallocations).

    Behavior note: the old version seeded each cycle's block with a zero row (np.zeros(m)) and never
    deleted it, so Phi carried one all-zero row per cycle. Those rows are inert (0 <= 0 in the LP),
    so l1_minimization / l1_min_heuristic are unaffected; this version simply omits them, so Phi now
    has exactly sum_c len(cycle_c) rows."""
    m = G.num_edges()
    ind_enc = make_index_encoding(G)
    rows, cols, data = [], [], []
    r = 0
    count = 0
    for C in get_chordless_cycles(G):
        C_ind = [ind_enc[e] for e in C]
        L = len(C_ind)
        for t in range(L):                      # one row per cycle edge; -1 on edge t, +1 elsewhere
            rows.extend([r] * L)
            cols.extend(C_ind)
            data.extend([-1.0 if s == t else 1.0 for s in range(L)])
            r += 1
        count += 1
    if count == 0:
        return np.zeros(m), count
    return sparse.csr_matrix((data, (rows, cols)), shape=(r, m)), count


# ----------------------------------------------------------------------------
# Graph completion / preprocessing
# ----------------------------------------------------------------------------

def complete(G):
    """Complete the weighted graph G by adding every missing edge xy with weight dist(x, y).

    TODO: handle disconnected G (distances between components are infinite)."""
    H = G.copy()
    Gc = G.complement()
    D = G.distance_all_pairs(by_weight=True)
    H.add_edges([(e[0], e[1], D[e[0]][e[1]]) for e in Gc.edges(sort=True)])
    return H


# ----------------------------------------------------------------------------
# Metric-repair algorithms
# ----------------------------------------------------------------------------

def reduce_solution(S, G, Diff=[]):
    """Restrict an extended cover S to the edges actually present in G."""
    return {e for e in G.edges(labels=False, sort=True) if e in S}

def domr_alg(G, with_weights=0):
    """Decrease-Only Metric Repair: the edges whose weight exceeds their shortest-path distance.
    With with_weights, returns (u, v, w) triples; otherwise (u, v) pairs."""
    S = set()
    apsp = G.distance_all_pairs(by_weight=True)
    for e in G.edges(sort=True):
        if e[2] != apsp[e[0]][e[1]]:
            S.add(e if with_weights else (e[0], e[1]))
    return S

def Gilbert_Jain_IOMR(Kn):
    """Gilbert & Jain heuristic: for each broken triangle, arbitrarily fix the 'left' edge.
    Assumes the input graph Kn is complete (IOMR).

    Vectorized: the inner double loop is replaced by a NumPy reduction per pivot k. Within one k,
    each row i only ever reads its own M[i,k] and the fixed rows i,k, and the final M[i,k] is just
    max(M[i,k], max_{j<i}(M[i,j]-M[k,j])) regardless of the order j is processed -- so computing all
    i at once is identical to the original sequential scan. Sage matrix element access (slow) is
    also dropped in favour of a single np.array conversion."""
    A, verts = graph_to_matrix(Kn)                         # position-indexed; verts maps back to labels
    n = A.shape[0]
    S = set()
    lower = np.tril(np.ones((n, n), dtype=bool), k=-1)     # mask j < i, computed once
    idx = np.arange(n)
    for k in range(n):
        cand = np.where(lower, A - A[k][None, :], -np.inf).max(axis=1)   # max_{j<i}(A[i,j]-A[k,j])
        viol = cand > A[:, k]
        if viol.any():
            for i in idx[viol]:
                S.add((int(i), k) if i < k else (k, int(i)))
            A[viol, k] = cand[viol]
    return positions_to_labels(S, verts)

def _mvd_pivot_rec(ind, X, S):
    """Recursive pivot step for MVD_Pivot. X is the (mutated) NumPy adjacency matrix; S accumulates
    the corrected edges.

    Vectorized: for a fixed pivot i, every pair (j, k) of remaining vertices is clipped into
    [ |X[i,j]-X[i,k]| , X[i,j]+X[i,k] ] independently -- row i is untouched in this call and each
    pair writes a distinct entry, so the whole pair loop is one broadcast clip.

    BUGFIX (symmetry): the original scalar code (and the first vectorized port) wrote only X[j,k]
    and left X[k,j] stale, so the distance matrix turned asymmetric and later pivots read stale
    distances. We now write both X[j,k] and X[k,j]. This intentionally changes MVD_Pivot's output
    (it no longer reproduces the buggy original)."""
    if len(ind) <= 2:
        return
    i = np.random.choice(ind)
    ind_i = ind.copy()                         # don't mutate the caller's list
    ind_i.remove(i)                            # remove the current pivot
    R = np.array(ind_i)
    p = X[i, R]                                # X[i, m]; X is kept symmetric (see write below)
    lo = np.abs(p[:, None] - p[None, :])
    hi = p[:, None] + p[None, :]
    sub = X[np.ix_(R, R)]
    new = np.minimum(np.maximum(sub, lo), hi)  # clip into [lo, hi]
    a, b = np.triu_indices(len(R), k=1)        # pairs (a before b in ind order)
    changed = new[a, b] != sub[a, b]
    X[R[a], R[b]] = new[a, b]                  # write X[j,k] ...
    X[R[b], R[a]] = new[a, b]                  # ... and X[k,j]: keep the distance matrix symmetric
    for t in np.nonzero(changed)[0]:
        j, k = int(R[a[t]]), int(R[b[t]])
        S.add((j, k) if j < k else (k, j))
    _mvd_pivot_rec(ind_i, X, S)

def MVD_Pivot(Kn):
    """'Fitting Metrics with Minimum Disagreement' pivot algorithm (solves general MR, not IOMR).

    The recursion works in matrix-POSITION space, so its cover is mapped back to vertex LABELS via the
    graph_to_matrix / positions_to_labels boundary (see that section for why this is mandatory)."""
    X, verts = graph_to_matrix(Kn)
    S = set()
    _mvd_pivot_rec(list(range(len(verts))), X, S)
    return positions_to_labels(S, verts)

def l1_minimization(Gc):
    """L1 metric repair on an already-completed graph Gc.

    Minimizes the L1 norm of the edge-weight corrections subject to all chordless-cycle (metric)
    constraints, and returns the support of the correction (the cover)."""
    phi, count = induced_cycle_matrix(Gc)      # (num_constraints x num_edges), sparse CSR
    if count == 0:
        return set()
    D = make_index_encoding(Gc)
    w = get_weights_vector(Gc, D)
    m = phi.shape[1]                           # one LP variable (weight correction) per edge
    # same LP as before (A_ub = -phi, b_ub = phi @ w) but fed the sparse matrix directly,
    # skipping the two np.transpose round-trips the old code did.
    soln = linprog(np.ones(m), A_ub=-phi, b_ub=phi @ w, method='highs')
    x = soln.x
    return {(u, v) for u, v in Gc.edges(sort=True, labels=False) if x[D[(u, v)]] > 0}

def l1_min_heuristic(G):
    """L1 metric-repair heuristic for graph metric repair:

    1. complete G to Gc, filling missing edges with shortest-path distances,
    2. solve the L1 minimization over all metric (chordless-cycle) constraints of Gc,
    3. return the support of the correction restricted to E(G) (a light cover of G).

    BUGFIX: previously called an undefined `induc` and mishandled the (matrix, count) return of
    `induced_cycle_matrix`; now delegates to l1_minimization on the completed graph. (The original
    `method='simplex'` was also removed from modern SciPy, so 'highs' is used.)"""
    G_edges = set(G.edges(sort=True, labels=False))
    return {e for e in l1_minimization(complete(G)) if e in G_edges}

def find_shortest_path(u, v, Dict):
    """Reconstruct a shortest u-v path (as a list of sorted edge tuples) from a predecessor dict."""
    end = v
    P = []
    while end != u:
        w = Dict[u][end]                       # a predecessor of `end`
        P.append(tuple(sorted((end, w))))
        end = w
    return P

def shortest_path_cover(G, general=True):
    """Greedy shortest-path cover, an L(+1)-approximation for (graph) metric repair.

    Repeatedly finds a broken edge, covers it together with the edges of a shortest alternative
    path, removes them, and iterates until no broken edge remains.

    (The duplicate empty stub that previously shadowed this function has been removed.)"""
    H = G.copy()
    D = get_weights(G)
    S = set()
    while True:
        dist, Dict = floyd_warshall_shortest_paths(H, predecessors=1, distances=1)
        found_broken = 0
        for u, v, w in H.edges(sort=True):
            if D[(u, v)] > dist[u][v]:
                found_broken = 1
                P = find_shortest_path(u, v, Dict)
                if general:
                    S.add(tuple(sorted((u, v))))
                    H.delete_edge((u, v))
                S.update(P)
                H.delete_edges(P)
        if not found_broken:
            return S

def left_edge_heuristic(G):
    """Complete G, run the Gilbert & Jain left-edge heuristic, then reduce to a cover of G."""
    return reduce_solution(Gilbert_Jain_IOMR(complete(G)), G)

def pivot_heuristic(G):
    """Complete G, run the MVD pivot algorithm, then reduce to a cover of G."""
    return reduce_solution(MVD_Pivot(complete(G)), G)


def get_truly_light_edges(G, Heavy_Edges, APSP_D):
    """Light edges that provably DO participate in some broken cycle, i.e. survive the four-point
    test against at least one heavy edge (the provably-light ones are dropped).

    Vectorized: the O(|light| * |heavy|) Python double loop becomes one broadcast over the
    (light, heavy) grid, after materializing the APSP dict-of-dicts into a NumPy matrix once."""
    Light_Edges = list(set(G.edges()) - Heavy_Edges)
    Heavy = list(Heavy_Edges)
    if not Light_Edges or not Heavy:           # no heavy edges => nothing is on a broken cycle
        return set()
    # materialize the distance dict-of-dicts into a dense matrix with a vertex index map
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
    on_broken = ~(cond1 & cond2)               # matches on_broken_cycle (metric_extras.sage), elementwise
    keep = on_broken.any(axis=1)               # survives against at least one heavy edge
    return {Light_Edges[t] for t in np.nonzero(keep)[0]}

def truly_light_heuristic(G):
    """Discard provably-light edges, then return the cycle dimension of the remaining subgraph
    (heavy + ambiguous edges) together with that subgraph H."""
    APSP_D = G.shortest_path_all_pairs(by_weight=1)[0]
    Heavy_Edges = domr_alg(G, with_weights=1)
    Truly_Light = get_truly_light_edges(G, Heavy_Edges, APSP_D)
    H = Graph(weighted=1)
    H.add_edges(Heavy_Edges | Truly_Light)
    return H.size() - H.order() + 1, H
