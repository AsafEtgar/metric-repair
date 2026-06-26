"""
metric_repair.py  --  pure-Python (networkx + scipy) port of metric_repair.sage

The metric-repair library: the repair algorithms and exactly the support they use (edge/weight
encoding, cycle matrix, graph completion, verifier). Auxiliary helpers not used by any algorithm
live in metric_extras.py.

REPRESENTATION (read this once and the rest is obvious)
-------------------------------------------------------
* A weighted graph is a ``networkx.Graph`` with a numeric ``weight`` on every edge.
* An *edge* is always normalised to a sorted tuple ``(u, v)`` with ``u <= v``.
* A *cover* ``S`` is a ``set`` of such edges.
* ``sorted_edges(G)`` is the canonical edge order (the replacement for Sage ``G.edges(sort=True)``).
* Shortest paths come from ``scipy.sparse.csgraph`` (C code), in two forms:
    - ``all_pairs_distances(G)``  -> dict-of-dicts {u: {v: dist}}   (replaces ``distance_all_pairs``)
    - the predecessor matrix inside ``shortest_path_cover``         (replaces ``floyd_warshall``)

EQUIVALENCE TO THE SAGE VERSION
-------------------------------
Deterministic functions (complete, domr_alg, verifier, iomr_verifier, the cycle matrix, l1) are
bit-for-bit identical to Sage on the same input graph -- see equivalence/. The two randomised /
tie-broken pieces match in *distribution*, not exactly:
  * MVD_Pivot is reproducible given a NumPy seed, and reproduces Sage exactly on the same matrix +
    same seed (the kernel below is copied verbatim from the Sage file).
  * shortest_path_cover depends on which equal-length shortest path is chosen; scipy and Sage break
    ties differently, so covers differ slightly in *which* edges (sizes match within a few percent,
    and every cover is valid).
"""

import numpy as np
import networkx as nx
from itertools import combinations
from scipy import sparse
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy.sparse.csgraph import shortest_path


# ----------------------------------------------------------------------------
# Tiny shared helpers
# ----------------------------------------------------------------------------

def _norm(u, v):
    # TODO: This is also defined in graph_models.py, do we need both implementations?
    """Order an edge's endpoints as (small, large)."""
    return (u, v) if u <= v else (v, u)


def sorted_edges(G, weight=False):
    """Canonical edge order: sorted list of normalised (u, v) [or (u, v, w)] tuples.

    This is the pure-Python replacement for Sage's ``G.edges(sort=True[, labels=...])`` and is used
    wherever the original relied on that deterministic ordering.
    """
    order = sorted(_norm(a, b) for a, b in G.edges())
    if weight:
        return [(u, v, G[u][v]["weight"]) for (u, v) in order]
    return order


def edges_with_weights(G):
    """Set of (u, v, w) triples (u <= v) -- the analogue of Sage's ``set(G.edges())``."""
    return {(u, v, G[u][v]["weight"]) for (u, v) in sorted_edges(G)}


def all_pairs_distances(G):
    """All-pairs shortest-path distances as a dict-of-dicts {u: {v: dist}}.

    Pure-Python replacement for Sage's ``G.distance_all_pairs(by_weight=True)``: the heavy lifting is
    done in C by ``scipy.sparse.csgraph.shortest_path`` (Dijkstra; edge weights are non-negative),
    then unpacked into the dict-of-dicts the translated algorithms read. Unreachable pairs hold
    ``math.inf`` -- Sage simply omitted them, and every caller treats 'missing' as inf, so the two
    behave identically.
    """
    verts = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(verts)}
    A = nx.to_scipy_sparse_array(G, nodelist=verts, weight="weight", format="csr")
    D = shortest_path(A, method="D", directed=False)
    return {u: {w: D[idx[u], idx[w]] for w in verts} for u in verts}


# ----------------------------------------------------------------------------
# Edge encoding & weights
# ----------------------------------------------------------------------------

def make_index_encoding(G):
    """Encode the edges of G as indices 0..m-1 (each edge sorted small, large). Returns dict E -> [m]."""
    return {e: i for i, e in enumerate(sorted_edges(G))}


def get_weights(G):
    """Return the weight function w: E -> R as a dict keyed by (u, v) with u <= v."""
    return {(u, v): G[u][v]["weight"] for (u, v) in sorted_edges(G)}


def get_weights_vector(G, D):
    """Return the weight vector indexed by the edge encoding D: E -> [m]."""
    w = np.zeros(G.number_of_edges())
    for edge, weight in get_weights(G).items():
        w[D[edge]] = weight
    return w


# ----------------------------------------------------------------------------
# Matrix <-> vertex-label boundary
#
# The matrix algorithms (MVD_Pivot, Gilbert_Jain_IOMR) work on a position-indexed adjacency matrix,
# but a graph's vertex LABELS are frequently NOT 0..n-1 (generators drop isolated vertices, leaving
# gaps). The single rule for these algorithms: enter with graph_to_matrix, leave with
# positions_to_labels. Keeping both halves of the translation here -- rather than re-derived inline --
# is what makes the "treat a matrix position as a label" bug structurally impossible.
# ----------------------------------------------------------------------------

def graph_to_matrix(G):
    """Return (A, verts): G's weighted adjacency matrix (float, position-indexed) and
    verts = sorted(G.nodes()), the position -> label map. Missing edges and the diagonal are 0
    (callers pass complete graphs, so every off-diagonal entry is a real edge)."""
    verts = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=verts, weight="weight", dtype=float)
    return A, verts


def positions_to_labels(S, verts):
    """Translate a set of matrix-position pairs (i, j) into vertex-LABEL pairs (verts[i], verts[j]),
    each normalised to (small, large). Inverse companion to graph_to_matrix."""
    out = set()
    for i, j in S:
        a, b = verts[i], verts[j]
        out.add((a, b) if a < b else (b, a))
    return out


# ----------------------------------------------------------------------------
# Verifiers (feasibility predicates)
# ----------------------------------------------------------------------------

def verifier(G, S, tol=1e-9):
    """Verify that S is a cover of G: return 1 if heavying-up the edges of S yields a metric, else 0.

    Heavy every cover edge to M (the max weight) -- large enough to act like removal, since any path
    using a cover edge then costs >= M >= every direct weight and can never undercut one. A non-cover
    edge (u, v) is therefore unfixable iff a path avoiding the cover already undercuts it: D[u][v] < w.
    A float tolerance keeps a tie (path == direct edge) from being misread as a break on non-integer
    weights. (General metric repair -- it may increase OR decrease cover edges.)
    """
    H = G.copy()
    M = max(d["weight"] for _, _, d in G.edges(data=True))
    Sset = {_norm(u, v) for u, v in S}
    for u, v in Sset:
        if H.has_edge(u, v):
            H[u][v]["weight"] = M
    D = all_pairs_distances(H)
    for u, v, w in sorted_edges(H, weight=True):
        if _norm(u, v) in Sset:
            continue
        if D[u][v] < float(w) - tol:
            return 0
    return 1


def iomr_verifier(G, S, tol=1e-9):
    """Verify that S is a valid INCREASE-ONLY (IOMR) cover of G: return 1 if G can be made metric by
    increasing ONLY the weights of S, else 0.

    Like verifier(), but increase-only. Remove the cover edges (they may be raised arbitrarily, even
    to +inf) and require EVERY edge -- cover and non-cover alike -- to be unbroken by a path among the
    remaining (frozen) edges: d_{G-S}(u, v) >= w(u, v). Cover edges are checked too, because
    increase-only cannot lower a cover edge that a frozen detour undercuts (general repair can, which
    is why verifier() exempts them). IOMR feasibility implies general feasibility, so this is strictly
    stronger than verifier().
    """
    H = G.copy()
    for a, b in S:
        if H.has_edge(a, b):
            H.remove_edge(a, b)
    D = all_pairs_distances(H)
    for u, v, w in sorted_edges(G, weight=True):
        duv = D.get(u, {}).get(v, float("inf"))
        if duv < float(w) - tol:
            return 0
    return 1


# ----------------------------------------------------------------------------
# Cycle functions
# ----------------------------------------------------------------------------

def get_list_of_edges(cyc_vtx):
    """Given the vertex list of a cycle, return its edges as sorted (u, v) tuples."""
    k = len(cyc_vtx)
    return [_norm(cyc_vtx[i], cyc_vtx[(i + 1) % k]) for i in range(k)]


def get_chordless_cycles(G):
    """Generator of all chordless cycles of G (as edge lists), via networkx (same call Sage made)."""
    for C in nx.chordless_cycles(G):
        yield get_list_of_edges(C)


def induced_cycle_matrix(G):
    """Metric-testing matrix Phi for the chordless cycles of G, as a sparse CSR matrix.

    Returns (Phi, count) where count is the number of chordless cycles. For each cycle we emit one
    row per edge: +1 on every cycle edge and -1 on the singled-out one (the triangle/cycle
    inequality). If count == 0 returns (zero-vector, 0).
    """
    m = G.number_of_edges()
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


def broken_cycles(G, max_len=None):
    """Yield the broken simple cycles of G, each as a list of sorted (u, v) edge tuples.

    A cycle is 'broken' iff its longest edge exceeds the sum of the others (2*max > total) -- a
    violated polygon inequality. Enumerates simple cycles via networkx; pass max_len to cap the cycle
    length. Enumeration is worst-case exponential, so run this on the (sparse) ORIGINAL graph G, not
    its dense completion; the planned separation-oracle LP is the scalable alternative.
    """
    for cyc in nx.simple_cycles(G, length_bound=max_len):
        if len(cyc) < 3:
            continue
        edges = get_list_of_edges(cyc)
        ws = [G[u][v]["weight"] for (u, v) in edges]
        if 2 * max(ws) > sum(ws):
            yield edges


def broken_cycle_incidence(G, max_len=None):
    """Return (B, n_cycles, D): the 0/1 incidence matrix of broken cycles (rows) against edges
    (columns) as a sparse CSR matrix, the number of broken cycles, and the edge encoding D: E -> [m].
    B[c, D[e]] = 1 iff edge e lies on broken cycle c. This is the constraint matrix of the hitting-set
    ILP (exact_metric_repair_ilp) and the covering LP (broken_cycle_rounding_heuristic) below.
    """
    D = make_index_encoding(G)
    m = G.number_of_edges()
    rows, cols = [], []
    r = 0
    for edges in broken_cycles(G, max_len):
        for e in edges:
            rows.append(r)
            cols.append(D[e])
        r += 1
    if r == 0:
        return sparse.csr_matrix((0, m)), 0, D
    B = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(r, m))
    return B, r, D


# ----------------------------------------------------------------------------
# Graph completion
# ----------------------------------------------------------------------------

def complete(G):
    """Complete the weighted graph G by adding every missing edge xy with weight dist(x, y).

    Assumes G is connected (otherwise the missing-edge distance is infinite)."""
    H = G.copy()
    D = all_pairs_distances(G)
    verts = sorted(G.nodes())
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            u, v = verts[i], verts[j]
            if not H.has_edge(u, v):
                H.add_edge(u, v, weight=D[u][v])
    return H


# ----------------------------------------------------------------------------
# Metric-repair algorithms
# ----------------------------------------------------------------------------

def reduce_solution(S, G):
    """Restrict an extended cover S to the edges actually present in G."""
    return {_norm(u, v) for u, v in G.edges() if _norm(u, v) in S}


def domr_alg(G, with_weights=0):
    """Decrease-Only Metric Repair: the edges whose weight exceeds their shortest-path distance.
    With with_weights, returns (u, v, w) triples; otherwise (u, v) pairs."""
    S = set()
    apsp = all_pairs_distances(G)
    for u, v, w in sorted_edges(G, weight=True):
        if w != apsp[u][v]:
            S.add((u, v, w) if with_weights else (u, v))
    return S


def Gilbert_Jain_IOMR(Kn):
    """Gilbert & Jain heuristic: for each broken triangle, arbitrarily fix the 'left' edge.
    Assumes the input graph Kn is complete (IOMR / increase-only).

    Vectorized: the inner double loop is one NumPy reduction per pivot k -- the final M[i,k] is
    max(M[i,k], max_{j<i}(M[i,j] - M[k,j])) regardless of the order j is processed, so computing all i
    at once equals the original sequential scan. (Kernel copied from the Sage version unchanged.)
    """
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
    the corrected edge POSITIONS.

    For a fixed pivot i, every pair (j, k) of remaining vertices is clipped into
    [ |X[i,j]-X[i,k]| , X[i,j]+X[i,k] ] -- one broadcast clip. Both X[j,k] and X[k,j] are written so
    the matrix stays symmetric (later pivots must read up-to-date distances). Kernel copied verbatim
    from the Sage version, so on the same matrix + same NumPy seed it reproduces the Sage cover.
    """
    if len(ind) <= 2:
        return
    i = np.random.choice(ind)
    ind_i = ind.copy()                         # don't mutate the caller's list
    ind_i.remove(i)                            # remove the current pivot
    R = np.array(ind_i)
    p = X[i, R]
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
    graph_to_matrix / positions_to_labels boundary."""
    X, verts = graph_to_matrix(Kn)
    S = set()
    _mvd_pivot_rec(list(range(len(verts))), X, S)
    return positions_to_labels(S, verts)


def _l1_solve(Gc, cycle_matrix=induced_cycle_matrix, solver="highs-ipm", reweight=0, eps=1e-3):
    """Solve the L1 weight-correction LP on Gc; return (x, D): the correction vector and edge encoding.

    Minimises sum_e x_e subject to the polygon inequality on the repaired weights w+x for every row of
    cycle_matrix(Gc), with x >= 0.

    solver: linprog method governing WHICH optimal solution is returned (the optimum value is
    solver-independent -- only the support differs). MEASURED on this LP the effect is small (~a few
    %): contrary to the usual "simplex is sparse" intuition, 'highs-ipm' (interior point) comes out
    marginally SPARSER than 'highs-ds' (dual simplex) here, so it is the default. reweight > 0 runs
    that many extra reweighted-L1 passes (cost_e <- 1/(|x_e|+eps), Candes-Wakin-Boyd) -- the more
    reliable sparsifier; 'highs-ipm' with reweight gave the sparsest support measured.
    """
    phi, count = cycle_matrix(Gc)
    D = make_index_encoding(Gc)
    m = Gc.number_of_edges()
    if count == 0:
        return np.zeros(m), D
    w = get_weights_vector(Gc, D)
    c = np.ones(m)
    x = np.zeros(m)
    for _ in range(reweight + 1):
        x = linprog(c, A_ub=-phi, b_ub=phi @ w, method=solver).x
        c = 1.0 / (np.abs(x) + eps)            # reweight for the next pass
    return x, D


def l1_minimization(Gc, solver="highs-ipm", reweight=0):
    """L1 metric repair on an already-completed graph Gc.

    Minimises the L1 norm of the edge-weight corrections subject to all chordless-cycle (metric)
    constraints, and returns the support of the correction (the cover). Defaults to the interior-point
    solver ('highs-ipm'), marginally the sparsest single-solve choice measured here; set reweight > 0
    (reweighted L1) for an even sparser support. (The LP OPTIMUM is solver-independent; only which
    support is returned changes. Support cutoff is 1e-7 to ignore interior-point near-zeros.)"""
    x, D = _l1_solve(Gc, induced_cycle_matrix, solver=solver, reweight=reweight)
    return {(u, v) for u, v in sorted_edges(Gc) if x[D[(u, v)]] > 1e-7}


def l1_min_heuristic(G, solver="highs-ipm", reweight=0):
    """L1 metric-repair heuristic: complete G, solve the L1 minimisation over all metric (chordless-
    cycle) constraints of the completion, and return the support restricted to E(G)."""
    G_edges = set(sorted_edges(G))
    return {e for e in l1_minimization(complete(G), solver=solver, reweight=reweight) if e in G_edges}


# ----------------------------------------------------------------------------
# Exact ILP & LP-rounding heuristics (hitting set over the broken cycles of G)
#
# A set S is a valid general-MR cover iff it hits every broken cycle of G, so the exact minimum cover
# is a minimum hitting set (exact_metric_repair_ilp). The covering LP relaxation, randomized-rounded,
# is broken_cycle_rounding_heuristic. l1_rounding_heuristic instead rounds the weight-correction LP
# above. NOTE: validity is guaranteed only when ALL broken cycles are enumerated (max_len=None); a
# length bound makes these heuristic (a long broken cycle may go un-hit).
# ----------------------------------------------------------------------------

def exact_metric_repair_ilp(G, max_len=None):
    """Exact minimum metric-repair cover (general MR): the smallest edge set hitting every broken cycle
    of G. Solves the hitting-set ILP

        min  sum_e y_e   s.t.   sum_{e in C} y_e >= 1  for every broken cycle C,   y_e in {0, 1}

    with scipy.optimize.milp, returning the cover {e : y_e = 1}. This is the exact optimum the
    heuristics approximate. Enumerates broken cycles (worst-case exponential -- pass max_len on larger
    graphs; the result is then a lower bound / may be invalid for long broken cycles)."""
    B, n_cyc, D = broken_cycle_incidence(G, max_len)
    if n_cyc == 0:
        return set()
    m = B.shape[1]
    res = milp(c=np.ones(m), constraints=LinearConstraint(B, lb=1, ub=np.inf),
               integrality=np.ones(m), bounds=Bounds(0, 1))
    y = np.round(res.x).astype(int)
    inv = {i: e for e, i in D.items()}
    return {inv[i] for i in np.nonzero(y)[0]}


def l1_rounding_heuristic(G, rounds=20, scale=1.0, seed=None, solver="highs-ipm"):
    """Randomized rounding of the current L1 weight-correction LP into a (sparser) valid cover.

    Solve l1 on complete(G) to get corrections x, restrict to E(G), and over several rounds sample each
    support edge e independently with probability min(1, scale * x_e / max_x). Keep the smallest SAMPLED
    set the verifier accepts; fall back to the full l1 support (always valid) if no sample is valid. So
    the result is never worse than l1_min_heuristic, and usually sparser."""
    Gc = complete(G)
    x, D = _l1_solve(Gc, induced_cycle_matrix, solver=solver)
    support = [e for e in sorted_edges(G) if x[D[e]] > 1e-7]
    if not support:
        return set()
    xmax = max(x[D[e]] for e in support)
    p = {e: min(1.0, scale * x[D[e]] / xmax) for e in support}
    rng = np.random.default_rng(seed)
    best = set(support)                                  # full support: the deterministic, valid cover
    for _ in range(rounds):
        S = {e for e in support if rng.random() < p[e]}
        if S and len(S) < len(best) and verifier(G, S):
            best = S
    return best


def broken_cycle_rounding_heuristic(G, max_len=None, rounds=20, scale=None, seed=None):
    """Randomized rounding of the covering LP over ALL broken cycles of G into a valid cover.

    Solve the LP relaxation of the hitting set  min 1.y  s.t. B y >= 1, 0 <= y <= 1, then over several
    rounds sample each edge e with probability min(1, scale * y_e), unioning the picks until every
    broken cycle is hit; any cycle still un-hit after the rounds is covered greedily, so the returned
    set is always a valid cover (given max_len=None). scale defaults to ~ln(#cycles), the standard
    set-cover rounding factor."""
    B, n_cyc, D = broken_cycle_incidence(G, max_len)
    if n_cyc == 0:
        return set()
    m = B.shape[1]
    y = linprog(np.ones(m), A_ub=-B, b_ub=-np.ones(n_cyc), bounds=(0, 1), method="highs").x
    if scale is None:
        scale = max(1.0, float(np.log(n_cyc + 1)))
    p = np.minimum(1.0, scale * y)
    rng = np.random.default_rng(seed)
    Bcsr = B.tocsr()
    chosen = set()

    def hits():
        return np.asarray(Bcsr[:, list(chosen)].sum(axis=1)).ravel() if chosen else np.zeros(n_cyc)

    for _ in range(rounds):
        chosen.update(np.nonzero(rng.random(m) < p)[0].tolist())
        if (hits() >= 1).all():
            break
    for c in np.nonzero(hits() < 1)[0]:                  # greedy top-up -> always a valid cover
        row = Bcsr.getrow(c).indices
        if not (set(row.tolist()) & chosen):
            chosen.add(int(row[0]))
    inv = {i: e for e, i in D.items()}
    return {inv[i] for i in chosen}

def find_shortest_path(u, v, pred_row):
    """Reconstruct a shortest u->v path as a list of (position) edges from a scipy predecessor row
    (pred_row[x] = predecessor of x on the shortest path out of u). Replaces the Sage version that
    walked a predecessor dict-of-dicts; used inside shortest_path_cover in position space."""
    end, P = v, []
    while end != u:
        w = int(pred_row[end])
        P.append(_norm(end, w))
        end = w
    return P


def shortest_path_cover(G, general=True):
    # TODO: document the "general" flag, does it behave as I want it to?
    """Greedy shortest-path cover, an L(+1)-approximation for (graph) metric repair.

    Each pass computes shortest paths once; for every broken edge (direct weight > shortest distance)
    it covers the edge together with the edges of one shortest alternative path, deletes them all, and
    repeats until no broken edge remains. Faithful to the Sage greedy (it batches per pass: deletions
    use that pass's distances), but shortest paths run on a dense NumPy adjacency via
    ``scipy.sparse.csgraph`` (Floyd-Warshall) -- ~100x faster than the Sage Boost+graph-object loop.

    Tie-breaking note: when several shortest paths tie, scipy may pick a different one than Sage, so
    the resulting cover can differ from Sage in *which* edges (sizes match within a few percent, and
    the cover is always valid).
    """
    verts = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(verts)}
    n = len(verts)
    A = np.zeros((n, n))
    for u, v, w in sorted_edges(G, weight=True):
        A[idx[u], idx[v]] = A[idx[v], idx[u]] = float(w)
    S = set()
    while True:
        dist, pred = shortest_path(A, method="FW", directed=False, return_predecessors=True)
        ii, jj = np.where(np.triu(A, 1) > 0)            # current edges, in label-sorted order
        found = False
        to_delete = []
        for a, b in zip(ii.tolist(), jj.tolist()):
            if A[a, b] > dist[a, b]:                     # broken: a shorter path than the direct edge
                found = True
                P = find_shortest_path(a, b, pred[a])    # position edges of the alternative path
                if general:
                    S.add(_norm(verts[a], verts[b]))
                    to_delete.append((a, b))
                for i, j in P:
                    S.add(_norm(verts[i], verts[j]))
                    to_delete.append((i, j))
        if not found:
            return S
        for i, j in to_delete:                            # batch-delete after the pass
            A[i, j] = A[j, i] = 0.0


def left_edge_heuristic(G):
    """Complete G, run the Gilbert & Jain left-edge heuristic, then reduce to a cover of G."""
    return reduce_solution(Gilbert_Jain_IOMR(complete(G)), G)


def pivot_heuristic(G):
    """Complete G, run the MVD pivot algorithm, then reduce to a cover of G."""
    return reduce_solution(MVD_Pivot(complete(G)), G)
