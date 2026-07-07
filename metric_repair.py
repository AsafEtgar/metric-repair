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
import heapq
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


def broken_cycle_length_bound(G):
    """Upper bound on the length (#edges) of any BROKEN cycle of G, or None when no finite bound applies
    (a non-positive edge weight, or no edges). Capping simple-cycle enumeration at this length misses NO
    broken cycle, so the hitting-set ILP stays exact and the rounding covers stay valid, while the
    worst-case-exponential enumeration is bounded.

    Derivation. A broken cycle of length k has a heavy edge of weight w* strictly greater than the sum
    of its other k-1 edges. Each of those is >= w_min (the smallest edge weight in G), so

        (k - 1) * w_min  <  w*  <=  w_max     =>     k <= floor(w_max / w_min) + 1

    -- the DYNAMIC-RANGE bound, valid for any positive weights (integer or real). When every weight is a
    positive INTEGER the strict inequality between integers gives w* >= (k-1)*w_min + 1, tightening it to

        k <= floor((w_max - 1) / w_min) + 1       (which equals w_max when w_min == 1).

    So the bound is governed by the RATIO w_max / w_min: tight (small) when the weights span a narrow
    dynamic range, large (but still correct) when they don't -- in which case prefer the planned
    separation-oracle LP, which never enumerates cycles. A non-positive weight breaks the derivation, so
    the bound is withheld then. Integral-valued floats (e.g. 2.0) count as integers; any fractional
    weight uses the real-valued form.
    """
    wmin = wmax = None
    all_int = True
    for _, _, w in G.edges(data="weight"):
        if w is None or w <= 0:                  # a non-positive weight invalidates the derivation
            return None
        if float(w) != int(w):
            all_int = False
        wmin = w if wmin is None else min(wmin, w)
        wmax = w if wmax is None else max(wmax, w)
    if wmin is None:                             # no edges -> nothing to enumerate
        return None
    if all_int:
        return (int(wmax) - 1) // int(wmin) + 1
    return int(wmax // wmin) + 1


def broken_cycles(G, max_len=None):
    """Yield the broken simple cycles of G, each as a list of sorted (u, v) edge tuples.

    A cycle is 'broken' iff its longest edge exceeds the sum of the others (2*max > total) -- a
    violated polygon inequality. Enumerates simple cycles via networkx; pass max_len to cap the cycle
    length. When max_len is left None the cap defaults to broken_cycle_length_bound(G) (the
    dynamic-range bound floor(w_max / w_min) + 1, tightened for integer weights) -- a cap that provably
    loses no broken cycle, so the enumeration stays COMPLETE while running far faster. Enumeration is
    still worst-case exponential, so run this on the (sparse) ORIGINAL graph G, not its dense
    completion; the planned separation-oracle LP is the scalable alternative.
    """
    if max_len is None:
        max_len = broken_cycle_length_bound(G)      # provably complete; None -> full enumeration
    for cyc in nx.simple_cycles(G, length_bound=max_len):
        if len(cyc) < 3:
            continue
        edges = get_list_of_edges(cyc)
        ws = [G[u][v]["weight"] for (u, v) in edges]
        if 2 * max(ws) > sum(ws):
            yield edges


def broken_cycle_incidence(G, max_len=None, drop_max=False):
    """Return (B, n_cycles, D): the 0/1 incidence matrix of broken cycles (rows) against edges
    (columns) as a sparse CSR matrix, the number of broken cycles, and the edge encoding D: E -> [m].
    B[c, D[e]] = 1 iff edge e lies on broken cycle c. This is the constraint matrix of the hitting-set
    ILP (exact_metric_repair_ilp) and the covering LP (broken_cycle_rounding_heuristic) below.

    drop_max=True gives the INCREASE-ONLY (IOMR) formulation: the maximum-weight edge of each broken
    cycle is left OUT of that cycle's row, so the cover is forced to hit a LIGHT edge. This is exactly
    right for increase-only repair -- a broken cycle (2*max > total) can only be fixed by raising its
    light edges; raising the heavy edge makes it worse, so hitting a cycle at its heavy edge is useless.
    The maximum edge of a broken cycle is UNIQUE (two edges tied at max would give total >= 2*max,
    contradicting 2*max > total), so there is never any tie to break.
    """
    D = make_index_encoding(G)
    m = G.number_of_edges()
    rows, cols = [], []
    r = 0
    for edges in broken_cycles(G, max_len):
        if drop_max:
            emax = max(edges, key=lambda e: G[e[0]][e[1]]["weight"])
            row_edges = [e for e in edges if e != emax]        # IOMR: force a hit on a light edge
        else:
            row_edges = edges
        for e in row_edges:
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


def _l1_solve(Gc, cycle_matrix=induced_cycle_matrix, solver="highs-ipm", reweight=0, eps=1e-3,
              general=False, min_weight=1.0):
    """Solve the L1 weight-correction LP on Gc; return (x, D): the correction vector and edge encoding.

    Two formulations, selected by `general`:
      * general=False (default, increase-oriented): corrections x >= 0, so repaired weights w+x only
        INCREASE. Minimises sum_e x_e subject to the polygon inequality on w+x for every row of
        cycle_matrix(Gc). This is the historical formulation (matches the Sage equivalence reference).
      * general=True (GENERAL MR, decreases allowed): free-sign corrections x = p - n (p, n >= 0),
        minimising the true L1 norm sum_e|x_e| = sum_e(p_e + n_e). Weights may decrease as well as
        increase. The decrease is bounded so the repaired weight stays STRICTLY POSITIVE, never 0: with
        n_e <= max(0, w_e - min_weight) we get w + x >= min(min_weight, w_e) > 0. So a repaired distance
        is always a valid positive metric weight (min_weight=1 => integer weights repair to >= 1; an edge
        already below min_weight is simply frozen from below, never forced up). The polygon constraints
        are unchanged: phi(w+x) >= 0.

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
    if not general:
        c = np.ones(m)
        x = np.zeros(m)
        for _ in range(reweight + 1):
            x = linprog(c, A_ub=-phi, b_ub=phi @ w, method=solver).x
            c = 1.0 / (np.abs(x) + eps)        # reweight for the next pass
        return x, D
    # general MR: x = p - n,  minimise sum(p + n),  phi(w + p - n) >= 0,  0 <= p,  0 <= n <= w - min_w.
    A = sparse.hstack([-phi, phi]).tocsr()     # rows: -phi.p + phi.n <= phi.w  <=>  phi(w + p - n) >= 0
    b = phi @ w
    # n_e <= max(0, w_e - min_weight)  =>  w + x = w + p - n >= min(min_weight, w_e) > 0 (never 0).
    bounds = [(0, None)] * m + [(0, max(0.0, float(wi) - min_weight)) for wi in w]
    r = np.ones(m)
    x = np.zeros(m)
    for _ in range(reweight + 1):
        cost = np.concatenate([r, r])          # reweighted L1: cost on |x_e| applies to both p_e and n_e
        z = linprog(cost, A_ub=A, b_ub=b, bounds=bounds, method=solver).x
        x = z[:m] - z[m:]                       # signed correction p - n
        r = 1.0 / (np.abs(x) + eps)
    return x, D


def l1_minimization(Gc, solver="highs-ipm", reweight=0, general=False, min_weight=1.0):
    """L1 metric repair on an already-completed graph Gc.

    Minimises the L1 norm of the edge-weight corrections subject to all chordless-cycle (metric)
    constraints, and returns the support of the correction (the cover). general=False keeps the
    increase-oriented LP (x >= 0); general=True allows decreases (free-sign x) -- true general MR, with
    the repaired weight kept STRICTLY POSITIVE (w+x >= min(min_weight, w_e) > 0, never 0); see _l1_solve.
    Defaults to the interior-point solver ('highs-ipm'), marginally the sparsest single-solve choice
    measured here; set reweight > 0 (reweighted L1) for an even sparser support. (The LP OPTIMUM is
    solver-independent; only which support is returned changes. Support cutoff 1e-7 on |x|.)"""
    x, D = _l1_solve(Gc, induced_cycle_matrix, solver=solver, reweight=reweight, general=general,
                     min_weight=min_weight)
    return {(u, v) for u, v in sorted_edges(Gc) if abs(x[D[(u, v)]]) > 1e-7}


def l1_min_heuristic(G, solver="highs-ipm", reweight=0, general=False, min_weight=1.0):
    """L1 metric-repair heuristic: complete G, solve the L1 minimisation over all metric (chordless-
    cycle) constraints of the completion, and return the support restricted to E(G). general=True solves
    the general-MR LP (decreases allowed, repaired weights kept > 0 via min_weight); general=False
    (default) the increase-oriented one."""
    G_edges = set(sorted_edges(G))
    return {e for e in l1_minimization(complete(G), solver=solver, reweight=reweight, general=general,
                                       min_weight=min_weight)
            if e in G_edges}


# ----------------------------------------------------------------------------
# Exact ILP & LP-rounding heuristics (hitting set over the broken cycles of G)
#
# A set S is a valid general-MR cover iff it hits every broken cycle of G, so the exact minimum cover
# is a minimum hitting set (exact_metric_repair_ilp). The covering LP relaxation, randomized-rounded,
# is broken_cycle_rounding_heuristic. l1_rounding_heuristic instead rounds the weight-correction LP
# above. NOTE: exactness/validity needs ALL broken cycles enumerated. With max_len=None this holds:
# broken_cycles auto-caps at broken_cycle_length_bound(G) (the dynamic-range bound, provably complete
# for any positive weights), falling back to full enumeration only when even that bound is undefined.
# Passing an explicit max_len that is too small makes these heuristic (a longer broken cycle may go
# un-hit).
# ----------------------------------------------------------------------------

def exact_metric_repair_ilp(G, max_len=None, iomr=False):
    """Exact minimum metric-repair cover: the smallest edge set hitting every broken cycle of G. Solves
    the hitting-set ILP

        min  sum_e y_e   s.t.   sum_{e in C} y_e >= 1  for every broken cycle C,   y_e in {0, 1}

    with scipy.optimize.milp, returning the cover {e : y_e = 1}. This is the exact optimum the
    heuristics approximate. Enumerates broken cycles (worst-case exponential); with max_len=None this
    stays exact automatically (the enumeration auto-caps at broken_cycle_length_bound(G), which loses no
    broken cycle). Passing an explicit, too-small max_len makes the result a lower bound / possibly
    invalid for longer broken cycles.

    iomr=False (default) is general MR (hit any edge of each cycle). iomr=True is the exact INCREASE-ONLY
    (IOMR) optimum: each cycle's row omits its maximum-weight edge (drop_max), so the cover must hit a
    LIGHT edge -- the correct increase-only constraint (see broken_cycle_incidence). Validate with
    iomr_verifier(G, cover); general MR with verifier(G, cover)."""
    B, n_cyc, D = broken_cycle_incidence(G, max_len, drop_max=iomr)
    if n_cyc == 0:
        return set()
    m = B.shape[1]
    res = milp(c=np.ones(m), constraints=LinearConstraint(B, lb=1, ub=np.inf),
               integrality=np.ones(m), bounds=Bounds(0, 1))
    y = np.round(res.x).astype(int) #todo: why do we need to round for an ilp?
    inv = {i: e for e, i in D.items()}
    return {inv[i] for i in np.nonzero(y)[0]}


def l1_rounding_heuristic(G, rounds=20, scale=1.0, seed=None, solver="highs-ipm", general=False,
                          min_weight=1.0):
    """Randomized rounding of the L1 weight-correction LP into a (sparser) valid cover.

    Solve l1 on complete(G) to get corrections x, restrict to E(G), and over several rounds sample each
    support edge e independently with probability min(1, scale * |x_e| / max|x|). Keep the smallest
    SAMPLED set the verifier accepts; fall back to the full l1 support (always valid) if no sample is
    valid. So the result is never worse than l1_min_heuristic, and usually sparser.

    Rounding is SIGN-AGNOSTIC, which is what makes general=True work: the support is a COVER (which edges
    we may change), and a cover is the same object whether an edge is to be increased or decreased. We
    therefore sample on the magnitude |x_e| (the LP's fractional evidence that edge e matters), and the
    general verifier already lets a chosen cover edge move either way -- so once general=True only swaps
    the LP for the free-sign one, no other rounding logic changes.

    The acceptance check MATCHES the variant: general=True keeps sampled sets the general verifier accepts;
    general=False keeps only sets iomr_verifier accepts, so the result is a genuine increase-only cover
    (the increase-oriented x >= 0 LP already targets IOMR)."""
    Gc = complete(G)
    x, D = _l1_solve(Gc, induced_cycle_matrix, solver=solver, general=general, min_weight=min_weight)
    check = verifier if general else iomr_verifier       # validate against the matching MR variant
    support = [e for e in sorted_edges(G) if abs(x[D[e]]) > 1e-7]
    if not support:
        return set()
    xmax = max(abs(x[D[e]]) for e in support)
    p = {e: min(1.0, scale * abs(x[D[e]]) / xmax) for e in support}
    rng = np.random.default_rng(seed)
    best = set(support)                                  # full support: the deterministic fallback cover
    for _ in range(rounds):
        S = {e for e in support if rng.random() < p[e]}
        if S and len(S) < len(best) and check(G, S):
            best = S
    return best


def _l1_separation_phi(rows, m):
    """Build the polygon-inequality matrix phi (n_rows x m) from cutting-plane rows. Each row is
    (heavy_col, light_cols): -1 on the heavy (distinguished) edge, +1 on each light edge, so
    phi @ (w+x) >= 0 encodes w'_heavy <= sum_light w' -- the same sign convention as induced_cycle_matrix."""
    data, ri, ci = [], [], []
    for r, (hcol, lcols) in enumerate(rows):
        ri.append(r); ci.append(hcol); data.append(-1.0)
        for c in lcols:
            ri.append(r); ci.append(c); data.append(1.0)
    return sparse.csr_matrix((data, (ri, ci)), shape=(len(rows), m))


def l1_separation(G, general=False, complete_graph=False, solver="highs-ipm", reweight=0,
                  max_rounds=200, tol=1e-6, min_weight=1.0, verbose=False):
    """L1 weight-correction repair via CUTTING PLANES -- no chordless-cycle enumeration.

    The enumerated L1 (l1_minimization) materialises every polygon inequality of the completion up front
    (induced_cycle_matrix). This generates them on demand instead: solve the L1 LP over the current rows,
    form w' = w + x, run one all-pairs-shortest-path pass, and for every edge (u,v) whose shortest
    w'-detour undercuts it add that cycle's polygon inequality; stop when w' has no broken cycle. Minimises
    sum|x_e| (general=True, GMR -- repaired weights kept STRICTLY POSITIVE, w'>=min(min_weight, w_e)>0,
    never 0) or sum x_e with x >= 0 (general=False, increase-only/IOMR), and returns the support restricted
    to E(G) -- the cover.

    complete_graph=False (default) runs directly on G: no metric completion. On convergence w' has no
    broken cycle, and because x >= 0 forces some LIGHT edge of every broken cycle up (general=False) /
    some edge of every broken cycle to move (general=True), the support is a PROVABLY VALID cover of G
    (iomr_verifier / verifier accept it) -- no restrict-to-E(G) gamble. complete_graph=True reproduces the
    completion-based l1_minimization semantics via cutting planes (same optimum, no enumeration), but then
    inherits the restrict-to-E(G) heuristic step. Still a HEURISTIC overall (L1 is a surrogate for the
    sparsest cover); reweight>0 adds reweighted-L1 passes. On convergence the returned cover is valid; if
    max_rounds is hit first it may not be (check with the matching verifier)."""
    H = complete(G) if complete_graph else G
    D = make_index_encoding(H)
    m = H.number_of_edges()
    w = get_weights_vector(H, D)
    verts = sorted(H.nodes())
    idx = {v: i for i, v in enumerate(verts)}
    n = len(verts)
    G_edges = set(sorted_edges(G))
    rows, seen = [], set()
    x = np.zeros(m)
    for r in range(max_rounds):
        wprime = w + x
        A = np.zeros((n, n))
        for (u, v) in sorted_edges(H):                       # adjacency in the CURRENT weights w'
            a, b = idx[u], idx[v]
            A[a, b] = A[b, a] = wprime[D[(u, v)]]
        dist, pred = shortest_path(A, method="FW", directed=False, return_predecessors=True)
        new_rows = []
        for (u, v) in sorted_edges(H):
            a, b = idx[u], idx[v]
            if dist[a, b] < wprime[D[(u, v)]] - tol:         # (u,v) undercut by a w'-detour -> add polygon
                P = find_shortest_path(a, b, pred[a])
                lcols = frozenset(D[_norm(verts[i], verts[j])] for (i, j) in P)
                key = (D[(u, v)], lcols)
                if not lcols or key in seen:
                    continue
                seen.add(key)
                new_rows.append((D[(u, v)], lcols))
        if not new_rows:                                     # w' metric -> optimal, done
            break
        rows.extend(new_rows)
        phi = _l1_separation_phi(rows, m)
        b_ub = phi @ w
        if not general:                                      # x >= 0: increase-only (IOMR)
            c = np.ones(m)
            for _ in range(reweight + 1):
                x = linprog(c, A_ub=-phi, b_ub=b_ub, bounds=(0, None), method=solver).x
                c = 1.0 / (np.abs(x) + 1e-3)
        else:                                                # free-sign x = p - n: general MR
            Aub = sparse.hstack([-phi, phi]).tocsr()
            # n_e <= max(0, w_e - min_weight)  =>  w + x >= min(min_weight, w_e) > 0 (never 0)
            bounds = [(0, None)] * m + [(0, max(0.0, float(wi) - min_weight)) for wi in w]
            rw = np.ones(m)
            for _ in range(reweight + 1):
                z = linprog(np.concatenate([rw, rw]), A_ub=Aub, b_ub=b_ub, bounds=bounds,
                            method=solver).x
                x = z[:m] - z[m:]
                rw = 1.0 / (np.abs(x) + 1e-3)
        if verbose:
            print(f"[l1-sep] round {r}: {len(rows)} polygon rows, "
                  f"|support|={int((np.abs(x) > 1e-7).sum())}")
    return {e for e in sorted_edges(H) if e in G_edges and abs(x[D[e]]) > 1e-7}


def broken_cycle_rounding_heuristic(G, max_len=None, rounds=20, scale=None, seed=None, iomr=False):
    """Randomized rounding of the covering LP over ALL broken cycles of G into a valid cover.

    Solve the LP relaxation of the hitting set  min 1.y  s.t. B y >= 1, 0 <= y <= 1, then over several
    rounds sample each edge e with probability min(1, scale * y_e), unioning the picks until every
    broken cycle is hit; any cycle still un-hit after the rounds is covered greedily, so the returned
    set is always a valid cover over the enumerated cycles. With max_len=None that enumeration is
    complete (auto-capped at broken_cycle_length_bound(G)), so the cover is valid for G as a whole.
    scale defaults to ~ln(#cycles), the standard set-cover rounding factor -- so with scale=ln(#cycles)
    the rounding is the classic O(log)-approximation of the covering LP (the greedy top-up only makes the
    returned set no larger and still valid).

    iomr=False (default) is general MR. iomr=True is increase-only: each cycle's row omits its maximum
    edge (drop_max), so every sampled/greedy hit lands on a LIGHT edge -- a valid IOMR cover over the
    enumerated cycles (validate with iomr_verifier). (Thin wrapper: the enum+randomized corner of
    covering_lp_cover.)"""
    return covering_lp_cover(G, solve="enum", rounding="randomized", iomr=iomr, max_len=max_len,
                             rounds=rounds, scale=scale, seed=seed)[0]


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


# ----------------------------------------------------------------------------
# Separation-oracle LP / ILP  (cutting planes over broken cycles -- scales WITHOUT
# enumerating every cycle; the route to exact/baseline covers at large n. See OVERVIEW.md section 6.)
#
# Model:  min 1.y   s.t.   sum_{e in C} y_e >= 1   for every broken cycle C.
# Enumerating all C is intractable past ~n=100, so we start with NO constraints and add only VIOLATED
# cycles found by a separation oracle, re-solving until none are violated.
#
# Separation oracle (naive, shortest-path based): edge (u,v) lies on a broken cycle iff some u-v path
# is shorter (in original weights) than w_uv. One all-pairs-shortest-path call finds, for every edge,
# its 'canonical' broken cycle = (u,v) + the shortest detour.
#   * ILP (integral cover S): the cover edges are first heavied so detours must avoid them -- this is
#     exactly the verifier's feasibility certificate, so the oracle is SOUND AND COMPLETE and the
#     cutting-plane ILP returns the EXACT minimum cover.
#   * LP (fractional y): the same canonical cycles give a valid (if not tightest) lower bound.
# ----------------------------------------------------------------------------

def _apsp_positions(G, heavy=None, big=0.0):
    """All-pairs shortest paths (Floyd-Warshall, with predecessors) in position space. Edges in the set
    `heavy` (normalised label pairs) are reweighted to `big` so shortest detours avoid them. Returns
    (dist, pred, verts, idx) where verts is the position -> label map and idx its inverse."""
    verts = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(verts)}
    A = np.zeros((len(verts), len(verts)))
    for u, v, w in sorted_edges(G, weight=True):
        a, b = idx[u], idx[v]
        A[a, b] = A[b, a] = big if (heavy is not None and (u, v) in heavy) else float(w)
    dist, pred = shortest_path(A, method="FW", directed=False, return_predecessors=True)
    return dist, pred, verts, idx


def _violated_cuts(G, sol, D, integral, tol=1e-6, max_cuts=None, iomr=False):
    """Separation oracle. Return a list of violated broken-cycle constraints for the current solution
    `sol` (a length-m array indexed by D). Each cut is a frozenset of edge-index columns -- the support
    of one constraint sum_{e in cut} y_e >= 1.

    integral=True : treat sol as a 0/1 cover S; heavy the cover edges so detours avoid them, then every
                    edge undercut by such a detour yields a cycle S fails to hit. This is the verifier's
                    certificate -- sound AND complete, so a clean pass proves S optimal.
    integral=False: treat sol as fractional; report each edge's canonical (shortest-detour) cycle when
                    its y-sum is < 1. Sound (real violated constraints) but not complete.

    iomr=False (general MR): the cut is the whole broken cycle {detour} + {(u,v)}; a cover edge (u,v) in
        S is exempt (it can be re-weighted down, so it never needs a detour). iomr=True (increase-only):
        the cut OMITS the undercut edge (u,v) -- which is always the cycle's unique maximum, since its
        weight exceeds the detour's total -- forcing a hit on a LIGHT edge. Cover edges are then NOT
        exempt: increase-only cannot lower a heavy cover edge that a frozen detour undercuts, so such a
        cycle must still be hit at a light edge (this is exactly what iomr_verifier checks). Dropping the
        skip is what makes the integral oracle sound AND complete for IOMR, not just a column change.
    """
    if integral:
        S = {e for e in D if sol[D[e]] > 0.5}
        big = sum(w for _, _, w in sorted_edges(G, weight=True)) + 1.0
        dist, pred, verts, idx = _apsp_positions(G, heavy=S, big=big)
    else:
        S = None
        dist, pred, verts, idx = _apsp_positions(G)
    cuts, seen = [], set()
    for (u, v) in sorted_edges(G):
        if integral and not iomr and (u, v) in S:           # general MR: cover edges are re-weightable
            continue
        a, b = idx[u], idx[v]
        if dist[a, b] < G[u][v]["weight"] - tol:            # (u,v) undercut -> broken cycle through it
            cols = {D[(verts[i], verts[j])] for (i, j) in find_shortest_path(a, b, pred[a])}
            if not iomr:
                cols.add(D[(u, v)])                         # general MR: heavy edge may also be hit
            key = frozenset(cols)
            if not cols or key in seen:
                continue
            if integral or sum(sol[c] for c in cols) < 1.0 - tol:
                seen.add(key)
                cuts.append(key)
                if max_cuts and len(cuts) >= max_cuts:
                    break
    return cuts


def _cuts_to_matrix(rows, m):
    """Stack a list of cuts (each a set of columns) into a sparse 0/1 constraint matrix (n_cuts x m)."""
    data, ri, ci = [], [], []
    for k, cyc in enumerate(rows):
        for c in cyc:
            ri.append(k)
            ci.append(c)
            data.append(1.0)
    return sparse.csr_matrix((data, (ri, ci)), shape=(len(rows), m))


def _restricted_matrix(rows):
    """Constraint matrix over only the ACTIVE columns (edge-indices that appear in some cut), plus the
    sorted `active` list so a reduced solution can be scattered back to full length. The vast majority
    of edges lie on no broken cycle; keeping them as free zero-cost variables makes the LP/ILP ~100x
    slower for no benefit (measured 138s -> 1.6s at n=500), so we solve only over the active edges."""
    active = sorted(set().union(*rows))
    remap = {c: i for i, c in enumerate(active)}
    data, ri, ci = [], [], []
    for k, cyc in enumerate(rows):
        for c in cyc:
            ri.append(k)
            ci.append(remap[c])
            data.append(1.0)
    return sparse.csr_matrix((data, (ri, ci)), shape=(len(rows), len(active))), active


def _rsp_separation(G, yvec, D, tol=1e-6, max_cuts=None, iomr=False):
    """EXACT restricted-shortest-path separation for the fractional LP (tightest possible cuts).

    For each edge (u,v) it finds the minimum-y-cost u->v path whose ORIGINAL-weight length is < w_uv --
    so the cycle (u,v)+path is genuinely broken -- and reports that cycle if its hitting-set constraint
    is violated. Because each per-edge minimum is exact, a pass that finds nothing PROVES no broken cycle
    is violated, so the cutting-plane LP then equals the TRUE optimum over all broken cycles (the tightest
    lower bound) -- unlike the naive shortest-detour oracle (_violated_cuts).

    iomr=False (general MR): the cycle is violated iff y_uv + cost < 1 and the cut is {(u,v)} + path.
    iomr=True (increase-only): (u,v) is the cycle's unique maximum (its weight exceeds the whole path),
    so the cut OMITS it and the cycle is violated iff cost < 1 (the path's y-mass alone must reach 1).
    Iterating (u,v) over ALL edges enumerates every broken cycle exactly once by its unique heavy edge,
    so this stays sound AND complete for IOMR.

    This per-edge minimisation is the weight-constrained shortest path problem (Garey & Johnson ND30),
    weakly NP-hard. The weight-budget DP below is pseudo-polynomial in the max edge weight w_max -- which
    is small for integer-weight instances (it equals the broken-cycle length bound), so it is cheap here.
    cost[b][s, v] = min y-cost of an s->v path of total ORIGINAL weight EXACTLY b. Requires positive
    INTEGER weights; returns None otherwise so the caller can fall back to the naive oracle.
    """
    wmax = 0
    for _, _, w in sorted_edges(G, weight=True):
        if w < 1 or float(w) != int(w):
            return None
        wmax = max(wmax, int(w))
    verts = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(verts)}
    n = len(verts)
    B = wmax - 1                                      # path weight < w_uv <= w_max  =>  budget <= w_max-1
    if B < 1:                                         # all weights 1 -> no broken cycle is possible
        return []
    # Directed edges grouped by integer weight, so the DP relaxation at each budget is ONE vectorised
    # scatter-min per weight (over all sources at once), not a Python loop over edges -- the n=1000 win.
    grp = {}                                          # w -> [tails, heads, ycosts]
    for (a, b) in sorted_edges(G):
        ia, ib, w, yv = idx[a], idx[b], int(G[a][b]["weight"]), yvec[D[(a, b)]]
        grp.setdefault(w, ([], [], []))
        grp[w][0].extend((ia, ib))                    # tail of each directed copy
        grp[w][1].extend((ib, ia))                    # head of each directed copy
        grp[w][2].extend((yv, yv))
    weights = sorted(grp)
    EW = {w: (np.asarray(grp[w][0]), np.asarray(grp[w][1]), np.asarray(grp[w][2], float))
          for w in weights}
    cost = np.full((B + 1, n, n), np.inf)             # cost[b][s, v]: min y-cost s->v of total weight b
    np.fill_diagonal(cost[0], 0.0)                    # weight-0 path: each source reaches itself
    srcrow = np.arange(n)[:, None]
    for b in range(1, B + 1):
        cb = cost[b]
        for w in weights:
            if w > b:                                 # weights ascending -> the rest exceed b too
                break
            tails, heads, yc = EW[w]
            cand = cost[b - w][:, tails] + yc[None, :]               # (n_sources, n_edges_w)
            np.minimum.at(cb, (np.broadcast_to(srcrow, cand.shape),
                               np.broadcast_to(heads[None, :], cand.shape)), cand)
    # query each edge; reconstruct its detour from the cost table (walk back along the optimality eqn)
    cuts, seen = [], set()
    for (u, v) in sorted_edges(G):
        s, t = idx[u], idx[v]
        wuv, yuv = int(G[u][v]["weight"]), yvec[D[(u, v)]]
        bcap = min(wuv - 1, B)
        col = cost[0:bcap + 1, s, t]
        bstar = int(np.argmin(col))
        best = col[bstar]
        slack = best if iomr else yuv + best            # IOMR omits the heavy edge (u,v) from the sum
        if not np.isfinite(best) or slack >= 1.0 - tol:
            continue
        cols = set() if iomr else {D[(u, v)]}
        node, bb = t, bstar
        while node != s and bb > 0:
            lbl = verts[node]
            pick_e = pick_x = pick_w = None
            pick_res = np.inf
            for nb in G.neighbors(lbl):               # incoming edge that achieves cost[bb][s][node]
                w2 = int(G[lbl][nb]["weight"])
                if w2 > bb:
                    continue
                x = idx[nb]
                res = abs(cost[bb - w2, s, x] + yvec[D[_norm(lbl, nb)]] - cost[bb, s, node])
                if res < pick_res:
                    pick_res, pick_x, pick_e, pick_w = res, x, _norm(lbl, nb), w2
            if pick_x is None or pick_res > 1e-6:
                break
            cols.add(D[pick_e])
            bb -= pick_w
            node = pick_x
        key = frozenset(cols)
        if key not in seen:
            seen.add(key)
            cuts.append(key)
            if max_cuts and len(cuts) >= max_cuts:
                break
    return cuts


def metric_repair_lp_separation(G, max_rounds=200, tol=1e-6, solver="highs-ds", oracle="rsp",
                                iomr=False, verbose=False, return_rows=False, return_rounds=False):
    """Cutting-plane LP lower bound on the minimum metric-repair cover, via a separation oracle.

    Solves  min 1.y  s.t. (added broken-cycle constraints) y >= 1, 0 <= y <= 1, adding only violated
    cycles until the oracle finds none. Returns (lp_value, y, D, n_cuts); return_rows=True appends `rows`
    (the accumulated cut column-sets = the constraint matrix, for re-optimising over the same polytope) and
    return_rounds=True appends `n_rounds` (separation rounds run, the last of which found no new cut -- the
    same iteration-count convention as exact_metric_repair_ilp_separation). The optional returns append in
    that order, so the tuple is (lp_value, y, D, n_cuts[, rows][, n_rounds]). `lp_value` is a valid LOWER BOUND on the exact cover size
    (hence on every heuristic's cover size), and it never enumerates all cycles, so it scales to large n.
    Only the active edges (those in some cut) enter the LP, and the default dual-simplex solver returns a
    vertex -- so when the polytope is integral the returned y is integral and `lp_value` IS the exact
    optimum.

    iomr=False (default) is general MR. iomr=True is increase-only: each cut omits its cycle's maximum
    edge, so `lp_value` is a lower bound on the exact IOMR cover (see _rsp_separation / _violated_cuts).

    oracle:
      "rsp"   (default) -- EXACT weight-constrained shortest path separation (_rsp_separation). The
              resulting `lp_value` is the TRUE optimum over all broken cycles (tightest bound). Needs
              positive integer weights; automatically falls back to "naive" otherwise.
      "naive" -- separates only on canonical (shortest-detour) cycles (_violated_cuts). Faster per round
              but the bound may be loose (it converges to the canonical-cycle LP).
    """
    D = make_index_encoding(G)
    m = G.number_of_edges()
    rows, seen = [], set()
    y = np.zeros(m)
    val = 0.0
    use_rsp = (oracle == "rsp")
    r = -1                                               # so n_rounds = 0 if the loop never runs (max_rounds=0)
    for r in range(max_rounds):
        cuts = _rsp_separation(G, y, D, tol=tol, iomr=iomr) if use_rsp else None
        if cuts is None:                                 # non-integer weights -> fall back to naive
            use_rsp = False
            cuts = _violated_cuts(G, y, D, integral=False, tol=tol, iomr=iomr)
        cuts = [c for c in cuts if c not in seen]
        if not cuts:
            break
        for c in cuts:
            seen.add(c)
            rows.append(c)
        Ba, active = _restricted_matrix(rows)            # solve only over active edges (huge speedup)
        ma = len(active)
        res = linprog(np.ones(ma), A_ub=-Ba, b_ub=-np.ones(len(rows)), bounds=(0, 1), method=solver)
        y = np.zeros(m)
        y[active] = res.x
        val = res.fun
        if verbose:
            print(f"[lp-sep] round {r}: {len(rows)} cuts, {ma} active vars, lp={val:.4f}  "
                  f"({'rsp' if use_rsp else 'naive'})")
    out = [val, y, D, len(rows)]
    if return_rows:
        out.append(rows)
    if return_rounds:
        out.append(r + 1)                                # rounds run (incl. the final no-new-cut check)
    return tuple(out)


def exact_metric_repair_ilp_separation(G, max_rounds=500, time_limit=None, tol=1e-6, iomr=False,
                                       verbose=False):
    """EXACT minimum metric-repair cover via a cutting-plane ILP with lazily-added broken-cycle constraints.

    Repeats: solve the hitting-set ILP over the current cuts -> cover S; run the (exact, verifier-based)
    separation oracle for broken cycles S misses; add them; re-solve. When the oracle finds none, S hits
    EVERY broken cycle and is the exact minimum (a relaxation feasible at its own optimum is optimal). It
    never enumerates all cycles, so memory stays small -- but minimum hitting set is NP-hard, so worst
    case is exponential time. Bound it with max_rounds and/or time_limit (seconds per ILP solve).

    iomr=False (default) is general MR; validate with verifier(G, cover). iomr=True is the exact
    INCREASE-ONLY optimum -- each cut omits its cycle's maximum edge AND cover edges are no longer exempt
    from separation (a heavy cover edge still needs a light-edge hit); validate with iomr_verifier(G,
    cover). Convergence for IOMR means iomr_verifier(G, cover) == 1.

    Returns (cover, info). If info['converged'] is True the cover is the proven exact optimum; if False
    (budget hit) the ILP objective is still a valid LOWER BOUND on the optimum, but the returned cover
    may not be valid -- check with the matching verifier."""
    D = make_index_encoding(G)
    m = G.number_of_edges()
    inv = {i: e for e, i in D.items()}
    rows, seen = [], set()
    S = set()
    sol = np.zeros(m)
    converged = False
    r = 0
    for r in range(max_rounds):
        cuts = [c for c in _violated_cuts(G, sol, D, integral=True, tol=tol, iomr=iomr) if c not in seen]
        if not cuts:
            converged = True
            break
        for c in cuts:
            seen.add(c)
            rows.append(c)
        Ba, active = _restricted_matrix(rows)               # solve only over active edges (huge speedup)
        opts = {"time_limit": time_limit} if time_limit else {}
        res = milp(c=np.ones(len(active)), constraints=LinearConstraint(Ba, lb=1, ub=np.inf),
                   integrality=np.ones(len(active)), bounds=Bounds(0, 1), options=opts)
        if res.x is None:                                   # solver gave up (time limit, no incumbent)
            break
        sol = np.zeros(m)
        sol[active] = np.round(res.x)
        S = {inv[i] for i in np.nonzero(sol > 0.5)[0]}
        if verbose:
            print(f"[ilp-sep] round {r}: {len(rows)} cuts, {len(active)} vars, |S|={len(S)}")
    return S, {"rounds": r + 1, "constraints": len(rows), "converged": converged, "size": len(S)}


def _positive_integer_weights(G):
    """True iff every edge weight is a positive integer (the regime where the RSP oracle actually runs
    and the covering LP is separated over ALL broken cycles)."""
    for _, _, w in sorted_edges(G, weight=True):
        if w < 1 or float(w) != int(w):
            return False
    return True


def threshold_rounding_cover(G, oracle="rsp", iomr=False, tol=1e-6, verbose=False):
    """Deterministic 1/L threshold rounding of the separation-LP solution into a valid cover.

    Solve the covering LP with metric_repair_lp_separation (fractional optimum y*), then round UP every
    edge with y*_e >= 1/f, where f is the max number of edges in any constraint row: f = L (the
    broken-cycle length bound) for general MR, f = L-1 for IOMR (whose rows drop each cycle's max edge).
    Every broken cycle has <= f constrained edges and sum_{e in row} y*_e >= 1, so some edge per row
    clears 1/f; the rounded set therefore hits every cycle, with

        |S| <= f * (LP optimum) <= f * OPT      -- a deterministic f-approximation (L, or L-1 for IOMR).

    The ratio needs y* feasible for ALL broken cycles, which holds only under the EXACT oracle
    ("rsp", and only for positive integer weights -- otherwise metric_repair_lp_separation falls back to
    naive internally). With oracle="naive" the LP is separated on canonical cycles only, so y* may
    violate a non-canonical broken cycle and the threshold set can miss it; a verifier-based greedy
    top-up then restores validity. The returned cover is ALWAYS valid (checked by the integral oracle);
    info['guaranteed'] says whether the f-approximation bound provably holds for this run.

    Returns (cover, info) with info: lp_value, f, threshold, rounded (size before top-up), added (edges
    the top-up appended), size, guaranteed. (Thin wrapper: the separation+deterministic corner of
    covering_lp_cover.)
    """
    S, info = covering_lp_cover(G, solve="separation", rounding="deterministic", iomr=iomr,
                                oracle=oracle, tol=tol, verbose=verbose)
    info["threshold"] = (1.0 / info["f"]) if info["f"] else 0.5
    return S, info


# ----------------------------------------------------------------------------
# Unified covering-LP cover: the 2x2 of  solve (enum | separation)  x  rounding (randomized | det).
# broken_cycle_rounding_heuristic and threshold_rounding_cover are two corners of this grid; the engine
# below implements all four behind one entry point and they now delegate to it.
# ----------------------------------------------------------------------------

def _covering_lp_fractional(G, solve, iomr, oracle, max_len, tol, lp_solver, verbose):
    """Solve the covering LP  min 1.y  s.t.  B y >= 1, 0<=y<=1  (broken-cycle hitting set, drop_max=iomr)
    and return (y, D, check, n_constraints, Bmat):
      * y     -- the fractional optimum (length m, indexed by D),
      * check -- check(sol) -> list of violated cuts (column frozensets) for a candidate 0/1 array sol,
                 empty iff sol is a valid cover,
      * n_constraints -- number of constraints known up front (the cycle count for "enum", else None),
      * Bmat  -- the (n_constraints x m) constraint matrix, for re-optimising over the same polytope
                 (multi-vertex rounding). For "separation" it is the accumulated cut matrix.
    solve="enum" builds the full length-bounded matrix; "separation" runs metric_repair_lp_separation and
    uses the exact integral oracle for `check`."""
    m = G.number_of_edges()
    if solve == "separation":
        _, y, D, _, rows, nrounds = metric_repair_lp_separation(
            G, oracle=oracle, iomr=iomr, tol=tol, solver=lp_solver, verbose=verbose,
            return_rows=True, return_rounds=True)
        Bmat = _cuts_to_matrix(rows, m) if rows else sparse.csr_matrix((0, m))

        def check(sol):
            return _violated_cuts(G, sol, D, integral=True, tol=tol, iomr=iomr)
        return y, D, check, None, Bmat, nrounds
    # enumeration: no separation loop, so n_rounds is undefined (None)
    B, n_cyc, D = broken_cycle_incidence(G, max_len, drop_max=iomr)
    if n_cyc == 0:
        return np.zeros(m), D, (lambda sol: []), 0, sparse.csr_matrix((0, m)), None
    y = linprog(np.ones(m), A_ub=-B, b_ub=-np.ones(n_cyc), bounds=(0, 1), method=lp_solver).x
    Bcsr = B.tocsr()

    def check(sol):
        hit = np.asarray(Bcsr @ np.asarray(sol)).ravel()
        return [frozenset(Bcsr.getrow(c).indices.tolist()) for c in np.nonzero(hit < 0.5)[0]]
    return y, D, check, n_cyc, Bcsr, None


def _optimal_face_vertices(Bmat, m, opt, k, mode, seed, lp_solver, tol, y0):
    """Return distinct optimal vertices of the covering LP's OPTIMAL FACE
    {y : 1.y <= opt+tol, Bmat y >= 1, 0<=y<=1}, each as a length-m array, found by minimising a secondary
    objective over that face (tie-breaking within the optimum).

    mode="reweight": iterated reweighted-L1 (cost_e <- 1/(|y_e|+eps)) seeded from y0 -- drives toward the
                     SPARSEST-support optimal vertex (fewest nonzeros -> usually the smallest threshold
                     rounding). mode="random": signed Gaussian directions for diversity. mode="both" runs
                     k of each and pools them (empirically neither dominates, so this wins most often).
    Only ACTIVE columns (those in some constraint) are optimised; the rest stay 0 (an inactive edge on no
    cycle must not be pulled in by a negative random cost). Returns [] if the face is a single point."""
    active = np.unique(Bmat.nonzero()[1])
    if active.size == 0:
        return []
    Br = Bmat[:, active].tocsr()
    ones_row = sparse.csr_matrix(np.ones((1, active.size)))
    A_ub = sparse.vstack([-Br, ones_row]).tocsr()                   # Br y>=1  and  1.y<=opt+tol
    b_ub = np.concatenate([-np.ones(Br.shape[0]), [opt + tol]])
    seen, verts = set(), []
    for md in (["reweight", "random"] if mode == "both" else [mode]):
        rng = np.random.default_rng(seed)
        c = 1.0 / (np.abs(y0[active]) + 1e-3) if md == "reweight" else rng.standard_normal(active.size)
        for _ in range(k):
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), method=lp_solver)
            if res.x is None:
                c = rng.standard_normal(active.size)     # solve failed; try another direction
                continue
            yv = np.zeros(m)
            yv[active] = res.x
            key = tuple(np.round(res.x, 6))
            if key not in seen:
                seen.add(key)
                verts.append(yv)
            c = 1.0 / (np.abs(res.x) + 1e-3) if md == "reweight" else rng.standard_normal(active.size)
    return verts


def _topup(cols, check, yv, m):
    """Oracle-driven feasibility top-up: while `check` reports a missed cut, add its most-fractional edge
    (by yv). Returns (cols, added). Guarantees the returned cols is a valid cover."""
    added = 0
    for _ in range(m + 1):
        sol = np.zeros(m)
        if cols:
            sol[list(cols)] = 1.0
        cuts = check(sol)
        if not cuts:
            break
        for cut in cuts:
            cols.add(max(cut, key=lambda c: yv[c]))
            added += 1
    return cols, added


def _threshold_round_topup(yv, f, check, m, tol):
    """Round UP every edge with yv_e >= 1/f, then top-up to a valid cover. Returns (cols, added)."""
    thr = (1.0 / f) if f else 0.5
    cols = set(np.nonzero(yv >= thr - tol)[0].tolist())
    return _topup(cols, check, yv, m)


def _heavy_pairs(G):
    """H: heavy edges (u,v) with w0(u,v) > shortest-path distance (a strictly shorter detour exists) --
    the terminal endpoints region growing must separate. (Exactly the DOMR broken edges.)"""
    apsp = all_pairs_distances(G)
    return [(u, v) for u, v, w in sorted_edges(G, weight=True) if float(w) > apsp[u][v] + 1e-9]


def _region_growing_multicut(G, xvec, D, pairs, tol=1e-6):
    """Garg-Vazirani-Yannakakis region-growing multicut, in G, over edge lengths x = xvec (indexed by D)
    with unit edge costs. For each still-connected terminal pair (s,t): grow a ball around s under the
    x-metric with the DIRECT edge (s,t) excluded (so we separate the DETOURS, not the heavy edge itself),
    pick a radius < 1/2 meeting the GVY volume bound (cut(ball) <= 2 ln(k+1) * vol(ball), seed vol =
    Fvol/k), cut its boundary, delete the ball, and continue. Returns (cut_cols, min_detour_dist,
    full_separation): the multicut as a set of edge-columns, the smallest shortest-x-detour over the
    pairs, and whether that minimum is >= 1 (the precondition). Under full_separation the total cut is
    O(log|pairs|) * sum(x) = O(log n) * OPTfrac; otherwise a pair may go unseparated (the caller's top-up
    then restores validity)."""
    k = max(1, len(pairs))
    Fvol = float(sum(xvec[D[e]] for e in sorted_edges(G)))
    seed = Fvol / k
    bnd = 2.0 * np.log(k + 1)
    edges = list(sorted_edges(G))
    alive = set(G.nodes())
    cut, min_dist = set(), np.inf
    INF = float("inf")
    for (s, t) in pairs:
        if s not in alive or t not in alive:
            continue
        dist = {s: 0.0}                                   # Dijkstra from s in `alive`, minus edge (s,t)
        pq = [(0.0, s)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, INF):
                continue
            for v in G.neighbors(u):
                if v not in alive or {u, v} == {s, t}:
                    continue
                nd = d + xvec[D[_norm(u, v)]]
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        min_dist = min(min_dist, dist.get(t, INF))        # shortest x-detour s->t
        levels = sorted({d for d in dist.values() if d < 0.5 - tol})
        picked_boundary, best_boundary, best_bcount = None, None, None
        for r in levels:
            ball = {u for u, du in dist.items() if du <= r + tol}
            if t in ball:                                 # ball would swallow t -> can't separate here
                break
            inside, bcount, boundary = 0.0, 0, set()
            for (a, b) in edges:
                if a not in alive or b not in alive or {a, b} == {s, t}:
                    continue
                ina, inb = a in ball, b in ball
                if ina and inb:
                    inside += xvec[D[(a, b)]]
                elif ina or inb:
                    bcount += 1
                    boundary.add(D[(a, b)])
            if best_bcount is None or bcount < best_bcount:
                best_bcount, best_boundary, best_ball = bcount, boundary, ball
            if bcount <= bnd * (seed + inside) + tol:     # GVY volume bound met -> use this radius
                picked_boundary, picked_ball = boundary, ball
                break
        if picked_boundary is None:                       # no bound-satisfying radius: min-boundary cut
            if best_boundary is None:
                continue
            picked_boundary, picked_ball = best_boundary, best_ball
        cut |= picked_boundary
        alive -= picked_ball                              # remove the ball; recurse on the rest
    full_sep = bool(min_dist >= 1.0 - tol)
    return cut, (0.0 if min_dist == INF else float(min_dist)), full_sep


def covering_lp_cover(G, solve="separation", rounding="deterministic", iomr=False, oracle="rsp",
                      max_len=None, rounds=20, scale=None, best_of_k=1, vertex_mode="both",
                      seed=None, tol=1e-6, solver=None, verbose=False):
    """Covering-LP metric-repair cover, exposing the two ORTHOGONAL choices as flags:

        solve    = "enum" | "separation"           -- how  min 1.y s.t. By>=1  is built/solved
        rounding = "randomized" | "deterministic"  -- how the fractional optimum y is rounded to a cover

    iomr=True hits each broken cycle at a LIGHT edge (drop_max). A shared oracle-driven top-up guarantees
    the returned cover is always valid. Returns (cover, info).

    solve:
      "enum"       -- enumerate all (length-bounded) broken cycles, solve once. Exact relaxation but the
                      enumeration ceiling is ~n=100.
      "separation" -- cutting planes (metric_repair_lp_separation), no enumeration -> scales to ~n=1000.
                      oracle="rsp" (exact, integer weights) makes y the true LP optimum over ALL cycles.
    rounding:
      "deterministic" -- round up every y_e >= 1/f (f = L, or L-1 for IOMR): a provable f-approximation,
                      |S| <= f*LP <= f*OPT (needs y feasible for all cycles -- always true for "enum";
                      for "separation" needs the exact rsp oracle). info['guaranteed'] reports this.
      "randomized"    -- sample edge e w.p. min(1, scale*y_e), union over `rounds`; O(log*OPT) in
                      expectation with scale~ln(#constraints). scale defaults to ln(#cycles) for "enum"
                      and to L*ln(n) (an ln(n^L) bound on #cycles) for "separation".
      "region_growing" -- (iomr only) Garg-Vazirani-Yannakakis region-growing multicut in G separating
                      every heavy pair's DETOURS (_region_growing_multicut). When the LP optimum certifies
                      FULL SEPARATION -- every pair's shortest y-detour >= 1 (info['full_separation'],
                      info['min_pair_dist']) -- this is a provable O(log|H|) = O(log n) approximation with
                      NO W factor, so info['guaranteed'] is True; otherwise a pair may go unseparated and
                      the top-up restores validity (no ratio). NB: the trivial "deterministic" f-rounding
                      already gives O(W) unconditionally, which dominates unless W is large AND full
                      separation holds -- so region growing is the win exactly in that regime.

    best_of_k (deterministic only): when > 1, round several DISTINCT optimal vertices of the LP's optimal
    face (found by _optimal_face_vertices) and keep the smallest valid cover. Only helps when the optimum
    is non-unique (the IOMR-gap regime; the GMR LP is integral/usually unique) -- there it frequently
    recovers the exact optimum. vertex_mode picks the secondary objective that tie-breaks within the
    optimal face: "reweight" drives toward the sparsest-support vertex, "random" samples signed Gaussian
    directions, "both" (default) pools k of each and keeps the best (neither dominates in practice). Every
    candidate vertex is optimal, so the f-approximation still holds for the winner; on the separation LP an
    alternate restricted-optimal vertex can violate a not-yet-added cycle -- the top-up fixes validity but
    such a winner is then not marked guaranteed.

    info keys: solve, rounding, iomr, lp_value, f, scale, best_of_k, vertices_tried, rounded (size before
    top-up), added (top-up edges), size, guaranteed, rounds (separation-oracle rounds, None for solve="enum");
    region_growing adds full_separation, min_pair_dist."""
    lp_solver = solver or ("highs-ds" if solve == "separation" else "highs")
    y, D, check, ncon, Bmat, nrounds = _covering_lp_fractional(
        G, solve, iomr, oracle, max_len, tol, lp_solver, verbose)
    inv = {i: e for e, i in D.items()}
    m = len(inv)
    n = G.number_of_nodes()
    L = broken_cycle_length_bound(G)
    lp_value = float(np.sum(y))
    f = (max(1, (L - 1) if iomr else L)) if L is not None else None
    used_scale = None
    vertices_tried = 1

    if rounding == "deterministic":
        cols, added = _threshold_round_topup(y, f, check, m, tol)
        if best_of_k > 1:                                # round several optimal vertices, keep smallest
            verts = _optimal_face_vertices(Bmat, m, lp_value, best_of_k - 1, vertex_mode,
                                           seed, lp_solver, tol, y)
            vertices_tried = 1 + len(verts)
            for yv in verts:
                cols_v, added_v = _threshold_round_topup(yv, f, check, m, tol)
                if len(cols_v) < len(cols):
                    cols, added = cols_v, added_v
        rounded = len(cols) - added
    elif rounding == "randomized":
        if scale is None:
            base = float(np.log((ncon if ncon else n) + 1)) if solve == "enum" \
                else (L if L else 1) * float(np.log(n + 1))
            scale = max(1.0, base)
        used_scale = scale
        rng = np.random.default_rng(seed)
        p = np.minimum(1.0, scale * y)
        cols, sol = set(), np.zeros(m)
        for _ in range(rounds):
            picks = np.nonzero(rng.random(m) < p)[0]
            cols.update(picks.tolist())
            sol[picks] = 1.0
            if not check(sol):
                break
        rounded = len(cols)
        cols, added = _topup(cols, check, y, m)          # feasibility top-up (valid cover always)
    elif rounding == "region_growing":                   # GVY region-growing multicut (IOMR only)
        if not iomr:
            raise ValueError("rounding='region_growing' is defined for iomr=True (light-edge covers)")
        cut_cols, min_pair_dist, full_sep = _region_growing_multicut(G, y, D, _heavy_pairs(G), tol)
        cols = set(cut_cols)
        rounded = len(cols)
        cols, added = _topup(cols, check, y, m)          # net for any pair region growing left unseparated
    else:
        raise ValueError("rounding must be 'deterministic', 'randomized' or 'region_growing', "
                         f"got {rounding!r}")

    if rounding == "deterministic" and f is not None and added == 0:
        guaranteed = (solve == "enum") or (oracle == "rsp" and _positive_integer_weights(G))
    elif rounding == "region_growing":
        guaranteed = bool(full_sep and added == 0)       # O(log|H|) bound holds only under full separation
    else:
        guaranteed = False                               # randomized bound is whp/expected, not certified
    S = {inv[c] for c in cols}
    info = {"solve": solve, "rounding": rounding, "iomr": iomr, "lp_value": lp_value, "f": f,
            "scale": used_scale, "best_of_k": best_of_k, "vertices_tried": vertices_tried,
            "rounded": rounded, "added": added, "size": len(S), "guaranteed": guaranteed,
            "rounds": nrounds}                           # separation-oracle rounds (None for solve="enum")
    if rounding == "region_growing":
        info["full_separation"] = full_sep
        info["min_pair_dist"] = min_pair_dist
    if verbose:
        extra = (f" full_sep={full_sep} min_dist={min_pair_dist:.3f}"
                 if rounding == "region_growing" else "")
        print(f"[cov-lp] solve={solve} round={rounding} lp={lp_value:.3f} f={f} tried={vertices_tried} "
              f"rounded={rounded} added={added} size={len(S)} guaranteed={guaranteed}{extra}")
    return S, info
