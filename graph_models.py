"""
graph_models.py  --  pure-Python (networkx) port of graph_models.sage

Random weighted-graph generators for the metric-repair experiments, plus the
seed_all() reproducibility helper.

REPRESENTATION
--------------
A weighted graph is a `networkx.Graph` whose edges carry a numeric ``weight``
attribute, i.e. ``G[u][v]["weight"]``. Exactly like the Sage version, a graph is
built *from its edge list*, so isolated vertices are dropped and vertex labels
need not be contiguous 0..n-1 (this matters for the matrix algorithms; see
metric_repair.py).

DIFFERENCE FROM SAGE (the one intentional behavioural change)
-------------------------------------------------------------
Graph *structure* comes from ``networkx.fast_gnp_random_graph`` instead of Sage's
``graphs.RandomGNP``. These draw from different RNG streams, so **the same seed
produces a different graph than the Sage version** -- by design. Everything
downstream of a given graph (completion, the repair algorithms, the verifier) is
faithful; see equivalence/ for the proof. Edge *weights* still use the same NumPy
distributions, so the statistical model is unchanged.

Self-contained: depends only on numpy / networkx / scipy.
"""

import random

import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix, cKDTree


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------

def seed_all(seed):
    """Seed every RNG the generators draw from, so a run is reproducible from one number.

    The pure-Python generators use two sources (the Sage version needed a third, Sage's own RNG):
      - NumPy's RNG   (np.random.geometric / exponential / uniform / randint)  -> np.random.seed
      - Python's random (random.randint, and networkx.fast_gnp_random_graph)   -> random.seed

    For a cluster job array, pass a distinct seed per task (e.g. base_seed + task_id) and record it
    in the output so results can be regenerated exactly.
    """
    # The `random` library IS needed: networkx.fast_gnp_random_graph (in _gnp_edges) and
    # random_weighted_graph's random.randint both draw from Python's global RNG, so seeding NumPy alone
    # would not make the graph *structure* reproducible.
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _norm(u, v):
    """Order the endpoints of an edge as (small, large)."""
    return (u, v) if u <= v else (v, u)


def _num(w):
    """Convert a NumPy scalar weight to a plain Python int/float (so graphs serialise cleanly)."""
    return w.item() if hasattr(w, "item") else w


def _weighted_graph(edges, weights):
    """Build a weighted networkx Graph from an iterable of (u, v) edges and matching weights.

    Mirrors the Sage ``_weighted_graph``: the graph is built from edges only, so any vertex with no
    incident edge is absent and the surviving labels can have gaps.
    """
    H = nx.Graph()
    for (u, v), w in zip(edges, weights):
        H.add_edge(int(u), int(v), weight=_num(w))
    return H


def _gnp_edges(n, p):
    """Sorted edge list of an Erdos-Renyi G(n, p) (networkx replacement for graphs.RandomGNP)."""
    g = nx.fast_gnp_random_graph(n, p)            # uses Python's `random` (seeded by seed_all)
    return sorted(_norm(u, v) for u, v in g.edges())


def _complete_edges(n):
    """Sorted edge list of the complete graph K_n."""
    return [(u, v) for u in range(n) for v in range(u + 1, n)]


# ----------------------------------------------------------------------------
# Random weighted graphs
# ----------------------------------------------------------------------------

def random_weighted_graph(n, p, lower_weight=1, upper_weight=100):
    """G(n, p) with i.i.d. integer edge weights in [lower_weight, upper_weight] (ignores metrics)."""
    edges = _gnp_edges(n, p)
    weights = [random.randint(lower_weight, upper_weight) for _ in range(len(edges))]
    return _weighted_graph(edges, weights)


def random_geometric_weighted_graph(n, p):
    """G(n, p) with i.i.d. Geometric(1 - p) edge weights.

    NOTE the coupling: the weight parameter is 1 - p, so as p -> 0 the weights collapse to 1 and the
    graph becomes metric (non-metricity requires alpha < 3/5 when p = n^-alpha). Use this for FIXED-p
    experiments; for a p-sweep use random_decoupled_geometric_weighted_graph (weights fixed)."""
    edges = _gnp_edges(n, p)
    weights = np.random.geometric(1 - p, len(edges))
    return _weighted_graph(edges, weights)


def random_decoupled_geometric_weighted_graph(n, p, weight_p=0.5):
    """G(n, p) with i.i.d. Geometric(1 - weight_p) edge weights -- the weight distribution is FIXED,
    DECOUPLED from the edge probability p. Unlike random_geometric_weighted_graph (whose weight parameter
    tracks p, so the graph turns metric as p -> 0), this keeps a fixed weight spread at every density, so
    it stays non-metric across a p = n^-alpha sweep. weight_p=0.5 matches the coupled model 'as if p=0.5'
    (mean weight 2, w_max ~ log|E|)."""
    edges = _gnp_edges(n, p)
    weights = np.random.geometric(1 - weight_p, len(edges))
    return _weighted_graph(edges, weights)


def random_exponential_weighted_graph(n, p):
    """G(n, p) with i.i.d. Exponential(log(1/p)) edge weights, resampled until >= 1."""
    edges = _gnp_edges(n, p)
    weights = []
    for _ in range(len(edges)):
        w = 0
        while w < 1:
            w = np.random.exponential(np.log(1 / p))
        weights.append(w)
    return _weighted_graph(edges, weights)


def random_metric_graph(n, p):
    """Connected G(n, p) whose edge weights are Euclidean distances between random 5-d points."""
    edges = _gnp_edges(n, p)
    H = _weighted_graph(edges, [0] * len(edges))
    while H.number_of_nodes() < n or not nx.is_connected(H):   # ensure connected on all n vertices
        edges = _gnp_edges(n, p)
        H = _weighted_graph(edges, [0] * len(edges))
    vects = np.random.randint(0, high=15, size=(n, 5))         # one draw instead of n
    D = distance_matrix(vects, vects)
    for u, v in list(H.edges()):
        H[u][v]["weight"] = _num(D[u, v])
    return H


def random_uniform_weighted_graph(n, p):
    """Complete graph with U(0, 1) edge weights, thresholded to keep edges with weight > 1 - p."""
    edges = _complete_edges(n)
    weights = np.random.uniform(size=len(edges))
    kept = [((u, v), w) for (u, v), w in zip(edges, weights) if w > 1 - p]
    return _weighted_graph([e for e, _ in kept], [w for _, w in kept])


# ----------------------------------------------------------------------------
# Random geometric graphs (metric base + a breaker to inject non-metricity)
# ----------------------------------------------------------------------------

def random_geometric_metric_graph(n, mode="radius", radius=0.25, k=10, dim=2, weight_scale=None):
    """Random Geometric Graph (RGG): n points i.i.d. Uniform[0,1]^dim; edge weights = Euclidean distance,
    so the graph is METRIC BY CONSTRUCTION -- true distances obey the triangle inequality, so no cycle is
    broken and OPT = 0 until you break it with break_metric_graph(). Unlike random_metric_graph (Euclidean
    weights on an *Erdos-Renyi* topology), here the TOPOLOGY itself comes from geometry (proximity). Each
    surviving node carries its coordinate in the 'pos' attribute (ground truth for geometry checks).

    mode
      'radius' -- edge iff Euclidean distance <= radius (classic RGG; radius sets density / incompleteness;
                  for 2D connectivity you want radius >~ sqrt(log n / (pi n))). Built with
                  networkx.random_geometric_graph (scipy cKDTree-accelerated) on our own seeded points.
      'knn'    -- symmetric k-nearest-neighbour graph (each point joined to its k closest, via a KDTree;
                  the two directions are unioned into an undirected edge). Not in networkx, so it's custom.
    weight_scale
      None  -- float Euclidean weights: EXACTLY metric, but the rsp / threshold-rsp methods (which need
               integer weights for the weight-budget DP) will SKIP, exactly as for the exponential model.
      int C -- integer weights round(C * distance), clamped to >= 1, so the FULL suite (incl. rsp) runs.
               Rounding is an *incoherent* per-edge perturbation: it can nudge a few near-degenerate triples
               just past the triangle inequality, leaving a tiny non-metric 'noise floor' (rare, size
               O(1/C)). Pick C large enough that this floor sits well below your planted break magnitude.
               (This is the knob that connects to point-jitter -- see break_metric_graph's notes.)
    """
    pts = np.random.uniform(0.0, 1.0, size=(n, dim))           # points, seeded by seed_all (np.random)
    if mode == "radius":
        # networkx does the radius neighbour query (scipy cKDTree under the hood); we hand it our own
        # seeded points, so the structure is reproducible and the weights come from the same coordinates.
        R = nx.random_geometric_graph(n, radius, pos={i: pts[i].tolist() for i in range(n)})
        edges = [_norm(u, v) for u, v in R.edges()]
    elif mode == "knn":
        kk = max(1, min(k, n - 1))
        _, idx = cKDTree(pts).query(pts, k=kk + 1)             # cols 1..k = k nearest (col 0 is self, d=0)
        es = set()
        for u in range(n):
            for v in idx[u, 1:]:
                es.add(_norm(u, int(v)))                        # symmetrise (union of both directions)
        edges = sorted(es)
    else:
        raise ValueError(f"mode must be 'radius' or 'knn', got {mode!r}")

    def _wt(u, v):
        d = float(np.linalg.norm(pts[u] - pts[v]))
        return max(1, int(round(weight_scale * d))) if weight_scale is not None else d

    H = _weighted_graph(edges, [_wt(u, v) for u, v in edges])
    nx.set_node_attributes(H, {i: pts[i] for i in H.nodes()}, name="pos")
    return H


def break_metric_graph(G, frac_q=0.1, direction="inflate", magnitude=2.0):
    """Turn a metric graph non-metric by reweighting a random fraction frac_q of its edges. Returns
    (H, corrupted): H is a copy with reweighted edges; corrupted is the set of reweighted (u, v) pairs --
    the GROUND-TRUTH edit set. 'pos' node attributes (if any) are preserved. Weight type is inherited:
    integer graphs stay integer (rounded, >= 1); float graphs stay float.

    direction
      'inflate' -- raise the weight ABOVE the edge's shortest detour (its best alternative path), so the
                   edge itself becomes the too-long / broken edge. A decrease-only (DOMR) edit restores it,
                   hence OPT <= #inflated; on near-disjoint edges OPT == #inflated -- a clean planted
                   optimum, and edit_recall against `corrupted` is meaningful.
      'deflate' -- lower the weight below the |a-b| gap of its most asymmetric 2-path, creating a SHORTCUT
                   that makes *other* edges look too long. The repaired set then differs from the corrupted
                   set (planted != OPT) -- the harder regime.
      'mixed'   -- inflate half the chosen edges, deflate the other half.
    magnitude   multiplicative factor past the break threshold (>1): inflate -> detour*magnitude,
                deflate -> gap/magnitude.

    Only edges that lie on a cycle can be broken; bridges (and, for deflate, edges with no usable 2-path)
    are skipped, so |corrupted| may be < round(frac_q*|E|). Non-metricity is judged by the standard
    broken-cycle test in metric_repair.py -- nothing here presumes a repair variant.

    RELATION TO POINT-JITTER (why float vs integer weights matters). Both this reweighting and point-jitter
    inject non-metricity, but at different layers. Reweighting edits weights DIRECTLY, so `corrupted` is
    exactly the perturbed set. Point-jitter (not implemented here) instead moves a POINT and recomputes
    distances for only a SUBSET of its incident edges -- a *coherent-but-partial* geometric perturbation,
    so several edges break together and the culprit is a vertex, not a single edge. The weight mode decides
    how cleanly either break stands out: float weights keep the RGG base EXACTLY metric, so 100% of the
    non-metricity is your planted break; integer weights add a rounding 'noise floor' that is itself a
    tiny incoherent jitter -- so with integer weights you must keep the scale C large enough that the floor
    is negligible next to the break, or the planted `corrupted` set gets contaminated by rounding artifacts.
    """
    H = G.copy()
    integer = all(isinstance(d["weight"], (int, np.integer)) for _, _, d in G.edges(data=True))
    edges = list(H.edges())
    m = int(round(frac_q * len(edges)))
    random.shuffle(edges)                                       # Python RNG (seeded by seed_all)
    plan = (["inflate"] * (m // 2) + ["deflate"] * (m - m // 2)) if direction == "mixed" else [direction] * m

    def _set(u, v, w):
        H[u][v]["weight"] = max(1, int(round(w))) if integer else float(w)

    corrupted = set()
    for (u, v), dirn in zip(edges, plan):
        w0 = G[u][v]["weight"]
        if dirn == "inflate":
            Gm = G.copy(); Gm.remove_edge(u, v)                # shortest detour = best path avoiding the edge
            try:
                detour = nx.shortest_path_length(Gm, u, v, weight="weight")
            except nx.NetworkXNoPath:
                continue                                        # bridge: not on any cycle -> can't be broken
            _set(u, v, max(detour * magnitude, w0 * 1.001 + 1))
            corrupted.add(_norm(u, v))
        elif dirn == "deflate":
            common = (set(H[u]) & set(H[v])) - {u, v}
            if not common:
                continue                                        # no 2-path -> nothing to shortcut
            gap = max(abs(G[u][c]["weight"] - G[c][v]["weight"]) for c in common)
            if gap <= (1 if integer else 1e-9):
                continue                                        # need room below the gap: integer >=1, float >0
                                                                # (float RGG weights are all <1 -- the old `<=1`
                                                                # guard silently skipped EVERY deflate edge)
            _set(u, v, gap / magnitude)
            corrupted.add(_norm(u, v))
        else:
            raise ValueError(f"direction must be 'inflate', 'deflate', or 'mixed', got {direction!r}")
    return H, corrupted


def jitter_points(G, n_jitter=3, jitter=0.25, subset_s=0.5):
    """Point-jitter break for RGG graphs (requires 'pos' node attributes, from
    random_geometric_metric_graph). Models sensor drift: a few points move, but only SOME of their
    incident edges get re-measured. For each vertex v in an INDEPENDENT SET of up to n_jitter vertices
    (degree >= 2, pairwise non-adjacent -- see the note below):

      1. displace it -- p1 = pos[v] + jitter * (random unit vector);
      2. pick a random PROPER subset (fraction subset_s) of v's incident edges and RESET each to the new
         distance dist(p1, pos[u]); leave the remaining incident edges at their original weight.

    The reset edges are mutually coherent (a consistent star from p1) but INCONSISTENT with the stale ones
    (still at the old position) -- that mismatch is what breaks the metric. Returns (H, corrupted,
    jittered): corrupted = the reset edges (ground-truth edit set; reverting them restores the original
    metric RGG, so verifier(H, corrupted) == 1); jittered = the moved vertices. 'pos' is left at the
    ORIGINAL coordinates -- the reference geometry the corrupted edges deviate from.

    Why a PROPER subset is essential: subset_s = 0 (nothing reset) or 1 (all incident edges reset = a
    coherent relocation) both leave v consistent with a single position -> still metric. Only a partial
    (0 < subset_s < 1) update breaks it, and only if `jitter` is large enough vs local edge lengths.

    Why an INDEPENDENT set: an edge between two jittered vertices would be reset from one endpoint's NEW
    position against the other's OLD position, so it could never be globally coherent -- s=1 would then
    still break. Restricting the jittered vertices to be pairwise non-adjacent means every jittered star
    reaches only un-jittered neighbours, so a full (s=1) update is a genuine coherent relocation (stays
    metric) and each corrupted edge is attributable to exactly one jittered vertex (clean localization).

    Contrast with break_metric_graph's inflate/deflate: here the corrupted edges are (a) CLUSTERED around
    the jittered vertices and (b) each independently too-long or too-short (depending on whether p1 moved v
    toward or away from u), so corrupted is generally NOT the DOMR set and #jittered_points != OPT -- the
    repair edits weights, not points. Integer graphs keep integer weights (rounded, >= 1).
    """
    if any("pos" not in G.nodes[v] for v in G):
        raise ValueError("jitter_points needs 'pos' node attributes (use random_geometric_metric_graph)")
    H = G.copy()
    integer = all(isinstance(d["weight"], (int, np.integer)) for _, _, d in G.edges(data=True))
    pos = {v: np.asarray(G.nodes[v]["pos"], dtype=float) for v in G}
    dim = len(next(iter(pos.values())))

    candidates = [v for v in H if H.degree(v) >= 2]
    random.shuffle(candidates)                                   # greedy independent set (pairwise non-adj)
    chosen, banned = [], set()
    for v in candidates:
        if len(chosen) >= n_jitter:
            break
        if v not in banned:
            chosen.append(v)
            banned.add(v)
            banned.update(H.neighbors(v))

    def _set(u, v, w):
        H[u][v]["weight"] = max(1, int(round(w))) if integer else float(w)

    corrupted = set()
    for v in chosen:
        step = np.random.normal(size=dim)
        step = step / (np.linalg.norm(step) or 1.0) * jitter    # random unit direction, length `jitter`
        p1 = pos[v] + step
        incident = list(H.neighbors(v))
        k = min(len(incident), max(0, int(round(subset_s * len(incident)))))    # 0 or deg => coherent (metric)
        for u in random.sample(incident, k):
            _set(u, v, float(np.linalg.norm(p1 - pos[u])))
            corrupted.add(_norm(u, v))
    return H, corrupted, chosen


# ----------------------------------------------------------------------------
# Complete-graph generators
# ----------------------------------------------------------------------------

def uniform_complete_graph(n, L=0, U=1):
    """Complete graph with i.i.d. U(L, U) edge weights."""
    edges = _complete_edges(n)
    weights = np.random.uniform(low=L, high=U, size=len(edges))
    return _weighted_graph(edges, weights)


def geometric_complete_graph(n, p):
    """Complete graph with i.i.d. Geometric(1 - p) edge weights (support {1, 2, 3, ...}).

    DELIBERATE DEVIATION from the Sage original, which drew Geometric(1 - p) - 1 (support {0, 1, ...})
    and could therefore produce weight-0 edges. Weights are now >= 1 so that (a) the graph carries a
    genuine positive metric and (b) the broken-cycle length bound applies (a 0-weight edge lengthens a
    cycle without adding to the competing sum; see broken_cycle_length_bound in metric_repair.py)."""
    edges = _complete_edges(n)
    weights = [np.random.geometric(1 - p) for _ in range(len(edges))]
    return _weighted_graph(edges, weights)


# ----------------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------------

def get_mst(G):
    """Minimum spanning tree of G (by edge weight)."""
    return nx.minimum_spanning_tree(G, weight="weight")
