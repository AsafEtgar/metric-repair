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
from scipy.spatial import distance_matrix


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
    """G(n, p) with i.i.d. Geometric(1 - p) edge weights."""
    edges = _gnp_edges(n, p)
    weights = np.random.geometric(1 - p, len(edges))
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
# Complete-graph generators
# ----------------------------------------------------------------------------

def uniform_complete_graph(n, L=0, U=1):
    """Complete graph with i.i.d. U(L, U) edge weights."""
    edges = _complete_edges(n)
    weights = np.random.uniform(low=L, high=U, size=len(edges))
    return _weighted_graph(edges, weights)


def geometric_complete_graph(n, p):
    """Complete graph with i.i.d. Geometric(1 - p) - 1 edge weights."""
    edges = _complete_edges(n)
    weights = [np.random.geometric(1 - p) - 1 for _ in range(len(edges))]
    return _weighted_graph(edges, weights)


# ----------------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------------

def get_mst(G):
    """Minimum spanning tree of G (by edge weight)."""
    return nx.minimum_spanning_tree(G, weight="weight")
