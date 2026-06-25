# ============================================================================
# graph_models.sage
#
# Random weighted-graph generators for the metric-repair experiments, plus the
# seed_all() reproducibility helper. Self-contained: load it on its own, or via
# the Packages_and_Functions.ipynb loader together with metric_repair.sage.
# ============================================================================

import random
import numpy as np
from scipy.spatial import distance_matrix
from sage.graphs.spanning_tree import *          # filter_kruskal (get_mst)


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------

def seed_all(seed):
    """Seed every RNG the library draws from, so a run is reproducible from one number.

    The generators use three independent sources:
      - Sage's RNG       (graphs.RandomGNP)            -> set_random_seed
      - NumPy's RNG      (np.random.geometric/choice)  -> np.random.seed
      - Python's random  (random.randint)              -> random.seed
    For a cluster job array, pass a distinct seed per task (e.g. base_seed + task_id) and record it
    in the output so results can be regenerated exactly."""
    seed = int(seed)                  # in a Sage notebook a literal like 5 is a Sage Integer;
    set_random_seed(seed)             # numpy/random.seed reject that, so coerce to a plain int
    np.random.seed(seed)
    random.seed(seed)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _weighted_graph(edges, weights):
    """Build a weighted Sage Graph from an iterable of (u, v) edges and matching weights."""
    return Graph([(e[0], e[1], w) for e, w in zip(edges, weights)], weighted=True)


# ----------------------------------------------------------------------------
# Random weighted graphs
# ----------------------------------------------------------------------------

def random_weighted_graph(n, p, lower_weight=1, upper_weight=100):
    """G(n, p) with i.i.d. integer edge weights in [lower_weight, upper_weight] (ignores metrics)."""
    g = graphs.RandomGNP(n, p)
    edges = g.edges(sort=True)
    weights = [random.randint(lower_weight, upper_weight) for _ in range(len(edges))]
    return _weighted_graph(edges, weights)

def random_geometric_weighted_graph(n, p):
    """G(n, p) with i.i.d. Geometric(1 - p) edge weights."""
    g = graphs.RandomGNP(n, p)
    edges = g.edges(sort=True)
    weights = np.random.geometric(1 - p, len(edges))
    return _weighted_graph(edges, weights)

def random_exponential_weighted_graph(n, p):
    """G(n, p) with i.i.d. Exponential(log(1/p)) edge weights, resampled until >= 1."""
    g = graphs.RandomGNP(n, p)
    edges = g.edges(sort=True)
    weights = []
    for _ in range(len(edges)):
        w = 0
        while w < 1:
            w = np.random.exponential(np.log(1 / p))
        weights.append(w)
    return _weighted_graph(edges, weights)

def random_metric_graph(n, p):
    """Connected G(n, p) whose edge weights are Euclidean distances between random 5-d points."""
    g = graphs.RandomGNP(n, p)
    while not g.is_connected():               # can switch to biconnected if needed
        g = graphs.RandomGNP(n, p)
    vects = np.random.randint(0, high=15, size=(n, 5))   # one draw instead of n
    D = distance_matrix(vects, vects)
    edges = g.edges(sort=False)
    return _weighted_graph(edges, [D[e[0], e[1]] for e in edges])

def random_uniform_weighted_graph(n, p):
    """Complete graph with U(0, 1) edge weights, thresholded to keep edges with weight > 1 - p."""
    g = graphs.CompleteGraph(n)
    edges = g.edges(sort=True)
    weights = np.random.uniform(size=len(edges))
    kept = [(e[0], e[1], w) for e, w in zip(edges, weights) if w > 1 - p]
    return Graph(kept, weighted=True)


# ----------------------------------------------------------------------------
# Complete-graph generators
# ----------------------------------------------------------------------------

def uniform_complete_graph(n, L=0, U=1):
    """Complete graph with i.i.d. U(L, U) edge weights."""
    g = graphs.CompleteGraph(n)
    edges = g.edges(sort=True)
    weights = np.random.uniform(low=L, high=U, size=len(edges))
    return _weighted_graph(edges, weights)

def geometric_complete_graph(n, p):
    """Complete graph with i.i.d. Geometric(1 - p) - 1 edge weights."""
    K = graphs.CompleteGraph(n)
    for u, v in K.edges(labels=0, sort=1):
        K.set_edge_label(u, v, np.random.geometric(1 - p) - 1)
    return K


# ----------------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------------

def get_mst(G):
    """Minimum spanning tree of G (Kruskal)."""
    return G.subgraph(G.vertices(sort=False), filter_kruskal(G))
