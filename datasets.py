"""
datasets.py  --  load REAL data into the weighted-graph representation used everywhere else.

Every loader returns a ``networkx.Graph`` whose edges carry a numeric ``weight`` attribute
(``G[u][v]["weight"]``) -- exactly the representation produced by graph_models.py and consumed by
metric_repair.py / metric_extras.py. So once your data is loaded, every repair algorithm, the
verifier, and the experiment harness work on it unchanged.

TWO INPUT FORMATS are supported (see data/README.md and data/examples/ for samples):

  1. EDGE LIST  (``load_edgelist``)        -- a CSV with columns  u, v, weight  (one row per edge).
                                              Natural for sparse / partially-observed dissimilarities.
  2. DISTANCE MATRIX (``load_distance_matrix``) -- a square, symmetric dissimilarity matrix (CSV with
                                              an optional label row/column, or a .npy array). Natural
                                              for fully-observed pairwise dissimilarities; this is the
                                              canonical metric-repair input (the matrix may violate the
                                              triangle inequality -- that is exactly what we repair).

Design choices worth knowing:
  - Edges are canonicalised to (min, max) endpoints (``_norm``), matching the rest of the codebase.
  - The diagonal of a distance matrix is ignored (a vertex is at distance 0 from itself).
  - A NaN / missing entry means "this pair was not measured" -> no edge (an incomplete metric); the
    repair algorithms run on whatever graph results, and ``complete(G)`` fills missing pairs with
    shortest-path distances when a complete graph is required.
  - Weights are kept in whatever dtype the source provides (int stays int, float stays float). Note
    that the broken-cycle length bound (metric_repair.broken_cycle_length_bound) is tight only when
    weights are positive integers or span a small dynamic range; see that function's docstring.

Depends only on numpy / networkx / pandas (and graph_models for the shared ``_norm``).
"""

import os

import numpy as np
import networkx as nx
import pandas as pd

from graph_models import _norm          # shared edge canonicaliser (single source of truth)


# ----------------------------------------------------------------------------
# Core: build a graph from a dense dissimilarity matrix
# ----------------------------------------------------------------------------

def graph_from_matrix(M, labels=None, threshold=None, zero_is_missing=False, check_symmetry=True):
    """Build a weighted Graph from a square dissimilarity matrix M (n x n).

    For every pair i < j an edge (labels[i], labels[j]) with weight M[i, j] is added, unless:
      - the entry is NaN / infinite (treated as 'not measured' -> no edge), or
      - ``zero_is_missing`` and the entry is 0 (some datasets encode 'no edge' as 0), or
      - ``threshold`` is set and the entry exceeds it (keep only the near pairs -> a sparser graph).

    labels defaults to 0..n-1. With check_symmetry, asserts M is (numerically) symmetric -- a
    dissimilarity must be, and an asymmetric input usually signals a malformed file.
    """
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    if M.shape != (n, n):
        raise ValueError(f"distance matrix must be square, got {M.shape}")
    if check_symmetry and not np.allclose(M, M.T, equal_nan=True):
        raise ValueError("distance matrix is not symmetric")
    if labels is None:
        labels = list(range(n))
    G = nx.Graph()
    G.add_nodes_from(labels)
    for i in range(n):
        for j in range(i + 1, n):
            w = M[i, j]
            if not np.isfinite(w):
                continue
            if zero_is_missing and w == 0:
                continue
            if threshold is not None and w > threshold:
                continue
            u, v = _norm(labels[i], labels[j])
            G.add_edge(u, v, weight=w.item())
    return G


# ----------------------------------------------------------------------------
# File loaders
# ----------------------------------------------------------------------------

def load_distance_matrix(path, threshold=None, zero_is_missing=False, labeled=True):
    """Load a square dissimilarity matrix from ``path`` and return a weighted Graph.

    Supported files (by extension): ``.npy`` (a raw numpy array, no labels) or any text/CSV that
    pandas can read. For CSV, ``labeled=True`` (default) expects the first column AND first row to hold
    the vertex labels (a corner cell may be blank); ``labeled=False`` reads a bare numeric grid and
    uses 0..n-1. See graph_from_matrix for threshold / zero_is_missing semantics.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        M = np.load(path)
        labels = None
    elif labeled:
        df = pd.read_csv(path, index_col=0)
        labels = list(df.index)
        if list(df.columns) != [str(c) for c in labels] and len(df.columns) == len(labels):
            labels = list(df.columns)            # prefer column labels if they look like the names
        M = df.to_numpy(dtype=float)
    else:
        M = pd.read_csv(path, header=None).to_numpy(dtype=float)
        labels = None
    return graph_from_matrix(M, labels=labels, threshold=threshold, zero_is_missing=zero_is_missing)


def load_edgelist(path, u_col="u", v_col="v", weight_col="weight", sep=","):
    """Load an edge list (CSV with columns u, v, weight) and return a weighted Graph.

    Column names are configurable; if the file has no header pass integer-like names via u_col/v_col/
    weight_col after reading, or rename upstream. Duplicate edges keep the LAST weight seen (pandas/
    networkx semantics); endpoints are canonicalised with _norm.
    """
    df = pd.read_csv(path, sep=sep)
    missing = {u_col, v_col, weight_col} - set(df.columns)
    if missing:
        raise ValueError(f"edge list {path!r} is missing column(s) {sorted(missing)}; "
                         f"found {list(df.columns)}")
    G = nx.Graph()
    for u, v, w in zip(df[u_col], df[v_col], df[weight_col]):
        a, b = _norm(u, v)
        G.add_edge(a, b, weight=w.item() if hasattr(w, "item") else w)
    return G


# ----------------------------------------------------------------------------
# Saving (round-trips with load_edgelist)
# ----------------------------------------------------------------------------

def save_edgelist(G, path, u_col="u", v_col="v", weight_col="weight"):
    """Write G to ``path`` as a CSV edge list (columns u, v, weight), sorted for stable diffs."""
    rows = [(u, v, G[u][v]["weight"]) for u, v in sorted(_norm(a, b) for a, b in G.edges())]
    pd.DataFrame(rows, columns=[u_col, v_col, weight_col]).to_csv(path, index=False)


# ----------------------------------------------------------------------------
# Quick description (ties a loaded dataset to the repair tooling)
# ----------------------------------------------------------------------------

def describe(G, count_broken=True):
    """Return a small dict of dataset stats: #vertices, #edges, connected?, weight range, already a
    metric?, and (optionally) the number of broken cycles under the auto length bound.

    Imports from metric_repair / metric_extras are done lazily so loading data never requires scipy.
    """
    from metric_repair import broken_cycle_length_bound, broken_cycles
    from metric_extras import is_metric

    ws = [w for _, _, w in G.edges(data="weight")]
    info = {
        "n_vertices": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "connected": nx.is_connected(G) if G.number_of_nodes() else False,
        "weight_min": min(ws) if ws else None,
        "weight_max": max(ws) if ws else None,
        "cycle_length_bound": broken_cycle_length_bound(G),
        "is_metric": is_metric(G) if G.number_of_edges() else True,
    }
    if count_broken:
        info["n_broken_cycles"] = sum(1 for _ in broken_cycles(G))
    return info
