"""build_inversions.py -- similarity-inversion variants for the real-data experiments (REAL_EXPERIMENTS.md §1.1).

`bct_coactivation`, `flycns_male`, `fish1_ten` derive a distance from a similarity/strength `s` via
`d = 1/s`. Non-metricity is NOT invariant to that transform, so each becomes a 4-way conversion study. This
recovers `s = 1/d` from the already-built `d=1/s` graph (IDENTICAL topology -- no need to re-parse the raw
sources) and re-emits the 3 other conversions:

    _lin : d = (max s − s) + eps     (strong → small; eps avoids a degenerate 0-distance on the max edge)
    _log : d = log(max s / s) + eps  (strong → small)
    _raw : d = s                     (strong → LARGE — inverted semantics; R1/R5 characterization only)

The base `d = 1/s` graph already exists and is unchanged. Standalone (does NOT import build_real_graphs,
whose module body runs the full raw build on import).

    sage -python build_inversions.py
"""
import math
import os
import sys

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import load_edgelist, save_edgelist          # noqa: E402

OUT = "data/processed"
BASES = ["bct_coactivation", "flycns_male", "fish1_ten"]
EPS = 1e-9

CONVERSIONS = {
    "_lin": lambda s, smax: (smax - s) + smax * EPS,
    "_log": lambda s, smax: math.log(smax / s) + EPS,
    "_raw": lambda s, smax: s,
}


def _nonmetric_frac(G):
    """Fraction of edges whose weight exceeds the shortest path between its endpoints (same test as
    build_real_graphs.graph_report; dense Dijkstra, fine for n <= ~1200 here)."""
    nodes = list(G.nodes()); idx = {u: i for i, u in enumerate(nodes)}; n = len(nodes)
    ii, jj, ww = [], [], []
    for u, v, w in G.edges(data="weight"):
        ii.append(idx[u]); jj.append(idx[v]); ww.append(float(w))
    ww = np.asarray(ww); m = len(ww)
    if m == 0:
        return n, 0, 0.0
    A = sp.csr_matrix((np.r_[ww, ww], (np.r_[ii, jj], np.r_[jj, ii])), shape=(n, n))
    D = shortest_path(A, method="D", directed=False)
    bad = sum(1 for a, b, w in zip(ii, jj, ww) if D[a, b] < w - max(1e-9, 1e-9 * w))
    return n, m, bad / m


def _emit(base, suffix, conv):
    G = load_edgelist(os.path.join(OUT, f"{base}.csv"))
    sim = {(u, v): 1.0 / d["weight"] for u, v, d in G.edges(data=True)}   # recover similarity s = 1/d
    smax = max(sim.values())
    H = nx.Graph()
    for (u, v), s in sim.items():
        H.add_edge(u, v, weight=conv(s, smax))
    save_edgelist(H, os.path.join(OUT, f"{base}{suffix}.csv"))
    n, m, frac = _nonmetric_frac(H)
    print(f"  [ok] {base + suffix:24s} n={n} m={m} nonmetric={100 * frac:.1f}%", flush=True)
    return frac


def main():
    for base in BASES:
        Gb = load_edgelist(os.path.join(OUT, f"{base}.csv"))
        _, _, fb = _nonmetric_frac(Gb)
        fr = {"inv(1/s)": fb}
        print(f"[base] {base}: nonmetric={100 * fb:.1f}%", flush=True)
        for suffix, conv in CONVERSIONS.items():
            fr[suffix.lstrip("_")] = _emit(base, suffix, conv)
        print(f"  -> {base}: " + "  ".join(f"{k}={v:.3f}" for k, v in fr.items()) + "\n")


if __name__ == "__main__":
    main()
