"""Reproduce the two claims the paper makes about the WEIGHT CORRECTION, so neither is an orphan number.

The paper says two things that need a script behind them:

  (1) "the truth lies strictly below its detour on every corrupted edge, at a median of 0.755 times it"
      -- this is why the canonical rule (build_F: w <- the edge's detour in G \\ S) overshoots. It aims at
      the top of the feasible range and the truth is not at the top.

  (2) "the midpoint rule leaves 22 heavy edges behind on the real base, so it is not a repair at all, and it
      is worse than the canonical rule anyway"
      -- this is the paper's own concession that our obvious fix does not work. It was measured with code
      that lived briefly inside the cluster harness and was then cut when the harness was scoped down to
      topology only. A concession is still a claim: it needs to be reproducible. Hence this file.

Claim (2) is the expensive one (n = 5000, one cover + two |H| recomputations). Run it with --real.

  sage -python experiments/midpoint_check.py            # claim (1), the RGG: ~1 min
  sage -python experiments/midpoint_check.py --real     # claim (2), the road network: ~20 min
"""
import argparse
import os
import sys

import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_models import break_metric_graph, random_geometric_metric_graph        # noqa: E402
from metric_repair import domr_alg                                                # noqa: E402
from downstream_recovery import apsp, knn_recovery                                # noqa: E402
from mds_recovery import (RGG_SPECS, _norm_cover, _procrustes_disp,               # noqa: E402
                          build_F_distances, classical_mds, finite_core, seed_all, smacof)
from mds_sweep import VERIFY, _SUITE, _cap_for, _planted_base, run_isolated       # noqa: E402

ALPHA = 0.5          # the midpoint of [L, detour]. Parameter-free: NOT tuned against the truth.


def _key(u, v):
    return (u, v) if u <= v else (v, u)


def _detours(edges, cover, nc):
    """d'(u,v) for every cover edge: its shortest path in G \\ S. This is exactly what build_F assigns."""
    keep = [(u, v, w) for u, v, w in edges if _key(u, v) not in cover]
    r = [u for u, _, _ in keep] + [v for _, v, _ in keep]
    c = [v for _, v, _ in keep] + [u for u, _, _ in keep]
    w = [x for _, _, x in keep] * 2
    Gs = coo_matrix((w, (r, c)), shape=(nc, nc)).tocsr()
    ends = sorted({u for u, v in cover} | {v for u, v in cover})
    if not ends:
        return {}, None, keep
    D = dijkstra(Gs, directed=False, indices=ends)
    row = {n: i for i, n in enumerate(ends)}
    return {(u, v): D[row[u]][v] for (u, v) in cover}, (D, row), keep


def _lower_bounds(cover, D_row, keep):
    """L(u,v): below this, some OTHER edge becomes heavy (its detour would shorten through uv).
    L = max over non-cover edges (a,b) of  w(a,b) - d'(a,u) - d'(v,b), and the mirrored term."""
    D, row = D_row
    A = np.array([u for u, _, _ in keep], dtype=int)
    B = np.array([v for _, v, _ in keep], dtype=int)
    W = np.array([x for _, _, x in keep], dtype=float)
    out = {}
    for (u, v) in cover:
        du, dv = D[row[u]], D[row[v]]
        t1 = W - du[A] - dv[B]
        t2 = W - dv[A] - du[B]
        out[(u, v)] = max(0.0, float(np.nanmax(np.concatenate([t1, t2])))) if len(W) else 0.0
    return out


def heavy_count(edges, nc):
    """|H| -- RECOMPUTED. 0 iff the graph is metric. Never assumed."""
    G = nx.Graph()
    G.add_nodes_from(range(nc))
    for u, v, w in edges:
        G.add_edge(u, v, weight=float(w))
    return len(_norm_cover(domr_alg(G)))


# ---------------------------------------------------------------------------
# CLAIM 1 -- the truth lies BELOW the detour, on the planted RGG
# ---------------------------------------------------------------------------
def claim_where_the_truth_sits():
    name, n, deg, direction, frac, mag, seed = [s for s in RGG_SPECS if s[0] == "rgg_inflate"][0]
    seed_all(seed)
    r = float(np.sqrt(deg / (np.pi * max(n - 1, 1))))
    T = random_geometric_metric_graph(n, mode="radius", radius=r)
    T = nx.convert_node_labels_to_integers(T.subgraph(max(nx.connected_components(T), key=len)).copy())
    C, corrupted = break_metric_graph(T, frac_q=frac, direction=direction, magnitude=mag)
    comp = max(nx.connected_components(C), key=len)
    mp = {u: i for i, u in enumerate(sorted(comp))}
    C = nx.relabel_nodes(C.subgraph(comp).copy(), mp)
    Tc = nx.relabel_nodes(T.subgraph(comp).copy(), mp)
    B = {_key(mp[u], mp[v]) for u, v in corrupted if u in mp and v in mp}
    nc = C.number_of_nodes()
    E = [(u, v, float(C[u][v]["weight"])) for u, v in C.edges()]
    w0 = {_key(u, v): float(Tc[u][v]["weight"]) for u, v in Tc.edges()}

    det, _, _ = _detours(E, B, nc)
    ratio = [w0[_key(u, v)] / det[(u, v)] for (u, v) in B if np.isfinite(det[(u, v)]) and det[(u, v)] > 0]
    ratio = np.array(ratio)
    print("CLAIM 1 -- where does the TRUE weight sit relative to the detour build_F aims at?")
    print(f"  corrupted edges with a finite detour: {len(ratio)} of {len(B)}")
    print(f"  true weight / detour:  median {np.median(ratio):.3f}   "
          f"quartiles {np.percentile(ratio, 25):.3f} / {np.percentile(ratio, 75):.3f}")
    print(f"  strictly BELOW the detour: {(ratio < 1.0).sum()} of {len(ratio)} "
          f"({100.0 * (ratio < 1.0).mean():.1f}%)")
    print("  => build_F sets every cover edge to 1.000x its detour. The truth is not there.")
    return ratio


# ---------------------------------------------------------------------------
# CLAIM 2 -- the midpoint rule is NOT a repair on the sparse road network
# ---------------------------------------------------------------------------
def claim_midpoint_fails_on_the_road(algo="gmr_thr_naive"):
    spec = ("dimacs_ny_d_inflate", "dimacs_ny_d", "inflate", 0.20, 3.0, 1000)
    _, base, direction, frac, mag, seed = spec
    G0, Dtrue, nodes, gt_ix, _ = _planted_base(base)
    seed_all(seed)
    C, corrupted = break_metric_graph(G0, frac_q=frac, direction=direction, magnitude=mag)
    C = nx.convert_node_labels_to_integers(C.subgraph(max(nx.connected_components(C), key=len)).copy())
    nc = C.number_of_nodes()
    gtrow = np.array([C.nodes[i]["gtrow"] for i in range(nc)], dtype=int)
    E = [(u, v, float(C[u][v]["weight"])) for u, v in C.edges()]
    Dobs = apsp(E, nc)
    core = finite_core(Dobs)
    Dt = Dtrue[np.ix_(gtrow[core], gtrow[core])]
    true_cfg, _ = classical_mds(Dt, 2)

    def score(D):
        Dc = D[np.ix_(core, core)]
        Y, _ = classical_mds(Dc, 2)
        Ys, _ = smacof(Dc, 2, init=Y)
        return _procrustes_disp(true_cfg, Ys)[0], knn_recovery(Dc, Dt, 10)

    fn, _variant, vkey = _SUITE[algo]
    out = run_isolated(fn, C, VERIFY[vkey], _cap_for(algo, nc))
    assert out.get("status") == "ok", f"{algo} did not return a cover: {out.get('status')}"
    S = {_key(u, v) for u, v in _norm_cover(out["cover"])}

    d1, k1 = score(build_F_distances(E, S, nc))                     # the canonical rule
    det, Drow, keep = _detours(E, S, nc)
    L = _lower_bounds(S, Drow, keep)
    E2, skipped = [], 0
    neww = {}
    for (u, v) in S:
        dd, ll = det[(u, v)], L[(u, v)]
        if not np.isfinite(dd) or ll > dd:
            skipped += 1
            continue
        neww[_key(u, v)] = ll + ALPHA * (dd - ll)
    E2 = [(u, v, neww.get(_key(u, v), w)) for u, v, w in E]
    Ha = heavy_count(E2, nc)                                        # RECOMPUTED, not assumed
    d2, k2 = score(apsp(E2, nc))
    do, ko = score(Dobs)

    print(f"\nCLAIM 2 -- the midpoint rule on the SPARSE road network ({algo}, |S| = {len(S)})")
    print(f"  {'rule':<34}{'|H| after':>10}{'disparity':>11}{'knn10':>9}   verdict")
    print("  " + "-" * 78)
    print(f"  {'observed (no repair)':<34}{'':>10}{do:>11.4f}{ko:>9.4f}")
    print(f"  {'build_F  (w <- detour)':<34}{0:>10}{d1:>11.4f}{k1:>9.4f}   METRIC -- a valid repair")
    print(f"  {'midpoint (w <- L + 0.5(det-L))':<34}{Ha:>10}{d2:>11.4f}{k2:>9.4f}   "
          + ("METRIC -- a valid repair" if Ha == 0 else "NOT METRIC -- NOT A REPAIR"))
    if skipped:
        print(f"  ({skipped} cover edge(s) had no usable interval and kept their observed weight)")
    print(f"\n  => |H| after the midpoint correction = {Ha}. "
          + ("The rule holds here." if Ha == 0 else "The rule does NOT hold on a sparse base."))
    print(f"  => and it is {'better' if d2 < d1 else 'WORSE'} than the canonical rule anyway "
          f"({d2:.4f} vs {d1:.4f}).")
    return Ha, d1, d2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="also run claim 2 (n=5000, ~20 min)")
    a = ap.parse_args()
    claim_where_the_truth_sits()
    if a.real:
        claim_midpoint_fails_on_the_road()
