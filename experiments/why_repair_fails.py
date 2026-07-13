"""IS THE WRONG SET, OR THE WRONG CORRECTION, WHAT KILLS RECOVERY?

Metric repair minimises |S|: the NUMBER of edges edited. Nothing in that objective says the edited edges
should be the CORRUPTED ones, nor that their new weights should be the TRUE ones. Those are three different
problems, and the algorithms solve only the first. When repair fails to recover structure, the failure can
therefore live in either of two places, and they demand opposite fixes:

  (a) THE SET is wrong -- the algorithm edits edges that were never corrupted, and misses the ones that were.
  (b) THE CORRECTION is wrong -- even given the right edges, the repair rule (build_F: reweight each cover
      edge to its shortest detour in G \\ S) does not restore the true weight.

On a PLANTED graph both are measurable, because `break_metric_graph` hands back the ground-truth edit set B
and the clean graph holds the true weights w0. So we cross them:

    SET        x  CORRECTION      what it isolates
    ---------------------------------------------------------------------------
    oracle B      oracle w0       the ceiling: this IS the clean graph
    oracle B      build_F rule    (b) alone -- perfect set, the rule's own loss
    algo S        build_F rule    what we actually do today
    algo S        oracle w0       (a) alone -- the algorithm's set, perfectly corrected

If `oracle B + build_F` recovers, the rule is fine and the SET is the problem: we need a recovery-aware
choice of S, not a minimum-cardinality one. If it does NOT recover, no choice of set can save us and the
CORRECTION must change.

We also report the cover's precision and recall against B -- does the algorithm even FIND the corrupted edges?
"""
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_models import break_metric_graph, random_geometric_metric_graph  # noqa: E402
from downstream_recovery import apsp, knn_recovery  # noqa: E402
from mds_recovery import (RGG_SPECS, _norm_cover, _procrustes_disp, build_F_distances,  # noqa: E402
                          classical_mds, finite_core, seed_all, smacof)
from mds_sweep import SWEEP_ALGOS, VERIFY, _SUITE, _cap_for, run_isolated  # noqa: E402

K = 10


def _euclid(P):
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    np.fill_diagonal(d, 0.0)
    return d


def _key(u, v):
    return (u, v) if u <= v else (v, u)


def score(D, core, true_cfg, Dtrue):
    Dc = D[np.ix_(core, core)]
    Yc, _ = classical_mds(Dc, 2)
    Ys, _ = smacof(Dc, 2, init=Yc)
    return _procrustes_disp(true_cfg, Ys)[0], knn_recovery(Dc, Dtrue, K)


def run(name, n, deg, direction, frac, mag, seed):
    seed_all(seed)
    radius = float(np.sqrt(deg / (np.pi * max(n - 1, 1))))
    T = random_geometric_metric_graph(n, mode="radius", radius=radius)
    T = nx.convert_node_labels_to_integers(T.subgraph(max(nx.connected_components(T), key=len)).copy())
    C, corrupted = break_metric_graph(T, frac_q=frac, direction=direction, magnitude=mag)
    # relabel BOTH together so the ground-truth edit set survives the relabelling (it is keyed by node id)
    comp = max(nx.connected_components(C), key=len)
    mapping = {u: i for i, u in enumerate(sorted(comp))}
    C = nx.relabel_nodes(C.subgraph(comp).copy(), mapping)
    T = nx.relabel_nodes(T.subgraph(comp).copy(), mapping)
    B = {_key(mapping[u], mapping[v]) for u, v in corrupted if u in mapping and v in mapping}

    nc = C.number_of_nodes()
    pos = np.array([C.nodes[i]["pos"] for i in range(nc)], dtype=float)
    obs_edges = [(u, v, float(C[u][v]["weight"])) for u, v in C.edges()]
    true_w = {_key(u, v): float(T[u][v]["weight"]) for u, v in T.edges()}

    Dobs = apsp(obs_edges, nc)
    core = finite_core(Dobs)
    true_cfg = pos[core]
    Dtrue = _euclid(pos[core])

    def with_true_weights(S):
        """The graph with every edge of S restored to its TRUE weight (edges of S that were never corrupted
        already carry their true weight, so this is exactly 'undo the corruption the algorithm found')."""
        return [(u, v, true_w[_key(u, v)] if _key(u, v) in S else w) for u, v, w in obs_edges]

    out = []
    d, k = score(Dobs, core, true_cfg, Dtrue)
    out.append(dict(graph=name, cond="observed (no repair)", setk="--", corr="--",
                    size=0, prec=np.nan, rec=np.nan, disp=d, knn=k))

    # THE CEILING: oracle set + oracle weights == the clean graph
    d, k = score(apsp([(u, v, true_w[_key(u, v)]) for u, v, _ in obs_edges], nc), core, true_cfg, Dtrue)
    out.append(dict(graph=name, cond="ORACLE set + ORACLE weights (= clean)", setk="oracle", corr="oracle",
                    size=len(B), prec=1.0, rec=1.0, disp=d, knn=k))

    # (b) ALONE: the right edges, corrected by the repair rule
    d, k = score(build_F_distances(obs_edges, B, nc), core, true_cfg, Dtrue)
    out.append(dict(graph=name, cond="ORACLE set + build_F rule", setk="oracle", corr="build_F",
                    size=len(B), prec=1.0, rec=1.0, disp=d, knn=k))

    for algo, variant in SWEEP_ALGOS:
        fn, sv, vkey = _SUITE[algo]
        res = run_isolated(fn, C, VERIFY[vkey], _cap_for(algo, nc))
        if res.get("status") != "ok" or res.get("converged") is False:
            continue
        S = {_key(u, v) for u, v in _norm_cover(res["cover"])}
        inter = len(S & B)
        prec = inter / max(len(S), 1)
        rec = inter / max(len(B), 1)
        # what we do today
        d, k = score(build_F_distances(obs_edges, S, nc), core, true_cfg, Dtrue)
        out.append(dict(graph=name, cond=f"{algo}: set + build_F rule", setk=algo, corr="build_F",
                        size=len(S), prec=prec, rec=rec, disp=d, knn=k))
        # (a) ALONE: the algorithm's set, corrected PERFECTLY
        d, k = score(apsp(with_true_weights(S), nc), core, true_cfg, Dtrue)
        out.append(dict(graph=name, cond=f"{algo}: set + ORACLE weights", setk=algo, corr="oracle",
                        size=len(S), prec=prec, rec=rec, disp=d, knn=k))
    return out


if __name__ == "__main__":
    want = sys.argv[1].split(",") if len(sys.argv) > 1 else ["rgg_inflate", "rgg_deflate", "rgg_mixed"]
    rows = []
    for spec in RGG_SPECS:
        if spec[0] in want:
            print(f"[why] {spec[0]}", flush=True)
            rows += run(*spec)
    df = pd.DataFrame(rows)
    df.to_csv("analysis/why_repair_fails.csv", index=False)

    for g, G in df.groupby("graph", sort=False):
        print(f"\n{'=' * 100}\n{g}\n{'=' * 100}")
        print(f"{'condition':<40}{'|S|':>6}{'prec':>7}{'recall':>8}{'disp':>9}{'knn10':>8}")
        print("-" * 100)
        for _, r in G.iterrows():
            p = "  --  " if not np.isfinite(r.prec) else f"{r.prec:.3f} "
            q = "  --  " if not np.isfinite(r.rec) else f"{r.rec:.3f} "
            print(f"{r.cond:<40}{int(r['size']):>6}{p:>7}{q:>8}{r.disp:>9.4f}{r.knn:>8.4f}")
    print("\nwrote analysis/why_repair_fails.csv")
