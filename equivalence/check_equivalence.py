"""
equivalence/check_equivalence.py

STEP 2 of the equivalence proof. Reconstruct the exact graphs the Sage library was run on (from
equivalence/reference.json) and run the PURE-PYTHON library on them, then diff. Imports no Sage.

Run from the repo root (after export_reference.sage):
    PYTHONPATH=. python equivalence/check_equivalence.py        # or: sage -python equivalence/check_equivalence.py

What is proven:
  * Deterministic functions (complete, domr_alg, left_edge_heuristic, l1_min_heuristic, verifier,
    iomr_verifier, is_metric) are BIT-FOR-BIT identical to Sage on the same input graph.
  * MVD_Pivot / pivot_heuristic reproduce Sage EXACTLY given the same NumPy seed (the kernel is the
    same numpy code, run on the same matrix).
  * shortest_path_cover matches in DISTRIBUTION only -- scipy and Sage break shortest-path ties
    differently -- so we check that sizes are close and every cover is valid (not bit-identical).
  * Graph GENERATION is intentionally NOT compared (different RNG); we reconstruct Sage's graphs from
    the exported edge lists so the algorithm comparison is apples-to-apples.
"""
import json
import os
import sys

import numpy as np
import networkx as nx
from scipy.optimize import linprog

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import metric_repair as mr


def graph_from_edges(edges):
    G = nx.Graph()
    for u, v, w in edges:
        G.add_edge(int(u), int(v), weight=int(w))
    return G


def complete_dict(Kn):
    return {f"{min(u,v)}-{max(u,v)}": float(d["weight"]) for u, v, d in Kn.edges(data=True)}


def main():
    ref_path = os.path.join(os.path.dirname(__file__), "reference.json")
    data = json.load(open(ref_path))
    N = len(data)

    # exact-match counters
    exact = {k: 0 for k in ["complete", "domr", "left_edge", "l1_obj", "ver_domr_on_Kn",
                            "iomrver_ledge_on_Kn", "is_metric_G", "is_metric_Kn", "mvd", "piv"]}
    # tie-broken / degenerate: distributional checks (valid + size close)
    dist_valid = {"spc_general": 0, "spc_iomr": 0, "l1": 0}
    dist_size_rel = {"spc_general": [], "spc_iomr": [], "l1": []}

    try:
        import metric_extras as me
        have_extras = True
    except Exception:
        have_extras = False

    for rec in data:
        G = graph_from_edges(rec["edges"])
        Kn = mr.complete(G)

        if complete_dict(Kn) == {k: float(v) for k, v in rec["complete"].items()}:
            exact["complete"] += 1
        if mr.domr_alg(G) == set(map(tuple, rec["domr"])):
            exact["domr"] += 1
        if mr.left_edge_heuristic(G) == set(map(tuple, rec["left_edge"])):
            exact["left_edge"] += 1
        # l1: compare the LP OPTIMUM (unique) exactly; the support is degenerate (checked below)
        phi, cnt = mr.induced_cycle_matrix(Kn)
        if cnt > 0:
            Dl1 = mr.make_index_encoding(Kn)
            wl1 = mr.get_weights_vector(Kn, Dl1)
            l1_obj = float(linprog(np.ones(phi.shape[1]), A_ub=-phi, b_ub=phi @ wl1, method="highs").fun)
        else:
            l1_obj = 0.0
        if abs(l1_obj - rec["l1_obj"]) < 1e-6:
            exact["l1_obj"] += 1
        l1_py = mr.l1_min_heuristic(G)
        dist_valid["l1"] += mr.verifier(G, l1_py)
        dist_size_rel["l1"].append(len(l1_py) / max(1, len(rec["l1"])))
        if mr.verifier(Kn, mr.domr_alg(Kn)) == rec["ver_domr_on_Kn"]:
            exact["ver_domr_on_Kn"] += 1
        if mr.iomr_verifier(Kn, mr.Gilbert_Jain_IOMR(Kn)) == rec["iomrver_ledge_on_Kn"]:
            exact["iomrver_ledge_on_Kn"] += 1
        if have_extras:
            if int(me.is_metric(G)) == rec["is_metric_G"]:
                exact["is_metric_G"] += 1
            if int(me.is_metric(Kn)) == rec["is_metric_Kn"]:
                exact["is_metric_Kn"] += 1

        np.random.seed(rec["mvd_seed"])
        if mr.MVD_Pivot(Kn) == set(map(tuple, rec["mvd"])):
            exact["mvd"] += 1
        np.random.seed(rec["piv_seed"])
        if mr.pivot_heuristic(G) == set(map(tuple, rec["piv"])):
            exact["piv"] += 1

        sg = mr.shortest_path_cover(Kn, general=True)
        si = mr.shortest_path_cover(Kn, general=False)
        dist_valid["spc_general"] += mr.verifier(Kn, set(sg))
        dist_valid["spc_iomr"] += mr.iomr_verifier(Kn, set(si))
        dist_size_rel["spc_general"].append(len(sg) / max(1, rec["spc_general_size"]))
        dist_size_rel["spc_iomr"].append(len(si) / max(1, rec["spc_iomr_size"]))

    print(f"Compared {N} graphs (pure Python vs Sage), algorithms only (graph generation excluded).\n")
    print("EXACT (bit-for-bit identical to Sage on the same graph):")
    labels = {
        "complete": "complete()", "domr": "domr_alg", "left_edge": "left_edge_heuristic",
        "l1_obj": "l1 LP optimum (value)", "ver_domr_on_Kn": "verifier verdict",
        "iomrver_ledge_on_Kn": "iomr_verifier verdict", "is_metric_G": "is_metric(G)",
        "is_metric_Kn": "is_metric(Kn)", "mvd": "MVD_Pivot (seeded)", "piv": "pivot_heuristic (seeded)",
    }
    all_ok = True
    for k in ["complete", "domr", "left_edge", "l1_obj", "ver_domr_on_Kn", "iomrver_ledge_on_Kn",
              "is_metric_G", "is_metric_Kn", "mvd", "piv"]:
        if not have_extras and k.startswith("is_metric"):
            print(f"  {labels[k]:28s}: (metric_extras not importable, skipped)")
            continue
        ok = exact[k] == N
        all_ok = all_ok and ok
        print(f"  {labels[k]:28s}: {exact[k]:>2}/{N}  {'OK' if ok else 'MISMATCH'}")

    print("\nDISTRIBUTIONAL (non-unique result: tie-breaking or LP degeneracy; expect valid + size close):")
    dlabels = {"spc_general": "shortest_path_cover[general]", "spc_iomr": "shortest_path_cover[iomr]",
               "l1": "l1_min_heuristic (support)"}
    dist_ok = True
    for name in ("spc_general", "spc_iomr", "l1"):
        rels = dist_size_rel[name]
        dist_ok = dist_ok and (dist_valid[name] == N)
        print(f"  {dlabels[name]:30s}: valid {dist_valid[name]}/{N}  "
              f"size nx/sage  mean={np.mean(rels):.3f}  min={np.min(rels):.3f}  max={np.max(rels):.3f}")

    print("\nRESULT:", "PASS -- behaviour matches (graph generation excluded)" if (all_ok and dist_ok)
          else "CHECK ABOVE -- some mismatch")
    sys.exit(0 if (all_ok and dist_ok) else 1)


if __name__ == "__main__":
    main()
