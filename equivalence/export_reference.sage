# equivalence/export_reference.sage
#
# STEP 1 of the equivalence proof. Run with the ORIGINAL Sage library and export, for a batch of
# graphs, the inputs (edge lists) and the reference outputs of every function. Step 2
# (check_equivalence.py) reconstructs the same graphs in pure Python and diffs against this file.
#
# Run from the repo root:   sage equivalence/export_reference.sage
import json, numpy as np
load("sage_version/graph_models.sage")
load("sage_version/metric_repair.sage")
load("sage_version/metric_extras.sage")

def pairs(S):
    return sorted([(int(min(u, v)), int(max(u, v))) for (u, v) in S])

seed_all(0)
out = []
for n in [12, 16, 20, 24, 28, 32, 36, 40, 44, 48]:
    for _try in range(40):
        G = random_geometric_weighted_graph(int(n), 0.35)
        if G.is_connected() and G.num_verts() >= 4:
            break
    if not G.is_connected():
        continue
    Kn = complete(G)
    s_mvd, s_piv = 1000 + n, 5000 + n

    rec = dict(n=int(n))
    rec["edges"] = [(int(u), int(v), int(w)) for u, v, w in G.edges(sort=True)]
    # deterministic outputs
    rec["complete"] = {f"{int(min(u,v))}-{int(max(u,v))}": float(w) for u, v, w in Kn.edges(sort=True)}
    rec["domr"] = pairs(domr_alg(G))
    rec["left_edge"] = pairs(left_edge_heuristic(G))           # Gilbert-Jain: deterministic
    rec["l1"] = pairs(l1_min_heuristic(G))                     # support (may differ: LP is degenerate)
    # L1 LP optimum: the minimum total correction. This is UNIQUE even when the support is not, so it
    # is the right thing to compare bit-for-bit (proves the two LPs are the same problem).
    _phi, _cnt = induced_cycle_matrix(Kn)
    if _cnt > 0:
        _Dl1 = make_index_encoding(Kn); _wl1 = get_weights_vector(Kn, _Dl1)
        rec["l1_obj"] = float(linprog(np.ones(_phi.shape[1]), A_ub=-_phi, b_ub=_phi @ _wl1,
                                      method="highs").fun)
    else:
        rec["l1_obj"] = 0.0
    rec["ver_domr_on_Kn"] = int(verifier(Kn, domr_alg(Kn)))
    rec["iomrver_ledge_on_Kn"] = int(iomr_verifier(Kn, Gilbert_Jain_IOMR(Kn)))
    rec["is_metric_G"] = int(is_metric(G))
    rec["is_metric_Kn"] = int(is_metric(Kn))
    # seeded MVD (reproducible exactly given the seed + same matrix)
    np.random.seed(int(s_mvd)); rec["mvd_seed"] = int(s_mvd); rec["mvd"] = pairs(MVD_Pivot(Kn))
    np.random.seed(int(s_piv)); rec["piv_seed"] = int(s_piv); rec["piv"] = pairs(pivot_heuristic(G))
    # tie-broken shortest-path cover: record cover + size + validity (compared approximately)
    sg = shortest_path_cover(Kn, general=True)
    si = shortest_path_cover(Kn, general=False)
    rec["spc_general_size"] = len(sg); rec["spc_general_valid"] = int(verifier(Kn, set(sg)))
    rec["spc_iomr_size"] = len(si);    rec["spc_iomr_valid"] = int(iomr_verifier(Kn, set(si)))
    out.append(rec)
    print(f"n={n}: domr={len(rec['domr'])} ledge={len(rec['left_edge'])} l1={len(rec['l1'])} "
          f"mvd={len(rec['mvd'])} spc_gen={rec['spc_general_size']} spc_iomr={rec['spc_iomr_size']}")

json.dump(out, open("equivalence/reference.json", "w"))
print(f"\nexported {len(out)} graphs -> equivalence/reference.json")
