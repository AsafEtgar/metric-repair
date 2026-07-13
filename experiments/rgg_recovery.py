"""MATCHED topology + geometry recovery on the MDS RGG instances.

WHY THIS EXISTS. `mds_sweep.run_rgg_sweep` already builds each algorithm's repaired distance matrix
(`build_F_distances`) and scores its GEOMETRY (Procrustes disparity). It then throws the matrix away. The
paper's TOPOLOGY numbers (k-NN lift) came from a different place entirely -- the campaign grid, at other n,
other p, other corruption magnitudes -- so the two axes were never measured on the same instance and could not
honestly be put in one table.

This script recomputes BOTH on the SAME instance, from the SAME cover, on the SAME repaired matrix. It is a
faithful replay of `run_rgg_sweep`: same seed_all, same generator, same break, same suite, same caps.

THE GATE. A replay is only worth anything if it reproduces. Every disparity computed here is checked against
the stored `analysis/summary_mds_sweep.csv`; a mismatch beyond 1e-6 is a hard failure, not a warning. If the
replay drifted (a different cover, a different RNG draw), the k-NN numbers would be measuring a DIFFERENT
repair than the figures show, and would be worthless.

Truth = the planted point positions (Euclidean), the same `pos` the disparity is scored against.
"""
import argparse
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_models import break_metric_graph, random_geometric_metric_graph  # noqa: E402
from metric_repair import domr_alg  # noqa: E402
from downstream_recovery import apsp, knn_recovery  # noqa: E402
from mds_recovery import (RGG_SPECS, _norm_cover, _procrustes_disp, build_F_distances,  # noqa: E402
                          classical_mds, finite_core, seed_all, smacof)
from mds_sweep import SWEEP_ALGOS, VERIFY, _SUITE, _cap_for, run_isolated  # noqa: E402

KS = [5, 10, 20]


def _euclid(P):
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    np.fill_diagonal(d, 0.0)
    return d


def _score(D, true_cfg, Dtrue):
    """Geometry (Procrustes disparity, smacof) AND topology (k-NN Jaccard) off ONE repaired matrix."""
    Yc, _ = classical_mds(D, 2)
    Ys, _ = smacof(D, 2, init=Yc)
    disp = _procrustes_disp(true_cfg, Ys)[0]
    knn = {k: knn_recovery(D, Dtrue, k) for k in KS}
    return disp, knn


def run_spec(name, n, deg, direction, frac, mag, seed):
    print(f"[rgg_recovery] {name}", flush=True)
    seed_all(seed)                                    # byte-for-byte the replay of run_rgg_sweep
    radius = float(np.sqrt(deg / (np.pi * max(n - 1, 1))))
    T = random_geometric_metric_graph(n, mode="radius", radius=radius)
    T = nx.convert_node_labels_to_integers(T.subgraph(max(nx.connected_components(T), key=len)).copy())
    C, _ = break_metric_graph(T, frac_q=frac, direction=direction, magnitude=mag)
    C = nx.convert_node_labels_to_integers(C.subgraph(max(nx.connected_components(C), key=len)).copy())
    nc = C.number_of_nodes()
    pos = np.array([C.nodes[i]["pos"] for i in range(nc)], dtype=float)
    edges = [(u, v, C[u][v]["weight"]) for u, v in C.edges()]
    Dobs = apsp(edges, nc)
    core = finite_core(Dobs)
    true_cfg = pos[core]
    Dtrue = _euclid(pos[core])                        # the planted geometry: the SAME truth the disparity uses

    def restrict(D):
        return D[np.ix_(core, core)]

    H = max(len(_norm_cover(domr_alg(C))), 1)
    disp, knn = _score(restrict(Dobs), true_cfg, Dtrue)
    rows = [dict(graph=name, corruption=direction, algo="observed", variant="observed",
                 cover_size=0, H=H, m=len(edges), n_used=len(core), status="ok",
                 disp_smacof=disp, **{f"knn{k}": knn[k] for k in KS})]

    for algo, variant in SWEEP_ALGOS:
        fn, suite_variant, vkey = _SUITE[algo]
        assert suite_variant == variant, f"{algo}: suite says {suite_variant}, we say {variant}"
        out = run_isolated(fn, C, VERIFY[vkey], _cap_for(algo, nc))
        status = out.get("status", "error:no-status")
        if status != "ok" or out.get("converged") is False:
            why = status if status != "ok" else "not_converged"
            print(f"    [{algo}] {why} -- no cover", flush=True)
            rows.append(dict(graph=name, corruption=direction, algo=algo, variant=variant,
                             cover_size=np.nan, H=H, m=len(edges), n_used=len(core), status=why,
                             disp_smacof=np.nan, **{f"knn{k}": np.nan for k in KS}))
            continue
        cover = _norm_cover(out["cover"])
        d, kn = _score(restrict(build_F_distances(edges, cover, nc)), true_cfg, Dtrue)
        print(f"    [{algo}] |S|={len(cover)}  disp {d:.4f}  knn10 {kn[10]:.4f}", flush=True)
        rows.append(dict(graph=name, corruption=direction, algo=algo, variant=variant,
                         cover_size=len(cover), H=H, m=len(edges), n_used=len(core), status="ok",
                         disp_smacof=d, **{f"knn{k}": kn[k] for k in KS}))
    return rows


def gate(df, sweep_csv):
    """THE GATE: every replayed disparity must reproduce the stored one. Otherwise the replay drifted and the
    k-NN numbers describe a repair that is not the one in the figures."""
    if not os.path.exists(sweep_csv):
        print(f"!! {sweep_csv} missing -- CANNOT verify the replay; refusing to certify"); return False
    ref = pd.read_csv(sweep_csv)
    ref = ref[ref.status.fillna("ok").eq("ok")][["graph", "algo", "disp_smacof"]]
    j = df[df.status.eq("ok")].merge(ref, on=["graph", "algo"], suffixes=("", "_ref"))
    if j.empty:
        print("!! no overlap with the stored sweep -- CANNOT verify"); return False
    err = (j.disp_smacof - j.disp_smacof_ref).abs()
    bad = j[err > 1e-6]
    print(f"\nGATE: {len(j)} replayed disparities checked against {os.path.basename(sweep_csv)}; "
          f"max |delta| = {err.max():.2e}")
    if len(bad):
        print("!! REPLAY DRIFTED -- these do not reproduce; the k-NN numbers are NOT matched:")
        print(bad[["graph", "algo", "disp_smacof", "disp_smacof_ref"]].to_string(index=False))
        return False
    print("GATE PASSED: the replay reproduces the stored covers exactly. Topology and geometry are MATCHED.")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="analysis")
    ap.add_argument("--sweep", default="analysis/summary_mds_sweep.csv")
    ap.add_argument("--specs", default="all", help="comma-separated spec names, or 'all'")
    a = ap.parse_args()

    want = None if a.specs == "all" else set(a.specs.split(","))
    rows = []
    for spec in RGG_SPECS:
        if want and spec[0] not in want:
            continue
        rows += run_spec(*spec)

    df = pd.DataFrame(rows)
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, "summary_rgg_recovery.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(df)} rows)")

    ok = gate(df, a.sweep)

    # Lemma 6.1: a decrease-only cover changes no shortest path, so its lift and its disparity delta are both
    # EXACTLY zero. This is the pipeline's built-in self-check, and it MUST gate the exit code -- it used to
    # print "!! VIOLATED" and then exit 0, because `ok` was computed above and never updated here. A check
    # that cannot fail the run is decoration, not a check.
    for g, grp in df.groupby("graph"):
        o = grp[grp.algo == "observed"]
        d = grp[grp.algo == "domr"]
        if len(o) and len(d) and pd.notna(d.disp_smacof.iloc[0]):
            dd = abs(float(d.disp_smacof.iloc[0]) - float(o.disp_smacof.iloc[0]))
            dk = abs(float(d.knn10.iloc[0]) - float(o.knn10.iloc[0]))
            good = dd < 1e-9 and dk < 1e-9
            ok = ok and good
            print(f"  DOMR self-check {g}: |d disp| = {dd:.2e}  |d knn10| = {dk:.2e}  "
                  f"{'OK' if good else '!! VIOLATED'}")
    if not ok:
        print("\n!! A GATE OR THE DOMR SELF-CHECK FAILED. These numbers are void until it is explained.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
