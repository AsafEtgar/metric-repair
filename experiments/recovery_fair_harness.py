"""Downstream recovery, RE-RUN unbiased: ONE task per (graph, corruption, algorithm, seed), UNIFORM 2h cap.

WHY. The published recovery ran the WHOLE suite in one task per (graph, corruption), under a 300s/900s
per-algorithm cap plus a shared per-task budget. On the n=5000 road net most of the suite times out, so the
median is taken over the cheap combinatorial SURVIVORS and is biased toward them (see tab:corruption's own
caption). This array removes the bias: every algorithm gets its OWN task and a uniform 2h wall cap, so a
timeout is an ALGORITHMIC fact, not a scheduling artefact, and one slow method never starves another.

GRID. {RGG n=3000, dimacs_ny_d (n=5000)} x {inflate, deflate, mixed} x 5 seeds x (observed + the suite).
The graph and its planted corruption are seeded by (graph, corruption, seed) and NOT by the algorithm, so all
of an instance's algorithm-tasks rebuild the BYTE-IDENTICAL corrupted graph -- the invariant that makes a
cross-algorithm comparison valid. The preflight gates it (rebuild-twice edge-set + |B| check).

REUSE. graph_models (the FIXED inflate -- no +1 floor), the mds_sweep suite/isolation, and the
recovery_harness/rgg_recovery build+score, so geometry (Procrustes on a SMACOF embedding) and topology (k-NN
Jaccard at k in {5,10,20}) are matched off ONE repaired matrix, exactly as the published table computed them.
Nothing is reimplemented.

  usage   sage -python experiments/recovery_fair_harness.py --preflight
          sage -python experiments/recovery_fair_harness.py --count
          sage -python experiments/recovery_fair_harness.py --task-index 0 --outdir results_recovery_fair
"""
import argparse
import os
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_models import (seed_all, random_geometric_metric_graph,               # noqa: E402
                          break_metric_graph)
from metric_repair import domr_alg                                               # noqa: E402
from downstream_recovery import apsp, knn_recovery                               # noqa: E402
from mds_recovery import (_norm_cover, _procrustes_disp, build_F_distances,       # noqa: E402
                          classical_mds, finite_core, smacof)
from mds_sweep import SWEEP_ALGOS, VERIFY, _SUITE, run_isolated                   # noqa: E402
# The dimacs build is exactly recovery_harness.build_instance; the shared scoring is its score(). Imported
# READ-ONLY so this array cannot drift from the published table's construction.
from recovery_harness import build_instance as _build_dimacs, score, KS, _key     # noqa: E402

CAP_S = 2 * 3600                       # 2h, uniform, every algorithm. THE POINT of the re-run.
SEEDS = [1000, 1001, 1002, 1003, 1004]
DIRECTIONS = ["inflate", "deflate", "mixed"]

# The two graphs and their corruption knobs, matching the published recovery conventions so the re-run is
# comparable (only the cap, the RGG size n=3000, and the 5 seeds change). RGG: deg=10, frac=0.15 (rgg_recovery
# RGG_SPECS). dimacs: the recovery_harness fracs (0.20 inflate/mixed, 0.30 deflate -- a road net is hard to
# shortcut). magnitude 3.0 throughout.
GRAPHS = {
    "rgg_n3000":   dict(kind="rgg", n=3000, deg=10, frac={"inflate": 0.15, "deflate": 0.15, "mixed": 0.15}),
    "dimacs_ny_d": dict(kind="dimacs", base="dimacs_ny_d",
                        frac={"inflate": 0.20, "deflate": 0.30, "mixed": 0.20}),
}
MAG = 3.0


def _euclid(P):
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    np.fill_diagonal(d, 0.0)
    return d


def build_rgg(name, n, deg, direction, frac, mag, seed):
    """An RGG instance in the SAME dict shape recovery_harness.build_instance returns, so score() serves both.
    Truth = the generating positions; B = the planted set, mapped into C's index space by node identity."""
    seed_all(seed)
    radius = float(np.sqrt(deg / (np.pi * max(n - 1, 1))))
    T = random_geometric_metric_graph(n, mode="radius", radius=radius)
    T = nx.convert_node_labels_to_integers(T.subgraph(max(nx.connected_components(T), key=len)).copy())
    for i in T.nodes():
        T.nodes[i]["olabel"] = i                            # identity tag, survives the next relabel (for B)
    C, corrupted = break_metric_graph(T, frac_q=frac, direction=direction, magnitude=mag)
    C = nx.convert_node_labels_to_integers(C.subgraph(max(nx.connected_components(C), key=len)).copy())
    nc = C.number_of_nodes()
    pos = np.array([C.nodes[i]["pos"] for i in range(nc)], dtype=float)
    edges = [(u, v, float(C[u][v]["weight"])) for u, v in C.edges()]
    Dobs = apsp(edges, nc)
    core = finite_core(Dobs)
    lab2i = {C.nodes[i]["olabel"]: i for i in range(nc)}
    B = {_key(lab2i[u], lab2i[v]) for u, v in corrupted if u in lab2i and v in lab2i}
    H = max(len(_norm_cover(domr_alg(C))), 1)
    return dict(name=name, C=C, nc=nc, edges=edges, B=B, Dobs=Dobs, core=core,
                Dtrue=_euclid(pos[core]), true_cfg=pos[core], H=H, m=len(edges), n_corrupted=len(corrupted))


def build(graph, direction, seed):
    cfg = GRAPHS[graph]
    frac = cfg["frac"][direction]
    if cfg["kind"] == "rgg":
        return build_rgg(f"{graph}_{direction}", cfg["n"], cfg["deg"], direction, frac, MAG, seed)
    spec = (f"{cfg['base']}_{direction}", cfg["base"], direction, frac, MAG, seed)
    return _build_dimacs(spec)


def all_tasks():
    """One entry per (graph, direction, seed, algo); algo=None is the `observed` baseline of that instance."""
    out = []
    for graph in GRAPHS:
        for direction in DIRECTIONS:
            for seed in SEEDS:
                out.append((graph, direction, seed, None))
                for algo, _variant in SWEEP_ALGOS:
                    out.append((graph, direction, seed, algo))
    return out


def run_one(task_index, outdir):
    graph, direction, seed, algo = all_tasks()[task_index]
    ins = build(graph, direction, seed)
    common = dict(graph=graph, corruption=direction, seed=seed, n=ins["nc"], m=ins["m"], H=ins["H"],
                  n_corrupted=ins["n_corrupted"], n_core=len(ins["core"]), cap_s=CAP_S)

    if algo is None:
        d, kn = score(ins["Dobs"], ins)
        rows = [dict(common, algo="observed", variant="observed", status="ok", cover_size=0,
                     precision=np.nan, recall=np.nan, wall=0.0, disp=d, **{f"knn{k}": kn[k] for k in KS})]
    else:
        fn, variant, vkey = _SUITE[algo]
        t0 = time.time()
        out = run_isolated(fn, ins["C"], VERIFY[vkey], CAP_S)      # <-- uniform 2h cap, no _cap_for, no budget
        wall = round(time.time() - t0, 2)
        status = out.get("status", "error:no-status")
        if status != "ok" or out.get("converged") is False:
            why = status if status != "ok" else "not_converged"   # an uncertified/timed-out cover is a NULL row
            rows = [dict(common, algo=algo, variant=variant, status=why, cover_size=np.nan,
                         precision=np.nan, recall=np.nan, wall=wall, disp=np.nan,
                         **{f"knn{k}": np.nan for k in KS})]
        else:
            S = {_key(u, v) for u, v in _norm_cover(out["cover"])}
            inter = len(S & ins["B"])
            d, kn = score(build_F_distances(ins["edges"], S, ins["nc"]), ins)
            rows = [dict(common, algo=algo, variant=variant, status="ok", cover_size=len(S),
                         precision=inter / max(len(S), 1), recall=inter / max(len(ins["B"]), 1),
                         wall=wall, disp=d, **{f"knn{k}": kn[k] for k in KS})]

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"rec_{task_index:04d}_{graph}_{direction}_s{seed}_{algo or 'observed'}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ----------------------------------------------------------------------------
# Preflight -- refuses to submit unless the invariants hold.
# ----------------------------------------------------------------------------
def preflight():
    fails = []

    def chk(c, name, obs):
        print(f"  [{'PASS' if c else 'FAIL'}] {name:<54} {obs}")
        if not c:
            fails.append(name)

    n_suite = len(SWEEP_ALGOS)
    N = len(all_tasks())
    chk(N == len(GRAPHS) * len(DIRECTIONS) * len(SEEDS) * (1 + n_suite),
        "G1 grid = graphs x dirs x seeds x (observed + suite)",
        f"{N} = {len(GRAPHS)} x {len(DIRECTIONS)} x {len(SEEDS)} x (1 + {n_suite})")

    # G2  THE FIX IS PRESENT. Build a small RGG inflation and confirm the effective magnitude equals mu (the
    #     '+1' floor would pin it at ~11.8x). This array MUST plant the corrected corruption.
    seed_all(7)
    Gm = random_geometric_metric_graph(n=300, mode="radius",
                                       radius=float(np.sqrt(10 / (np.pi * 299))))
    seed_all(7)
    Hh, Bb = break_metric_graph(Gm, frac_q=0.10, direction="inflate", magnitude=MAG)
    effs = []
    for (u, v) in list(Bb)[:60]:
        Gc = Gm.copy(); Gc.remove_edge(u, v)
        try:
            effs.append(Hh[u][v]["weight"] / nx.shortest_path_length(Gc, u, v, weight="weight"))
        except nx.NetworkXNoPath:
            continue
    eff = float(np.median(effs)) if effs else float("nan")
    chk(abs(eff / MAG - 1.0) < 0.05, "G2 inflate magnitude is LIVE (effective == requested)",
        f"effective {eff:.2f} vs requested {MAG}")

    # G3  BOTH GRAPHS BUILD and the corruption LANDS (|B| > 0). The RGG is checked at every direction (cheap);
    #     dimacs (n=5000, ~minutes to build) is checked once -- enough to confirm the base loads and breaks.
    for graph in GRAPHS:
        dirs = DIRECTIONS if GRAPHS[graph]["kind"] == "rgg" else DIRECTIONS[:1]
        for direction in dirs:
            try:
                ins = build(graph, direction, SEEDS[0])
                ok = ins["nc"] > 0 and ins["n_corrupted"] > 0 and len(ins["core"]) > 0
                chk(ok, f"G3 {graph}/{direction} builds and breaks",
                    f"n={ins['nc']} m={ins['m']} |B|={ins['n_corrupted']} |H|={ins['H']} core={len(ins['core'])}")
            except Exception as e:
                chk(False, f"G3 {graph}/{direction} builds and breaks", f"{type(e).__name__}: {e}")

    # G4  DETERMINISM -- THE INVARIANT that makes the comparison valid. Rebuild one instance twice; the edge
    #     set and the planted set B must be byte-identical, because the seed keys on (graph, corruption, seed)
    #     and NOT the algorithm. Checked on the RGG (fast); the seeding path -- seed_all(seed) then
    #     break_metric_graph -- is IDENTICAL for dimacs, and _planted_base loads deterministically, so the same
    #     invariant holds there without paying two n=5000 builds in the preflight.
    rgg = next(g for g in GRAPHS if GRAPHS[g]["kind"] == "rgg")
    a = build(rgg, "inflate", SEEDS[0])
    b = build(rgg, "inflate", SEEDS[0])
    same_e = ({_key(u, v) for u, v, _ in a["edges"]} == {_key(u, v) for u, v, _ in b["edges"]})
    chk(same_e and a["B"] == b["B"] and a["nc"] == b["nc"],
        f"G4 {rgg}: rebuild is byte-identical (seed excludes algo)",
        f"edges match={same_e}, |B| {len(a['B'])}=={len(b['B'])}, n {a['nc']}=={b['nc']}")

    # G5  the cap is what we said, and it is uniform.
    chk(CAP_S == 2 * 3600, "G5 cap is a uniform 2h", f"{CAP_S}s = {CAP_S/3600:.1f}h per algorithm")

    print()
    print(f"  {N} tasks; each caps its algorithm at {CAP_S/3600:.1f}h + instance build/score. "
          f"Suite = {n_suite} algorithms + observed.")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-index", type=int, default=None)
    ap.add_argument("--outdir", default="results_recovery_fair")
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    a = ap.parse_args()

    if a.preflight:
        fails = preflight()
        if fails:
            raise SystemExit(f"\n*** PREFLIGHT FAILED: {fails}. NOT submitting. ***")
        print("\nPreflight clean.")
        return
    if a.count:
        print(len(all_tasks()))
        return
    if a.task_index is None:
        ap.error("--task-index is required (or --count / --preflight)")
    print(f"recovery_fair task {a.task_index} -> {run_one(a.task_index, a.outdir)}")


if __name__ == "__main__":
    main()
