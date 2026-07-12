"""The topology array: matched k-NN AND geometry on the PLANTED road network.

WHY THIS EXISTS. The recovery table's planted `dimacs_ny_d` rows carry a geometry number but no topology
number, because a matched k-NN needs the cover suite re-run at n = 5000. Every other cell in that table is
measured; these three are not. We fill them here rather than borrow a k-NN from a differently parameterised
run -- the earlier realrec downstream pass used magnitude 5.0 while the MDS sweep used 3.0, and quoting one
beside the other would be a confound dressed as a result.

WHAT "MATCHED" MEANS, AND WHY IT IS THE POINT. Both axes are computed from ONE cover, on ONE repaired distance
matrix, in one process. A k-NN entry and a disparity entry on the same row therefore describe the same repair
of the same graph, and may be compared directly. That was not previously true anywhere in the study.

THE GATE. Task 0 is the `observed` row of dimacs_ny_d_inflate and MUST reproduce disp = 0.364054, the value
already stored in analysis/summary_mds_sweep.csv; every algorithm's build_F disparity must likewise reproduce
its stored value. If they do not, the instance rebuild has drifted (a different seed draw, a different node
order) and every number here describes a different graph than the paper's figures do. collect_recovery.py
refuses to print results until this passes.

One task = one (planted spec, algorithm). 3 specs x (1 observed + 15 algorithms) = 48 tasks.
Per-task cost at n = 5000: cover <= 300 s (the HUGE_N cap), plus ~5-15 min of instance build and scoring.
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

from graph_models import break_metric_graph                                        # noqa: E402
from metric_repair import domr_alg                                                 # noqa: E402
from downstream_recovery import apsp, knn_recovery                                 # noqa: E402
from mds_recovery import (_norm_cover, _procrustes_disp, build_F_distances,        # noqa: E402
                          classical_mds, finite_core, seed_all, smacof)
from mds_sweep import SWEEP_ALGOS, VERIFY, _SUITE, _cap_for, _planted_base, run_isolated  # noqa: E402

KS = [5, 10, 20]

# Mirrors mds_sweep.REALREC_SPECS EXACTLY -- same base, direction, frac, magnitude, seed. That identity is
# what makes the gate meaningful: this array's geometry column must reproduce the stored sweep.
SPECS = [
    ("dimacs_ny_d_inflate", "dimacs_ny_d", "inflate", 0.20, 3.0, 1000),
    ("dimacs_ny_d_deflate", "dimacs_ny_d", "deflate", 0.30, 3.0, 1000),
    ("dimacs_ny_d_mixed",   "dimacs_ny_d", "mixed",   0.20, 3.0, 1000),
]


def all_tasks():
    """(spec, algo) pairs, plus one `observed` baseline row per spec (algo=None)."""
    out = []
    for spec in SPECS:
        out.append((spec, None))
        for algo, _variant in SWEEP_ALGOS:
            out.append((spec, algo))
    return out


def _key(u, v):
    return (u, v) if u <= v else (v, u)


def build_instance(spec):
    """The planted graph, byte-for-byte as mds_sweep.run_realrec_sweep builds it -- including the two node-order
    traps it guards against, both of which yield a plausible-but-meaningless disparity rather than a crash."""
    name, base, direction, frac, mag, seed = spec
    G0, Dtrue, nodes, gt_ix, _ = _planted_base(base)

    seed_all(seed)
    C, corrupted = break_metric_graph(G0, frac_q=frac, direction=direction, magnitude=mag)
    C = nx.convert_node_labels_to_integers(C.subgraph(max(nx.connected_components(C), key=len)).copy())
    nc = C.number_of_nodes()
    gtrow = np.array([C.nodes[i]["gtrow"] for i in range(nc)], dtype=int)
    olabel = [C.nodes[i]["olabel"] for i in range(nc)]
    assert (gtrow >= 0).all(), "a surviving node carries no ground-truth row -- the relabel dropped the truth"
    assert olabel == [str(nodes[int(gt_ix[r])]) for r in gtrow], \
        "truth row order != embedding row order -- true_cfg is permuted against the embedding"

    edges = [(u, v, float(C[u][v]["weight"])) for u, v in C.edges()]

    # the ground-truth edit set, carried into C's index space by LABEL (C was relabelled by sorted node order)
    lab2i = {C.nodes[i]["olabel"]: i for i in range(nc)}
    B = set()
    for u, v in corrupted:
        lu, lv = str(G0.nodes[u]["olabel"]), str(G0.nodes[v]["olabel"])
        if lu in lab2i and lv in lab2i:
            B.add(_key(lab2i[lu], lab2i[lv]))

    Dobs = apsp(edges, nc)
    core = finite_core(Dobs)
    Dt = Dtrue[np.ix_(gtrow[core], gtrow[core])]           # the truth, in EMBEDDING row order
    true_cfg, _ = classical_mds(Dt, 2)
    H = len(_norm_cover(domr_alg(C)))
    return dict(name=name, C=C, nc=nc, edges=edges, B=B, Dobs=Dobs, core=core,
                Dtrue=Dt, true_cfg=true_cfg, H=H, m=len(edges), n_corrupted=len(corrupted))


def score(D, ins):
    """Geometry (Procrustes) and topology (k-NN at every k) from ONE distance matrix -- matched by construction."""
    Dc = D[np.ix_(ins["core"], ins["core"])]
    Yc, _ = classical_mds(Dc, 2)
    Ys, _ = smacof(Dc, 2, init=Yc)
    disp = _procrustes_disp(ins["true_cfg"], Ys)[0]
    return disp, {k: knn_recovery(Dc, ins["Dtrue"], k) for k in KS}


def run_one(task_index, outdir):
    tasks = all_tasks()
    spec, algo = tasks[task_index]
    ins = build_instance(spec)
    common = dict(graph=ins["name"], corruption=spec[2], n=ins["nc"], m=ins["m"], H=ins["H"],
                  n_corrupted=ins["n_corrupted"], n_core=len(ins["core"]))

    if algo is None:
        d, kn = score(ins["Dobs"], ins)
        rows = [dict(common, algo="observed", variant="observed", status="ok", cover_size=0,
                     precision=np.nan, recall=np.nan, wall=0.0, disp=d,
                     **{f"knn{k}": kn[k] for k in KS})]
    else:
        fn, variant, vkey = _SUITE[algo]
        t0 = time.time()
        out = run_isolated(fn, ins["C"], VERIFY[vkey], _cap_for(algo, ins["nc"]))
        wall = round(time.time() - t0, 2)
        status = out.get("status", "error:no-status")
        if status != "ok" or out.get("converged") is False:
            # An uncertified cover is not a result. Keep the row as an EXPLICIT null; never drop it.
            why = status if status != "ok" else "not_converged"
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
    path = os.path.join(outdir, f"recovery_{task_index:03d}_{ins['name']}_{algo or 'observed'}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-index", type=int, default=None)
    ap.add_argument("--outdir", default="results_recovery")
    ap.add_argument("--count", action="store_true", help="print the number of tasks and exit")
    a = ap.parse_args()
    if a.count:
        print(len(all_tasks()))
        return
    if a.task_index is None:
        ap.error("--task-index is required (or pass --count)")
    print(f"recovery task {a.task_index} -> {run_one(a.task_index, a.outdir)}")


if __name__ == "__main__":
    main()
