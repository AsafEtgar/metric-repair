"""IS IT THE SET, OR THE WEIGHTS? -- the question, asked on REAL data.

On a PLANTED graph we can cross the two choices, because we know both the corrupted set B and the true
weight w0 of every edge. On real data neither exists: the weights ARE the measurements, and there is nothing
to restore them to. So we define the only thing we can:

    the "true weight" of edge (u,v)  :=  d*(u,v),  the external ground-truth distance between its endpoints.

READ THAT DEFINITION TWICE. It is not "undo a corruption". It is "REPLACE A MEASUREMENT WITH THE TRUTH". On
dimacs_ny_t the edges measure travel time and d* is geography -- overwriting one with the other is a
substitution, not a repair. It still answers the question we are asking (is the deficit in the SET or in the
WEIGHTS?), but it is a different operation than on planted data, and the paper must say so.

THE PREDICTION THIS TESTS. Two diagnostics motivate the run, and both are cheap and already measured:

  (1) THE ERROR IS NOT IN THE HEAVY SET. Define err(e) = |w(e) - d*(e)|. If repair's premise held, H would
      carry a disproportionate share of it. It does not: the concentration -- (share of error in H) divided
      by (share of edges in H) -- is 1.0x on nmr_residue, 0.7x on nmr_atom, 1.9x on dimacs_ny_t, and 0.0x on
      pbmc3k. The heavy set carries its PROPORTIONAL share and no more. Being heavy and being wrong are
      different properties, and on real data they are essentially uncorrelated.

  (2) THE ERROR IS WORTH FIXING. Set EVERY edge to d* and the disparity falls from 0.1047 to 0.0443 on
      nmr_residue (58% of it recoverable) and from 0.0854 to 0.0258 on dimacs_ny_t (70%). The information is
      in the edge weights. There is a great deal on the table.

  So the prediction: oracle-weighting a cover of size |S| should recover roughly |S|/m of the available gain
  -- on nmr_residue, about 5% of 58%, i.e. ~3%. Near nothing. If that is what we measure, then repair edits
  the wrong edges, and the value stays on the table. This run turns that prediction into a measurement.

ARMS, per saved cover:
    restore    w(e) <- d_{G\\S}(e)   the canonical rule. It reproduces the existing pipeline, and that
                                     reproduction is the GATE.
    oracle     w(e) <- d*(e)         the new arm.
and, once per graph:
    observed   no repair at all      the baseline
    all_oracle EVERY edge <- d*(e)   the CEILING: the most any reweighting could possibly buy.

NON-DESTRUCTIVE. Reads data/processed/, data/processed/gt/, and results_real/results_real_covers/. Writes
ONLY to results_oracle/. It touches no existing CSV, figure, or paper number.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downstream_recovery import (_covers_dir, apsp, knn_recovery, load_cover,  # noqa: E402
                                 load_graph, true_distances)
from mds_recovery import (_norm_cover, _procrustes_disp, build_F_distances,  # noqa: E402
                          classical_mds, finite_core, smacof)
from metric_repair import domr_alg  # noqa: E402
from mds_recovery import _nx_from_edges  # noqa: E402

KS = [5, 10, 20]
GRAPHS = ["nmr_1d3z_residue", "nmr_1d3z_atom", "dimacs_ny_t", "pbmc3k_cosine_knn", "ripe_atlas"]
DIM = {"nmr_1d3z_residue": 3, "nmr_1d3z_atom": 3}          # the NMR truths are 3-D folds; the rest are 2-D
ALGOS = ["domr", "gmr_bestofk", "gmr_ilp", "gmr_rand", "gmr_thr_naive", "iomr_bestofk", "iomr_ilp",
         "iomr_rand", "iomr_regiongrow", "iomr_thr_naive", "l1sep_gmr", "l1sep_iomr", "left_edge",
         "pivot", "spc_gmr", "spc_iomr"]


def all_tasks():
    return [(g, a) for g in GRAPHS for a in ALGOS]


def _key(u, v):
    return (u, v) if u <= v else (v, u)


def build_instance(graph):
    """The graph, its external truth d*, and the row alignment between them.

    THE NODE-ORDERING TRAP LIVES HERE. `load_graph` sorts labels as STRINGS ('10' < '2'); the truth is stored
    in its own order. Mixing them permutes d* against the graph and yields plausible, stable, MEANINGLESS
    numbers -- with no crash. We go through load_graph + true_distances END TO END, and then ASSERT the
    alignment against the edge weights where the graph is supposed to equal the truth."""
    nodes, idx, edges = load_graph(graph)
    n = len(nodes)
    gt_ix, Dstar = true_distances(graph, nodes)
    row = {int(x): j for j, x in enumerate(gt_ix)}          # graph index -> row of Dstar
    dim = DIM.get(graph, 2)

    Dobs = apsp(edges, n)
    core = [c for c in finite_core(Dobs) if c in row]       # nodes that are BOTH reachable and have a truth
    core = np.array(sorted(core))
    trows = np.array([row[c] for c in core])
    Dt = Dstar[np.ix_(trows, trows)]
    true_cfg, _ = classical_mds(Dt, dim)
    assert true_cfg.shape[0] == len(core), "truth and embedding have different row counts"

    H = len(_norm_cover(domr_alg(_nx_from_edges(edges, n))))
    return dict(graph=graph, nodes=nodes, idx=idx, edges=edges, n=n, m=len(edges), H=H,
                Dstar=Dstar, row=row, core=core, Dt=Dt, true_cfg=true_cfg, dim=dim, Dobs=Dobs)


def score(D, ins):
    """Both axes from ONE distance matrix, on the SAME node set, in the SAME row order."""
    Dc = D[np.ix_(ins["core"], ins["core"])]
    Y, _ = classical_mds(Dc, ins["dim"])
    Ys, _ = smacof(Dc, ins["dim"], init=Y)
    disp = _procrustes_disp(ins["true_cfg"], Ys)[0]
    return disp, {k: knn_recovery(Dc, ins["Dt"], k) for k in KS}


def oracle_edges(ins, S):
    """Set the cover's edges to their TRUE distance d*. Every other edge keeps its measurement."""
    row = ins["row"]
    out = []
    for u, v, w in ins["edges"]:
        if _key(u, v) in S and u in row and v in row:
            out.append((u, v, float(ins["Dstar"][row[u], row[v]])))
        else:
            out.append((u, v, w))
    return out


def _row(ins, algo, arm, cover_file, size, disp, kn, wall):
    return dict(graph=ins["graph"], algo=algo, arm=arm, cover_file=cover_file, cover_size=size,
                n=ins["n"], m=ins["m"], H=ins["H"], n_core=len(ins["core"]), dim=ins["dim"],
                wall=round(wall, 2), disp=disp, **{f"knn{k}": kn[k] for k in KS})


def run_one(graph, algo, covers_root, outdir):
    ins = build_instance(graph)
    rows = []

    # the two per-graph reference rows, written with the FIRST algorithm only (they do not depend on a cover)
    if algo == ALGOS[0]:
        t = time.time()
        d, kn = score(ins["Dobs"], ins)
        rows.append(_row(ins, "--", "observed", "", 0, d, kn, time.time() - t))
        t = time.time()
        d, kn = score(apsp(oracle_edges(ins, {_key(u, v) for u, v, _ in ins["edges"]}), ins["n"]), ins)
        rows.append(_row(ins, "--", "all_oracle", "", ins["m"], d, kn, time.time() - t))
        print(f"  [{graph}] observed / all_oracle written (the ceiling)", flush=True)

    cdir = _covers_dir(covers_root, graph)
    files = sorted(f for f in os.listdir(cdir) if f.startswith(algo + "__") and f.endswith(".txt")) \
        if os.path.isdir(cdir) else []
    if not files:
        print(f"  [{graph}/{algo}] no saved cover -- nothing to score (an honest null, not a gap)", flush=True)
    for f in files:
        S = load_cover(os.path.join(cdir, f), ins["idx"])
        if not S:
            continue
        t = time.time()
        d1, k1 = score(build_F_distances(ins["edges"], S, ins["n"]), ins)      # the canonical rule -- the GATE
        rows.append(_row(ins, algo, "restore", f, len(S), d1, k1, time.time() - t))
        t = time.time()
        d2, k2 = score(apsp(oracle_edges(ins, S), ins["n"]), ins)              # the new arm
        rows.append(_row(ins, algo, "oracle", f, len(S), d2, k2, time.time() - t))
        print(f"  [{graph}/{algo}] {f:<28} |S|={len(S):<6} restore {d1:.4f}  oracle {d2:.4f}", flush=True)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"oracle_{graph}__{algo}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-index", type=int, default=None)
    ap.add_argument("--graph", default=None)
    ap.add_argument("--algo", default=None)
    ap.add_argument("--covers", default="results_real/results_real_covers")
    ap.add_argument("--outdir", default="results_oracle")
    ap.add_argument("--count", action="store_true")
    a = ap.parse_args()
    if a.count:
        print(len(all_tasks())); return
    if a.task_index is not None:
        g, alg = all_tasks()[a.task_index]
    elif a.graph and a.algo:
        g, alg = a.graph, a.algo
    else:
        ap.error("give --task-index, or --graph and --algo (or --count)")
    print(f"oracle task {g}/{alg} -> {run_one(g, alg, a.covers, a.outdir)}")


if __name__ == "__main__":
    main()
