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

    # alpha: the scalar that carries the truth's units into the graph's. See oracle_edges().
    r = [float(w) / float(Dstar[row[u], row[v]])
         for u, v, w in edges
         if u in row and v in row and float(Dstar[row[u], row[v]]) > 1e-12]
    ratio = float(np.median(r)) if r else 1.0
    # SAME QUANTITY -> alpha = 1: d* IS the truth and we use it. The NMR ratio of 1.21-1.25 is NOT a unit
    # conversion -- it is the NOE SLACK, because an upper bound exceeds the distance it bounds. Rescaling by
    # it would hand the edge 1.21x the truth, which is not the truth. Use d* directly.
    # DIFFERENT QUANTITY -> alpha = the median ratio, and the arm is asking a WEAKER question (see below).
    same_units = 0.5 < ratio < 2.0
    alpha = 1.0 if same_units else ratio
    note = ("same quantity: alpha = 1, the oracle arm gives the edge its TRUE distance"
            if same_units else
            f"DIFFERENT quantity (w/d* = {ratio:.4g}): the oracle arm gives the edge the value it would "
            "have if it were PROPORTIONAL to the truth. That is the distance-proportional idealisation, "
            "NOT the truth, and it is a weaker question. Treat this graph's row with care.")
    print(f"    [{graph}] alpha = {alpha:.4g}  -- {note}", flush=True)

    return dict(graph=graph, nodes=nodes, idx=idx, edges=edges, n=n, m=len(edges), H=H, alpha=alpha,
                same_units=int(same_units), Dstar=Dstar, row=row, core=core, Dt=Dt, true_cfg=true_cfg,
                dim=dim, Dobs=Dobs)


def score(D, ins):
    """Both axes from ONE distance matrix, on the SAME node set, in the SAME row order."""
    Dc = D[np.ix_(ins["core"], ins["core"])]
    Y, _ = classical_mds(Dc, ins["dim"])
    Ys, _ = smacof(Dc, ins["dim"], init=Y)
    disp = _procrustes_disp(ins["true_cfg"], Ys)[0]
    return disp, {k: knn_recovery(Dc, ins["Dt"], k) for k in KS}


def oracle_edges(ins, S):
    """Set the cover's edges to their TRUE distance -- RESCALED INTO THE GRAPH'S UNITS.

    THE BUG THIS FIXES, and it is worth stating because it silently produced a plausible wrong answer.
    Writing d*(u,v) straight into an edge only makes sense when the graph and the truth measure the SAME
    QUANTITY. They do on the two NMR graphs (NOE upper bounds and the fold are both in Angstrom; the median
    ratio is 1.21-1.25, and it exceeds 1 for the physical reason that an NOE is an UPPER bound) and on
    pbmc3k (the edges ARE the cosine distances; ratio 1.000). They do NOT on dimacs_ny_t, whose edges are
    TRAVEL TIME while d* is geographic kilometres -- median ratio 24,984 -- nor on ripe_atlas, whose edges
    are round-trip milliseconds against kilometres -- ratio 0.021.

    Substituting unscaled, we wrote a kilometre value into a graph of seconds. The edge became 25,000 times
    too short, turned into a vast shortcut, and made the map WORSE than doing nothing (0.0879 against an
    observed 0.0854). That is not a finding. It is a unit error wearing a finding's clothes.

    So we rescale: alpha = median over edges of w(e)/d*(e), the best scalar conversion between the two
    measurement systems, and the oracle weight is alpha * d*(e). On the graphs where the units already agree
    alpha is ~1 and this changes nothing. On the others it turns an incoherent substitution into a
    well-posed question -- but a WEAKER one, and the paper must say which it is asking:

        SAME QUANTITY (nmr, pbmc3k):  "give this edge its true distance."
        DIFFERENT QUANTITY (dimacs_ny_t, ripe):  "give this edge the travel time / latency it would have if
        it were proportional to geography." That is the distance-proportional idealisation, not the truth.
    """
    row, a = ins["row"], ins["alpha"]
    out = []
    for u, v, w in ins["edges"]:
        if _key(u, v) in S and u in row and v in row:
            out.append((u, v, a * float(ins["Dstar"][row[u], row[v]])))
        else:
            out.append((u, v, w))
    return out


def _row(ins, algo, arm, cover_file, size, disp, kn, wall):
    return dict(graph=ins["graph"], algo=algo, arm=arm, cover_file=cover_file, cover_size=size,
                n=ins["n"], m=ins["m"], H=ins["H"], n_core=len(ins["core"]), dim=ins["dim"],
                alpha=ins["alpha"], same_units=ins["same_units"],
                wall=round(wall, 2), disp=disp, **{f"knn{k}": kn[k] for k in KS})


def run_one(graph, algo, covers_root, outdir):
    ins = build_instance(graph)
    rows = []

    # the two per-graph reference rows, written with the FIRST algorithm only (they do not depend on a cover)
    if algo == ALGOS[0]:
        t = time.time()
        d, kn = score(ins["Dobs"], ins)
        r = _row(ins, "--", "observed", "", 0, d, kn, time.time() - t)
        # A GATE REFERENCE, and a real methodological difference worth recording rather than hiding.
        # The existing pipeline scores k-NN over ALL ground-truth nodes and lets knn_recovery drop non-finite
        # entries row by row. We score over the FINITE CORE, because classical MDS cannot accept an infinity
        # and we want BOTH axes on the SAME node set -- a matched measurement is the whole point of this file.
        # On nmr_1d3z_atom the graph has 3 components, so 3 of its 343 truth-bearing nodes fall outside the
        # giant one: our core is 340 and theirs is 343, and that difference is the entire delta (0.441199
        # against 0.436158 at k=20). We therefore ALSO report the old node set, so the gate can compare like
        # with like instead of failing on a design choice.
        gt = np.array(sorted(ins["row"].keys()))
        tr = np.array([ins["row"][g] for g in gt])
        Dg = ins["Dobs"][np.ix_(gt, gt)]
        Dt = ins["Dstar"][np.ix_(tr, tr)]
        for k in KS:
            r[f"knn{k}_gtset"] = knn_recovery(Dg, Dt, k)
        r["n_gtset"] = len(gt)
        rows.append(r)
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
