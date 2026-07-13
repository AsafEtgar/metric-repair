"""Assert that the pbmc3k truth is aligned to downstream_recovery's node order -- the ordering trap.

load_graph sorts node labels as STRINGS ('10' < '2'); the labels are expression-matrix row indices. If the
truth is permuted, every kNN/Spearman/triplet number for this graph is plausible, stable, and meaningless.
The check has teeth because the graph's edge weights ARE d* on their pairs (w = 1 - sim[i,j] by construction),
so a permuted truth cannot pass it.

    sage -python experiments/verify_pbmc_alignment.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from downstream_recovery import DOWNSTREAM_GRAPHS, load_graph, true_distances   # noqa: E402

GRAPH = "pbmc3k_cosine_knn"

nodes, idx, edges = load_graph(GRAPH)
n = len(nodes)
print(f"[{GRAPH}] kind={DOWNSTREAM_GRAPHS[GRAPH]}  n={n}  m={len(edges)}")
print(f"  load_graph order (string sort): {nodes[:6]} ...  -> NOT numeric" if nodes[:3] != ["0", "1", "2"]
      else "  load_graph order is numeric -- unexpected")

gt_ix, Dtrue = true_distances(GRAPH, nodes)
print(f"  true_distances: n_gt={len(gt_ix)}  Dtrue={Dtrue.shape}  "
      f"symmetric={np.allclose(Dtrue, Dtrue.T)}  diag0={np.abs(np.diag(Dtrue)).max():.1e}  "
      f"min={Dtrue.min():.4f}  max={Dtrue.max():.4f}")
assert len(gt_ix) == n, "every graph node must carry truth"

# graph index -> row of Dtrue (gt_ix[j] is the graph index of Dtrue's row j)
row_of = np.full(n, -1, dtype=int)
row_of[gt_ix] = np.arange(len(gt_ix))
assert (row_of >= 0).all()

# THE CHECK: for every graph edge, weight == Dtrue[row_of[iu], row_of[iv]] to float precision.
errs = np.array([abs(w - Dtrue[row_of[iu], row_of[iv]]) for (iu, iv, w) in edges])
print(f"  EDGE CHECK over all {len(edges)} edges: max |w - Dtrue| = {errs.max():.3e}   mean = {errs.mean():.3e}")

# A permuted truth must FAIL: shuffle the rows and confirm the error explodes (the check is not vacuous).
rng = np.random.default_rng(0)
perm = rng.permutation(len(gt_ix))
Dperm = Dtrue[np.ix_(perm, perm)]
errs_p = np.array([abs(w - Dperm[row_of[iu], row_of[iv]]) for (iu, iv, w) in edges])
print(f"  control (rows shuffled): max |w - Dtrue| = {errs_p.max():.3e}  mean = {errs_p.mean():.3e}"
      f"   <- the check is not vacuous")

assert errs.max() < 1e-12, f"*** MISALIGNED *** max edge error {errs.max():.3e}"
print("  OK -- truth is aligned to the loader's node order (max error < 1e-12)")
