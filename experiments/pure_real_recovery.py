"""pure_real_recovery.py -- the PURE-REAL downstream table, completed, on all five graphs.

`downstream_recovery.run_one_graph` already knows how to do this: load a real graph, resolve its EXTERNAL
ground truth, score the observed shortest-path metric against it, then replay every saved cover through the
`restore` construction and re-score. What was missing was coverage and one column.

  COVERAGE.  The published summary (analysis/summary_downstream.csv) holds three graphs -- ripe_atlas,
  nmr_1d3z_atom, nmr_1d3z_residue. Two more carry saved covers and an external truth and were never scored:
  pbmc3k_cosine_knn (131 covers, truth = the ambient PCA-50 cosine metric) and dimacs_ny_t (131 covers,
  truth = the DIMACS geography). dimacs_ny_d is genuinely N/A -- it is exactly metric (|H| = 0), so there
  are no covers and no repair to score.

  THE TRIPLET COLUMN.  downstream_recovery emits delta_triplet; downstream_analyze.py drops it. The paper
  therefore records triplet accuracy in Limitations as unobtainable. It is not: it costs nothing to keep,
  and this module keeps it. All three axes are reported -- kNN lift (local), Spearman (global), triplet
  (ordinal, in between).

SELF-CHECK WITH TEETH.  DOMR's cover is exactly the heavy set, and reweighting a heavy edge to its detour
leaves every shortest-path distance unchanged (Lemma 6.1). So DOMR's lift and delta_triplet must be EXACTLY
0.0 on every graph. They are asserted, not hoped for: a nonzero DOMR number aborts the run.

SPEED.  We do not touch downstream_recovery. We monkeypatch ONE hot function into it -- `_knn_sets`, whose
per-row Python filter costs 18s per call on dimacs_ny_t (5,000 nodes) and would push this run past two hours.
The replacement is vectorized and PROVABLY IDENTICAL: `verify_fast_knn` re-runs the original on the real
matrices of the graph in hand, at every k, and aborts on the first differing set. Equivalence is not an
argument in a comment -- it is a precondition checked at run time, on the same data the run then uses.

    sage -python experiments/pure_real_recovery.py                 # all five, in parallel
    sage -python experiments/pure_real_recovery.py --graphs ripe_atlas --jobs 1
    sage -python experiments/pure_real_recovery.py --aggregate-only      # re-summarize saved rows

Writes analysis/pure_real_rows.csv (per cover per k) and analysis/summary_pure_real.csv (per graph/algo/
variant/k: median lift, delta_spearman, delta_triplet + the observed baselines).
"""
import argparse
import csv
import glob
import os
import statistics
import sys
import time
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import downstream_recovery as dr   # noqa: E402
from downstream_recovery import FIELDS, K_LIST, load_graph, true_distances, apsp   # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The five pure-real graphs that have BOTH an external ground truth and saved covers.
# dimacs_ny_d is excluded on purpose: |H| = 0, it is already metric, there is nothing to repair.
GRAPHS = ["nmr_1d3z_residue", "nmr_1d3z_atom", "dimacs_ny_t", "pbmc3k_cosine_knn", "ripe_atlas"]
COVERS_ROOT = os.path.join("results_real", "results_real_covers")
ROWS_CSV = os.path.join("analysis", "pure_real_rows.csv")
SUMMARY_CSV = os.path.join("analysis", "summary_pure_real.csv")
PER_GRAPH_DIR = "results_pure_real"

SUM_FIELDS = ["graph", "gt_kind", "variant", "algo", "k", "n_covers", "n", "n_gt",
              "recovery_obs", "recovery_rep_med", "lift_med", "lift_q25", "lift_q75",
              "spearman_obs", "spearman_rep_med", "delta_spearman_med",
              "triplet_obs", "triplet_rep_med", "delta_triplet_med",
              "beats_observed"]


# ---------------------------------------------------------------------------
# The one hot patch: a vectorized _knn_sets, verified identical before it is used.
# ---------------------------------------------------------------------------
def fast_knn_sets(D, k, block=512):
    """Drop-in for downstream_recovery._knn_sets.

    The original masks the diagonal, stable-argsorts the row, filters to finite entries, and takes the
    first k. Non-finite entries sort LAST under a stable argsort, so 'first k of the finite entries' and
    'the finite entries among the first k' are the same set -- which is what lets the filter move off the
    row and onto a k-column slice. Same sets, same tie-breaking, no Python loop over n columns.
    """
    n = D.shape[0]
    out = []
    for s in range(0, n, block):
        B = np.array(D[s:s + block], dtype=float)        # copy: we overwrite the diagonal
        rows = np.arange(B.shape[0])
        B[rows, s + rows] = np.inf                       # exclude self, as the original does
        order = np.argsort(B, axis=1, kind="stable")[:, :k]
        for r in rows:
            o = order[r]
            out.append(frozenset(o[np.isfinite(B[r, o])].tolist()))
    return out


def verify_fast_knn(mats, tag=""):
    """Abort unless the vectorized _knn_sets reproduces the original EXACTLY, on these matrices, at every k.
    `mats` is a list of (name, matrix) -- pass the real Dtrue/Dobs of the graph about to be scored."""
    for name, M in mats:
        for k in K_LIST:
            a = dr._orig_knn_sets(M, k)
            b = fast_knn_sets(M, k)
            if a != b:
                bad = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
                raise SystemExit(
                    f"FATAL [{tag}] fast_knn_sets != _knn_sets on {name} at k={k}, first diff at row {bad}: "
                    f"{sorted(a[bad])} vs {sorted(b[bad])}. Refusing to run with a patched kernel.")
    return True


dr._orig_knn_sets = dr._knn_sets      # keep the reference implementation for the equivalence check


# ---------------------------------------------------------------------------
# Progress: wrap build_F_distances, which run_one_graph calls exactly once per cover, in cover order.
# ---------------------------------------------------------------------------
def _install_progress(graph, cover_files, t0):
    n_cov = len(cover_files)
    state = {"i": 0}
    orig = dr._orig_build_F

    def wrapped(edges, cover, n):
        i = state["i"]
        name = os.path.basename(cover_files[i])[:-4] if i < n_cov else "?"
        t = time.time()
        out = orig(edges, cover, n)
        state["i"] = i + 1
        el = time.time() - t0
        eta = el / (i + 1) * (n_cov - i - 1)
        print(f"  [{graph}] cover {i + 1:>3}/{n_cov}  {name:<28} |S|={len(cover):<7} "
              f"{time.time() - t:5.1f}s  elapsed {el / 60:5.1f}m  eta {eta / 60:5.1f}m", flush=True)
        return out

    dr.build_F_distances = wrapped


def score_graph(graph, covers_root=COVERS_ROOT, verify=True):
    """Score one graph through downstream_recovery.run_one_graph (unmodified) and return its rows."""
    t0 = time.time()
    cdir = dr._covers_dir(covers_root, graph)
    cover_files = sorted(glob.glob(os.path.join(cdir, "*.txt"))) if cdir else []
    if not cover_files:
        raise SystemExit(f"FATAL [{graph}] no covers under {covers_root}/{graph}/ -- nothing to score.")

    nodes, idx, edges = load_graph(graph)
    n = len(nodes)
    gt_ix, Dtrue = true_distances(graph, nodes)
    print(f"[{graph}] n={n} m={len(edges)} n_gt={len(gt_ix)} covers={len(cover_files)} "
          f"gt={dr.DOWNSTREAM_GRAPHS[graph]}", flush=True)

    if verify:
        Dobs = apsp(edges, n)[np.ix_(gt_ix, gt_ix)]
        verify_fast_knn([("Dtrue", Dtrue), ("Dobs", Dobs)], tag=graph)
        n_inf = int(np.isinf(Dobs).sum())
        print(f"[{graph}] fast_knn_sets == _knn_sets on Dtrue and Dobs at k={list(K_LIST)} "
              f"(Dobs carries {n_inf} infinite pairs) -- patch verified, {time.time() - t0:.0f}s", flush=True)
        del Dobs

    dr._knn_sets = fast_knn_sets
    _install_progress(graph, cover_files, time.time())
    rows = dr.run_one_graph(graph, covers_root=covers_root)
    dr._knn_sets = dr._orig_knn_sets
    dr.build_F_distances = dr._orig_build_F
    print(f"[{graph}] DONE {len(rows)} rows from {len(cover_files)} covers in "
          f"{(time.time() - t0) / 60:.1f}m", flush=True)
    return rows


dr._orig_build_F = dr.build_F_distances


def _worker(args):
    graph, covers_root, verify, outdir = args
    rows = score_graph(graph, covers_root, verify)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{graph}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# Self-check: Lemma 6.1. DOMR must move NOTHING.
# ---------------------------------------------------------------------------
def domr_selfcheck(rows, graphs=GRAPHS):
    domr = [r for r in rows if r["algo"] == "domr"]
    by_graph = {}
    for r in domr:
        by_graph.setdefault(r["graph"], []).append(r)
    print("\n" + "=" * 96)
    print("SELF-CHECK (Lemma 6.1): a decrease-only cover leaves every shortest-path distance unchanged,")
    print("so DOMR's lift and delta_triplet must be EXACTLY 0.0 on every graph.")
    print("=" * 96)
    bad = []
    for g in graphs:
        rs = by_graph.get(g, [])
        if not rs:
            bad.append(f"{g}: NO domr row at all")
            print(f"  {g:<20} MISSING")
            continue
        ml = max(abs(float(r["lift"])) for r in rs)
        mt = max(abs(float(r["delta_triplet"])) for r in rs)
        ms = max(abs(float(r["delta_spearman"])) for r in rs)
        ok = (ml == 0.0) and (mt == 0.0)
        print(f"  {g:<20} max|lift|={ml:.1e}  max|delta_triplet|={mt:.1e}  max|delta_spearman|={ms:.1e}"
              f"   {'PASS' if ok else '*** FAIL ***'}")
        if not ok:
            bad.append(f"{g}: max|lift|={ml} max|delta_triplet|={mt}")
    if bad:
        raise SystemExit("\nFATAL -- DOMR SELF-CHECK FAILED. The pipeline is wrong; the numbers are void.\n  "
                         + "\n  ".join(bad))
    print("  ALL PASS -- DOMR is exactly inert on all %d graphs. Pipeline sound." % len(graphs))
    return True


# ---------------------------------------------------------------------------
# Aggregate + report
# ---------------------------------------------------------------------------
def aggregate(rows):
    key = lambda r: (r["graph"], r["variant"], r["algo"], int(r["k"]))   # noqa: E731
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    out = []
    for (graph, variant, algo, k), rs in sorted(groups.items()):
        f = lambda c: [float(r[c]) for r in rs]                          # noqa: E731
        q = lambda v, p: float(np.percentile(v, p))                      # noqa: E731
        lifts = f("lift")
        out.append({
            "graph": graph, "gt_kind": rs[0]["gt_kind"], "variant": variant, "algo": algo, "k": k,
            "n_covers": len(rs), "n": rs[0]["n"], "n_gt": rs[0]["n_gt"],
            "recovery_obs": round(float(rs[0]["recovery_obs"]), 6),
            "recovery_rep_med": round(statistics.median(f("recovery_rep")), 6),
            "lift_med": round(statistics.median(lifts), 6),
            "lift_q25": round(q(lifts, 25), 6), "lift_q75": round(q(lifts, 75), 6),
            "spearman_obs": round(float(rs[0]["spearman_obs"]), 6),
            "spearman_rep_med": round(statistics.median(f("spearman_rep")), 6),
            "delta_spearman_med": round(statistics.median(f("delta_spearman")), 6),
            "triplet_obs": round(float(rs[0]["triplet_obs"]), 6),
            "triplet_rep_med": round(statistics.median(f("triplet_rep")), 6),
            "delta_triplet_med": round(statistics.median(f("delta_triplet")), 6),
            "beats_observed": int(sum(1 for x in lifts if x > 0)),
        })
    return out


def report(rows, k_focus=10, graphs=GRAPHS):
    """Per graph: the observed kNN recovery, the BEST lift and who achieved it, and whether ANY cover beats
    the observed graph -- on each of the three axes."""
    print("\n" + "=" * 96)
    print(f"PURE-REAL DOWNSTREAM RECOVERY -- per graph (kNN at k={k_focus}; Spearman/triplet are k-free)")
    print("=" * 96)
    for g in graphs:
        rs = [r for r in rows if r["graph"] == g]
        if not rs:
            print(f"\n{g}: NO ROWS"); continue
        rk = [r for r in rs if int(r["k"]) == k_focus]
        uniq = {}                                    # one row per cover for the k-free axes
        for r in rs:
            uniq[(r["algo"], r["mode"], r["seed"])] = r
        u = list(uniq.values())
        n_cov = len(u)
        obs = float(rk[0]["recovery_obs"])
        best = max(rk, key=lambda r: float(r["lift"]))
        pos = sum(1 for r in rk if float(r["lift"]) > 0)
        bs = max(u, key=lambda r: float(r["delta_spearman"]))
        bt = max(u, key=lambda r: float(r["delta_triplet"]))
        pos_s = sum(1 for r in u if float(r["delta_spearman"]) > 0)
        pos_t = sum(1 for r in u if float(r["delta_triplet"]) > 0)
        print(f"\n{g}   (n={rs[0]['n']}, n_gt={rs[0]['n_gt']}, {n_cov} covers, gt={rs[0]['gt_kind']})")
        print(f"  kNN@{k_focus:<2}  observed {obs:8.4f} | BEST lift {float(best['lift']):+8.4f} "
              f"by {best['algo']} ({best['variant']}) -> {float(best['recovery_rep']):.4f}"
              f" | {pos}/{len(rk)} covers beat observed")
        print(f"  Spearman  observed {float(rk[0]['spearman_obs']):8.4f} | BEST delta "
              f"{float(bs['delta_spearman']):+8.4f} by {bs['algo']} ({bs['variant']})"
              f" | {pos_s}/{n_cov} beat observed")
        print(f"  Triplet   observed {float(rk[0]['triplet_obs']):8.4f} | BEST delta "
              f"{float(bt['delta_triplet']):+8.4f} by {bt['algo']} ({bt['variant']})"
              f" | {pos_t}/{n_cov} beat observed")
        if pos == 0:
            print(f"  -> NO cover beats the observed graph on kNN@{k_focus}. Repair does not help topology here.")
        med = statistics.median([float(r["lift"]) for r in rk])
        print(f"  median lift {med:+.4f} over all {len(rk)} covers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", nargs="*", default=GRAPHS)
    ap.add_argument("--covers", default=COVERS_ROOT)
    ap.add_argument("--outdir", default=PER_GRAPH_DIR, help="per-graph CSVs (crash-resume)")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--k-focus", type=int, default=10)
    ap.add_argument("--no-verify", action="store_true", help="skip the fast_knn_sets equivalence check")
    ap.add_argument("--aggregate-only", action="store_true", help="re-summarize the saved per-graph CSVs")
    ap.add_argument("--resume", action="store_true", help="reuse per-graph CSVs that already exist")
    a = ap.parse_args()
    os.chdir(REPO)
    os.makedirs("analysis", exist_ok=True)
    os.makedirs(a.outdir, exist_ok=True)

    todo = list(a.graphs)
    if a.aggregate_only:
        todo = []
    elif a.resume:
        todo = [g for g in todo if not os.path.exists(os.path.join(a.outdir, f"{g}.csv"))]
        print(f"resume: {len(todo)} of {len(a.graphs)} graphs still to score")

    t0 = time.time()
    if todo:
        jobs = min(a.jobs, len(todo))
        args = [(g, a.covers, not a.no_verify, a.outdir) for g in todo]
        if jobs > 1:
            with get_context("fork").Pool(jobs) as p:
                for path in p.imap_unordered(_worker, args):
                    print(f"wrote {path}", flush=True)
        else:
            for x in args:
                print(f"wrote {_worker(x)}", flush=True)
        print(f"\nall scoring done in {(time.time() - t0) / 60:.1f} min", flush=True)

    rows = []
    for g in a.graphs:
        path = os.path.join(a.outdir, f"{g}.csv")
        if not os.path.exists(path):
            raise SystemExit(f"FATAL: {path} missing -- cannot aggregate.")
        with open(path, newline="") as f:
            rows += list(csv.DictReader(f))

    with open(ROWS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    summ = aggregate(rows)
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUM_FIELDS)
        w.writeheader()
        w.writerows(summ)

    domr_selfcheck(rows, graphs=a.graphs)
    report(rows, k_focus=a.k_focus, graphs=a.graphs)
    print(f"\nwrote {ROWS_CSV} ({len(rows)} rows)")
    print(f"wrote {SUMMARY_CSV} ({len(summ)} rows)")


if __name__ == "__main__":
    main()
