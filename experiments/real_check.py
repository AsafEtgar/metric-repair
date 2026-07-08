"""real_check.py -- health check for the real-dataset results (real_harness.py). See REAL_EXPERIMENTS.md.

    sage -python experiments/real_check.py --results results_real [--covers results_real_covers]

Reports, so you can catch a partial/broken run before trusting it:
  * FILE COVERAGE  -- which of the 621 expected task CSVs (589 heur + 32 ilp) are missing (task crashed
                      before writing, or still running) or unexpected.
  * STATUS         -- timeouts are a CONTROLLED cap, not a crash: the 17h ILP cap (-> LP-bound fallback) and
                      the REAL_HEUR_TIMEOUT_S heuristic cap (slow heuristic on a big-H graph like ripe/flycns)
                      are both benign, as are skipped_H (region gate) / skipped_time (per-task budget). Only
                      error/oom/killed count as hard failures.
  * INVALID COVERS -- valid==0 (a produced cover that doesn't verify -- always a bug).
  * ALGO COVERAGE  -- each graph should have its full set of algorithms across its files.
  * METRIC CONTROL -- dimacs_ny_d is metric, so every cover must be size 0 (repair does nothing).
The final PROBLEMS count is the number of hard issues (missing files + genuine failures + invalid covers).
"""
import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_harness import all_tasks, real_graphs, dist_sensible, RAND_ALGOS   # noqa: E402

NUM = ("size", "valid", "cpu", "wall", "peak_mb", "H", "nonmetric_frac", "rounds", "seed", "n", "m")
ILP = {"gmr_ilp", "iomr_ilp"}
# expected #algorithm-rows per file (18-algo suite minus rsp): det=11, rand=5, each ilp file=1
EXP_ROWS = {"det": 11, "rand": 5, "gmr_ilp": 1, "iomr_ilp": 1}


def _fname(graph, mode, seed):
    return f"{graph}__rand_s{seed:02d}.csv" if mode == "rand" else f"{graph}__{mode}.csv"


def expected_files():
    exp = {}
    for g, mode, seed in all_tasks("heur"):
        exp[_fname(g, mode, seed)] = ("heur", g, mode)
    for g, mode, seed in all_tasks("ilp"):
        exp[_fname(g, mode, seed)] = ("ilp", g, mode)
    return exp


def check(results, covers=None):
    problems = 0
    exp = expected_files()
    files = sorted(glob.glob(os.path.join(results, "*.csv")))
    present = {os.path.basename(f) for f in files}
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True) if files else pd.DataFrame()
    for c in NUM:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "converged" in df:
        df["converged"] = df["converged"].map({"True": True, "False": False, True: True, False: False})

    print("=" * 72)
    print(f"LOADED {len(present)} files, {len(df)} rows, "
          f"{df['graph'].nunique() if len(df) else 0} graphs, {df['algo'].nunique() if len(df) else 0} algos")

    # ---- file coverage ----
    missing = [f for f in exp if f not in present]
    extra = [f for f in present if f not in exp]
    print(f"\nFILE COVERAGE: {len(present)}/{len(exp)} present, {len(missing)} missing, {len(extra)} unexpected")
    if missing:
        by_array = {}
        for f in missing:
            by_array.setdefault(exp[f][0], []).append(exp[f][1])
        for arr, gs in by_array.items():
            from collections import Counter
            top = ", ".join(f"{g}x{c}" for g, c in Counter(gs).most_common(6))
            print(f"  {arr}: {len(gs)} missing  (by graph: {top}{' ...' if len(set(gs)) > 6 else ''})")
        problems += len(missing)
    if extra:
        print(f"  unexpected files: {extra[:8]}{' ...' if len(extra) > 8 else ''}")

    if df.empty:
        print(f"\nPROBLEMS: {problems}"); return problems

    # ---- status: timeouts are a CONTROLLED cap, not a crash. Both the 17h ILP cap (-> LP-bound fallback) and
    #      the REAL_HEUR_TIMEOUT_S heuristic cap (a slow heuristic on a big-H graph -> that cell has no cover)
    #      are benign. skipped_H (region gate) / skipped_time (per-task budget) are benign too (not in `bad`).
    #      Only error/oom/killed are genuine failures. ----
    print("\nstatus counts:")
    print(df["status"].value_counts().to_string())
    bad = df[df["status"].str.startswith(("error", "oom", "killed", "timeout"), na=False)]
    is_to = bad["status"].str.startswith("timeout", na=False)
    timeouts, hard = bad[is_to], bad[~is_to]
    ilp_to = timeouts[timeouts["algo"].isin(ILP)]
    heur_to = timeouts[~timeouts["algo"].isin(ILP)]
    print(f"\nbenign timeouts: {len(ilp_to)} ILP (17h cap -> LP bound) + {len(heur_to)} heuristic "
          f"(hit REAL_HEUR_TIMEOUT_S on a big-H graph -> that algo x graph cell has no cover)")
    if len(heur_to):
        print(heur_to.groupby("graph")["algo"].nunique().rename("n_algos_timed_out").to_string())
    print(f"\nHARD failures (error/oom/killed only): {len(hard)}")
    if len(hard):
        print(hard.groupby(["algo", "status"]).size().to_string())
        problems += len(hard)

    # ---- invalid covers ----
    inval = df[df["valid"] == 0]
    print(f"\ninvalid covers (valid==0, should be 0): {len(inval)}")
    if len(inval):
        print(inval.groupby(["graph", "algo"]).size().to_string())
        problems += len(inval)

    # ---- per-graph algo coverage (over the files that DID land) ----
    algos_per_graph = df.groupby("graph")["algo"].nunique()
    full = 18                                    # 11 det + 5 rand + 2 ilp
    thin = algos_per_graph[algos_per_graph < full - 2]     # raw variants legitimately lack the 2 ilp
    if len(thin):
        print(f"\ngraphs with < {full - 2} algorithms (may be mid-run or failed):")
        print(thin.to_string())

    # ---- metric control: dimacs_ny_d must repair to nothing ----
    ctrl = df[(df["graph"] == "dimacs_ny_d") & df["size"].notna()]
    if len(ctrl):
        nonzero = ctrl[ctrl["size"] > 0]
        print(f"\nmetric control dimacs_ny_d: {len(nonzero)} rows with size>0 (should be 0)")
        if len(nonzero):
            print(nonzero.groupby("algo")["size"].max().to_string()); problems += len(nonzero)

    # ---- covers present (for the GT-recovery pass) ----
    if covers and os.path.isdir(covers):
        ncov = len(glob.glob(os.path.join(covers, "*", "*.txt")))
        gt = [g for g in ("nmr_1d3z_residue", "nmr_1d3z_atom", "ripe_atlas") if
              glob.glob(os.path.join(covers, g, "*.txt"))]
        print(f"\ncovers: {ncov} files; GT-tier graphs with covers: {gt}")

    print("=" * 72)
    print(f"PROBLEMS: {problems}"
          f"   ({len(missing)} missing files + {len(hard)} hard failures + {len(inval)} invalid covers)")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results_real/ dir")
    ap.add_argument("--covers", default=None, help="results_real_covers/ dir (optional)")
    a = ap.parse_args()
    n = check(a.results, a.covers)
    if n:
        print(f"\n*** {n} hard problems -- investigate before trusting / analyzing ***")


if __name__ == "__main__":
    main()
