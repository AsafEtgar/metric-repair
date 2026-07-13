"""Concatenate the recovery array's per-task CSVs, RUN THE GATES, and print the rows it exists to fill.

THE GATES MUST FAIL CLOSED. An earlier version of this file failed OPEN in three separate ways, and an
adversarial audit found all three. They are recorded here because the shape of the mistake recurs:

  1. GATE 1 compared the rebuilt disparities against HARDCODED constants -- and two of the three constants
     were FABRICATED (0.028410 and 0.151216, against the real 0.028436 and 0.151152; they appear in no CSV).
     A gate that checks against an invented number is worse than no gate: it manufactures a delta on a
     byte-exact reproduction, and it moves the tolerance window off-centre. There is now no dict. Every
     expected value is READ FROM analysis/summary_mds_sweep.csv, which is the thing we are checking against.

  2. GATE 3 ran inside `if len(o) and len(d) and pd.notna(...)` with no else. If the DOMR row was missing or
     was a timeout, ZERO checks ran and the gate returned True. An empty section reads exactly like a passing
     section. Absence must not be indistinguishable from success.

  3. GATE 2 passed vacuously on an empty merge and on a missing sweep file. It even printed "the gate did not
     run; do not treat that as a pass" -- and then treated it as a pass.

  Composed, those three let a run in which EVERY repair algorithm timed out -- only the three `observed` rows
  returning -- print "All gates passed." and exit 0, having validated nothing.

  A fourth: report() was called BEFORE the exit check, so a rejected run still printed its table to stdout.
  Numbers a gate refused must not reach a log that someone will later grep.

Every gate now updates `ok`, every skip is a FAILURE, and nothing prints until the gates pass.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

N_TASKS = 48                # 3 planted specs x (1 observed + 15 algorithms)
TOL = 0.01                  # a "win" is a >1% relative gain; below that, best-of-N is selection noise
DRIFT = 1e-4                # a rebuilt disparity must reproduce its stored value to this
GRAPHS = ["dimacs_ny_d_inflate", "dimacs_ny_d_deflate", "dimacs_ny_d_mixed"]


def gates(df, sweep):
    """Every gate updates ok. Every skip is a FAILURE. A check that cannot fail is not a check."""
    ok = True

    if not os.path.exists(sweep):
        print(f"!! GATE FAILURE: {sweep} not found. The gates compare against it; without it nothing can be")
        print("   verified, and an unverifiable run is a failed run.")
        return False
    ref = pd.read_csv(sweep)
    ref = ref[ref.status.fillna("ok").eq("ok") & ref.disp_smacof.notna()]

    print("GATE 1 -- the `observed` rows must reproduce the stored sweep")
    print("   (expected values are READ FROM the CSV -- there is no hardcoded constant here, by design)")
    for g in GRAPHS:
        r = df[(df.graph == g) & (df.algo == "observed")]
        e = ref[(ref.graph == g) & (ref.algo == "observed")]
        if r.empty:
            print(f"  {g:<22} !! MISSING observed row -- FAIL"); ok = False; continue
        if e.empty:
            print(f"  {g:<22} !! no stored value to check against -- FAIL"); ok = False; continue
        got, want = float(r.disp.iloc[0]), float(e.disp_smacof.iloc[0])
        d = abs(got - want)
        good = d < DRIFT
        print(f"  {g:<22} {got:.6f} vs stored {want:.6f}   |delta| {d:.2e}   {'OK' if good else '!! DRIFT'}")
        ok = ok and good

    print("\nGATE 2 -- every algorithm's disparity must reproduce its stored value")
    j = df[df.disp.notna()].merge(ref[["graph", "algo", "disp_smacof"]], on=["graph", "algo"])
    n_expected = len(ref[ref.graph.isin(GRAPHS)])
    if j.empty:
        print("  !! ZERO rows matched the stored sweep. Either the graph/algo names drifted or nothing")
        print("     returned. A vacuous gate is a failed gate. -- FAIL")
        ok = False
    else:
        e = (j.disp - j.disp_smacof).abs()
        good = e.max() < DRIFT
        print(f"  {len(j)} rows matched (the stored sweep has {n_expected} for these graphs); "
              f"max |delta| = {e.max():.2e}   {'OK' if good else '!! DRIFT'}")
        ok = ok and good
        if len(j) < n_expected:
            print(f"  !! only {len(j)} of {n_expected} stored rows were reproduced. The rest did not return")
            print("     here. That is not necessarily wrong, but it is NOT a full verification. -- FAIL")
            ok = False

    print("\nGATE 3 -- DOMR must move NEITHER axis (Lemma 6.1)")
    for g in GRAPHS:
        o = df[(df.graph == g) & (df.algo == "observed")]
        d = df[(df.graph == g) & (df.algo == "domr")]
        if o.empty or d.empty:
            print(f"  {g:<22} !! no DOMR row -- the control did not run, so it CANNOT have passed. FAIL")
            ok = False; continue
        if pd.isna(d.disp.iloc[0]):
            print(f"  {g:<22} !! DOMR returned no cover (status={d.status.iloc[0]}). The control is the one")
            print(f"  {'':22}    algorithm that must always succeed. FAIL")
            ok = False; continue
        dd = abs(float(d.disp.iloc[0]) - float(o.disp.iloc[0]))
        dk = abs(float(d.knn10.iloc[0]) - float(o.knn10.iloc[0]))
        good = dd < 1e-9 and dk < 1e-9
        print(f"  {g:<22} |d disp| {dd:.2e}  |d knn10| {dk:.2e}   {'OK' if good else '!! VIOLATED'}")
        ok = ok and good

    return ok


def report(df, k):
    kc = f"knn{k}"
    v = lambda x: "WIN " if x > TOL else ("LOSE" if x < -TOL else "TIE ")  # noqa: E731
    print(f"\n{'=' * 104}")
    print(f"MATCHED TOPOLOGY AND GEOMETRY -- one cover, one repaired matrix (k = {k})")
    print(f"{'=' * 104}")
    for g in GRAPHS:
        G = df[df.graph == g]
        o = G[G.algo == "observed"]
        R = G[(~G.algo.isin(["observed", "domr"])) & G.disp.notna()]
        n_all = len(G[~G.algo.isin(["observed", "domr"])])
        if o.empty or R.empty:
            print(f"\n{g:<22} no usable rows"); continue
        ko, go, H = float(o[kc].iloc[0]), float(o.disp.iloc[0]), int(o.H.iloc[0])
        bt = R.sort_values(kc, ascending=False).iloc[0]
        bg = R.sort_values("disp").iloc[0]
        km, gm = float(R[kc].median()), float(R.disp.median())
        print(f"\n{g}   |H| = {H}   observed: kNN {ko:.4f}, disparity {go:.4f}")
        print(f"  topology  med {km:.4f} [{v((km - ko) / max(ko, 1e-9))}]   "
              f"best {float(bt[kc]):.4f} [{v((float(bt[kc]) - ko) / max(ko, 1e-9))}]  by {bt.algo}")
        print(f"  geometry  med {gm:.4f} [{v((go - gm) / go)}]   "
              f"best {float(bg.disp):.4f} [{v((go - float(bg.disp)) / go)}]  by {bg.algo}")
        # THE CONDITIONING MUST BE ON THE FACE OF THE NUMBER. `med` is a median over the algorithms that
        # RETURNED within the cap -- on the inflate/mixed specs the entire LP/ILP family times out, so the
        # median is taken over the cheap combinatorial methods and is biased by algorithm class, not sampled
        # at random. Quoting it as "the median algorithm" without this is a misstatement.
        print(f"  ^ med is over the {len(R)} of {n_all} algorithms that RETURNED within the cap"
              + ("" if len(R) == n_all else "  <-- conditioned on convergence; state this wherever it is quoted"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results_recovery")
    ap.add_argument("--out", default="analysis/summary_recovery.csv")
    ap.add_argument("--sweep", default="analysis/summary_mds_sweep.csv")
    ap.add_argument("-k", type=int, default=20)
    ap.add_argument("--force", action="store_true", help="print the table even if a gate failed (do not)")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.indir, "*.csv")))
    if not files:
        sys.exit(f"no CSVs in {a.indir}/ -- did the array run?")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"collected {len(files)}/{N_TASKS} task CSVs -> {a.out} ({len(df)} rows)\n")

    ok = True
    if len(files) < N_TASKS:
        print(f"!! GATE FAILURE: only {len(files)} of {N_TASKS} tasks produced a CSV. A task that TIMES OUT")
        print("   still writes a status=timeout row; a task SLURM KILLS writes nothing. Missing files mean")
        print("   the second kind. Check logs/ before trusting anything.\n")
        ok = False

    n_null = int((df.status != "ok").sum())
    if n_null:
        print(f"note: {n_null} row(s) returned no cover (timeout / not_converged). They are kept as explicit")
        print("      nulls and excluded from the medians -- never silently dropped.\n")

    ok = gates(df, a.sweep) and ok

    if not ok and not a.force:
        sys.exit("\n!! A GATE FAILED. Nothing is printed: numbers a gate rejected must not reach a log that\n"
                 "   someone will later grep. Explain the failure, or re-run with --force if you know why.")
    report(df, a.k)
    print("\nAll gates passed." if ok else "\n!! PRINTED UNDER --force WITH A FAILED GATE. Do not quote these.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
