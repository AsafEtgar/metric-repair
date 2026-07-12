"""Concatenate the recovery array's per-task CSVs, RUN THE GATES, and print the three table rows it exists
to fill.

THE GATES COME FIRST, and a failure is fatal rather than a warning. This array rebuilds the planted instance
from scratch; if that rebuild drifted by so much as a seed draw, every number in it describes a different
graph than the paper's figures do. So before anything is printed we check that the rebuilt disparities
reproduce the ones already stored in analysis/summary_mds_sweep.csv, and that DOMR moves neither axis.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

# The stored disparities of the three `observed` rows (analysis/summary_mds_sweep.csv, disp_smacof).
GATE = {"dimacs_ny_d_inflate": 0.364054, "dimacs_ny_d_deflate": 0.028410, "dimacs_ny_d_mixed": 0.151216}
TOL = 0.01          # a "win" is a >1% relative gain; below that, best-of-15 is selection noise


def gates(df, sweep):
    ok = True
    print("GATE 1 -- the `observed` rows must reproduce the stored sweep")
    for g, want in GATE.items():
        r = df[(df.graph == g) & (df.algo == "observed")]
        if r.empty:
            print(f"  {g:<22} MISSING observed row"); ok = False; continue
        got = float(r.disp.iloc[0]); d = abs(got - want)
        print(f"  {g:<22} {got:.6f} vs stored {want:.6f}   |delta| {d:.2e}   "
              f"{'OK' if d < 1e-4 else '!! DRIFT'}")
        ok = ok and d < 1e-4

    print("\nGATE 2 -- every algorithm's disparity must reproduce its stored value")
    if os.path.exists(sweep):
        ref = pd.read_csv(sweep)
        ref = ref[ref.status.fillna("ok").eq("ok")][["graph", "algo", "disp_smacof"]]
        j = df[df.disp.notna()].merge(ref, on=["graph", "algo"])
        if len(j):
            e = (j.disp - j.disp_smacof).abs()
            print(f"  {len(j)} matched; max |delta| = {e.max():.2e}   "
                  f"{'OK' if e.max() < 1e-4 else '!! DRIFT'}")
            ok = ok and e.max() < 1e-4
        else:
            print("  no overlap to check")
    else:
        print(f"  {sweep} not found -- SKIPPED (the gate did not run; do not treat that as a pass)")

    print("\nGATE 3 -- DOMR must move NEITHER axis (Lemma 6.1)")
    for g, G in df.groupby("graph"):
        o, d = G[G.algo == "observed"], G[G.algo == "domr"]
        if len(o) and len(d) and pd.notna(d.disp.iloc[0]):
            dd = abs(float(d.disp.iloc[0]) - float(o.disp.iloc[0]))
            dk = abs(float(d.knn10.iloc[0]) - float(o.knn10.iloc[0]))
            good = dd < 1e-9 and dk < 1e-9
            print(f"  {g:<22} |d disp| {dd:.2e}  |d knn10| {dk:.2e}   {'OK' if good else '!! VIOLATED'}")
            ok = ok and good
    return ok


def report(df, k):
    kc = f"knn{k}"
    v = lambda x: "WIN " if x > TOL else ("LOSE" if x < -TOL else "TIE ")  # noqa: E731
    print(f"\n{'=' * 108}")
    print(f"THE THREE MISSING ROWS -- matched topology and geometry, one cover, one repaired matrix (k={k})")
    print(f"{'=' * 108}")
    print(f"{'graph':<22}{'|H|':>6}{'kNN obs':>9}{'med':>8}{'best':>8}{'  by':<15}"
          f"{'disp obs':>9}{'med':>8}{'best':>8}{'  by':<15}")
    print("-" * 108)
    for g, G in df.groupby("graph", sort=False):
        o = G[G.algo == "observed"]
        R = G[(~G.algo.isin(["observed", "domr"])) & G.disp.notna()]
        if o.empty or R.empty:
            print(f"{g:<22}  no usable rows"); continue
        ko, go, H = float(o[kc].iloc[0]), float(o.disp.iloc[0]), int(o.H.iloc[0])
        bt = R.sort_values(kc, ascending=False).iloc[0]
        bg = R.sort_values("disp").iloc[0]
        km, gm = float(R[kc].median()), float(R.disp.median())
        print(f"{g:<22}{H:>6}{ko:>9.4f}{km:>8.4f}{float(bt[kc]):>8.4f}  {bt.algo:<13}"
              f"{go:>9.4f}{gm:>8.4f}{float(bg.disp):>8.4f}  {bg.algo:<13}")
        print(f"{'':22}{'':>6}{'':9}{'  ' + v((km - ko) / max(ko, 1e-9)):>8}"
              f"{'  ' + v((float(bt[kc]) - ko) / max(ko, 1e-9)):>8}  {'(med / best)':<13}"
              f"{'':9}{'  ' + v((go - gm) / go):>8}{'  ' + v((go - float(bg.disp)) / go):>8}  "
              f"{'(med / best)':<13}")
    print("\n  med = the median algorithm, which is what one gets WITHOUT an oracle; choosing `best` requires")
    print("  the ground truth one is trying to recover. Report med; quote best as the ceiling.")

    n_null = int((df.status != "ok").sum())
    if n_null:
        print(f"\n  {n_null} row(s) returned no cover (timeout / not_converged). They are kept as explicit")
        print("  nulls and excluded from the medians -- never silently dropped.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results_recovery")
    ap.add_argument("--out", default="analysis/summary_recovery.csv")
    ap.add_argument("--sweep", default="analysis/summary_mds_sweep.csv")
    ap.add_argument("-k", type=int, default=20, help="k for the k-NN column (default 20)")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.indir, "*.csv")))
    if not files:
        sys.exit(f"no CSVs in {a.indir}/ -- did the array run?")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"collected {len(files)}/48 task CSVs -> {a.out} ({len(df)} rows)\n")
    if len(files) < 48:
        print(f"!! only {len(files)} of 48 tasks produced a CSV -- check logs/ before trusting the medians\n")

    ok = gates(df, a.sweep)
    report(df, a.k)
    if not ok:
        sys.exit("\n!! A GATE FAILED. Do not use these numbers until it is explained.")
    print("\nAll gates passed.")


if __name__ == "__main__":
    main()
