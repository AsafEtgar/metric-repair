"""Collect the oracle-weights array, RUN THE GATES, and print the table it exists to produce.

THE GATES FAIL CLOSED. Every skip is a failure; a check that cannot fail is not a check.

  GATE 1 -- DOMR under `restore` must reproduce `observed` EXACTLY (Lemma: a decrease-only cover changes no
            shortest path). This is the pipeline's built-in control, and it is free.
  GATE 2 -- every graph must carry both reference rows (observed, all_oracle). Without the ceiling there is
            no denominator and the headline number cannot be computed.
  GATE 3 -- the `observed` k-NN must reproduce analysis/summary_pure_real.csv, where that file has the graph.
            That proves this pipeline and the existing one are measuring the same thing.

THE HEADLINE. For each cover we report what fraction of the AVAILABLE gain it captures when handed the TRUE
weights:

    captured = (disp_observed - disp_oracle) / (disp_observed - disp_all_oracle)

The denominator is the ceiling: what one gets by setting EVERY edge to its true distance. If metric repair
selected the edges where the error lives, a cover would capture far more than its share of the edges. It does
not. It captures about |S|/m -- its proportional share -- which is what a UNIFORM error implies, and which
means the heavy set is not the wrong set.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

GRAPHS = ["nmr_1d3z_residue", "nmr_1d3z_atom", "dimacs_ny_t", "pbmc3k_cosine_knn", "ripe_atlas"]


def gates(df, pure="analysis/summary_pure_real.csv", k=20):
    ok = True
    print("GATE 1 -- DOMR under `restore` must reproduce `observed` exactly (decrease-only invariance)")
    for g in sorted(df.graph.unique()):
        o = df[(df.graph == g) & (df.arm == "observed")]
        d = df[(df.graph == g) & (df.algo == "domr") & (df.arm == "restore")]
        if o.empty or d.empty:
            print(f"  {g:<20} !! no DOMR/observed row -- the control did not run, so it CANNOT have passed. FAIL")
            ok = False; continue
        dd = abs(float(d.disp.iloc[0]) - float(o.disp.iloc[0]))
        dk = abs(float(d[f"knn{k}"].iloc[0]) - float(o[f"knn{k}"].iloc[0]))
        good = dd < 1e-9 and dk < 1e-9
        print(f"  {g:<20} |d disp| {dd:.2e}  |d knn| {dk:.2e}   {'OK' if good else '!! VIOLATED'}")
        ok = ok and good

    print("\nGATE 2 -- every graph needs BOTH reference rows (observed, all_oracle)")
    for g in sorted(df.graph.unique()):
        have = set(df[df.graph == g].arm)
        miss = {"observed", "all_oracle"} - have
        print(f"  {g:<20} {'OK' if not miss else '!! MISSING ' + ','.join(sorted(miss)) + ' -- no ceiling, no denominator. FAIL'}")
        ok = ok and not miss

    print("\nGATE 3 -- `observed` k-NN must reproduce the existing pipeline")
    if os.path.exists(pure):
        P = pd.read_csv(pure); P = P[P.k == k]
        n = 0
        for g in sorted(df.graph.unique()):
            o = df[(df.graph == g) & (df.arm == "observed")]
            p = P[P.graph == g]
            if o.empty or p.empty:
                continue
            # Compare on the OLD pipeline's node set (all gt nodes), not our matched finite core -- otherwise
            # the gate fails on a design choice rather than on a defect. We print both, so the difference is
            # on the face of the output.
            col = f"knn{k}_gtset" if f"knn{k}_gtset" in o.columns and pd.notna(o[f"knn{k}_gtset"].iloc[0]) \
                else f"knn{k}"
            e = abs(float(o[col].iloc[0]) - float(p.recovery_obs.iloc[0]))
            good = e < 1e-6
            extra = ""
            if col.endswith("_gtset") and abs(float(o[f"knn{k}"].iloc[0]) - float(o[col].iloc[0])) > 1e-9:
                extra = (f"   [we report {float(o[f'knn{k}'].iloc[0]):.6f} over our {int(o.n_core.iloc[0])}-node "
                         f"matched core; the {int(o.n_gtset.iloc[0])}-node gt set is the gate reference]")
            print(f"  {g:<20} {float(o[col].iloc[0]):.6f} vs stored {float(p.recovery_obs.iloc[0]):.6f}   "
                  f"|delta| {e:.2e}   {'OK' if good else '!! DRIFT'}{extra}")
            ok = ok and good; n += 1
        if n == 0:
            print("  !! no overlap with the stored pipeline -- a vacuous gate is a failed gate. FAIL"); ok = False
    else:
        print(f"  !! {pure} not found -- CANNOT verify. FAIL"); ok = False
    return ok


def report(df, k=20):
    kc = f"knn{k}"
    print(f"\n{'=' * 108}")
    print("GIVEN THE TRUE WEIGHTS, HOW MUCH OF THE AVAILABLE GAIN DOES A METRIC-REPAIR COVER CAPTURE?")
    print(f"{'=' * 108}")
    for g in GRAPHS:
        G = df[df.graph == g]
        if G.empty:
            continue
        o = G[G.arm == "observed"]
        c = G[G.arm == "all_oracle"]
        if o.empty or c.empty:
            print(f"\n{g}: missing a reference row"); continue
        do, dc = float(o.disp.iloc[0]), float(c.disp.iloc[0])
        ko, kcl = float(o[kc].iloc[0]), float(c[kc].iloc[0])
        m = int(o.m.iloc[0]); H = int(o.H.iloc[0])
        gain = do - dc
        print(f"\n{g}   m = {m:,}   |H| = {H:,} ({100.0*H/m:.2f}% of edges)".replace(",", "{,}").replace("{,}", ","))
        print(f"  observed disparity {do:.4f}   ceiling (EVERY edge -> d*) {dc:.4f}   "
              f"=> {100*gain/do:.0f}% of the geometry is recoverable by reweighting")
        print(f"  {'algorithm':<16}{'|S|':>7}{'|S|/m':>8}{'restore':>10}{'ORACLE':>9}"
              f"{'captured':>10}{'  (predicted if the error were uniform)'}")
        print("  " + "-" * 100)
        R = G[G.arm.isin(["restore", "oracle"])]
        for a in sorted(R.algo.unique()):
            r = R[(R.algo == a) & (R.arm == "restore")]
            q = R[(R.algo == a) & (R.arm == "oracle")]
            if r.empty or q.empty:
                continue
            s = float(r.cover_size.median())
            d1 = float(r.disp.median()); d2 = float(q.disp.median())
            cap = (do - d2) / gain if abs(gain) > 1e-12 else np.nan
            print(f"  {a:<16}{int(s):>7}{s/m:>8.3f}{d1:>10.4f}{d2:>9.4f}{100*cap:>9.1f}%"
                  f"{100*s/m:>11.1f}%")
        print(f"  {'':16}{'':>7}{'':>8}{'':>10}{'':>9}{'^ what it got':>10}{'^ its share of the edges':>26}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results_oracle")
    ap.add_argument("--out", default="analysis/summary_oracle.csv")
    ap.add_argument("-k", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.indir, "*.csv")))
    if not files:
        sys.exit(f"no CSVs in {a.indir}/ -- did the array run?")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"collected {len(files)} task CSVs -> {a.out} ({len(df)} rows)\n")

    ok = gates(df, k=a.k)
    if not ok and not a.force:
        sys.exit("\n!! A GATE FAILED. Nothing is printed. Explain it, or re-run with --force if you know why.")
    report(df, a.k)
    print("\nAll gates passed." if ok else "\n!! PRINTED UNDER --force. Do not quote these.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
