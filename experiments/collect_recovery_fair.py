"""Collect the UNBIASED recovery re-run -> analysis/recovery_fair/, gated.

Input   results_recovery_fair/rec_*.csv   (one row per (graph, corruption, seed, algorithm))
Output  analysis/recovery_fair/rows.csv       every task row
        analysis/recovery_fair/summary.csv     per (graph, corruption, algorithm): median across the 5 seeds

The point of the array: a uniform 2h cap and one task per algorithm, so the median is no longer taken over the
cheap survivors of a 300s cap. The gates prove the run is trustworthy; G5 REPORTS survival per graph so the
prose can say how much the 2h cap bought over the old 300s one.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import recovery_fair_harness as rfh                                              # noqa: E402

KS = [5, 10, 20]
TOL = 1e-6                              # DOMR is restore(H): its repaired matrix IS the observed one, exactly


def load(indir):
    files = sorted(glob.glob(os.path.join(indir, "rec_*.csv")))
    if not files:
        raise SystemExit(f"FATAL: no rec_*.csv in {indir}/ -- did the array run?")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return d, files


def gate(d):
    fails = []

    def chk(c, name, obs):
        print(f"  [{'PASS' if c else 'FAIL'}] {name:<54} {obs}")
        if not c:
            fails.append(name)

    # G1  every task returned. The grid is the reference, not the CSV -- a short collect surfaces here.
    want = len(rfh.all_tasks())
    got = d.drop_duplicates(["graph", "corruption", "seed", "algo"])
    chk(len(got) == want, "G1 every task returned exactly one row", f"{len(got)} of {want}")

    # G2  schema: the columns the summary and gates need are present.
    need = {"graph", "corruption", "seed", "algo", "status", "disp", "cover_size", "n", "H", "cap_s"} \
        | {f"knn{k}" for k in KS}
    missing = need - set(d.columns)
    chk(not missing, "G2 schema has the needed columns", "ok" if not missing else f"MISSING {sorted(missing)}")

    # G3  THE DOMR CONTROL (Lemma). DOMR's cover is the heavy set; restore(H) leaves every shortest path
    #     unchanged, so its repaired matrix IS the observed one and its disparity/kNN must equal `observed`'s
    #     to float precision. A nonzero reading is a bug in the build/score chain, not a result.
    piv = d[d.status.eq("ok")].pivot_table(index=["graph", "corruption", "seed"], columns="algo",
                                           values="disp", aggfunc="first")
    if "domr" in piv.columns and "observed" in piv.columns:
        dd = (piv["domr"] - piv["observed"]).abs()
        pk = d[d.status.eq("ok")].pivot_table(index=["graph", "corruption", "seed"], columns="algo",
                                              values="knn20", aggfunc="first")
        dk = (pk["domr"] - pk["observed"]).abs() if "domr" in pk and "observed" in pk else pd.Series([np.nan])
        chk(dd.max() <= TOL and dk.max() <= TOL, "G3 DOMR control: disp & kNN == observed (Lemma)",
            f"max |domr - observed|: disp {dd.max():.1e}, knn20 {dk.max():.1e}")
    else:
        chk(False, "G3 DOMR control present", "domr or observed row missing")

    # G4  ranges. kNN Jaccard >= 0, disparity >= 0, precision/recall in [0,1] where present.
    o = d[d.status.eq("ok")]
    bad = int((o.disp < 0).sum() + sum((o[f"knn{k}"] < 0).sum() for k in KS)
              + ((o.precision < 0) | (o.precision > 1)).sum() + ((o.recall < 0) | (o.recall > 1)).sum())
    chk(bad == 0, "G4 disp>=0, kNN>=0, precision/recall in [0,1]", f"{bad} out-of-range cells")

    # G5  SURVIVAL, per graph x corruption -- a DISCLOSURE, not a failure. With the 2h cap this is the whole
    #     point: how many of the suite now return, vs the biased 300s run. Reported, never hidden.
    print("\n  survival per graph x corruption (algorithms returning ok, of "
          f"{len(rfh.SWEEP_ALGOS)}), across {len(rfh.SEEDS)} seeds:")
    alg = d[d.algo.ne("observed")]
    for (g, c), sub in alg.groupby(["graph", "corruption"]):
        ret = sub[sub.status.eq("ok")].algo.nunique()
        tmo = sorted(sub[sub.status.ne("ok")].algo.unique())
        print(f"     {g:12s} {c:8s}: {ret}/{len(rfh.SWEEP_ALGOS)} returned"
              + (f"; timed out/failed: {tmo}" if tmo else ""))
    return fails


def summarize(d):
    """Per (graph, corruption, algorithm): median disp and kNN across the seeds, and how many seeds it survived."""
    ok = d[d.status.eq("ok")]
    agg = {"disp": "median", "cover_size": "median", **{f"knn{k}": "median" for k in KS}}
    g = ok.groupby(["graph", "corruption", "algo"]).agg(agg)
    g["n_seeds_ok"] = ok.groupby(["graph", "corruption", "algo"]).seed.nunique()
    g["n_seeds"] = len(rfh.SEEDS)
    return g.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results_recovery_fair")
    ap.add_argument("--outdir", default="analysis/recovery_fair")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    d, files = load(a.indir)
    print(f"collected {len(files)} task CSVs -> {len(d)} rows\n")
    print("GATE -- nothing is written until these pass")
    fails = gate(d)
    if fails and not a.force:
        raise SystemExit(f"\n*** {len(fails)} GATE FAILURE(S): {fails}. NOTHING WRITTEN. ***")

    os.makedirs(a.outdir, exist_ok=True)
    d.to_csv(os.path.join(a.outdir, "rows.csv"), index=False)
    summarize(d).to_csv(os.path.join(a.outdir, "summary.csv"), index=False)
    print(f"\n  wrote {a.outdir}/rows.csv and summary.csv")
    print("\nAll gates passed." if not fails else "\n!! WRITTEN UNDER --force.")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
