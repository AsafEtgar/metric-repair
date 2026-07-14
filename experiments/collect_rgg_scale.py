"""Collect the RGG SCALE array, RUN THE GATES, and print what the section will quote.

THE GATES FAIL CLOSED. Nothing is printed until they pass.

  G1  COMPLETENESS. All 2,100 task CSVs, or the missing indices by name.
  G2  SCHEMA. Column-identical to rgg_harness.RGG_CSV_FIELDS, so these rows concatenate with the published
      RGG grids rather than living in a parallel universe.
  G3  THE DOMR IDENTITY. domr's cover IS the heavy set: size == H, exactly, on every task. One line that
      tests the generator, the corruption, the component decomposition and the CSV writer against a value
      theory fixes in advance.
  G4  *** skipped_time == 0. THE GATE THIS FILE EXISTS FOR. ***
      The section wants to claim that algorithms hit their limits even on a sparse family. `_run` marks
      algorithms `skipped_time` when the per-TASK budget expires -- walking build_suite_rgg in order, which
      ends: ... spc_gmr, spc_iomr, PIVOT, LEFT_EDGE. Those two are precisely the methods whose limitation the
      section exists to demonstrate. If the budget ever fires, their failure is a QUEUE POSITION, not an
      algorithmic fact, and the claim is unfalsifiable. The array runs with a 9 h budget against an 8 h
      ceiling so this cannot happen -- and this gate proves it did not, every run, rather than trusting the
      arithmetic.
  G5  NO JITTER. P2size must appear nowhere. Metric graphs cannot benchmark a repair.
  G6  BOTH DIRECTIONS AT EVERY n. The inversion is the section's punchline; a ladder with a hole in one
      direction cannot show it.

  input   results_rgg_scale/task_*.csv
  output  analysis/rgg_scale/rows.csv, summary.csv

  usage   sage -python experiments/collect_rgg_scale.py
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rgg_scale_harness as rs                                          # noqa: E402
from rgg_harness import RGG_CSV_FIELDS                                  # noqa: E402


def load(indir):
    files = sorted(glob.glob(os.path.join(indir, "task_*.csv")))
    if not files:
        raise SystemExit(f"FATAL: no task CSVs in {indir}/ -- did the array run?")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # trap 1: the RGG CSVs are LONG (one row per knn_k). Collapse for every COVER-level statistic.
    for c in ("size", "valid", "H", "E", "n", "wall", "peak_mb", "frac_q", "magnitude", "deg"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d, files


def gate(d, files):
    fails = []

    def chk(ok, name, obs):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<50} {obs}")
        if not ok:
            fails.append(name)

    uni = d.drop_duplicates(subset=["task", "algo"], keep="first")

    want = len(rs.all_tasks())
    got = sorted(uni.task.unique())
    missing = sorted(set(range(want)) - set(got))
    chk(not missing, "G1 every task returned a CSV",
        f"{len(got)} of {want}" + (f"; MISSING {missing[:6]}..." if missing else ""))

    extra = [c for c in d.columns if c not in RGG_CSV_FIELDS]
    absent = [c for c in RGG_CSV_FIELDS if c not in d.columns]
    chk(not extra and not absent, "G2 schema identical to rgg_harness.RGG_CSV_FIELDS",
        f"{len(d.columns)} cols" + (f"; extra={extra} missing={absent}" if (extra or absent) else ""))

    D = uni[uni.algo.eq("domr") & uni.status.eq("ok")].dropna(subset=["size", "H"])
    bad = int((D["size"] != D.H).sum()) if len(D) else -1
    chk(bad == 0, "G3 domr's cover IS the heavy set (size == H)",
        f"{len(D)} tasks, {bad} mismatches" if len(D) else "no ok domr rows -- the control did not run")

    # *** G4 ***
    st = uni[uni.status.eq("skipped_time")]
    chk(len(st) == 0, "G4 skipped_time == 0 (the BUDGET never fired)",
        f"{len(st)} rows" + (f" -- {st.algo.value_counts().head(3).to_dict()}" if len(st) else ""))
    if len(st):
        print("         *** THE TASK BUDGET FIRED. `_run` skips the algorithms LAST in build_suite_rgg")
        print("         *** order -- which ends with pivot and left_edge, the two whose limitation this")
        print("         *** section exists to demonstrate. Their failure is now a QUEUE POSITION, not an")
        print("         *** algorithmic fact. Raise BUDGET_S and re-run those tasks. Do NOT report this.")

    chk("P2size" not in set(uni.sweep.dropna()), "G5 no jitter sweep",
        f"sweeps: {sorted(set(uni.sweep.dropna()))}")

    lad = uni[uni.sweep.isin(["S1", "S1d"])]
    holes = []
    for n in rs.NS:
        for sw in ("S1", "S1d"):
            if lad[lad.sweep.eq(sw) & lad.n.eq(n)].empty:
                holes.append((sw, n))
    chk(not holes, "G6 both directions present at every n", f"{len(rs.NS)} rungs" +
        (f"; HOLES {holes[:4]}" if holes else ""))
    return fails


def report(d):
    uni = d.drop_duplicates(subset=["task", "algo"], keep="first")
    ok = uni[uni.status.eq("ok") & uni.valid.eq(1)].copy()
    ok["sm"] = ok["size"] / ok.E
    lad = uni[uni.sweep.isin(["S1", "S1d"])]

    print(f"\n{'=' * 92}")
    print("SCALE: WHO SURVIVES, AND WHAT IT COSTS  (the n-ladder, inflate + deflate)")
    print(f"{'=' * 92}")
    t = lad.drop_duplicates("task")
    print(f"  {'n':>6}{'m':>9}{'completion':>13}{'blowup':>9}")
    for n in (rs.NS[0], rs.NS[len(rs.NS) // 2], rs.NS[-1]):
        m = int(t[t.n.eq(n)].E.median())
        comp = n * (n - 1) // 2
        print(f"  {n:>6}{m:>9,}{comp:>13,}{comp / m:>8.0f}x")
    print("  (pivot and left_edge COMPLETE the graph -- they pay the right column, not the left)")

    print(f"\n  {'algo':<18}{'returns at n=' + str(rs.NS[-1]):>18}{'timeout':>10}{'peak MB':>10}")
    print("  " + "-" * 58)
    top = lad[lad.n.eq(rs.NS[-1])]
    ntop = top.task.nunique()
    for al in sorted(top.algo.unique()):
        A = top[top.algo.eq(al)]
        r = 100 * len(A[A.status.eq("ok") & A.valid.eq(1)]) / max(ntop, 1)
        to = 100 * A.status.eq("timeout").mean()
        mb = A.peak_mb.median()
        print(f"  {al:<18}{r:>17.0f}%{to:>9.0f}%{mb:>10.0f}")

    print(f"\n{'=' * 92}")
    print("THE CORRUPTION DECIDES: |S|/m by direction, at the top of the ladder")
    print(f"{'=' * 92}")
    print(f"  {'algo':<18}{'inflate':>10}{'deflate':>10}")
    print("  " + "-" * 40)
    for al in sorted(ok[ok.sweep.isin(["S1", "S1d"])].algo.unique()):
        if al == "domr":
            continue
        row = ""
        for sw in ("S1", "S1d"):
            K = ok[ok.algo.eq(al) & ok.sweep.eq(sw)]
            row += f"{K.sm.median():>10.3f}" if len(K) else f"{'--':>10}"
        print(f"  {al:<18}{row}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results_rgg_scale")
    ap.add_argument("--outdir", default="analysis/rgg_scale")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    d, files = load(a.indir)
    print(f"collected {len(files)} task CSVs -> {len(d)} rows\n")
    print("GATE -- nothing is reported until these pass")
    fails = gate(d, files)
    if fails and not a.force:
        raise SystemExit(f"\n*** {len(fails)} GATE FAILURE(S): {fails}. Nothing printed. ***")

    os.makedirs(a.outdir, exist_ok=True)
    d.to_csv(os.path.join(a.outdir, "rows.csv"), index=False)
    uni = d.drop_duplicates(subset=["task", "algo"], keep="first").copy()
    uni["sm"] = uni["size"] / uni.E
    uni.groupby(["sweep", "n", "algo"]).agg(
        n_ok=("task", "nunique"), sm_med=("sm", "median"),
        wall_med=("wall", "median"), peak_mb_med=("peak_mb", "median"),
    ).reset_index().to_csv(os.path.join(a.outdir, "summary.csv"), index=False)
    print(f"\n  wrote {a.outdir}/rows.csv and {a.outdir}/summary.csv")

    report(d)
    print("\nAll gates passed." if not fails else "\n!! PRINTED UNDER --force. Do not quote these.")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
