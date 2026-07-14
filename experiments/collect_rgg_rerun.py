"""Collect the RGG re-run (small + large), gate it, and prove the corruption is the one that was asked for.

THE GATES FAIL CLOSED. Nothing is reported until they pass.

  G1  COMPLETENESS. Every task returned a CSV, per grid, or the missing indices by name.
  G2  SCHEMA. Column-identical to rgg_harness.RGG_CSV_FIELDS, so these rows are a drop-in replacement for
      the published ones and rgg_analyze.py can eat them unchanged.
  G3  NO JITTER. The author asked for planted corruptions only. If a jitter row is here, the wrong grid ran.
  G4  *** THE MAGNITUDE KNOB IS LIVE -- THE ENTIRE REASON THIS CAMPAIGN EXISTS. *** Under the old code every
      inflation landed at ~11.8x the detour whatever magnitude was asked for, so |H| and the broken-cycle
      count were FLAT across the S3 magnitude sweep. If they are still flat, the fix did not reach the
      cluster (a stale checkout, a cached module) and the whole re-run is worthless. This CANNOT pass
      vacuously: it demands a monotone response to the knob, read from the delivered rows.
  G5  skipped_time == 0. The task budget must never have fired -- it did not on the published grids
      (0 of 175,218 and 0 of 32,582), and the fixed corruption is WEAKER, so it must not start now.
  G6  DOMR's cover IS the heavy set. size(domr) == H, exactly, on every task. One line, two pipelines.
  G7  H == B UNDER INFLATION -- WHERE IT HOLDS, AND WHERE IT DOES NOT. This is no longer a grid-wide identity
      and the collector must not pretend it is. A weak inflation does not survive its own interference: the
      detour is measured in G but heaviness is judged in H, so if e's detour runs through another inflated
      edge, that detour grew and a low-magnitude inflation no longer clears it. REPORTED per magnitude, and
      gated only where the paper relies on it (the baseline, magnitude 3.0).

  input   results_rgg_rerun_small/task_*.csv, results_rgg_rerun_large/task_*.csv
  output  analysis/rgg_rerun/{small,large}_rows.csv

  usage   sage -python experiments/collect_rgg_rerun.py
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rgg_rerun_harness as rr                                          # noqa: E402
from rgg_harness import RGG_CSV_FIELDS                                  # noqa: E402

BASE_MAG = 3.0          # the baseline the paper's direction comparison is pinned to (section5.matched)


def load(indir):
    files = sorted(glob.glob(os.path.join(indir, "task_*.csv")))
    if not files:
        raise SystemExit(f"FATAL: no task CSVs in {indir}/ -- did the array run?")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for c in ("n", "E", "H", "size", "valid", "magnitude", "frac_q", "deg", "n_corrupted", "cuts"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.drop_duplicates(subset=["task", "algo"], keep="first")        # the CSVs are long (one row per knn_k)
    return d, files


def ok(d):
    return d[d.status.eq("ok") & d.valid.eq(1)]


def gate(D):
    fails = []

    def chk(c, name, obs):
        print(f"  [{'PASS' if c else 'FAIL'}] {name:<54} {obs}")
        if not c:
            fails.append(name)

    for g, d in D.items():
        want = len(rr.all_tasks(g))
        got = sorted(d.task.unique())
        missing = sorted(set(range(want)) - set(got))
        chk(not missing, f"G1 every task returned a CSV ({g})",
            f"{len(got):,} of {want:,}" + (f"; MISSING {missing[:6]}..." if missing else ""))

    for g, d in D.items():
        extra = [c for c in d.columns if c not in RGG_CSV_FIELDS]
        absent = [c for c in RGG_CSV_FIELDS if c not in d.columns]
        chk(not extra and not absent, f"G2 schema == RGG_CSV_FIELDS ({g})",
            f"{len(d.columns)} cols" + (f"; extra={extra} missing={absent}" if (extra or absent) else ""))

    allrows = pd.concat(D.values(), ignore_index=True)
    jit = allrows[~allrows.break_type.eq("reweight")]
    chk(len(jit) == 0, "G3 planted corruptions only (no jitter)",
        f"{len(jit)} jitter rows" + (f" -- {sorted(set(jit.sweep))}" if len(jit) else ""))

    # G4  *** THE KNOB. *** S3 sweeps magnitude under inflation at a fixed n, deg and frac. Under the old code
    #     every rung planted an ~11.8x inflation, so |H| and the cut count were FLAT. A live knob must make
    #     BOTH rise with the magnitude. Read from the rows; no constant is assumed.
    S = D.get("small")
    s3 = S[S.sweep.eq("S3")].drop_duplicates("task") if S is not None else pd.DataFrame()
    if len(s3) == 0:
        chk(False, "G4 the inflate MAGNITUDE knob is live", "no S3 rows -- cannot verify the fix landed")
    else:
        h = s3.groupby("magnitude").H.median().sort_index()
        cut = (ok(S)[ok(S).sweep.eq("S3") & ok(S).algo.eq("l1sep_gmr")]
               .groupby("magnitude").cuts.median().sort_index())
        rise_h = h.iloc[-1] / max(h.iloc[0], 1)
        rise_c = (cut.iloc[-1] / max(cut.iloc[0], 1)) if len(cut) > 1 else float("nan")
        mono = bool((h.diff().dropna() >= 0).all())
        chk(rise_h > 1.5 and mono, "G4 the inflate MAGNITUDE knob is LIVE (|H| responds)",
            f"|H| {h.iloc[0]:.0f} -> {h.iloc[-1]:.0f} ({rise_h:.2f}x) over mu {h.index[0]}..{h.index[-1]}; "
            f"cuts {rise_c:.2f}x; monotone={mono}")
        if not (rise_h > 1.5 and mono):
            print("         *** |H| IS FLAT ACROSS THE MAGNITUDE SWEEP. That is the OLD bug's signature: the")
            print("         *** '+1' floor pinned every inflation to ~11.8x the detour. The cluster ran a")
            print("         *** STALE graph_models.py. This campaign is worthless -- do not analyse it.")

    for g, d in D.items():
        st = d[d.status.eq("skipped_time")]
        chk(len(st) == 0, f"G5 skipped_time == 0, the budget never fired ({g})",
            f"{len(st)} rows" + (f" -- {st.algo.value_counts().head(3).to_dict()}" if len(st) else ""))

    for g, d in D.items():
        Dm = ok(d)[ok(d).algo.eq("domr")].dropna(subset=["size", "H"])
        bad = int((Dm["size"] != Dm.H).sum()) if len(Dm) else -1
        chk(len(Dm) > 0 and bad == 0, f"G6 domr's cover IS the heavy set ({g})",
            f"{len(Dm):,} tasks, {bad} mismatches")

    # G7  H == B, per magnitude. Gated at the BASELINE only -- that is the population section5.matched() uses
    #     and the only one the paper's |S|/OPT depends on. Elsewhere it is REPORTED.
    T = allrows.drop_duplicates("task")
    inf = T[T.direction.eq("inflate")].dropna(subset=["H", "n_corrupted"])
    per = inf.groupby("magnitude").apply(
        lambda g: pd.Series({"tasks": len(g), "H_eq_B": float((g.H == g.n_corrupted).mean())}),
        include_groups=False).sort_index()
    base = per.loc[BASE_MAG, "H_eq_B"] if BASE_MAG in per.index else float("nan")
    chk(base == 1.0, f"G7 inflate: H == B at the BASELINE (magnitude {BASE_MAG})",
        f"{base:.3f} of tasks" if np.isfinite(base) else "no baseline rows")
    print("\n  H == B under inflation, per magnitude (it is NOT a grid-wide identity any more):")
    print(f"    {'magnitude':>10}{'tasks':>8}{'H == B':>10}")
    for m, r in per.iterrows():
        flag = "   <-- a weak inflation does not survive its own interference" if r.H_eq_B < 1.0 else ""
        print(f"    {m:>10}{int(r.tasks):>8}{r.H_eq_B:>10.3f}{flag}")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="analysis/rgg_rerun")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    D = {}
    for g in ("small", "large"):
        d, files = load(f"results_rgg_rerun_{g}")
        print(f"{g}: {len(files):,} task CSVs -> {len(d):,} cover rows")
        D[g] = d
    print("\nGATE -- nothing is written until these pass")
    fails = gate(D)
    if fails and not a.force:
        raise SystemExit(f"\n*** {len(fails)} GATE FAILURE(S): {fails}. NOTHING WRITTEN. ***")

    os.makedirs(a.outdir, exist_ok=True)
    for g, d in D.items():
        p = os.path.join(a.outdir, f"{g}_rows.csv")
        d.to_csv(p, index=False)
        print(f"\n  wrote {p}")
    print("\nNext:")
    print("  sage -python experiments/rgg_analyze.py --results results_rgg_rerun_small --outdir analysis/rgg")
    print("  sage -python experiments/rgg_analyze.py --results results_rgg_rerun_large "
          "--outdir analysis/rgg_large")
    print("  sage -python experiments/section5.py --texdir \"$PAPER/tables\" --figdir \"$PAPER/figures\"")
    print("\nAll gates passed." if not fails else "\n!! WRITTEN UNDER --force.")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
