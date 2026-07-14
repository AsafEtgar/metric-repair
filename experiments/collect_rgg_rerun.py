"""Collect the RGG re-run (small + large), gate it, and prove the corruption is the one that was asked for.

THE GATES FAIL CLOSED. Nothing is reported until they pass.

  G1  COMPLETENESS. Every task returned a CSV, per grid, or the missing indices by name.
  G2  SCHEMA. Column-identical to rgg_harness.RGG_CSV_FIELDS, so these rows are a drop-in replacement for
      the published ones and rgg_analyze.py can eat them unchanged.
  G3  NO JITTER. The author asked for planted corruptions only. If a jitter row is here, the wrong grid ran.
  G4  *** THE MAGNITUDE KNOB IS LIVE -- THE ENTIRE REASON THIS CAMPAIGN EXISTS. *** Under the old code every
      inflation landed at ~11.8x the detour whatever magnitude was asked for. Checked on the two statistics
      that CAN see that, the violated-cycle count and the H == B fraction -- and explicitly NOT on |H|, which
      cannot: the heavy set is a subset of the planted set and |B| is fixed by frac_q, so |H| saturates at
      |B| and is nearly flat in mu even when the knob works perfectly. The first version of this gate checked
      |H|, failed on healthy data, and accused the cluster of running a stale checkout. It was wrong.
  G5  skipped_time == 0. The task budget must never have fired -- it did not on the published grids
      (0 of 175,218 and 0 of 32,582), and the fixed corruption is WEAKER, so it must not start now.
  G6  DOMR's cover IS the heavy set. size(domr) == H, exactly, on every task. One line, two pipelines.
  G7  H == B UNDER INFLATION -- WHERE IT HOLDS, AND WHERE IT DOES NOT. This is no longer a grid-wide identity
      and the collector must not pretend it is. A weak inflation does not survive its own interference: the
      detour is measured in G but heaviness is judged in H, so if e's detour runs through another inflated
      edge, that detour grew and a low-magnitude inflation no longer clears it. It therefore depends on BOTH
      knobs -- the magnitude AND the fraction, since more planted edges means more interference -- and it is
      REPORTED on both axes and GATED only at the one baseline point the paper makes its claims at. Pooling
      "magnitude == 3.0" over a frac_q sweep running to 0.30, as the first version did, reports the mixing
      ratio of the sweep and not a property of the baseline.

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

BASE_MAG, BASE_FRAC, BASE_DEG = 3.0, 0.10, 12   # the baseline point section5.matched() pins every
                                                # direction comparison to. Gate HERE, report elsewhere.


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
    #     every rung planted an ~11.8x inflation.
    #
    #     THE FIRST VERSION OF THIS GATE CHECKED |H|, AND |H| IS THE ONE STATISTIC THAT CANNOT ANSWER.
    #     Under inflation the heavy set is a SUBSET of the planted set (raising w(e) can only make e heavy;
    #     it lengthens other edges' detours, which makes THEM less heavy, never more). And |B| is fixed by
    #     frac_q, not by magnitude. So |H| SATURATES at |B| and cannot rise with mu past it -- the observed
    #     152 -> 161 (1.06x) IS that saturation, and the gate read it as "flat, therefore the old bug", which
    #     was confidently wrong. A gate whose failure message accuses the cluster of running stale code had
    #     better be checking a statistic that can actually distinguish the two.
    #
    #     Two statistics can, and both are computed from the delivered rows:
    #       * the VIOLATED CYCLE COUNT (`cuts`): a heavier edge breaks more of the cycles through it. Under
    #         the old floor every rung planted the same ~11.8x inflation, so this was FLAT. It now moves 15x.
    #       * the H == B FRACTION: under the old floor the inflation was so violent that no planted edge could
    #         ever be un-broken by interference, so H == B held at EVERY rung. It now runs 0.00 -> 1.00.
    #     Either one alone refutes the stale-checkout hypothesis. We demand both.
    S = D.get("small")
    s3 = S[S.sweep.eq("S3")].drop_duplicates("task") if S is not None else pd.DataFrame()
    if len(s3) == 0:
        chk(False, "G4 the inflate MAGNITUDE knob is live", "no S3 rows -- cannot verify the fix landed")
    else:
        cut = (ok(S)[ok(S).sweep.eq("S3") & ok(S).algo.eq("l1sep_gmr")]
               .groupby("magnitude").cuts.median().sort_index())
        eq = s3.assign(eq=(s3.H == s3.n_corrupted)).groupby("magnitude").eq.mean().sort_index()
        rise_c = cut.iloc[-1] / max(cut.iloc[0], 1) if len(cut) > 1 else float("nan")
        rise_e = eq.iloc[-1] - eq.iloc[0] if len(eq) > 1 else float("nan")
        live = rise_c > 3.0 and rise_e > 0.5
        chk(live, "G4 the inflate MAGNITUDE knob is LIVE (cycles and H==B respond)",
            f"cuts {cut.iloc[0]:.0f} -> {cut.iloc[-1]:.0f} ({rise_c:.1f}x); "
            f"H==B {eq.iloc[0]:.2f} -> {eq.iloc[-1]:.2f}  over mu {eq.index[0]}..{eq.index[-1]}")
        if not live:
            print("         *** THE CORRUPTION DOES NOT RESPOND TO THE MAGNITUDE. Under the old '+1' floor every")
            print("         *** inflation landed at ~11.8x the detour whatever mu was asked for, which pins the")
            print("         *** cycle count flat AND forces H == B at every rung. If BOTH are inert, the cluster")
            print("         *** ran a stale graph_models.py. (|H| alone proves nothing: it saturates at |B|.)")

    for g, d in D.items():
        st = d[d.status.eq("skipped_time")]
        chk(len(st) == 0, f"G5 skipped_time == 0, the budget never fired ({g})",
            f"{len(st)} rows" + (f" -- {st.algo.value_counts().head(3).to_dict()}" if len(st) else ""))

    for g, d in D.items():
        Dm = ok(d)[ok(d).algo.eq("domr")].dropna(subset=["size", "H"])
        bad = int((Dm["size"] != Dm.H).sum()) if len(Dm) else -1
        chk(len(Dm) > 0 and bad == 0, f"G6 domr's cover IS the heavy set ({g})",
            f"{len(Dm):,} tasks, {bad} mismatches")

    # G7  H == B UNDER INFLATION -- AND THE FIRST VERSION OF THIS GATE POOLED, WHICH IS HOW IT GOT 0.732.
    #
    #     "magnitude == 3.0" is not one population. It is 1,680 tasks spanning the S1 ladder, the S2/S2k
    #     density sweeps AND the S4i FRACTION sweep, whose frac_q runs from 0.01 to 0.30. And frac_q is
    #     precisely the knob that drives the interference: a planted edge is un-broken when its detour runs
    #     through ANOTHER planted edge, and that gets likelier the more edges are planted. So a "H == B at
    #     mu = 3" pooled over frac_q reports the mixing ratio of the sweep, not a property of the baseline.
    #     Report it on BOTH axes, and gate it at the ONE point the paper's claims are made at.
    T = allrows.drop_duplicates("task")
    inf = T[T.direction.eq("inflate")].dropna(subset=["H", "n_corrupted"]).copy()
    inf["eq"] = inf.H == inf.n_corrupted
    inf["hb"] = inf.H / inf.n_corrupted.where(inf.n_corrupted > 0)

    base = inf[inf.magnitude.eq(BASE_MAG) & inf.frac_q.eq(BASE_FRAC)
               & inf.deg.eq(BASE_DEG) & inf["mode"].eq("radius")]
    b_eq = float(base.eq.mean()) if len(base) else float("nan")
    b_hb = float(base.hb.median()) if len(base) else float("nan")
    chk(len(base) > 0 and b_hb > 0.95,
        f"G7 inflate at the BASELINE: H is (almost) B  (mu={BASE_MAG}, frac={BASE_FRAC}, deg={BASE_DEG})",
        f"{len(base)} tasks: H==B exactly on {b_eq:.3f}, and |H|/|B| median {b_hb:.4f}"
        if len(base) else "no baseline rows")

    print("\n  H == B under inflation. It is NOT a grid-wide identity, and it depends on BOTH knobs:")
    print(f"    {'magnitude':>10}{'tasks':>8}{'H == B':>9}{'|H|/|B|':>10}")
    for m, g in inf.groupby("magnitude"):
        print(f"    {m:>10}{len(g):>8}{g.eq.mean():>9.3f}{g.hb.median():>10.4f}")
    print(f"\n  ...and at the baseline magnitude {BASE_MAG}, sliced by the FRACTION (this is what the pooled")
    print("     number was hiding -- more planted edges means more interference):")
    print(f"    {'frac_q':>8}{'tasks':>8}{'H == B':>9}{'|H|/|B|':>10}")
    for q, g in inf[inf.magnitude.eq(BASE_MAG)].groupby("frac_q"):
        print(f"    {q:>8}{len(g):>8}{g.eq.mean():>9.3f}{g.hb.median():>10.4f}")
    print("\n  H is always a SUBSET of B here (inflating an edge cannot make a DIFFERENT edge heavy -- it")
    print("  only lengthens other edges' detours). So DOMR's precision stays 1.000; what falls is its RECALL.")
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
