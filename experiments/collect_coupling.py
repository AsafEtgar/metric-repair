"""Collect the COUPLING A/B, run the gates, and print the one thing it exists to settle.

THE QUESTION. Appendix A reports that spc_gmr and pivot SWAP between the dense grid's two sweeps -- but those
sweeps differ in n, in p AND in the weight model, so the flip cannot be attributed. This array holds n and p
fixed and changes ONLY the weight model. If the ranking still flips, the coupling alone did it.

THE GATES FAIL CLOSED.

  G1  COMPLETENESS. All 660 task CSVs, or the missing indices by name.
  G2  SCHEMA. Column-identical to harness.CSV_FIELDS, so these rows concatenate with the published small grid.
  G3  THE DOMR IDENTITY. size(domr) == H, exactly, on every task.
  G4  *** skipped_time == 0. *** The budget must never have fired. `run_one_task` skips the algorithms LAST
      in build_suite order -- which ends spc_gmr, spc_iomr, PIVOT, LEFT_EDGE -- and spc_gmr and pivot are the
      two methods this array exists to compare. If the budget fired, their absence at the broken end is a
      queue position, not a result, and the experiment has destroyed its own headline.
  G5  THE MODELS ARE MATCHED. At every alpha the two models must carry the same n, the same p, and (to 10%)
      the same m. If they do not, the comparison confounds the weight model with the topology and answers
      nothing. This is the experiment's entire validity condition, and it is checked on the DELIVERED data,
      not just at preflight.
  G6  THE ONSET IS REAL, AND THE CONTROL IS FLAT. The coupled sweep must collapse to metric while the
      decoupled one does not. Two halves; neither can pass vacuously.

  input   results_coupling/task_*.csv
  output  analysis/coupling/rows.csv, summary.csv

  usage   sage -python experiments/collect_coupling.py
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import coupling_harness as ch                                           # noqa: E402
from harness import CSV_FIELDS                                          # noqa: E402

COUPLED, DECOUPLED = "geometric", "decoupled_geometric"
PAIR = ["spc_gmr", "pivot"]          # the two whose flip the array exists to attribute


def load(indir):
    files = sorted(glob.glob(os.path.join(indir, "task_*.csv")))
    if not files:
        raise SystemExit(f"FATAL: no task CSVs in {indir}/ -- did the array run?")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for c in ("n", "p", "alpha", "E", "H", "size", "valid", "wall", "peak_mb"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["sm"] = d["size"] / d.E
    d["hm"] = d.H / d.E
    return d, files


def ok(d):
    return d[d.status.eq("ok") & d.valid.eq(1)]


def gate(d):
    fails = []

    def chk(c, name, obs):
        print(f"  [{'PASS' if c else 'FAIL'}] {name:<50} {obs}")
        if not c:
            fails.append(name)

    want = len(ch.all_tasks())
    got = sorted(d.task.unique())
    missing = sorted(set(range(want)) - set(got))
    chk(not missing, "G1 every task returned a CSV",
        f"{len(got)} of {want}" + (f"; MISSING {missing[:6]}..." if missing else ""))

    extra = [c for c in d.columns if c not in CSV_FIELDS and c not in ("sm", "hm")]
    absent = [c for c in CSV_FIELDS if c not in d.columns]
    chk(not extra and not absent, "G2 schema identical to harness.CSV_FIELDS",
        f"{len(d.columns) - 2} cols" + (f"; extra={extra} missing={absent}" if (extra or absent) else ""))

    D = ok(d)[ok(d).algo.eq("domr")].dropna(subset=["size", "H"])
    bad = int((D["size"] != D.H).sum()) if len(D) else -1
    chk(bad == 0, "G3 domr's cover IS the heavy set (size == H)",
        f"{len(D)} tasks, {bad} mismatches" if len(D) else "no ok domr rows")

    st = d[d.status.eq("skipped_time")]
    chk(len(st) == 0, "G4 skipped_time == 0 (the BUDGET never fired)",
        f"{len(st)} rows" + (f" -- {st.algo.value_counts().head(3).to_dict()}" if len(st) else ""))
    if len(st):
        print("         *** THE BUDGET FIRED. run_one_task skips the algorithms LAST in build_suite order,")
        print("         *** which ends: spc_gmr, spc_iomr, PIVOT, LEFT_EDGE. spc_gmr and pivot are the two")
        print("         *** methods this array exists to compare. Their absence would be a queue position,")
        print("         *** not a result. Raise BUDGET_S and re-run. Do NOT report this.")

    # G5 -- THE VALIDITY CONDITION. Same n, same p, same m (to 10%) at every alpha.
    t = d.drop_duplicates("task")
    holes = []
    for a, g in t.groupby(t.alpha.round(3)):
        ms = {mo: g[g.model.eq(mo)].E.median() for mo in (COUPLED, DECOUPLED)}
        ps = {mo: g[g.model.eq(mo)].p.median() for mo in (COUPLED, DECOUPLED)}
        if any(pd.isna(v) for v in list(ms.values()) + list(ps.values())):
            holes.append((a, "missing a model")); continue
        if abs(ps[COUPLED] - ps[DECOUPLED]) > 1e-9:
            holes.append((a, "p differs"))
        elif abs(ms[COUPLED] - ms[DECOUPLED]) / ms[COUPLED] > 0.10:
            holes.append((a, "m differs by >10%"))
    chk(not holes, "G5 the two models are MATCHED at every alpha",
        f"{t.alpha.nunique()} rungs" + (f"; PROBLEMS {holes[:3]}" if holes else ""))

    # G6 -- the onset is real AND the control is flat. Both halves, or the array found nothing.
    hc = t[t.model.eq(COUPLED)].groupby(t.alpha.round(3)).hm.median()
    hd = t[t.model.eq(DECOUPLED)].groupby(t.alpha.round(3)).hm.median()
    coll = hc.max() / max(hc.min(), 1e-9)
    flat = hd.max() / max(hd.min(), 1e-9)
    chk(hc.max() > 0.20 and hc.min() < 0.01 and flat < 5,
        "G6 coupled COLLAPSES, decoupled does NOT",
        f"coupled {hc.max():.4f}->{hc.min():.4f} ({coll:.0f}x);  "
        f"decoupled {hd.max():.4f}->{hd.min():.4f} ({flat:.1f}x)")
    return fails


def report(d):
    t = d.drop_duplicates("task")
    print(f"\n{'=' * 88}")
    print("THE COUPLING, ISOLATED: same n, same p, same topology -- only the weight model differs")
    print(f"{'=' * 88}")
    print(f"  {'alpha':>6}{'p':>8}{'m':>9}   {'COUPLED |H|/m':>15}{'DECOUPLED |H|/m':>17}")
    print("  " + "-" * 60)
    for a, g in t.groupby(t.alpha.round(3)):
        c = g[g.model.eq(COUPLED)]
        u = g[g.model.eq(DECOUPLED)]
        print(f"  {a:>6.2f}{c.p.median():>8.3f}{int(c.E.median()):>9,}   "
              f"{c.hm.median():>15.4f}{u.hm.median():>17.4f}")

    print(f"\n{'=' * 88}")
    print("DOES THE WEIGHT MODEL, ALONE, FLIP THE RANKING?   |S|/m at matched (n, p)")
    print(f"{'=' * 88}")
    O = ok(d)
    print(f"  {'alpha':>6}" + "".join(f"{a + ' ' + m[:4]:>16}" for a in PAIR for m in (COUPLED, DECOUPLED)))
    print("  " + "-" * 70)
    flips = 0
    for a, g in O.groupby(O.alpha.round(3)):
        cells, v = "", {}
        for al in PAIR:
            for mo in (COUPLED, DECOUPLED):
                x = g[g.algo.eq(al) & g.model.eq(mo)].sm.median()
                v[(al, mo)] = x
                cells += f"{x:>16.3f}" if np.isfinite(x) else f"{'--':>16}"
        print(f"  {a:>6.2f}{cells}")
        try:
            c_win = v[(PAIR[0], COUPLED)] < v[(PAIR[1], COUPLED)]
            d_win = v[(PAIR[0], DECOUPLED)] < v[(PAIR[1], DECOUPLED)]
            if c_win != d_win:
                flips += 1
        except Exception:
            pass
    print(f"\n  rungs where {PAIR[0]} and {PAIR[1]} change places between the two models: {flips} "
          f"of {O.alpha.nunique()}")
    print("  If this is nonzero, THE WEIGHT MODEL ALONE FLIPS THE RANKING -- n, p and the topology")
    print("  distribution were identical. Appendix A's caveat becomes a mechanism.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results_coupling")
    ap.add_argument("--outdir", default="analysis/coupling")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    d, files = load(a.indir)
    print(f"collected {len(files)} task CSVs -> {len(d)} rows\n")
    print("GATE -- nothing is reported until these pass")
    fails = gate(d)
    if fails and not a.force:
        raise SystemExit(f"\n*** {len(fails)} GATE FAILURE(S): {fails}. Nothing printed. ***")

    os.makedirs(a.outdir, exist_ok=True)
    d.to_csv(os.path.join(a.outdir, "rows.csv"), index=False)
    ok(d).groupby(["alpha", "model", "algo"]).agg(
        n_ok=("task", "nunique"), sm_med=("sm", "median"), wall_med=("wall", "median"),
    ).reset_index().to_csv(os.path.join(a.outdir, "summary.csv"), index=False)
    print(f"\n  wrote {a.outdir}/rows.csv and {a.outdir}/summary.csv")

    report(d)
    print("\nAll gates passed." if not fails else "\n!! PRINTED UNDER --force.")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
