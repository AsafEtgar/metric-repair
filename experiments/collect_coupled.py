"""Collect the COUPLED density sweep (exp2c), RUN THE GATES, and print what the array was run to find.

THE GATES FAIL CLOSED. Every one can genuinely fail; none is a formality.

  G1  COMPLETENESS. All 320 task CSVs, or the missing indices by name. A short collect that silently
      averages over what came back is how a median becomes a fiction.
  G2  SCHEMA. The rows must be column-identical to the campaign's (harness.CSV_FIELDS). The whole point of
      running this array through the harness's own runner is that its rows can be read BESIDE exp1's and
      exp2b's; a drifted column means they cannot.
  G3  THE DOMR IDENTITY. domr's cover IS the heavy set, so size(domr) must equal H on every task, exactly.
      This tests the generator, the component decomposition, the heavy-set computation and the CSV writer in
      one line, against a value theory fixes in advance. A single mismatch is a bug, not a result.
  G4  THE ONSET IS REAL. |H|/m must climb from near-zero at the sparse end to a genuinely broken dense end.
      This is the claim the array exists to make; if it does not hold, the array found nothing and must not
      be quoted as if it had.
  G5  THE DROP IS DISCLOSED. gmr_bestofk and iomr_bestofk were not run. They must therefore appear NOWHERE,
      and the collector must SAY so rather than let their absence read as a return rate of zero. A missing
      row and a failed row are different facts and the table must not confuse them.

WHAT THE ARRAY IS FOR. Under coupling the edge weights are Geometric(1 - p), so the mean weight is 1/(1 - p):
DENSITY IS THE WEIGHT SPREAD. The two cannot be varied independently, and this sweep shows it -- non-metricity
switches on WITH density rather than beside it. Read against exp2b (same n, same density range, weights held
fixed at Geom(0.5)), it isolates what the weight model alone is responsible for.

  input   results_coupled/task_*.csv
  output  analysis/coupled/rows.csv          tidy, one row per (task, algo)
          analysis/coupled/summary.csv       one row per (alpha, algo)

  usage   sage -python experiments/collect_coupled.py
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import coupled_harness as ch                                            # noqa: E402
from harness import CSV_FIELDS                                          # noqa: E402


def load(indir):
    files = sorted(glob.glob(os.path.join(indir, "task_*.csv")))
    if not files:
        raise SystemExit(f"FATAL: no task CSVs in {indir}/ -- did the array run?")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for c in ("size", "valid", "H", "E", "n", "p", "alpha", "wall", "cpu"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d, files


def gate(d, files):
    fails = []

    def chk(ok, name, obs):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<54} {obs}")
        if not ok:
            fails.append(name)

    # G1  completeness
    want = len(ch.all_tasks())
    got = sorted(d.task.unique())
    missing = sorted(set(range(want)) - set(got))
    chk(not missing, "G1 every task returned a CSV",
        f"{len(got)} of {want}" + (f"; MISSING {missing[:8]}{'...' if len(missing) > 8 else ''}"
                                   if missing else ""))

    # G2  schema -- these rows must be readable beside exp1's and exp2b's
    extra = [c for c in d.columns if c not in CSV_FIELDS]
    absent = [c for c in CSV_FIELDS if c not in d.columns]
    chk(not extra and not absent, "G2 schema identical to harness.CSV_FIELDS",
        f"{len(d.columns)} cols" + (f"; extra={extra} missing={absent}" if (extra or absent) else ""))

    # G3  THE DOMR IDENTITY -- its cover IS H, exactly, on every task
    D = d[d.algo.eq("domr") & d.status.eq("ok")].dropna(subset=["size", "H"])
    if D.empty:
        chk(False, "G3 domr's cover IS the heavy set (size == H)", "no ok domr rows -- the control did not run")
    else:
        bad = int((D["size"] != D.H).sum())
        chk(bad == 0, "G3 domr's cover IS the heavy set (size == H)",
            f"{len(D)} tasks, {bad} mismatches")

    # G4  the onset. A sweep that never breaks anything found nothing.
    t = d.drop_duplicates("task").copy()
    t["hm"] = t.H / t.E
    by_a = t.groupby(t.alpha.round(3)).hm.median().sort_index(ascending=False)   # alpha DOWN = density UP
    lo, hi = float(by_a.iloc[0]), float(by_a.iloc[-1])
    chk(hi >= 0.05 and lo < 0.01, "G4 the onset is real (|H|/m climbs from ~0 to broken)",
        f"|H|/m {lo:.4f} (alpha={by_a.index[0]:.3f}) -> {hi:.4f} (alpha={by_a.index[-1]:.3f})")

    # G5  the drop, DISCLOSED. Absent != returned-zero.
    present = set(d.algo.unique())
    leaked = sorted(present & ch.DROP_ALGOS)
    chk(not leaked, "G5 the dropped methods appear nowhere", f"leaked: {leaked}" if leaked else "clean")
    print(f"         NOT RUN in this array: {sorted(ch.DROP_ALGOS)} -- they time out on 100% of exp2b's")
    print(f"         tasks (1800 s each, zero data). Their cells are ABSENT, not zero. Do not report a")
    print(f"         return rate for them here, and do not compare anyone's return rate against a")
    print(f"         different cap.")
    return fails


def report(d):
    ok = d[d.status.eq("ok") & d.valid.eq(1)].copy()
    ok["sm"] = ok["size"] / ok.E
    t = d.drop_duplicates("task").copy()
    t["hm"] = t.H / t.E

    print(f"\n{'=' * 96}")
    print("THE COUPLED SWEEP: density and non-metricity are ONE knob (weights are Geom(1-p))")
    print(f"{'=' * 96}")
    g = t.groupby(t.alpha.round(3)).agg(p=("p", "median"), m=("E", "median"), hm=("hm", "median"))
    g = g.sort_index(ascending=False)
    print(f"  {'alpha':>7}{'p':>9}{'m':>10}{'mean w':>9}{'|H|/m':>9}")
    print("  " + "-" * 46)
    for a, r in g.iterrows():
        print(f"  {a:>7.3f}{r.p:>9.4f}{int(r.m):>10,}{1 / (1 - r.p):>9.3f}{r.hm:>9.4f}")

    print(f"\n{'=' * 96}")
    print("|S|/m BY ALGORITHM, ACROSS THE SWEEP  (the paper's axis; medians over runs that returned+verified)")
    print(f"{'=' * 96}")
    alphas = sorted(ok.alpha.round(3).unique(), reverse=True)
    show = [alphas[0], alphas[len(alphas) // 3], alphas[2 * len(alphas) // 3], alphas[-1]]
    hdr = "".join(f"{'a=' + format(a, '.3f'):>12}" for a in show)
    print(f"  {'algo':<18}{hdr}{'  ret (all)':>12}")
    print("  " + "-" * (20 + 12 * len(show) + 12))
    ntask = d.task.nunique()
    for al in sorted(ok.algo.unique()):
        A = ok[ok.algo.eq(al)]
        cells = ""
        for a in show:
            K = A[A.alpha.round(3).eq(a)]
            cells += f"{K.sm.median():>12.3f}" if len(K) else f"{'--':>12}"
        ret = 100 * A.task.nunique() / ntask
        print(f"  {al:<18}{cells}{ret:>11.0f}%")
    print("\n  alpha DOWN = density UP = |H| UP. Read a row across: under coupling you cannot move density")
    print("  without moving brokenness, so a trend here is a trend in BOTH.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results_coupled")
    ap.add_argument("--outdir", default="analysis/coupled")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    d, files = load(a.indir)
    print(f"collected {len(files)} task CSVs -> {len(d)} rows\n")
    print("GATE -- nothing is reported until these pass")
    fails = gate(d, files)
    if fails and not a.force:
        raise SystemExit(f"\n*** {len(fails)} GATE FAILURE(S): {fails}. Nothing printed. Explain it, or "
                         "re-run with --force if you know why. ***")

    os.makedirs(a.outdir, exist_ok=True)
    d.to_csv(os.path.join(a.outdir, "rows.csv"), index=False)
    ok = d[d.status.eq("ok") & d.valid.eq(1)].copy()
    ok["sm"] = ok["size"] / ok.E
    s = ok.groupby(["alpha", "algo"]).agg(
        n_ok=("task", "nunique"), sm_med=("sm", "median"),
        ratio_domr_med=("size", lambda x: np.nan), wall_med=("wall", "median")).reset_index()
    s.to_csv(os.path.join(a.outdir, "summary.csv"), index=False)
    print(f"\n  wrote {a.outdir}/rows.csv and {a.outdir}/summary.csv")

    report(d)
    print("\nAll gates passed." if not fails else "\n!! PRINTED UNDER --force. Do not quote these.")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
