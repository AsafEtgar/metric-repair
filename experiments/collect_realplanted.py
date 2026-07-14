"""Collect the planted-real-bases array, gate it, and say which of its bases the fix actually changed.

THE GATES FAIL CLOSED.

  G1  COMPLETENESS. All 900 task CSVs, or the missing indices by name.
  G2  SCHEMA. Column-identical to rgg_harness.RGG_CSV_FIELDS.
  G3  EVERY BASE AND EVERY DIRECTION RETURNED. A base that dies on the cluster loses 180 tasks quietly; a
      direction that dies loses the comparison the array exists to make.
  G4  skipped_time == 0. The 6 h budget must never have fired.
  G5  DOMR's cover IS the heavy set (size == H) on every task.
  G6  *** THE ONE THAT EARNS ITS KEEP: THE FIX REACHED THE CLUSTER. *** The preflight measured, per base, how
      many inflated edges the old floor moved: 0.0% on dimacs_ny_d/_t and both fish bases, 8.8% on
      pbmc3k_cosine_knn. So the four unaffected bases MUST come back with |H| and |B| matching their published
      rows, and pbmc3k MUST NOT. If pbmc3k also matches, the cluster ran a stale graph_models.py and the
      re-run bought nothing; if a supposedly-unaffected base MOVED, the analysis of the damage was wrong and
      the scope of this re-run is too small. Both halves read from the published CSV -- no constant is typed
      in. It is a two-sided check and neither side can pass vacuously.

  input   results_realplanted/task_*.csv
  output  analysis/realplanted/rows.csv

  usage   sage -python experiments/collect_realplanted.py
"""
import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import realplanted_harness as rp                                        # noqa: E402
from rgg_harness import RGG_CSV_FIELDS                                  # noqa: E402

# The published rows this array replaces. Used ONLY to read the old |H| off disk -- never to assert a value.
PUBLISHED = "analysis/rgg_realrec/rgg_rows_with_ratio.csv"


def load(indir):
    files = sorted(glob.glob(os.path.join(indir, "task_*.csv")))
    if not files:
        raise SystemExit(f"FATAL: no task CSVs in {indir}/ -- did the array run?")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for c in ("n", "E", "H", "size", "valid", "magnitude", "frac_q", "n_corrupted"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.drop_duplicates(subset=["task", "algo"], keep="first"), files


def ok(d):
    return d[d.status.eq("ok") & d.valid.eq(1)]


def gate(d):
    fails = []

    def chk(c, name, obs):
        print(f"  [{'PASS' if c else 'FAIL'}] {name:<52} {obs}")
        if not c:
            fails.append(name)

    want = len(rp.all_tasks())
    got = sorted(d.task.unique())
    missing = sorted(set(range(want)) - set(got))
    chk(not missing, "G1 every task returned a CSV",
        f"{len(got)} of {want}" + (f"; MISSING {missing[:6]}..." if missing else ""))

    extra = [c for c in d.columns if c not in RGG_CSV_FIELDS]
    absent = [c for c in RGG_CSV_FIELDS if c not in d.columns]
    chk(not extra and not absent, "G2 schema == RGG_CSV_FIELDS",
        f"{len(d.columns)} cols" + (f"; extra={extra} missing={absent}" if (extra or absent) else ""))

    T = d.drop_duplicates("task")
    want_b = sorted({p["base"] for p in rp.rh.GRIDS[rp.GRID]})
    got_b = sorted(set(T.base.dropna()))
    got_dir = sorted(set(T.direction.dropna()))
    chk(got_b == want_b and set(got_dir) == {"inflate", "deflate", "mixed"},
        "G3 every base and every direction returned",
        f"bases {len(got_b)}/{len(want_b)}, dirs={got_dir}"
        + (f"; MISSING {sorted(set(want_b) - set(got_b))}" if set(want_b) - set(got_b) else ""))

    st = d[d.status.eq("skipped_time")]
    chk(len(st) == 0, "G4 skipped_time == 0 (the budget never fired)",
        f"{len(st)} rows" + (f" -- {st.algo.value_counts().head(3).to_dict()}" if len(st) else ""))

    Dm = ok(d)[ok(d).algo.eq("domr")].dropna(subset=["size", "H"])
    bad = int((Dm["size"] != Dm.H).sum()) if len(Dm) else -1
    chk(len(Dm) > 0 and bad == 0, "G5 domr's cover IS the heavy set (size == H)",
        f"{len(Dm)} tasks, {bad} mismatches")

    # G6  TWO-SIDED: the bases the fix should NOT have moved must match the published run, and the base it
    #     SHOULD have moved must not. Both read off the published CSV.
    if not os.path.exists(PUBLISHED):
        chk(False, "G6 the fix reached the cluster (vs the published rows)",
            f"{PUBLISHED} missing -- cannot compare; re-run is UNVERIFIED")
    else:
        old = pd.read_csv(PUBLISHED, low_memory=False).drop_duplicates("task")
        for c in ("H", "magnitude", "frac_q", "n_corrupted"):
            old[c] = pd.to_numeric(old[c], errors="coerce")
        key = ["base", "direction", "frac_q", "sample"]
        o = old[old.direction.eq("inflate")].set_index(key).H
        n = T[T.direction.eq("inflate")].set_index(key).H
        common = o.index.intersection(n.index)
        moved = {}
        for b in got_b:
            idx = [k for k in common if k[0] == b]
            if not idx:
                continue
            moved[b] = float((o.loc[idx] != n.loc[idx]).mean())
        expect_move = {"pbmc3k_cosine_knn"}                        # the preflight MEASURED this, per edge
        wrong = [b for b, f in moved.items()
                 if (f > 0.05) != (b in expect_move)]
        chk(bool(moved) and not wrong, "G6 the fix reached the cluster (|H| moved iff it should have)",
            "; ".join(f"{b} {f:.0%}" for b, f in sorted(moved.items()))
            + (f"; UNEXPECTED {wrong}" if wrong else ""))
        if wrong:
            print("         *** A base moved that the preflight said could not, or pbmc3k did not move at all.")
            print("         *** Either the cluster ran a STALE graph_models.py, or the damage analysis that")
            print("         *** scoped this re-run was wrong. Do not analyse these rows.")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results_realplanted")
    ap.add_argument("--outdir", default="analysis/realplanted")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    d, files = load(a.indir)
    print(f"collected {len(files)} task CSVs -> {len(d):,} cover rows\n")
    print("GATE -- nothing is written until these pass")
    fails = gate(d)
    if fails and not a.force:
        raise SystemExit(f"\n*** {len(fails)} GATE FAILURE(S): {fails}. NOTHING WRITTEN. ***")

    os.makedirs(a.outdir, exist_ok=True)
    p = os.path.join(a.outdir, "rows.csv")
    d.to_csv(p, index=False)
    print(f"\n  wrote {p}")
    print("\nAll gates passed." if not fails else "\n!! WRITTEN UNDER --force.")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
