"""rgg_check.py -- health check for the RGG experiment output (rgg_harness.py).

Runs LOCALLY (needs pandas). Input is a dir of task_*.csv or the combined results_rgg_all.csv.

    sage -python experiments/rgg_check.py --results results_rgg
    sage -python experiments/rgg_check.py --results results_rgg_all.csv

Reports: task/row/algo counts, status breakdown, HARD failures (timeout/oom/killed/error), invalid covers
(valid==0 -- on float RGG every cover should be valid, so any is a bug), empty breaks (n_corrupted==0),
samples-per-config (target 40), and Part-2 kNN coverage. The final PROBLEMS count is the # of hard issues.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

NUM = ("size", "valid", "cpu", "wall", "peak_mb", "H", "n_corrupted", "edit_precision", "edit_recall",
       "knn_k", "jaccard_TC", "jaccard_TF", "recall_TF", "lift", "triplet_acc_C", "triplet_acc_F",
       "n", "radius", "magnitude", "frac_q", "n_jitter", "jitter", "subset_s", "sample", "V", "E")
CONFIG_KEYS = ["part", "sweep", "mode", "n", "deg", "k", "direction", "magnitude",
               "frac_q", "n_jitter", "jitter", "subset_s"]


def load(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "task_*.csv")))
        if not files:
            raise SystemExit(f"no task_*.csv in {path}")
        df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    else:
        df = pd.read_csv(path)
    for c in NUM:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def check(df):
    print("=" * 70)
    print(f"LOADED {df['task'].nunique()} tasks, {len(df)} rows, {df['algo'].nunique()} algos, "
          f"parts={sorted(df['part'].unique())}, sweeps={df['sweep'].nunique()}")
    problems = 0

    print("\nstatus counts:")
    print(df["status"].value_counts().to_string())
    bad = df[df["status"].str.startswith(("timeout", "oom", "killed", "error"), na=False)]
    # The two exact ILPs are capped at 45s and EXPECTED to time out at large n -- that just triggers the LP
    # lower-bound fallback for the reference, so it's not a failure. Everything else (oom/killed/error, or a
    # timeout in any other algo) is a genuine hard problem.
    is_ilp_to = bad["status"].str.startswith("timeout", na=False) & bad["algo"].isin(["iomr_ilp", "gmr_ilp"])
    expected, hard = bad[is_ilp_to], bad[~is_ilp_to]
    print(f"\nexpected ILP timeouts (iomr_ilp/gmr_ilp -> LP-bound fallback, benign): {len(expected)}")
    if len(expected):
        print(expected.groupby("algo").size().to_string())
    print(f"\nHARD-fail rows (oom/killed/error, or timeout in any other algo): {len(hard)}")
    if len(hard):
        print(hard.groupby(["algo", "status"]).size().to_string())
        problems += len(hard)

    # invalid covers -- float RGG is exactly metric, so every produced cover should verify (valid==1)
    inval = df[df["valid"] == 0]
    print(f"\ninvalid covers (valid==0): {len(inval)}  (should be 0 on float RGG)")
    if len(inval):
        print(inval.groupby("algo").size().to_string())
        problems += len(inval)

    # each break should actually break something (n_corrupted>0); subset_s in (0,1) guarantees it
    per_task = df.groupby("task").first()
    empty = per_task[per_task["n_corrupted"] == 0]
    print(f"\ntasks with empty break (n_corrupted==0): {len(empty)}  (expected 0 for these sweeps)")
    if len(empty):
        problems += len(empty)

    print("\nper-algorithm status (rows):")
    print(df.groupby("algo")["status"].value_counts().unstack(fill_value=0).to_string())

    # samples per config (one row-family per sample: use domr)
    key = df[df["algo"] == "domr"]
    spc = key.groupby(CONFIG_KEYS, dropna=False)["sample"].nunique()
    tgt = int(spc.max()) if len(spc) else 0                # infer target from the fullest config (30 poc/40 full)
    print(f"\nsamples/config: min={spc.min()} max={spc.max()} (target {tgt}); "
          f"{(spc < tgt).sum()} configs under target (nonzero is fine only if the array is still running)")

    # Part 2 kNN coverage
    p2 = df[df["part"] == "p2"]
    if len(p2):
        kn = p2[p2["knn_k"].notna()]
        cov = float(kn["jaccard_TF"].notna().mean()) if len(kn) else 0.0
        print(f"\nPart 2: {len(kn)} kNN rows, jaccard_TF non-null frac={cov:.3f}, "
              f"median lift={kn['lift'].median():.4f}")
        if cov < 0.99:
            print("  ^ some kNN rows have null jaccard_TF -- investigate")

    print("=" * 70)
    print(f"PROBLEMS: {problems}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results_rgg/ dir or results_rgg_all.csv")
    a = ap.parse_args()
    n = check(load(a.results))
    if n:
        print(f"\n*** {n} hard problems -- investigate before trusting the RGG results ***")


if __name__ == "__main__":
    main()
