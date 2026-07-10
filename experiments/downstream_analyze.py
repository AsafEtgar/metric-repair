"""Aggregate the downstream-recovery rows into a per-(graph, variant, algo, k) summary, and print the headline.

    sage -python experiments/downstream_analyze.py --indir results_downstream --outdir analysis

Reads results_downstream/<graph>.csv, writes analysis/summary_downstream.csv. Keyed on graph (the real-data
convention), not on a synthetic sweep. Prints the DOMR self-check and the per-variant lift so the result is
readable without opening the CSV. No plotting here (matplotlib-free); real_plots-style figures are a separate,
optional step.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

NUM = ("recovery_obs", "recovery_rep", "lift", "spearman_obs", "spearman_rep", "delta_spearman",
       "n", "n_gt", "k", "seed")


def load(indir):
    files = sorted(glob.glob(os.path.join(indir, "*.csv")))
    if not files:
        raise SystemExit(f"no <graph>.csv in {indir}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    for c in NUM:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def aggregate(df):
    # A deterministic algo has one cover; a randomized one has up to 30 seeds. n_covers is the honest
    # denominator behind each median -- report it, per the survivorship discipline used elsewhere.
    keys = ["graph", "gt_kind", "variant", "algo", "k"]
    g = df.groupby(keys, dropna=False)
    named = {"n_covers": ("seed", "size"), "n": ("n", "first"), "n_gt": ("n_gt", "first")}
    for c in ("recovery_obs", "recovery_rep", "lift", "spearman_obs", "spearman_rep", "delta_spearman"):
        named[f"{c}_med"] = (c, "median")
        named[f"{c}_q25"] = (c, lambda s: s.quantile(.25))
        named[f"{c}_q75"] = (c, lambda s: s.quantile(.75))
    return g.agg(**named).reset_index().sort_values(keys).reset_index(drop=True)


def report(df, summ):
    print("=" * 72)
    print(f"LOADED {df['graph'].nunique()} graphs, {len(df)} rows, {df['algo'].nunique()} algorithms")

    # DOMR self-check: by Lemma (decrease-only invariance) DOMR changes no distance -> lift must be ~0.
    dom = df[df["variant"] == "DOMR"]
    if len(dom):
        worst = dom["lift"].abs().max()
        flag = "OK" if worst < 1e-6 else "*** NONZERO -- pipeline bug ***"
        print(f"\nDOMR self-check: max |lift| = {worst:.2e}   {flag}")

    print("\nlift (kNN recovery gained by repair over the observed graph), median over covers:")
    piv = (summ[summ["variant"] != "DOMR"]
           .pivot_table(index=["graph", "variant"], columns="k", values="lift_med"))
    print(piv.round(4).to_string())

    print("\nrank fidelity (Spearman vs ground truth), median delta:")
    sp = summ.groupby(["graph", "variant"])["delta_spearman_med"].median().unstack("variant")
    print(sp.round(4).to_string())

    print("\nobserved baseline (how much room repair has), per graph:")
    base = df.groupby(["graph", "k"])["recovery_obs"].first().unstack("k")
    print(base.round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results_downstream")
    ap.add_argument("--outdir", default="analysis")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    df = load(a.indir)
    summ = aggregate(df)
    report(df, summ)
    out = os.path.join(a.outdir, "summary_downstream.csv")
    summ.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(summ)} groups)")


if __name__ == "__main__":
    main()
