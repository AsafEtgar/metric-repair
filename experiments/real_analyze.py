"""real_analyze.py -- aggregate the real-dataset results (real_harness.py) into a per-(graph, algo) summary.

Runs LOCALLY (needs pandas). Input is the results_real/ directory (a mix of <graph>__det.csv,
<graph>__rand_sNN.csv, and optionally <graph>__{gmr,iomr}_ilp.csv).

    sage -python experiments/real_analyze.py --results results_real --outdir analysis

WORKS WITHOUT THE ILP RESULTS: the reference optimum per graph is the exact ILP where it converged, ELSE
the naive-LP lower bound (from the det array) -- so if the 17h ILP array hasn't landed (or timed out), every
`ratio` is still computed against the LP bound (ref_kind='lower_bound', an UPPER estimate of the true ratio).
`ratio_domr` = size/|H| needs no reference at all and is always available. Randomized algos (pivot, *_rand,
*_bestofk) are summarised over their 30 seeds (median/IQR/mean/std); deterministic and ILP algos are single
values.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

BOUND_ALGOS = {"gmr_lp_naive", "iomr_lp_naive"}     # emit an LP value, not a cover (size/valid blank)
GMR_VARIANTS = {"GMR", "DOMR"}                        # DOMR covers are valid GMR covers
RAND_ALGOS = {"pivot", "iomr_rand", "iomr_bestofk", "gmr_rand", "gmr_bestofk"}
NUM = ("size", "valid", "lp_bound", "exact_opt", "rounds", "cuts", "cpu", "wall", "peak_mb",
       "min_pair_dist", "H", "n", "m", "giant", "n_components", "nonmetric_frac", "w_min", "w_max", "seed")
DIST = ["size", "ratio", "ratio_domr", "cpu", "wall", "rounds", "valid"]


def load(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise SystemExit(f"no CSVs in {path}")
        df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    else:
        df = pd.read_csv(path)
    for c in NUM:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "converged" in df:
        df["converged"] = df["converged"].map({"True": True, "False": False, True: True, False: False})
    return df


def _graph_refs(g):
    """Per-graph reference optima (value, kind) for GMR and IOMR. ILP if it converged, else the naive-LP
    lower bound -- so a missing/timed-out ILP degrades gracefully to a bound."""
    def val(algo, col):
        r = g.loc[g["algo"] == algo, col]
        return r.iloc[0] if len(r) else np.nan

    if pd.notna(val("gmr_ilp", "size")) and val("gmr_ilp", "converged") is True:
        gmr, gk = val("gmr_ilp", "size"), "exact"
    else:
        gmr, gk = val("gmr_lp_naive", "lp_bound"), "lower_bound"
    if pd.notna(val("iomr_ilp", "size")) and val("iomr_ilp", "converged") is True:
        iomr, ik = val("iomr_ilp", "size"), "exact"
    else:
        iomr, ik = val("iomr_lp_naive", "lp_bound"), "lower_bound"
    if pd.isna(gmr):
        gk = "none"
    if pd.isna(iomr):
        ik = "none"
    return pd.Series({"gmr_ref": gmr, "gmr_ref_kind": gk, "iomr_ref": iomr, "iomr_ref_kind": ik})


def derive(df):
    refs = df.groupby("graph")[df.columns.tolist()].apply(_graph_refs).reset_index()
    df = df.merge(refs, on="graph", how="left")
    is_gmr = df["variant"].isin(GMR_VARIANTS)
    df["family"] = np.where(is_gmr, "gmr", "iomr")
    df["ref"] = np.where(is_gmr, df["gmr_ref"], df["iomr_ref"])
    df["ref_kind"] = np.where(is_gmr, df["gmr_ref_kind"], df["iomr_ref_kind"])
    df["ratio"] = df["size"] / df["ref"].where(df["ref"] > 0)
    df["ratio_domr"] = df["size"] / df["H"].where(df["H"] > 0)      # size / |H| -- always available
    return df


def _q(p):
    def f(s):
        return s.quantile(p)
    f.__name__ = f"q{int(p * 100)}"
    return f


def aggregate(df):
    """Per (graph, family, algo, variant): median/IQR/mean/std of the science columns (single-valued for the
    deterministic/ILP algos, a real distribution for the 30-seed randomized ones), plus per-graph meta."""
    keys = ["graph", "family", "algo", "variant"]
    g = df.groupby(keys, dropna=False)
    named = {"n_seeds": ("seed", "nunique"),
             "n": ("n", "first"), "m": ("m", "first"), "H": ("H", "first"),
             "nonmetric_frac": ("nonmetric_frac", "first"), "ref": ("ref", "first"),
             "ref_kind": ("ref_kind", "first"), "peak_mb": ("peak_mb", "median")}
    for c in DIST:
        named[f"{c}_med"] = (c, "median")
        named[f"{c}_q25"] = (c, _q(.25))
        named[f"{c}_q75"] = (c, _q(.75))
        named[f"{c}_mean"] = (c, "mean")
        named[f"{c}_std"] = (c, "std")
    summ = g.agg(**named).reset_index()
    summ["timeout_rate"] = g["status"].apply(lambda s: (s == "timeout").mean()).values
    summ["randomized"] = summ["algo"].isin(RAND_ALGOS)
    return summ.sort_values(keys).reset_index(drop=True)


def report(df, summ):
    print("=" * 72)
    graphs = sorted(df["graph"].unique())
    print(f"LOADED {len(graphs)} graphs, {len(df)} rows, {df['algo'].nunique()} algorithms")
    have_ilp = sorted(df.loc[df["algo"].isin(["gmr_ilp", "iomr_ilp"]), "graph"].unique())
    print(f"ILP present for {len(have_ilp)}/{len(graphs)} graphs" +
          ("" if have_ilp else "  -- ALL ratios are vs the LP lower bound"))
    # per-graph reference kind (exact vs bound) for each family
    rk = summ.groupby(["graph", "family"])["ref_kind"].first().unstack(fill_value="-")
    print("\nreference kind (exact = ILP converged; lower_bound = LP fallback):")
    print(rk.to_string())
    inval = df[df["valid"] == 0]
    if len(inval):
        print(f"\ninvalid covers (valid==0): {len(inval)}")
        print(inval.groupby(["graph", "algo"]).size().to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results_real/ dir or a combined csv")
    ap.add_argument("--outdir", default="analysis")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    df = load(a.results)
    df = derive(df)
    summ = aggregate(df)
    report(df, summ)

    rows_path = os.path.join(a.outdir, "real_rows_with_ratio.csv")
    summ_path = os.path.join(a.outdir, "summary_real.csv")
    df.to_csv(rows_path, index=False)
    summ.to_csv(summ_path, index=False)
    print(f"\nwrote {rows_path} ({len(df)} rows), {summ_path} ({len(summ)} groups)")
    print(f"next:  sage -python experiments/real_plots.py --summary {summ_path}")


if __name__ == "__main__":
    main()
