"""analyze.py -- load, validate, derive approximation ratios, and aggregate the experiment results.

Runs LOCALLY (needs pandas; plotting is a separate step and needs matplotlib) -- NOT in the cluster's
numpy/scipy env. Input is either the combined results_all.csv or a directory of task_*.csv.

    python experiments/analyze.py --results results_all.csv --outdir analysis
    python experiments/analyze.py --results results/ --outdir analysis      # a dir of per-task CSVs

Pipeline: load -> validate (health report) -> derive (per-instance reference + ratio) -> aggregate (over
the 40 samples -> tidy per-experiment summary). See EXPERIMENTS.md for the column schema.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

# Lower-bound rows: return an LP value, not a cover (size/valid are blank; the number is in lp_bound).
BOUND_ALGOS = {"gmr_lp_naive", "iomr_lp_naive", "iomr_lp_rsp"}
GMR_REF_VARIANTS = {"GMR", "DOMR"}          # DOMR covers are valid GMR covers -> compare to the GMR optimum


def load(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "task_*.csv")))
        if not files:
            raise SystemExit(f"no task_*.csv in {path}")
        df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    else:
        df = pd.read_csv(path)
    # coerce the tri-state / numeric columns (blanks -> NaN)
    for c in ("size", "valid", "lp_bound", "exact_opt", "rounds", "cuts", "cpu", "wall", "peak_mb",
              "min_pair_dist", "n", "p", "alpha", "V", "E", "w_min", "w_max", "giant", "H", "seed"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("converged", "guaranteed", "full_separation"):
        if c in df:
            df[c] = df[c].map({"True": True, "False": False, True: True, False: False})
    return df


def validate(df):
    """Print a health report; return the count of hard problems (invalid covers)."""
    print("=" * 70)
    print(f"LOADED: {df['task'].nunique()} tasks, {len(df)} rows, "
          f"{df['algo'].nunique()} algorithms, experiments={sorted(df['exp'].unique())}")
    print("\nSTATUS counts:")
    print(df["status"].value_counts().to_string())

    # invalid covers: valid==0 is either a bug OR (on gmr_lp_rsp) a GMR LP integrality gap
    inval = df[df["valid"] == 0]
    print(f"\nINVALID covers (valid==0): {len(inval)}")
    if len(inval):
        for _, r in inval.iterrows():
            tag = "  <- GMR LP gap (expected-ish)" if r["algo"] == "gmr_lp_rsp" else "  <- INVESTIGATE"
            print(f"  task {r['task']:>5} {r['algo']:>15} {r['exp']} n={r['n']} p={r['p']}{tag}")

    # per-algorithm status breakdown (are timeouts/skips where we expect?)
    print("\nper-algorithm status (rows):")
    tab = df.groupby("algo")["status"].value_counts().unstack(fill_value=0)
    print(tab.to_string())

    # samples per config (should be N_SAMPLES=40 in a full run)
    key = ["exp", "p", "n", "alpha"]
    spc = df[df["algo"] == "domr"].groupby(key, dropna=False)["sample"].nunique()
    print(f"\nsamples per config: min={spc.min()} max={spc.max()} (target 40); "
          f"{(spc < 40).sum()} configs under 40")
    return len(inval[inval["algo"] != "gmr_lp_rsp"])   # non-gmr-gap invalids = real problems


def _task_refs(g):
    """Per-task reference optima: (value, kind) for GMR and IOMR. kind in {exact, lower_bound, none}."""
    def val(algo, col):
        r = g.loc[g["algo"] == algo, col]
        return r.iloc[0] if len(r) else np.nan

    # GMR: exact ILP if converged; else the integral LP (valid rounded support); else its LP lower bound.
    if pd.notna(val("gmr_ilp", "size")) and val("gmr_ilp", "converged") is True:
        gmr, gk = val("gmr_ilp", "size"), "exact"
    elif pd.notna(val("gmr_lp_rsp", "size")) and val("gmr_lp_rsp", "valid") == 1:
        gmr, gk = val("gmr_lp_rsp", "size"), "exact"
    else:
        gmr, gk = val("gmr_lp_rsp", "lp_bound"), "lower_bound"

    # IOMR: exact ILP where it converged; else the tightest LP lower bound (rsp preferred over naive).
    if pd.notna(val("iomr_ilp", "size")) and val("iomr_ilp", "converged") is True:
        iomr, ik = val("iomr_ilp", "size"), "exact"
    else:
        lbs = [x for x in (val("iomr_lp_rsp", "lp_bound"), val("iomr_lp_naive", "lp_bound")) if pd.notna(x)]
        iomr, ik = (max(lbs), "lower_bound") if lbs else (np.nan, "none")
    return pd.Series({"gmr_ref": gmr, "gmr_ref_kind": gk, "iomr_ref": iomr, "iomr_ref_kind": ik})


def derive(df):
    """Add ref, ref_kind, ratio per row. ratio = cover size / reference optimum for that MR variant.
    ref_kind='exact' -> true approximation ratio; 'lower_bound' -> an UPPER estimate (size/LB >= true)."""
    refs = df.groupby("task")[df.columns.tolist()].apply(_task_refs).reset_index()
    df = df.merge(refs, on="task", how="left")
    is_gmr = df["variant"].isin(GMR_REF_VARIANTS)
    df["ref"] = np.where(is_gmr, df["gmr_ref"], df["iomr_ref"])
    df["ref_kind"] = np.where(is_gmr, df["gmr_ref_kind"], df["iomr_ref_kind"])
    df["ratio"] = df["size"] / df["ref"]
    return df


def _q(p):
    def f(s):
        return s.quantile(p)
    f.__name__ = f"q{int(p*100)}"
    return f


def aggregate(df):
    """Median + IQR over samples, per (experiment, x-parameter, algorithm). x = n (Exp 1) or alpha (Exp 2)."""
    df = df.copy()
    df["x"] = np.where(df["exp"] == "exp1", df["n"], df["alpha"])
    keys = ["exp", "model", "p", "x", "algo", "variant"]
    g = df.groupby(keys, dropna=False)
    summ = g.agg(
        n_samples=("sample", "nunique"),
        size_med=("size", "median"), size_q25=("size", _q(.25)), size_q75=("size", _q(.75)),
        ratio_med=("ratio", "median"), ratio_q25=("ratio", _q(.25)), ratio_q75=("ratio", _q(.75)),
        cpu_med=("cpu", "median"), wall_med=("wall", "median"), peak_med=("peak_mb", "median"),
        valid_rate=("valid", "mean"),
        converged_rate=("converged", "mean"),
        ref_kind=("ref_kind", lambda s: s.mode().iat[0] if len(s.mode()) else None),
    ).reset_index()
    summ["timeout_rate"] = g["status"].apply(lambda s: (s == "timeout").mean()).values
    return summ.sort_values(keys).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results_all.csv or a dir of task_*.csv")
    ap.add_argument("--outdir", default="analysis")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    df = load(a.results)
    problems = validate(df)
    df = derive(df)
    summ = aggregate(df)

    rows_path = os.path.join(a.outdir, "rows_with_ratio.csv")
    summ_path = os.path.join(a.outdir, "summary.csv")
    df.to_csv(rows_path, index=False)
    summ.to_csv(summ_path, index=False)
    print(f"\nwrote {rows_path} ({len(df)} rows) and {summ_path} ({len(summ)} groups)")
    if problems:
        print(f"\n*** {problems} non-GMR-gap invalid covers -- investigate before trusting the analysis ***")


if __name__ == "__main__":
    main()
