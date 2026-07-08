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
              "min_pair_dist", "n", "p", "alpha", "V", "E", "w_min", "w_max", "giant", "H", "seed", "light_frac"):
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
    tgt = int(spc.max()) if len(spc) else 0                # infer target from the fullest config (30 small/40 full)
    print(f"\nsamples per config: min={spc.min()} max={spc.max()} (target {tgt}); "
          f"{(spc < tgt).sum()} configs under target")
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
    # DOMR ratio: size / |DOMR| (= size / H, the broken-edge count). Since DOMR is a feasible GMR cover,
    # GMR_OPT <= |DOMR|, so for GMR variants this is a LOWER bound on the true GMR approximation ratio.
    df["ratio_domr"] = df["size"] / df["H"].where(df["H"] > 0)
    return df


def _q(p):
    def f(s):
        return s.quantile(p)
    f.__name__ = f"q{int(p*100)}"
    return f


# Columns summarised with the full distribution (median, IQR, mean, std) over the samples of a config.
# size/ratio/ratio_domr = the science; H = |DOMR| non-metricity magnitude; ref = the absolute OPT (plot
# OPT-vs-alpha directly); cpu/wall = runtime (IQR -> error bars); rounds/cuts = cutting-plane work.
DIST_COLS = ["size", "ratio", "ratio_domr", "light_frac", "H", "ref", "cpu", "wall", "rounds", "cuts"]


def aggregate(df):
    """Per (experiment, x-parameter, algorithm) summary over the samples. x = n (Exp 1) or alpha (Exp 2).
    Each column in DIST_COLS gets median/q25/q75/mean/std; plus memory, weight spread, validity/convergence,
    region-growing separation diagnostics, and status rates. Redundant-but-harmless: H and ref are per-task,
    so they repeat across algos of the same MR variant (the plot layer just filters to one)."""
    df = df.copy()
    df["x"] = np.where(df["exp"] == "exp1", df["n"], df["alpha"])
    keys = ["exp", "model", "p", "x", "algo", "variant"]
    g = df.groupby(keys, dropna=False)

    named = {"n_samples": ("sample", "nunique")}
    for c in DIST_COLS:
        named[f"{c}_med"] = (c, "median")
        named[f"{c}_q25"] = (c, _q(.25))
        named[f"{c}_q75"] = (c, _q(.75))
        named[f"{c}_mean"] = (c, "mean")
        named[f"{c}_std"] = (c, "std")
    named.update(
        peak_med=("peak_mb", "median"), peak_max=("peak_mb", "max"),
        w_max_med=("w_max", "median"), w_max_min=("w_max", "min"), w_max_max=("w_max", "max"),
        valid_rate=("valid", "mean"),
        converged_rate=("converged", "mean"),
        full_sep_rate=("full_separation", "mean"),           # region growing: fraction with full separation
        min_pair_dist_med=("min_pair_dist", "median"),       # region growing: median LP heavy-pair distance
        ref_kind=("ref_kind", lambda s: s.mode().iat[0] if len(s.mode()) else None),
    )
    summ = g.agg(**named).reset_index()
    summ["timeout_rate"] = g["status"].apply(lambda s: (s == "timeout").mean()).values
    return summ.sort_values(keys).reset_index(drop=True)


def runtime_weight_fit(df, min_points=8):
    """Per-algorithm log-log OLS: log(cpu) ~ intercept + size_exp*log(n) + weight_exp*log(w_max).

    `weight_exp` is the runtime-vs-weight-size dependency metric the design asks for: how CPU scales with
    the largest edge weight `w_max`. The rsp oracle's weight-budget DP is O(w_max*n^2), so we expect
    `weight_exp ~ 1` for the rsp methods (gmr_lp_rsp, iomr_lp_rsp, iomr_thr_rsp) and `~0` for the naive /
    purely combinatorial ones. `size_exp` is the graph-size exponent. Pools every instance where the algo
    ran (status ok, cpu>0); an exponent is NaN when its regressor doesn't vary in that pool (e.g. n is
    pinned at 500 across Exp 2, so `size_exp` is only identifiable once Exp 1's n-sweep is present). `r2`
    and `n_points` gauge trust. This is a coarse scaling summary for the runtime story, not a causal model
    (n and w_max are correlated within a single experiment; identifiability comes from pooling Exp 1 + 2).
    """
    d = df[(df["status"] == "ok") & (df["cpu"] > 0) & (df["n"] > 0) & (df["w_max"] > 0)].copy()
    rows = []
    for (algo, variant), g in d.groupby(["algo", "variant"], dropna=False):
        y = np.log(g["cpu"].to_numpy(dtype=float))
        ln = np.log(g["n"].to_numpy(dtype=float))
        lw = np.log(g["w_max"].to_numpy(dtype=float))
        cols, names = [np.ones_like(y)], ["intercept"]
        if ln.std() > 1e-9:                     # size_exp identifiable only if n varies in this pool
            cols.append(ln); names.append("size_exp")
        if lw.std() > 1e-9:                     # weight_exp identifiable only if w_max varies
            cols.append(lw); names.append("weight_exp")
        rec = {"algo": algo, "variant": variant, "n_points": int(len(y)),
               "size_exp": np.nan, "weight_exp": np.nan, "intercept": np.nan, "r2": np.nan,
               "n_min": int(g["n"].min()), "n_max": int(g["n"].max()),
               "w_max_min": float(g["w_max"].min()), "w_max_max": float(g["w_max"].max()),
               "cpu_med": float(np.median(g["cpu"]))}
        if len(y) >= min_points and len(cols) >= 2:
            X = np.column_stack(cols)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            ss_tot = float(((y - y.mean()) ** 2).sum())
            rec["r2"] = float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan
            for name, b in zip(names, beta):
                rec[name] = float(b)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["algo", "variant"]).reset_index(drop=True)


def rounds_scaling_fit(df, min_points=8):
    """Per-algorithm log-log OLS: log(rounds) ~ intercept + size_exp*log(n) + weight_exp*log(w_max).

    `rounds` is the separation-oracle iteration count (cutting-plane rounds). It's the analogue of
    runtime_weight_fit for ITERATIONS instead of CPU: `size_exp` is how the round count grows with n,
    `weight_exp` with the largest weight w_max (the rsp weight-budget oracle's cost). Populated for the exact
    ILP separation (gmr_ilp, iomr_ilp) and -- for results produced with the round-surfacing patch -- the
    LP-separation (gmr_lp_*, iomr_lp_*) and covering-LP (iomr_thr_*, iomr_bestofk/rand/regiongrow) methods.
    OLDER results simply have blank `rounds` for the latter, so those algos are skipped here (no crash) --
    only rows with status ok and rounds>0 are pooled. An exponent is NaN when its regressor doesn't vary in
    the pool (e.g. n pinned across Exp 2); r2/n_points gauge trust. Pairs with runtime_weight_fit: dividing
    the CPU-vs-n exponent by this rounds-vs-n exponent isolates per-round cost growth from round-count growth.
    """
    if "rounds" not in df:
        return pd.DataFrame(columns=["algo", "variant", "n_points", "size_exp", "weight_exp",
                                     "intercept", "r2", "n_min", "n_max", "rounds_med", "rounds_max"])
    d = df[(df["status"] == "ok") & (df["rounds"] > 0) & (df["n"] > 0) & (df["w_max"] > 0)].copy()
    rows = []
    for (algo, variant), g in d.groupby(["algo", "variant"], dropna=False):
        y = np.log(g["rounds"].to_numpy(dtype=float))
        ln = np.log(g["n"].to_numpy(dtype=float))
        lw = np.log(g["w_max"].to_numpy(dtype=float))
        cols, names = [np.ones_like(y)], ["intercept"]
        if ln.std() > 1e-9:                     # size_exp identifiable only if n varies in this pool
            cols.append(ln); names.append("size_exp")
        if lw.std() > 1e-9:                     # weight_exp identifiable only if w_max varies
            cols.append(lw); names.append("weight_exp")
        rec = {"algo": algo, "variant": variant, "n_points": int(len(y)),
               "size_exp": np.nan, "weight_exp": np.nan, "intercept": np.nan, "r2": np.nan,
               "n_min": int(g["n"].min()), "n_max": int(g["n"].max()),
               "rounds_med": float(np.median(g["rounds"])), "rounds_max": int(g["rounds"].max())}
        if len(y) >= min_points and len(cols) >= 2:
            X = np.column_stack(cols)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            ss_tot = float(((y - y.mean()) ** 2).sum())
            rec["r2"] = float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan
            for name, b in zip(names, beta):
                rec[name] = float(b)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["algo", "variant"]).reset_index(drop=True)


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
    fit = runtime_weight_fit(df)
    rfit = rounds_scaling_fit(df)

    rows_path = os.path.join(a.outdir, "rows_with_ratio.csv")
    summ_path = os.path.join(a.outdir, "summary.csv")
    fit_path = os.path.join(a.outdir, "runtime_weight_fit.csv")
    rfit_path = os.path.join(a.outdir, "rounds_scaling_fit.csv")
    df.to_csv(rows_path, index=False)
    summ.to_csv(summ_path, index=False)
    fit.to_csv(fit_path, index=False)
    rfit.to_csv(rfit_path, index=False)
    print(f"\nwrote {rows_path} ({len(df)} rows), {summ_path} ({len(summ)} groups), "
          f"{fit_path} ({len(fit)} algos), {rfit_path} ({len(rfit)} algos)")

    print("\nruntime scaling  log(cpu) ~ size_exp*log(n) + weight_exp*log(w_max):")
    print(fit[["algo", "variant", "n_points", "size_exp", "weight_exp", "r2"]].to_string(index=False))
    print("\niteration scaling  log(rounds) ~ size_exp*log(n) + weight_exp*log(w_max):")
    if len(rfit):
        print(rfit[["algo", "variant", "n_points", "size_exp", "weight_exp", "r2"]].to_string(index=False))
    else:
        print("  (no rows with rounds>0 -- results predate the round-surfacing patch)")

    if problems:
        print(f"\n*** {problems} non-GMR-gap invalid covers -- investigate before trusting the analysis ***")


if __name__ == "__main__":
    main()
