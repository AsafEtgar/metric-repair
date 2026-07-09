"""rgg_analyze.py -- aggregate the RGG experiment output (rgg_harness.py) into tidy summaries for plotting.

Runs LOCALLY (needs pandas; plotting is a separate step -- rgg_plots.py -- and needs matplotlib). Input is
either the combined results_rgg*_all.csv or a directory of task_*.csv.

    sage -python experiments/rgg_analyze.py --results results_rgg_poc_all.csv --outdir analysis
    sage -python experiments/rgg_analyze.py --results results_rgg_poc     --outdir analysis   # a dir

Two summaries (tidy, one row per group, median/IQR/mean/std over the samples of a config):

  summary_edit.csv  Part 1 & 2 edit quality: per (part, sweep, x, algo) the edit-precision/recall of the
                    repair cover vs the KNOWN planted `corrupted` set, plus cover `size`, `ratio_domr`
                    (= size/|H|, a lower bound on the GMR approximation ratio), `ratio` (= size/OPT), cpu.
                    Part-2's 4 kNN rows per (task,algo) collapse to their shared edit metrics first.

  summary_knn.csv   Part 2 kNN recovery: per (sweep, x, algo, knn_k) the Jaccard of the true-graph kNN vs
                    the corrupted (jaccard_TC, damage baseline) and vs the repaired (jaccard_TF), the repair
                    recall_TF, the `lift` (= jaccard_TF - jaccard_TC), and triplet-ordering accuracy (C, F).

The natural x-axis of each OFAT sweep is its swept knob (SWEEP_X); S6 (magnitude x frac_q) keeps frac_q as a
secondary `series`. `ratio` uses the RGG-appropriate optimum: the ILP where it converged, else the naive LP
lower bound (the 3 rsp methods are dropped on float RGG, so there's no rsp reference here).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

# columns to coerce to numeric (blanks -> NaN)
NUM = ("size", "valid", "lp_bound", "exact_opt", "rounds", "cuts", "cpu", "wall", "peak_mb",
       "min_pair_dist", "H", "n_corrupted", "edit_precision", "edit_recall", "light_frac",
       "knn_k", "jaccard_TC", "jaccard_TF", "recall_TF", "lift", "triplet_acc_C", "triplet_acc_F",
       "n", "radius", "deg", "k", "magnitude", "frac_q", "n_jitter", "jitter", "subset_s",
       "sample", "seed", "V", "E", "w_min", "w_max", "giant")

# each OFAT sweep varies exactly one knob -> that's its x-axis; S6 additionally varies frac_q (a 2nd series).
SWEEP_X = {
    "S1": "n", "P2size": "n", "POCsize_inflate": "n", "POCsize_jitter": "n",
    "S2": "deg", "S2k": "k",
    "S3": "magnitude", "S3d": "magnitude", "S6": "magnitude",
    "S4i": "frac_q", "S4d": "frac_q",
    "S5a": "n_jitter", "P2n": "n_jitter",
    "S5b": "jitter", "P2j": "jitter",
    "S5c": "subset_s", "P2s": "subset_s",
}
SWEEP_SERIES = {"S6": "frac_q"}                 # 2D sweep: keep the secondary knob as a line series

GMR_REF_VARIANTS = {"GMR", "DOMR"}              # DOMR covers are valid GMR covers -> compare to the GMR OPT
DIST_EDIT = ["edit_precision", "edit_recall", "light_frac", "size", "ratio", "ratio_domr", "cpu", "wall", "rounds", "cuts"]
DIST_KNN = ["jaccard_TC", "jaccard_TF", "recall_TF", "lift", "triplet_acc_C", "triplet_acc_F"]


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
    if "converged" in df:
        df["converged"] = df["converged"].map({"True": True, "False": False, True: True, False: False})
    return df


def _task_refs(g):
    """Per-task reference optima (value, kind) for GMR and IOMR. No rsp on float RGG: ILP if it converged,
    else the naive LP lower bound. kind in {exact, lower_bound, none}."""
    def val(algo, col):
        r = g.loc[g["algo"] == algo, col]
        return r.iloc[0] if len(r) else np.nan

    def exact_ok(algo):
        """`converged` alone is not enough. When status != "ok", harness._aggregate ANDed converged over only
        the components that returned and summed size/exact_opt over those same few -- a partial optimum that
        reads as complete (27 such ILP rows in results_rgg_full_all.csv). And an unverified cover is not an
        optimum. See AUDIT_REPORT.md A2. Falling back to the LP bound is honest and is labelled as such."""
        return (pd.notna(val(algo, "size")) and val(algo, "converged") is True
                and val(algo, "status") == "ok" and val(algo, "valid") == 1)

    if exact_ok("gmr_ilp"):
        gmr, gk = val("gmr_ilp", "size"), "exact"
    else:
        gmr, gk = val("gmr_lp_naive", "lp_bound"), "lower_bound"
    if exact_ok("iomr_ilp"):
        iomr, ik = val("iomr_ilp", "size"), "exact"
    else:
        iomr, ik = val("iomr_lp_naive", "lp_bound"), "lower_bound"
    if pd.isna(gmr):
        gk = "none"
    if pd.isna(iomr):
        ik = "none"
    return pd.Series({"gmr_ref": gmr, "gmr_ref_kind": gk, "iomr_ref": iomr, "iomr_ref_kind": ik})


def derive(df):
    """Add ratio (= size/OPT for the row's MR variant) and ratio_domr (= size/|H|, always available)."""
    refs = df.groupby("task")[df.columns.tolist()].apply(_task_refs).reset_index()
    df = df.merge(refs, on="task", how="left")
    is_gmr = df["variant"].isin(GMR_REF_VARIANTS)
    df["ref"] = np.where(is_gmr, df["gmr_ref"], df["iomr_ref"])
    df["ref_kind"] = np.where(is_gmr, df["gmr_ref_kind"], df["iomr_ref_kind"])
    df["ratio"] = df["size"] / df["ref"].where(df["ref"] > 0)
    df["ratio_domr"] = df["size"] / df["H"].where(df["H"] > 0)     # lower bound on the GMR approx ratio
    return df


def add_x(df):
    """Attach the per-sweep x-axis knob as `x` and the optional secondary knob as `series`."""
    df = df.copy()
    df["x"] = np.nan
    df["series"] = np.nan
    for sw, col in SWEEP_X.items():
        m = df["sweep"] == sw
        if m.any() and col in df:
            df.loc[m, "x"] = pd.to_numeric(df.loc[m, col], errors="coerce")
    for sw, col in SWEEP_SERIES.items():
        m = df["sweep"] == sw
        if m.any() and col in df:
            df.loc[m, "series"] = pd.to_numeric(df.loc[m, col], errors="coerce")
    unknown = sorted(set(df["sweep"].unique()) - set(SWEEP_X))
    if unknown:
        print(f"  note: sweeps with no x-mapping (left out of plots): {unknown}")
    return df


def _q(p):
    def f(s):
        return s.quantile(p)
    f.__name__ = f"q{int(p * 100)}"
    return f


def _dist_aggs(cols):
    named = {}
    for c in cols:
        named[f"{c}_med"] = (c, "median")
        named[f"{c}_q25"] = (c, _q(.25))
        named[f"{c}_q75"] = (c, _q(.75))
        named[f"{c}_mean"] = (c, "mean")
        named[f"{c}_std"] = (c, "std")
    return named


def aggregate_edit(df):
    """One row per (task, algo) -- Part-2's 4 kNN rows share edit metrics, so collapse them -- then summarise
    per (part, sweep, x, series, algo, variant) over the samples."""
    one = df.sort_values("knn_k").groupby(["task", "algo"], as_index=False).first()
    keys = ["part", "sweep", "mode", "break_type", "direction", "x", "series", "algo", "variant"]
    g = one.groupby(keys, dropna=False)
    named = {"n_samples": ("sample", "nunique"),
             "H_med": ("H", "median"), "n_corrupted_med": ("n_corrupted", "median"),
             "valid_rate": ("valid", "mean"), "ref_kind": ("ref_kind", lambda s: s.mode().iat[0]
                                                            if len(s.mode()) else None)}
    named.update(_dist_aggs(DIST_EDIT))
    return g.agg(**named).reset_index().sort_values(keys).reset_index(drop=True)


def aggregate_knn(df):
    """Part 2 only: summarise the T/C/F kNN metrics per (sweep, x, series, algo, variant, knn_k)."""
    k = df[df["knn_k"].notna()].copy()
    keys = ["sweep", "mode", "x", "series", "algo", "variant", "knn_k"]
    if k.empty:
        return pd.DataFrame(columns=keys + ["n_samples"] + [f"{c}_med" for c in DIST_KNN])
    g = k.groupby(keys, dropna=False)
    named = {"n_samples": ("sample", "nunique")}
    named.update(_dist_aggs(DIST_KNN))
    return g.agg(**named).reset_index().sort_values(keys).reset_index(drop=True)


def report(df):
    print("=" * 70)
    print(f"LOADED {df['task'].nunique()} tasks, {len(df)} rows, {df['algo'].nunique()} algos, "
          f"parts={sorted(df['part'].dropna().unique())}")
    print("sweeps present:", ", ".join(sorted(df['sweep'].dropna().unique())))
    key = df[df["algo"] == "domr"]
    spc = key.groupby(["sweep", "x"], dropna=False)["sample"].nunique()
    if len(spc):
        print(f"samples/config: min={int(spc.min())} max={int(spc.max())} "
              f"({(spc < spc.max()).sum()} configs below the max)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results_rgg*_all.csv or a dir of task_*.csv")
    ap.add_argument("--outdir", default="analysis")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    df = load(a.results)
    df = derive(df)
    df = add_x(df)
    report(df)

    # This analyzer previously neither filtered NOR reported invalid covers, and has no timeout_rate column,
    # so a run whose covers silently stopped verifying looked identical to a clean one. An invalid cover is
    # an insufficient cover -> |S| understated -> size_med / ratio_domr_med biased in the algorithm's favour.
    if "status" in df:
        n_to = int((df["status"] == "timeout").sum())
        n_sk = int(df["status"].astype(str).str.startswith("skipped").sum())
        if n_to or n_sk:
            print(f"\nnote: {n_to} timed-out and {n_sk} skipped rows -- these carry size=NaN and are silently "
                  f"dropped by median(); the per-group n_samples still counts them (AUDIT_REPORT.md A7).")
    inval = df["valid"] == 0
    if inval.any():
        print(f"\nDROPPING {int(inval.sum())} invalid-cover rows from the aggregate:")
        print(df[inval].groupby(["sweep", "algo"]).size().to_string())
    df_ok = df[~inval]

    edit = aggregate_edit(df_ok)
    knn = aggregate_knn(df_ok)

    rows_path = os.path.join(a.outdir, "rgg_rows_with_ratio.csv")   # rgg_ prefix: geometric writes rows_with_ratio.csv
    edit_path = os.path.join(a.outdir, "summary_edit.csv")
    knn_path = os.path.join(a.outdir, "summary_knn.csv")
    df.to_csv(rows_path, index=False)
    edit.to_csv(edit_path, index=False)
    knn.to_csv(knn_path, index=False)
    print(f"\nwrote {rows_path} ({len(df)} rows)")
    print(f"wrote {edit_path} ({len(edit)} groups)")
    print(f"wrote {knn_path} ({len(knn)} groups)")
    print("\nnext:  sage -python experiments/rgg_plots.py "
          f"--edit {edit_path} --knn {knn_path} --outdir {os.path.join(a.outdir, 'figs')}")


if __name__ == "__main__":
    main()
