"""plots.py -- the headline figures from the aggregated summary (experiments/analyze.py output).

Runs LOCALLY; needs matplotlib + pandas. On this Mac use sage's Python (it bundles both):

    sage -python experiments/plots.py --summary analysis/summary.csv --outdir analysis/figs/geometric

Figures are written split by MR axis into per-family subfolders of --outdir:
    <outdir>/gmr/   GMR (general metric repair): variants {GMR, DOMR}   -- any-direction edits
    <outdir>/iomr/  IOMR (increase-only metric repair): variant {IOMR}  -- a different problem/axis

Each family folder gets (median line + IQR band over the samples, one line per algorithm):
  fig1_ratio_vs_n    Exp 1: approximation ratio (size/OPT) vs n, one panel per p          (down: lower better)
  fig2_runtime_vs_n  Exp 1: CPU seconds vs n (log-log, to read the scaling exponent)      (down: lower better)
  fig3_onset_vs_p    Exp 2a: that family's OPT and |H| vs edge density p -- non-metricity onset (~alpha=3/5)
  fig4_ratio_vs_p    Exp 2a+2b: approximation ratio vs edge density p, one panel per exp  (down: lower better)

Exp-1 figures use n on the x-axis; Exp-2 figures use the actual edge density p (= n^-alpha), NOT alpha.
Every metric axis is annotated "higher/lower is better" (the descriptive onset plot has no such direction).
Each figure is PDF + PNG. Missing experiments/families are skipped with a note rather than crashing, so this
runs against partial results while the array is still landing.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")                       # non-interactive: save files, no display
import matplotlib.pyplot as plt             # noqa: E402
import numpy as np                          # noqa: E402
import pandas as pd                          # noqa: E402

FAMILY = {"gmr": {"GMR", "DOMR"}, "iomr": {"IOMR"}}     # MR axis -> the variants drawn in that family's folder
FAMILIES = ["gmr", "iomr"]
FAM_TITLE = {"gmr": "GMR (general MR)", "iomr": "IOMR (increase-only MR)"}
ONSET_ALPHA = 3 / 5                          # coupled-geometric non-metricity onset (Exp 2a)
CONNECT_ALPHA = 0.706                        # G(n,p) connectivity threshold at n=500 (Exp 2b crosses it)


def _ylab(base, better=None):
    """y-axis label with an explicit direction-of-goodness cue ('up'/'down'/None)."""
    tag = {"up": "↑ higher is better", "down": "↓ lower is better"}.get(better)
    return f"{base}\n({tag})" if tag else base


def _color_map(df):
    """Stable algo -> color across every panel and figure (tab20)."""
    algos = sorted(df["algo"].unique())
    cmap = plt.get_cmap("tab20")
    return {a: cmap(i % 20) for i, a in enumerate(algos)}


def _band(ax, sub, xcol, ycol, lo, hi, label, color, logy=False):
    """Draw one algo's median line + IQR band vs xcol. Returns True if anything non-NaN was plotted."""
    sub = sub.sort_values(xcol)
    x = sub[xcol].to_numpy(dtype=float)
    y = sub[ycol].to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if logy:
        m &= (y > 0)
    if not m.any():
        return False
    ax.plot(x[m], y[m], marker="o", ms=3, lw=1.3, color=color, label=label)
    if lo in sub and hi in sub:
        ylo, yhi = sub[lo].to_numpy(dtype=float), sub[hi].to_numpy(dtype=float)
        b = m & np.isfinite(ylo) & np.isfinite(yhi)
        if b.any():
            ax.fill_between(x[b], ylo[b], yhi[b], color=color, alpha=0.15, lw=0)
    return True


def _exp2_n(sub):
    """Recover the fixed n of an Exp-2 slice from p = n^-alpha (n = p^(-1/alpha)); n isn't a summary column."""
    r = sub.dropna(subset=["p", "x"])
    if r.empty:
        return 500.0
    r = r.iloc[0]
    return float(r["p"]) ** (-1.0 / float(r["x"]))


def _save(fig, outdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    wrote {name}.pdf / .png")


def _family(df, fam):
    return df[df["variant"].isin(FAMILY[fam])]


def fig1_ratio_vs_n(df, colors, fam, outdir):
    e = _family(df[df["exp"] == "exp1"], fam)
    if e.empty:
        print(f"    skip fig1 [{fam}] (no exp1 rows)"); return
    ps = sorted(e["p"].dropna().unique())
    fig, axes = plt.subplots(len(ps), 1, figsize=(6.5, 3.6 * len(ps)), squeeze=False, sharex=True)
    for i, p in enumerate(ps):
        ax = axes[i][0]
        drew = False
        for algo, gg in e[e["p"] == p].groupby("algo"):
            drew |= _band(ax, gg, "x", "ratio_med", "ratio_q25", "ratio_q75", algo, colors[algo])
        ax.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.5)
        ax.set_title(f"p={p}")
        if i == len(ps) - 1:
            ax.set_xlabel("n")
        ax.set_ylabel(_ylab("approx. ratio (size / OPT)", "down"))
        if drew:
            ax.legend(fontsize=6, ncol=2)
    fig.suptitle(f"Exp 1 — approximation ratio vs n — {FAM_TITLE[fam]}")
    fig.tight_layout()
    _save(fig, outdir, "fig1_ratio_vs_n")


def fig2_runtime_vs_n(df, colors, fam, outdir):
    e = _family(df[df["exp"] == "exp1"], fam)
    if e.empty:
        print(f"    skip fig2 [{fam}] (no exp1 rows)"); return
    ps = sorted(e["p"].dropna().unique())
    fig, axes = plt.subplots(len(ps), 1, figsize=(6.5, 3.6 * len(ps)), squeeze=False, sharex=True)
    for i, p in enumerate(ps):
        ax = axes[i][0]
        drew = False
        for algo, gg in e[e["p"] == p].groupby("algo"):
            drew |= _band(ax, gg, "x", "cpu_med", "cpu_q25", "cpu_q75", algo, colors[algo], logy=True)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"p={p}")
        if i == len(ps) - 1:
            ax.set_xlabel("n")
        ax.set_ylabel(_ylab("CPU seconds", "down"))
        if drew:
            ax.legend(fontsize=6, ncol=2)
    fig.suptitle(f"Exp 1 — CPU time vs n (log-log) — {FAM_TITLE[fam]}")
    fig.tight_layout()
    _save(fig, outdir, "fig2_runtime_vs_n")


def fig3_onset_vs_p(df, fam, outdir):
    e = df[df["exp"] == "exp2a"]
    if e.empty:
        print(f"    skip fig3 [{fam}] (no exp2a rows)"); return
    fig, ax = plt.subplots(figsize=(7, 5))
    opt_algo, opt_label = ("gmr_ilp", "GMR OPT") if fam == "gmr" else ("iomr_ilp", "IOMR OPT")
    _band(ax, e[e["algo"] == opt_algo], "p", "ref_med", "ref_q25", "ref_q75", opt_label, "tab:blue")
    _band(ax, e[e["algo"] == "domr"], "p", "H_med", "H_q25", "H_q75", "|H| = |DOMR|", "tab:green")
    onset_p = _exp2_n(e) ** (-ONSET_ALPHA)
    ax.axvline(onset_p, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.text(onset_p, ax.get_ylim()[1], r"  onset ($\alpha=3/5$)", va="top", fontsize=8)
    ax.set_xlabel(r"edge density $p\;(=n^{-\alpha})$")
    ax.set_ylabel("optimum / broken-edge count")          # descriptive magnitude -> no better/worse direction
    ax.set_title(f"Exp 2a — non-metricity onset (n=500) — {FAM_TITLE[fam]}")
    ax.legend()
    fig.tight_layout()
    _save(fig, outdir, "fig3_onset_vs_p")


def fig4_ratio_vs_p(df, colors, fam, outdir):
    exps = [x for x in ("exp2a", "exp2b") if (df["exp"] == x).any()]
    if not exps:
        print(f"    skip fig4 [{fam}] (no exp2 rows)"); return
    # sharex=False: exp2a and exp2b span different p-ranges, so let each experiment panel scale to its own.
    fig, axes = plt.subplots(len(exps), 1, figsize=(6.5, 3.6 * len(exps)), squeeze=False, sharex=False)
    for i, exp in enumerate(exps):
        ax = axes[i][0]
        sub = _family(df[df["exp"] == exp], fam)
        drew = False
        for algo, gg in sub.groupby("algo"):
            drew |= _band(ax, gg, "p", "ratio_med", "ratio_q25", "ratio_q75", algo, colors[algo])
        ax.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.5)
        if exp == "exp2b" and not sub.empty:
            ax.axvline(_exp2_n(sub) ** (-CONNECT_ALPHA), color="grey", lw=0.8, ls=":", alpha=0.7)
        ax.set_title(exp)
        ax.set_xlabel(r"edge density $p$")
        ax.set_ylabel(_ylab("approx. ratio (size / OPT)", "down"))
        if drew:
            ax.legend(fontsize=6, ncol=2)
    fig.suptitle(f"Exp 2 — approximation ratio vs edge density p — {FAM_TITLE[fam]}")
    fig.tight_layout()
    _save(fig, outdir, "fig4_ratio_vs_p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="summary.csv from experiments/analyze.py")
    ap.add_argument("--outdir", default="analysis/figs/geometric",
                    help="root for the figures; per-family gmr/ and iomr/ subfolders are created under it")
    a = ap.parse_args()

    df = pd.read_csv(a.summary)
    colors = _color_map(df)
    print(f"loaded {len(df)} summary rows; experiments={sorted(df['exp'].unique())}")
    for fam in FAMILIES:
        sub = os.path.join(a.outdir, fam)
        os.makedirs(sub, exist_ok=True)
        print(f"[{fam}] {FAM_TITLE[fam]} -> {sub}")
        fig1_ratio_vs_n(df, colors, fam, sub)
        fig2_runtime_vs_n(df, colors, fam, sub)
        fig3_onset_vs_p(df, fam, sub)
        fig4_ratio_vs_p(df, colors, fam, sub)
    print(f"figures -> {a.outdir}/{{gmr,iomr}}")


if __name__ == "__main__":
    main()
