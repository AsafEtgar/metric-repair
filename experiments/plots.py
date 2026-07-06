"""plots.py -- the headline figures from the aggregated summary (experiments/analyze.py output).

Runs LOCALLY; needs matplotlib + pandas. On this Mac use sage's Python (it bundles both):

    sage -python experiments/plots.py --summary analysis/summary.csv --outdir analysis/figs

Four headline figures (median line + IQR band over the 40 samples), each split by MR-variant *family*
(GMR family = {GMR, DOMR}; IOMR family = {IOMR}) with ALL algorithms of that family drawn -- we curate
which to keep later:

  fig1_ratio_vs_n    Exp 1: approximation ratio (size/OPT) vs n, panels = p x family
  fig2_runtime_vs_n  Exp 1: CPU seconds vs n (log-log, to read the scaling exponent), panels = p x family
  fig3_onset_vs_p    Exp 2a: OPT (GMR & IOMR) and |H| vs edge density p -- non-metricity onset (~ alpha=3/5)
  fig4_ratio_vs_p    Exp 2a+2b: approximation ratio vs edge density p, panels = experiment x family

Exp-1 figures use n on the x-axis; Exp-2 figures use the actual edge density p (= n^-alpha), NOT alpha.
Each figure is written as both PDF and PNG. Missing experiments (e.g. only Exp 2 present) are skipped with
a note rather than crashing, so this runs against partial results while the array is still landing.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")                       # non-interactive: save files, no display
import matplotlib.pyplot as plt             # noqa: E402
import numpy as np                          # noqa: E402
import pandas as pd                          # noqa: E402

FAMILY = {"GMR": {"GMR", "DOMR"}, "IOMR": {"IOMR"}}     # a family's panel draws every algo with these variants
FAMILIES = ["GMR", "IOMR"]
ONSET_ALPHA = 3 / 5                          # coupled-geometric non-metricity onset (Exp 2a)
CONNECT_ALPHA = 0.706                        # G(n,p) connectivity threshold at n=500 (Exp 2b crosses it)


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
    print(f"  wrote {name}.pdf / .png")


def _family(df, fam):
    return df[df["variant"].isin(FAMILY[fam])]


def fig1_ratio_vs_n(df, colors, outdir):
    e = df[df["exp"] == "exp1"]
    if e.empty:
        print("  skip fig1 (no exp1 rows)"); return
    ps = sorted(e["p"].dropna().unique())
    fig, axes = plt.subplots(len(ps), len(FAMILIES), figsize=(11, 3.6 * len(ps)),
                             squeeze=False, sharex=True)
    for i, p in enumerate(ps):
        for j, fam in enumerate(FAMILIES):
            ax = axes[i][j]
            sub = _family(e[e["p"] == p], fam)
            drew = False
            for algo, gg in sub.groupby("algo"):
                drew |= _band(ax, gg, "x", "ratio_med", "ratio_q25", "ratio_q75", algo, colors[algo])
            ax.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.5)
            ax.set_title(f"{fam} family, p={p}")
            if i == len(ps) - 1:
                ax.set_xlabel("n")
            if j == 0:
                ax.set_ylabel("approx. ratio (size / OPT)")
            if drew:
                ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Exp 1 — approximation ratio vs n")
    fig.tight_layout()
    _save(fig, outdir, "fig1_ratio_vs_n")


def fig2_runtime_vs_n(df, colors, outdir):
    e = df[df["exp"] == "exp1"]
    if e.empty:
        print("  skip fig2 (no exp1 rows)"); return
    ps = sorted(e["p"].dropna().unique())
    fig, axes = plt.subplots(len(ps), len(FAMILIES), figsize=(11, 3.6 * len(ps)),
                             squeeze=False, sharex=True)
    for i, p in enumerate(ps):
        for j, fam in enumerate(FAMILIES):
            ax = axes[i][j]
            sub = _family(e[e["p"] == p], fam)
            drew = False
            for algo, gg in sub.groupby("algo"):
                drew |= _band(ax, gg, "x", "cpu_med", "cpu_q25", "cpu_q75", algo, colors[algo], logy=True)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(f"{fam} family, p={p}")
            if i == len(ps) - 1:
                ax.set_xlabel("n")
            if j == 0:
                ax.set_ylabel("CPU seconds")
            if drew:
                ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Exp 1 — CPU time vs n (log-log)")
    fig.tight_layout()
    _save(fig, outdir, "fig2_runtime_vs_n")


def fig3_onset_vs_p(df, outdir):
    e = df[df["exp"] == "exp2a"]
    if e.empty:
        print("  skip fig3 (no exp2a rows)"); return
    fig, ax = plt.subplots(figsize=(7, 5))
    curves = [("gmr_ilp", "ref_med", "ref_q25", "ref_q75", "GMR OPT", "tab:blue"),
              ("iomr_ilp", "ref_med", "ref_q25", "ref_q75", "IOMR OPT", "tab:orange"),
              ("domr", "H_med", "H_q25", "H_q75", "|H| = |DOMR|", "tab:green")]
    for algo, y, lo, hi, label, color in curves:
        _band(ax, e[e["algo"] == algo], "p", y, lo, hi, label, color)
    onset_p = _exp2_n(e) ** (-ONSET_ALPHA)
    ax.axvline(onset_p, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.text(onset_p, ax.get_ylim()[1], r"  onset ($\alpha=3/5$)", va="top", fontsize=8)
    ax.set_xlabel(r"edge density $p\;(=n^{-\alpha})$")
    ax.set_ylabel("optimum / broken-edge count")
    ax.set_title("Exp 2a — non-metricity onset (coupled geometric, n=500)")
    ax.legend()
    fig.tight_layout()
    _save(fig, outdir, "fig3_onset_vs_p")


def fig4_ratio_vs_p(df, colors, outdir):
    exps = [x for x in ("exp2a", "exp2b") if (df["exp"] == x).any()]
    if not exps:
        print("  skip fig4 (no exp2 rows)"); return
    # sharex=False: exp2a and exp2b span different p-ranges, so let each experiment row scale to its own.
    fig, axes = plt.subplots(len(exps), len(FAMILIES), figsize=(11, 3.6 * len(exps)),
                             squeeze=False, sharex=False)
    for i, exp in enumerate(exps):
        for j, fam in enumerate(FAMILIES):
            ax = axes[i][j]
            sub = _family(df[df["exp"] == exp], fam)
            drew = False
            for algo, gg in sub.groupby("algo"):
                drew |= _band(ax, gg, "p", "ratio_med", "ratio_q25", "ratio_q75", algo, colors[algo])
            ax.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.5)
            if exp == "exp2b" and not sub.empty:
                ax.axvline(_exp2_n(sub) ** (-CONNECT_ALPHA), color="grey", lw=0.8, ls=":", alpha=0.7)
            ax.set_title(f"{exp} — {fam} family")
            ax.set_xlabel(r"edge density $p$")
            if j == 0:
                ax.set_ylabel("approx. ratio (size / OPT)")
            if drew:
                ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Exp 2 — approximation ratio vs edge density p")
    fig.tight_layout()
    _save(fig, outdir, "fig4_ratio_vs_p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="summary.csv from experiments/analyze.py")
    ap.add_argument("--outdir", default="analysis/figs")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    df = pd.read_csv(a.summary)
    colors = _color_map(df)
    print(f"loaded {len(df)} summary rows; experiments={sorted(df['exp'].unique())}")
    fig1_ratio_vs_n(df, colors, a.outdir)
    fig2_runtime_vs_n(df, colors, a.outdir)
    fig3_onset_vs_p(df, a.outdir)
    fig4_ratio_vs_p(df, colors, a.outdir)
    print(f"figures -> {a.outdir}")


if __name__ == "__main__":
    main()
