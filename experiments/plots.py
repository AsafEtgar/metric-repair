"""plots.py -- the headline figures from the aggregated summary (experiments/analyze.py output).

Runs LOCALLY; needs matplotlib + pandas. On this Mac use sage's Python (it bundles both):

    sage -python experiments/plots.py --summary analysis/summary.csv --outdir analysis/figs/geometric

Figures are written split by MR axis into per-family subfolders of --outdir:
    <outdir>/gmr/   GMR (general metric repair): variants {GMR, DOMR}   -- any-direction edits
    <outdir>/iomr/  IOMR (increase-only metric repair): variant {IOMR}  -- a different problem/axis

Styling (shared with rgg_plots.py via plot_common): every algorithm gets a stable colour+marker+linestyle
triple so the lines stay distinguishable even when hues repeat; ONE legend sits outside the axes. On the
approximation-ratio panels two things that used to mislead are now explicit:
  * the region where the exact ILP OPT is unavailable (it timed out) is shaded -- there `ratio = size / LP
    lower bound`, an OVER-estimate, so a "jump" as n grows is the reference weakening, not the algorithms; and
  * the exact algorithms all coincide at ratio 1.0, so they are drawn dimmed and named in a note instead of
    stacking six indistinguishable lines on top of each other.

  fig1_ratio_vs_n    Exp 1: approximation ratio (size/OPT) vs n, one panel per p          (down: lower better)
  fig2_runtime_vs_n  Exp 1: CPU seconds vs n (log-log, to read the scaling exponent)      (down: lower better)
  fig3_onset_vs_p    Exp 2a: that family's OPT and |H| vs edge density p -- non-metricity onset (~alpha=3/5)
  fig4_ratio_vs_p    Exp 2a+2b: approximation ratio vs edge density p, one panel per exp  (down: lower better)

Each figure is PDF + PNG. Missing experiments/families are skipped with a note rather than crashing.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")                       # non-interactive: save files, no display
import matplotlib.pyplot as plt             # noqa: E402
import numpy as np                          # noqa: E402
import pandas as pd                          # noqa: E402

from plot_common import (FAMILY, FAMILIES, FAM_TITLE, style_map, ylab, band,      # noqa: E402
                         figure_legend, note, save)

ONSET_ALPHA = 3 / 5                          # coupled-geometric non-metricity onset (Exp 2a)
CONNECT_ALPHA = 0.706                        # G(n,p) connectivity threshold at n=500 (Exp 2b crosses it)


def _family(df, fam):
    return df[df["variant"].isin(FAMILY[fam])]


def _flat_one(gg, col="ratio_med"):
    """True if this algo's median sits at ~1.0 across the whole panel (an exact / near-optimal method)."""
    v = gg[col].dropna()
    return len(v) > 0 and v.between(0.97, 1.03).all()


def _refkind_by_x(panel, xcol):
    if "ref_kind" not in panel:
        return None
    return panel.dropna(subset=[xcol]).groupby(xcol)["ref_kind"].agg(
        lambda s: s.mode().iat[0] if len(s.mode()) else "none")


def _shade_lp(ax, panel, xcol):
    """Flag where the reference is only the LP lower bound (ILP timed out) so `ratio = size / LB` OVER-estimates
    the true ratio. If the WHOLE panel is on the bound (e.g. IOMR, whose ILP always times out) a red flood is
    uninformative -> return 'all' so the caller adds a note; a partial TAIL is shaded and returns 'tail'."""
    rk = _refkind_by_x(panel, xcol)
    if rk is None or len(rk) < 2:
        return None
    non_exact = [x for x in rk.index if rk[x] != "exact"]
    if not non_exact:
        return None
    if len(non_exact) == len(rk):
        return "all"
    xs = sorted(rk.index)
    step = np.min(np.diff(xs))
    ax.axvspan(min(non_exact) - step / 2, max(xs) + step / 2, color="tab:red", alpha=0.06, lw=0, zorder=0)
    return "tail"


def _mark_bound_loosening(ax, panel, xcol):
    """When the reference is an LP bound, a >40% DROP in ref_med marks where a tighter bound was gated off
    (the rsp LP's n_max=150) and only a looser one remains -> the ratio jumps for reference reasons, not the
    algorithms. Draw a thin marker there so the 'jump' is not misread."""
    r = panel.dropna(subset=[xcol, "ref_med"]).groupby(xcol)["ref_med"].median().sort_index()
    if len(r) < 3:
        return
    frac = r.to_numpy()[1:] / r.to_numpy()[:-1]
    i = int(np.argmin(frac))
    if frac[i] < 0.6:
        xg = r.index[i + 1]
        ax.axvline(xg, color="grey", lw=0.9, ls=(0, (2, 2)), alpha=0.8, zorder=1)
        ax.text(xg, ax.get_ylim()[1], "  LP bound loosens\n  (tight rsp bound gated off)",
                va="top", ha="left", fontsize=6.5, color="0.3")


def _ratio_panel(ax, panel, sty, xcol):
    """Draw one approximation-ratio panel: flag the LP-bound region, dim+name the exact (=1.0) algos, draw the
    heuristics prominently, mark ratio=1. Returns (lp_flag in {tail,all,None}, list_of_optimal_algos)."""
    flag = _shade_lp(ax, panel, xcol)
    optimal = []
    for algo, gg in panel.groupby("algo"):
        flat = _flat_one(gg)
        # no IQR band here: 8 overlapping bands become mush and hide the lines (spread is in summary.csv).
        band(ax, gg, xcol, "ratio_med", "ratio_q25", "ratio_q75", algo, sty[algo], dim=flat, iqr=False)
        if flat:
            optimal.append(algo)
    ax.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.5)
    if flag == "all":                       # ILP never converged -> the whole curve is size / LP lower bound
        _mark_bound_loosening(ax, panel, xcol)
        note(ax, "ILP timed out at every point:\nratio = size / LP lower bound\n(over-estimates the true ratio)",
             loc="upper left")
    if optimal:
        note(ax, "coincide at ≈1.0 (exact/optimal):\n" + ", ".join(sorted(optimal)),
             loc="lower right" if flag == "all" else "upper left")
    return flag, optimal


def fig1_ratio_vs_n(df, sty, fam, outdir):
    e = _family(df[df["exp"] == "exp1"], fam)
    if e.empty:
        print(f"    skip fig1 [{fam}] (no exp1 rows)"); return
    ps = sorted(e["p"].dropna().unique())
    fig, axes = plt.subplots(len(ps), 1, figsize=(7.2, 3.9 * len(ps)), squeeze=False, sharex=True)
    any_tail = False
    for i, p in enumerate(ps):
        ax = axes[i][0]
        flag, _ = _ratio_panel(ax, e[e["p"] == p], sty, "x")
        any_tail |= (flag == "tail")
        ax.set_title(f"p = {p}")
        if i == len(ps) - 1:
            ax.set_xlabel("n")
        ax.set_ylabel(ylab("approx. ratio (size / OPT)", "down"))
    sub = " — red band: ILP OPT timed out, ratio is vs the LP lower bound (over-estimate)" if any_tail else ""
    fig.suptitle(f"Exp 1 — approximation ratio vs n — {FAM_TITLE[fam]}{sub}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    # figure_legend() attaches the legend to the figure; do NOT hand it to save(). Passing an explicit
    # bbox_extra_artists list there REPLACES matplotlib's default set (which includes the suptitle), so the
    # suptitle would be clipped off. With legend left off, save()'s tight bbox keeps BOTH legend + suptitle.
    figure_legend(fig)
    save(fig, outdir, "fig1_ratio_vs_n")


def fig2_runtime_vs_n(df, sty, fam, outdir):
    e = _family(df[df["exp"] == "exp1"], fam)
    if e.empty:
        print(f"    skip fig2 [{fam}] (no exp1 rows)"); return
    ps = sorted(e["p"].dropna().unique())
    fig, axes = plt.subplots(len(ps), 1, figsize=(7.2, 3.9 * len(ps)), squeeze=False, sharex=True)
    for i, p in enumerate(ps):
        ax = axes[i][0]
        for algo, gg in e[e["p"] == p].groupby("algo"):
            band(ax, gg, "x", "cpu_med", "cpu_q25", "cpu_q75", algo, sty[algo], logx=True, logy=True, iqr=False)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"p = {p}")
        if i == len(ps) - 1:
            ax.set_xlabel("n")
        ax.set_ylabel(ylab("CPU seconds", "down"))
    fig.suptitle(f"Exp 1 — CPU time vs n (log-log) — {FAM_TITLE[fam]}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    figure_legend(fig)                          # attach legend but keep it off save() -- see fig1 note
    save(fig, outdir, "fig2_runtime_vs_n")      # (explicit bbox_extra_artists would drop the suptitle)


def fig3_onset_vs_p(df, sty, fam, outdir):
    e = df[df["exp"] == "exp2a"]
    if e.empty:
        print(f"    skip fig3 [{fam}] (no exp2a rows)"); return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    opt_algo, opt_label = ("gmr_ilp", "GMR OPT") if fam == "gmr" else ("iomr_ilp", "IOMR OPT")
    band(ax, e[e["algo"] == opt_algo], "p", "ref_med", "ref_q25", "ref_q75", opt_label,
         {"color": "#0072B2", "marker": "o", "ls": "-"})
    band(ax, e[e["algo"] == "domr"], "p", "H_med", "H_q25", "H_q75", "|H| = |DOMR|",
         {"color": "#009E73", "marker": "s", "ls": "--"})
    n0 = e.dropna(subset=["p", "x"])
    if not n0.empty:
        n = float(n0.iloc[0]["p"]) ** (-1.0 / float(n0.iloc[0]["x"]))
        onset_p = n ** (-ONSET_ALPHA)
        ax.axvline(onset_p, color="k", lw=0.8, ls="--", alpha=0.6)
        ax.text(onset_p, ax.get_ylim()[1], r"  onset ($\alpha=3/5$)", va="top", fontsize=8)
    ax.set_xlabel(r"edge density $p\;(=n^{-\alpha})$")
    ax.set_ylabel("optimum / broken-edge count")          # descriptive magnitude -> no better/worse direction
    ax.set_title(f"Exp 2a — non-metricity onset (n=500) — {FAM_TITLE[fam]}")
    ax.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, outdir, "fig3_onset_vs_p")


def fig4_ratio_vs_p(df, sty, fam, outdir):
    exps = [x for x in ("exp2a", "exp2b") if (df["exp"] == x).any()]
    if not exps:
        print(f"    skip fig4 [{fam}] (no exp2 rows)"); return
    # sharex=False: exp2a and exp2b span different p-ranges, so let each experiment panel scale to its own.
    fig, axes = plt.subplots(len(exps), 1, figsize=(7.2, 3.9 * len(exps)), squeeze=False, sharex=False)
    any_tail = False
    for i, exp in enumerate(exps):
        ax = axes[i][0]
        sub = _family(df[df["exp"] == exp], fam)
        flag, _ = _ratio_panel(ax, sub, sty, "p")
        any_tail |= (flag == "tail")
        if exp == "exp2b" and not sub.empty:
            n0 = sub.dropna(subset=["p", "x"])
            if not n0.empty:
                n = float(n0.iloc[0]["p"]) ** (-1.0 / float(n0.iloc[0]["x"]))
                ax.axvline(n ** (-CONNECT_ALPHA), color="grey", lw=0.8, ls=":", alpha=0.7)
        ax.set_title(exp)
        ax.set_xlabel(r"edge density $p$")
        ax.set_ylabel(ylab("approx. ratio (size / OPT)", "down"))
    sub = " — red band: ratio is vs the LP lower bound (ILP timed out)" if any_tail else ""
    fig.suptitle(f"Exp 2 — approximation ratio vs edge density p — {FAM_TITLE[fam]}{sub}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    figure_legend(fig)                          # attach legend but keep it off save() -- see fig1 note
    save(fig, outdir, "fig4_ratio_vs_p")        # (explicit bbox_extra_artists would drop the suptitle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="summary.csv from experiments/analyze.py")
    ap.add_argument("--outdir", default="analysis/figs/geometric",
                    help="root for the figures; per-family gmr/ and iomr/ subfolders are created under it")
    a = ap.parse_args()

    df = pd.read_csv(a.summary)
    sty = style_map(df["algo"].dropna().unique())
    print(f"loaded {len(df)} summary rows; experiments={sorted(df['exp'].unique())}")
    for fam in FAMILIES:
        outdir = os.path.join(a.outdir, fam)
        os.makedirs(outdir, exist_ok=True)
        print(f"[{fam}] {FAM_TITLE[fam]} -> {outdir}")
        fig1_ratio_vs_n(df, sty, fam, outdir)
        fig2_runtime_vs_n(df, sty, fam, outdir)
        fig3_onset_vs_p(df, sty, fam, outdir)
        fig4_ratio_vs_p(df, sty, fam, outdir)
    print(f"figures -> {a.outdir}/{{gmr,iomr}}")


if __name__ == "__main__":
    main()
