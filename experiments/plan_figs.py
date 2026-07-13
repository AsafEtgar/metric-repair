"""Two floats for paper_plan.tex that the delivered figure set does not already contain.

  1. fig_real_triptych -- one ROW per real dataset: truth | observed | the cover that destroyed the LEAST.
     The full per-algorithm grids are in the appendix; this is the three panels a reader actually needs.
     Axes are SHARED WITHIN A ROW (never across rows -- those are unrelated frames). Without that, matplotlib
     autoscales each panel to its own cloud and a wrecked embedding renders at the same apparent size as an
     intact one, which is the one thing this figure exists to let the eye judge.

  2. table_size_runtime -- dataset x algorithm; each cell is |S|, and nothing else. Rows are blocked by
     variant (DOMR / GMR / IOMR) and bold marks the smallest cover WITHIN A BLOCK, for that dataset --
     comparing a GMR cover with an IOMR cover is comparing two different problems. An algorithm that never
     returned is an explicit marker, not a gap.

THE cpu=0 TRAP. A killed run records cpu = 0.0 (the wall correctly holds the cap). Every timeout row in the
campaign has cpu == 0. Aggregating cpu without filtering on status would therefore report the SLOWEST
algorithms as the fastest. The table no longer prints runtime, but the same status filter still governs it
and the trap is why: we aggregate over status == "ok" rows only, and report the rest as explicit non-returns.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The graphs the paper actually reasons about: the six that carry an external truth, plus the four
# no-truth real graphs that still exercise the algorithms. The _raw/_lin/_log rows are conversion
# variants of the same graph and would triple-count.
GRAPHS = ["dimacs_ny_t", "ripe_atlas", "pbmc3k_cosine_knn", "nmr_1d3z_atom", "nmr_1d3z_residue",
          "bct_coactivation", "cassiopeia_barcode_knn", "flycns_male", "fish1_ten"]
# lp_naive returns a BOUND, not a cover (size is NaN even when it succeeds) -- it is not an algorithm here.
DROP = {"gmr_lp_naive", "iomr_lp_naive", "gmr_lp_rsp", "iomr_lp_rsp"}

SHORT = {"dimacs_ny_t": "dimacs\\_t", "ripe_atlas": "ripe", "pbmc3k_cosine_knn": "pbmc3k",
         "nmr_1d3z_atom": "nmr\\_at", "nmr_1d3z_residue": "nmr\\_res", "bct_coactivation": "bct",
         "cassiopeia_barcode_knn": "cassio", "flycns_male": "flycns", "fish1_ten": "fish"}
TITLE = {"dimacs_ny_t": "dimacs\\_ny\\_t (road, travel time)", "ripe_atlas": "ripe\\_atlas (internet latency)",
         "pbmc3k_cosine_knn": "pbmc3k (single-cell RNA)", "nmr_1d3z_atom": "nmr\\_1d3z\\_atom (NMR, 3-D)",
         "nmr_1d3z_residue": "nmr\\_1d3z\\_residue (NMR, 3-D)"}


# ----------------------------------------------------------------------------
# 1. the triptych
# ----------------------------------------------------------------------------
def fig_real_triptych(outdir="analysis/figs/plan"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mds_plots import _scatter, _fold_info, _dim_of, shared_limits, _frame, _tag

    sw = pd.read_csv("analysis/summary_mds_sweep.csv")
    rna = pd.read_csv("analysis/summary_mds_rna.csv"); rna["graph"] = "pbmc3k_cosine_knn"
    rna["corruption"] = "none"
    df = pd.concat([sw, rna], ignore_index=True)
    df = df[df.get("status", pd.Series("ok", index=df.index)).fillna("ok").eq("ok") & df.disp_smacof.notna()]

    Z = dict(np.load("analysis/mds_sweep_embeddings.npz"))
    Z.update(np.load("analysis/mds_rna_embeddings.npz"))

    rows = []
    for g in ["dimacs_ny_t", "ripe_atlas", "pbmc3k_cosine_knn", "nmr_1d3z_atom", "nmr_1d3z_residue"]:
        G = df[df.graph == g]
        if G.empty:
            continue
        tag = _tag(g, G.corruption.iloc[0])
        if f"true::{tag}" not in Z:
            print(f"    skip {g} (no stored truth)"); continue
        obs = G[G.algo == "observed"]
        rep = G[~G.algo.isin(["observed", "domr"])].sort_values("disp_smacof")
        rep = rep[[f"emb::{tag}::{a}::smacof" in Z for a in rep.algo]]
        if obs.empty or rep.empty:
            continue
        rows.append((g, tag, obs.iloc[0], rep.iloc[0]))     # rep.iloc[0] = destroyed the least

    fig = plt.figure(figsize=(9.8, 3.25 * len(rows)))
    for r, (g, tag, obs, best) in enumerate(rows):
        dim = _dim_of(Z, tag)
        res, color = _fold_info(g, Z, tag)
        kobs = f"emb::{tag}::observed::smacof"
        kbest = f"emb::{tag}::{best.algo}::smacof"
        lim = shared_limits([Z[f"true::{tag}"], Z[kobs], Z[kbest]], dim)
        do, db = float(obs.disp_smacof), float(best.disp_smacof)
        # HONESTY: on several graphs the cover that destroyed the LEAST still destroys. Say so in the frame
        # and in the title -- a blue frame on a panel that is worse than doing nothing reads as a win.
        helps = db < do
        note = "" if helps else "\nSTILL WORSE THAN NO REPAIR"
        panels = [("true configuration", Z[f"true::{tag}"], "#222222"),
                  (f"observed (no repair)\ndisparity {do:.4f}", Z[kobs], "#222222"),
                  (f"{best.algo} -- destroyed the least\ndisparity {db:.4f}   "
                   f"$|S|={int(best.cover_size):,}$".replace(",", "{,}") + note,
                   Z[kbest], "#1f77b4" if helps else "#BB0000")]
        for c, (ttl, P, col) in enumerate(panels):
            ax = fig.add_subplot(len(rows), 3, r * 3 + c + 1,
                                 projection="3d" if dim >= 3 else None)
            _scatter(ax, P, color, dim, res=res, lim=lim)
            ax.set_title(ttl, fontsize=7.8, color="#B00" if col == "#BB0000" else "#222")
            _frame(ax, col)
        fig.text(0.007, 1.0 - (r + 0.5) / len(rows), TITLE.get(g, g).replace("\\_", "_"),
                 rotation=90, va="center", ha="left", fontsize=8.5, color="#333")

    fig.suptitle("Real graphs: the truth, the graph as observed, and the cover that damaged it LEAST.\n"
                 "Disparity = Procrustes distance to the external truth ($\\downarrow$ better). Axes are "
                 "shared WITHIN each row, so a distended embedding looks distended.\n"
                 "A RED frame means even the gentlest cover in the suite is worse than not repairing at all.",
                 fontsize=9)
    fig.tight_layout(rect=(0.02, 0, 1, 0.93))
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"real_triptych.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {outdir}/real_triptych.pdf / .png  ({len(rows)} rows)")


# ----------------------------------------------------------------------------
# 2. dataset x algorithm: |S|, blocked by variant
# ----------------------------------------------------------------------------
# Row order WITHIN each block. Membership is NOT taken from this list -- it is read from the `variant`
# column of the CSV, and anything the CSV puts in a block that is missing here is appended rather than
# silently dropped. The list only fixes the order (alphabetical, with the two Gilbert-Jain-style
# combinatorial methods -- pivot, left_edge -- last, since they are the outliers the reader is scanning for).
BLOCKS = [("DOMR", [r"\DOMR{}", ["domr"]]),
          ("GMR", [r"\GMR{}", ["gmr_bestofk", "gmr_ilp", "gmr_rand", "gmr_thr_naive", "l1sep_gmr",
                                "spc_gmr", "pivot"]]),
          ("IOMR", [r"\IOMR{}", ["iomr_bestofk", "iomr_ilp", "iomr_rand", "iomr_regiongrow",
                                 "iomr_thr_naive", "l1sep_iomr", "spc_iomr", "left_edge"]])]


def table_size_runtime(out="analysis/tab_plan_sizeruntime.tex"):
    d = pd.read_csv("analysis/real_rows_with_ratio.csv")
    d = d[d.graph.isin(GRAPHS) & ~d.algo.isin(DROP)]

    # Blocks come from the data, not from a hand-kept list: variant is the ground truth for GMR/IOMR/DOMR.
    seen = {a: str(v) for a, v in d[["algo", "variant"]].drop_duplicates().itertuples(index=False)}
    blocks = []
    for key, (label, order) in BLOCKS:
        members = [a for a in order if seen.get(a) == key]
        extra = sorted(a for a, v in seen.items() if v == key and a not in members)
        if extra:
            print(f"    [warn] {key}: not in BLOCKS order, appended: {extra}")
        blocks.append((label, members + extra))
    missing = sorted(set(seen) - {a for _, m in blocks for a in m})
    if missing:
        raise ValueError(f"algorithms with an unknown variant, would be dropped: {missing}")

    algos = [a for _, m in blocks for a in m]
    cell = {}
    for g in GRAPHS:
        sizes = {}
        for a in algos:
            R = d[(d.graph == g) & (d.algo == a)]
            if R.empty:
                cell[(g, a)] = None                       # never run on this graph
                continue
            ok = R[R.status.eq("ok") & R["size"].notna()]
            if ok.empty:
                # every attempt failed. WHICH failure matters: a timeout is intractable, skipped_H is a gate.
                why = "TO" if (R.status == "timeout").any() else str(R.status.iloc[0])
                cell[(g, a)] = ("fail", why)
                continue
            # median over seeds, over ok rows ONLY. Runtime is no longer reported, but the status filter
            # still matters and the next person will need to know why: THE cpu=0 TRAP -- a killed run
            # records cpu = 0.0, so any aggregate that does not filter on status reports the SLOWEST
            # algorithms as the fastest, and counts a timeout as a cover of size NaN.
            sizes[a] = int(round(float(ok["size"].median())))
            cell[(g, a)] = ("ok", sizes[a])
        # Best-in-block, per graph. Bold EVERY algorithm attaining the block minimum: ties are common
        # (five GMR methods all return 14 edges on nmr_1d3z_atom) and bolding an arbitrary one of two
        # identical numbers is a lie the reader can see.
        for _, members in blocks[1:]:                     # DOMR is a block of one -- nothing to compare
            have = {a: sizes[a] for a in members if a in sizes}
            if have:
                lo = min(have.values())
                for a in have:
                    if have[a] == lo:
                        cell[(g, a)] = cell[(g, a)] + ("best",)

    def fmt_n(x):
        return f"{x:,}".replace(",", r"{,}")

    ncol = len(GRAPHS) + 1
    # \tabcolsep 4pt (from 6pt) buys 36pt across nine column gaps -- enough to drop \resizebox and keep a
    # real \footnotesize font, rather than scaling the whole table to some arbitrary non-standard size.
    L = [r"\begin{table}[!p]\centering\footnotesize\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{Cover size, every algorithm on every real graph.} Each cell is $|S|$, the "
         r"number of edges the cover edits (median over seeds for the randomized methods). Rows are "
         r"blocked by variant, and \textbf{bold} marks the smallest cover \emph{within a block, for that "
         r"graph} --- \DOMR{} is a block of one, so it is never bold. Ties are all bolded. The spread "
         r"inside a single block is the story: on \code{pbmc3k} the exact \IOMR{} program returns $42$ "
         r"edges and \code{left\_edge} returns $26{,}830$. \code{TO} = never returned within its time cap "
         r"(an honest null, not a gap); \code{gate} = refused by the region-growing $|H| \le 200$ gate; "
         r"\code{--} = not run. Sizes are taken from completed runs only: a killed run records "
         r"$\mathrm{cpu} = 0$ and no cover at all, so a table that does not filter on status would quietly "
         r"credit a timeout with a cover.}",
         r"\label{tab:sizeruntime}",
         r"\begin{tabular}{@{}l" + "r" * len(GRAPHS) + r"@{}}", r"\toprule",
         "algorithm & " + " & ".join(r"\code{%s}" % SHORT[g] for g in GRAPHS) + r" \\"]
    for i, (label, members) in enumerate(blocks):
        L.append(r"\midrule" if i == 0 else r"\midrule")
        L.append(r"\multicolumn{%d}{@{}l@{}}{\itshape %s} \\" % (ncol, label))
        for a in members:
            cells = []
            for g in GRAPHS:
                c = cell[(g, a)]
                if c is None:
                    cells.append(r"\code{--}")
                elif c[0] == "fail":
                    cells.append(r"\code{TO}" if c[1] == "TO" else r"\code{gate}")
                else:
                    s = fmt_n(c[1])
                    cells.append(r"$\mathbf{%s}$" % s if len(c) > 2 else f"${s}$")
            L.append(r"\quad\code{%s} & " % a.replace("_", r"\_") + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"    wrote {out}  ({len(algos)} algorithms x {len(GRAPHS)} graphs, "
          f"{len(blocks)} blocks: {', '.join(f'{k}={len(m)}' for (k, _), (_, m) in zip(BLOCKS, blocks))})")
    return "\n".join(L)


if __name__ == "__main__":
    fig_real_triptych()
    table_size_runtime()
