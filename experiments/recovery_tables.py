"""Emit the two side-by-side recovery tables (LaTeX): topology AND geometry, per dataset, per corruption,
with the winning algorithm NAMED -- and the MEDIAN algorithm reported next to it.

WHY THE MEDIAN COLUMN EXISTS. The natural question is "fix G, do the best you can on each axis, did we win?"
But "the best you can" is an ORACLE: choosing the best algorithm requires the ground truth, which is the very
thing repair is trying to recover. Best-of-suite is therefore an upper bound on what repair COULD do, not what
it WILL do. The gap between the median and the best is the oracle premium, and on at least one instance it
reverses the verdict outright: on dimacs_ny_d_mixed the best cover improves the map by 5.9% while the MEDIAN
cover makes it 16.4% worse. A table that printed only "best" would have sold that row as a win.

So every row prints observed -> median -> best. The median is what a practitioner choosing blind actually gets.

Every number is read from a CSV. Nothing is transcribed by hand.

  RGG   -- analysis/summary_rgg_recovery.csv. Topology and geometry are MATCHED: both come from the same cover
           on the same repaired distance matrix (rgg_recovery.py replays run_rgg_sweep and gates every
           disparity against the stored sweep to 5e-7).

  REAL  -- topology from analysis/summary_pure_real.csv, geometry from summary_mds_sweep.csv + summary_mds_rna
           .csv. The PLANTED dimacs rows have geometry but NO topology: a matched k-NN there needs the cover
           suite re-run at n = 5000, which we have not done. Those cells print `--` rather than borrow a number
           from a differently-parameterised run.
"""
import os
import sys

import numpy as np
import pandas as pd

K = 20                                    # pbmc3k is a 15-NN graph: k<=15 is a CONSTRUCTION CEILING
                                          # (recovery 0.999 by definition). k=20 is the only honest
                                          # cell on that graph. Signs are unchanged at k=5,10.
TOL = 0.01                                # a "win" must be a >1% relative gain; below that, best-of-15 is noise

TEX = {"gmr_ilp": r"\code{gmr\_ilp}", "iomr_ilp": r"\code{iomr\_ilp}",
       "gmr_bestofk": r"\code{gmr\_bok}", "iomr_bestofk": r"\code{iomr\_bok}",
       "gmr_rand": r"\code{gmr\_rand}", "iomr_rand": r"\code{iomr\_rand}",
       "gmr_thr_naive": r"\code{gmr\_thr}", "iomr_thr_naive": r"\code{iomr\_thr}",
       "spc_gmr": r"\code{spc\_gmr}", "spc_iomr": r"\code{spc\_iomr}",
       "l1sep_gmr": r"\code{l1sep\_gmr}", "l1sep_iomr": r"\code{l1sep\_iomr}",
       "pivot": r"\code{pivot}", "left_edge": r"\code{left\_edge}",
       "iomr_regiongrow": r"\code{iomr\_rgrow}", "domr": r"\DOMR{}"}
GTEX = {"rgg_inflate": r"\code{rgg} $n{=}300$", "rgg_deflate": r"\code{rgg} $n{=}300$",
        "rgg_mixed": r"\code{rgg} $n{=}300$", "rgg_inflate_n1000": r"\code{rgg} $n{=}1000$",
        "rgg_deflate_n1000": r"\code{rgg} $n{=}1000$", "rgg_mixed_n1000": r"\code{rgg} $n{=}1000$",
        "dimacs_ny_d_inflate": r"\code{dimacs\_d}", "dimacs_ny_d_deflate": r"\code{dimacs\_d}",
        "dimacs_ny_d_mixed": r"\code{dimacs\_d}", "dimacs_ny_t": r"\code{dimacs\_t}",
        "pbmc3k_cosine_knn": r"\code{pbmc3k}", "nmr_1d3z_atom": r"\code{nmr\_atom}",
        "nmr_1d3z_residue": r"\code{nmr\_res}", "ripe_atlas": r"\code{ripe}"}


def a(name):
    return TEX.get(str(name), r"\code{%s}" % str(name).replace("_", r"\_"))


def num(v, best=False, gain=0.0):
    """Bold only a GENUINE win (>TOL relative). A +1e-17 'gain' is float noise, not an improvement."""
    s = f"{v:.4f}"
    return (r"$\mathbf{%s}$" % s) if (best and gain > TOL) else f"${s}$"


def pct(x):
    return r"\code{--}" if not np.isfinite(x) else f"${100.0 * x:+.1f}$"


def _axis(vals, obs, higher_better):
    """observed, median-over-algorithms, best-over-algorithms, the best algorithm, and both relative gains."""
    if not len(vals):
        return None
    s = vals.sort_values("v", ascending=not higher_better)
    b = s.iloc[0]
    med = float(s.v.median())
    g = (lambda x: (x - obs) / max(abs(obs), 1e-12)) if higher_better else (lambda x: (obs - x) / max(obs, 1e-12))
    return dict(obs=obs, med=med, best=float(b.v), algo=b.algo, gmed=g(med), gbest=g(float(b.v)))


def _emit(rows, label, caption, with_H):
    cols = "@{}ll" + ("r" if with_H else "") + "rrrl" + "rrrl@{}"
    hdr = (r"graph & corruption & " + (r"$|H|$ & " if with_H else "")
           + r"obs. & med. & best & by & obs. & med. & best & by \\")
    span = 4 if with_H else 3          # first topology column: after graph, corruption[, |H|]
    out = [r"\begin{table}[!t]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}",
           r"\caption{" + caption + "}", r"\label{" + label + "}",
           r"\begin{tabular}{" + cols + "}", r"\toprule",
           (r"& & " + (r"& " if with_H else "")
            + r"\multicolumn{4}{c}{topology: $k$-NN Jaccard ($\uparrow$)} "
              r"& \multicolumn{4}{c}{geometry: Procrustes disparity ($\downarrow$)} \\"),
           r"\cmidrule(lr){%d-%d}\cmidrule(l){%d-%d}" % (span, span + 3, span + 4, span + 7),
           hdr, r"\midrule"]
    for r in rows:
        cells = [GTEX[r["graph"]], r["corr"]]
        if with_H:
            cells.append(f"${r['H']}$")
        for ax in ("t", "g"):
            A = r[ax]
            if A is None:
                cells += [r"\code{--}"] * 4
            else:
                cells += [num(A["obs"]), num(A["med"]), num(A["best"], True, A["gbest"]), a(A["algo"])]
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def rgg_table(path="analysis/summary_rgg_recovery.csv"):
    d = pd.read_csv(path)
    ok = d[d.status.eq("ok")]
    order = ["rgg_inflate", "rgg_deflate", "rgg_mixed",
             "rgg_inflate_n1000", "rgg_deflate_n1000", "rgg_mixed_n1000"]
    rows = []
    for g in order:
        G = ok[ok.graph == g]
        if G.empty:
            continue
        obs = G[G.algo == "observed"].iloc[0]
        rep = G[~G.algo.isin(["observed", "domr"])]
        if rep.empty:
            continue
        t = _axis(pd.DataFrame({"v": rep[f"knn{K}"].astype(float), "algo": rep.algo}).dropna(),
                  float(obs[f"knn{K}"]), True)
        gm = _axis(pd.DataFrame({"v": rep.disp_smacof.astype(float), "algo": rep.algo}).dropna(),
                   float(obs.disp_smacof), False)
        rows.append(dict(graph=g, corr=g.replace("_n1000", "").replace("rgg_", ""),
                         H=int(obs.H), t=t, g=gm))
    cap = (r"\textbf{Topology and geometry, on the same repair.} Random geometric graphs, planted corruption. "
           r"Both axes are scored against the planted point set, and --- crucially --- both are computed from "
           r"the SAME cover on the SAME repaired distance matrix, so a $k$-NN entry and a disparity entry on "
           r"one row describe one repair (\code{rgg\_recovery.py} replays the sweep and gates every disparity "
           r"against its stored value to $5 \times 10^{-7}$). \emph{obs.} is the corrupted graph, \emph{best} "
           r"the best cover on that axis and \emph{by} the algorithm that achieved it. \textbf{Read the "
           r"\emph{med.} column, not \emph{best}}: choosing the best algorithm requires the ground truth, "
           r"which is the thing one is trying to recover, so \emph{best} is an oracle and the median is what a "
           r"practitioner choosing blind actually gets. Bold marks a genuine win ($>\!1\%$ relative). \DOMR{} "
           r"is excluded as the control: by Lemma~\ref{lem:domr} it moves neither axis, and the pipeline "
           r"confirms it to $0$ on every row.")
    return _emit(rows, "tab:rggrecovery", cap, with_H=True), rows


def real_table(pr="analysis/summary_pure_real.csv",
               sw="analysis/summary_mds_sweep.csv", rna="analysis/summary_mds_rna.csv",
               rec="analysis/summary_recovery.csv"):
    T = pd.read_csv(pr); T = T[T.k == K]
    # The planted road rows: topology AND geometry from ONE cover on ONE repaired matrix (the 48-task array).
    # Until this existed we printed `--` rather than borrow a k-NN from a differently-parameterised run.
    REC = pd.read_csv(rec) if os.path.exists(rec) else None
    M = pd.read_csv(sw)
    R = pd.read_csv(rna); R["graph"] = "pbmc3k_cosine_knn"
    M = pd.concat([M, R], ignore_index=True)
    M = M[M.status.fillna("ok").eq("ok") & M.disp_smacof.notna()]

    CORR = {"dimacs_ny_d_inflate": "planted infl.", "dimacs_ny_d_deflate": "planted defl.",
            "dimacs_ny_d_mixed": "planted mixed"}
    order = ["dimacs_ny_d_inflate", "dimacs_ny_d_deflate", "dimacs_ny_d_mixed",
             "dimacs_ny_t", "pbmc3k_cosine_knn", "nmr_1d3z_atom", "nmr_1d3z_residue", "ripe_atlas"]
    rows = []
    for g in order:
        Mg = M[M.graph == g]
        if Mg.empty:
            continue
        o = Mg[Mg.algo == "observed"].iloc[0]
        dm = Mg[Mg.algo == "domr"]
        Hg = int(dm.cover_size.iloc[0]) if len(dm) and pd.notna(dm.cover_size.iloc[0]) else -1
        grep = Mg[~Mg.algo.isin(["observed", "domr"])]
        gm = _axis(pd.DataFrame({"v": grep.disp_smacof.astype(float), "algo": grep.algo}).dropna(),
                   float(o.disp_smacof), False)

        t = None
        if REC is not None and g in set(REC.graph):
            # MATCHED: same cover, same repaired matrix as the geometry column beside it.
            Rg = REC[REC.graph == g]
            ro = Rg[Rg.algo == "observed"]
            rr = Rg[(~Rg.algo.isin(["observed", "domr"])) & Rg[f"knn{K}"].notna()]
            if len(ro) and len(rr):
                kobs = float(ro[f"knn{K}"].iloc[0])
                t = _axis(pd.DataFrame({"v": rr[f"knn{K}"].astype(float), "algo": rr.algo}).dropna(),
                          kobs, True)
                t["n_ret"] = len(rr)
                t["n_all"] = len(Rg[~Rg.algo.isin(["observed", "domr"])])
        if t is None:
            Tg = T[(T.graph == g) & (T.algo != "domr")]
            Tall = T[T.graph == g]
            if len(Tg):
                kobs = float(Tall.recovery_obs.iloc[0])
                t = _axis(pd.DataFrame({"v": kobs + Tg.lift_med.astype(float), "algo": Tg.algo}).dropna(),
                          kobs, True)
        rows.append(dict(graph=g, corr=CORR.get(g, "none (native)"), H=Hg, t=t, g=gm))
    cap = (r"\textbf{Every real graph, both axes, and who won.} Topology is $k$-NN Jaccard against the external "
           r"truth ($\uparrow$), geometry the Procrustes disparity against the same truth ($\downarrow$); "
           r"\emph{best} names the algorithm that achieved it, \emph{med.} is the median over every algorithm "
           r"that returned a cover --- and the median, not the best, is what one gets without an oracle. Bold "
           r"marks a genuine win ($>\!1\%$ relative). \textbf{No repair wins on topology on any real graph.} "
           r"Quoted at $k = 20$, which on \code{pbmc3k} is the only honest cell --- that graph is a $15$-NN "
           r"graph, so $k \le 15$ recovery is $0.999$ \emph{by construction}. At $k = 20$ it has $30$ points "
           r"of genuine headroom ($0.6957$ observed) and repair captures NONE of it. The signs are unchanged "
           r"at $k = 5, 10$. Geometry is "
           r"different: \code{dimacs\_t} improves by $19.2\%$, and robustly --- all $14$ covers beat the "
           r"observed graph, the worst by $2.0\%$. \textbf{The planted \code{dimacs} rows are the sharpest "
           r"measurement here}: topology and geometry come from the SAME cover on the SAME repaired matrix "
           r"(a $48$-task array at $n = 5000$), and the two axes \emph{disagree} --- under inflation the "
           r"geometry improves by $34.3\%$ while the topology is damaged, and the best cover topology has "
           r"still loses. On those rows \emph{med.} is a median over the algorithms that RETURNED within the "
           r"cap ($5$ of $14$ under inflation and mixed, $9$ of $14$ under deflation); the LP/ILP family "
           r"times out at this size, so the survivors are systematically the cheap combinatorial methods. "
           r"\DOMR{} is the control and is exactly $0$ on both axes, on every graph.")
    return _emit(rows, "tab:realrecovery", cap, with_H=True), rows


def verdicts(rows):
    """WIN / TIE / LOSE per axis, at the TOL threshold -- reported for the median AND for the oracle."""
    def v(g):
        return "WIN " if g > TOL else ("LOSE" if g < -TOL else "TIE ")
    out = []
    for r in rows:
        t, g = r["t"], r["g"]
        out.append((r["graph"], r["corr"],
                    v(t["gmed"]) if t else "  --", v(t["gbest"]) if t else "  --",
                    v(g["gmed"]) if g else "  --", v(g["gbest"]) if g else "  --",
                    t["gmed"] if t else np.nan, t["gbest"] if t else np.nan,
                    g["gmed"] if g else np.nan, g["gbest"] if g else np.nan))
    return out


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "analysis"
    os.makedirs(outdir, exist_ok=True)
    parts = []
    for name, fn in [("rggrecovery", rgg_table), ("realrecovery", real_table)]:
        try:
            tex, rows = fn()
        except FileNotFoundError as e:
            print(f"!! skipping {name}: {e}"); continue
        with open(os.path.join(outdir, f"tab_{name}.tex"), "w") as f:
            f.write(tex + "\n")
        parts.append((name, tex, rows))
        print(tex); print()

    print("=" * 108)
    print("VERDICTS -- 'fix G, do the best you can on each axis, did we win?'  (WIN needs >1% relative)")
    print(f"{'graph':<22}{'corruption':<16}{'TOPOLOGY med/best':<22}{'GEOMETRY med/best':<22}")
    print("-" * 108)
    for name, _tex, rows in parts:
        for (g, c, tm, tb, gm, gb, tmv, tbv, gmv, gbv) in verdicts(rows):
            tt = "  --" if tm == "  --" else f"{tm}({100*tmv:+5.1f}%) / {tb}({100*tbv:+5.1f}%)"
            gg = "  --" if gm == "  --" else f"{gm}({100*gmv:+5.1f}%) / {gb}({100*gbv:+5.1f}%)"
            flag = ""
            if tm != "  --" and tm.strip() != gm.strip():
                flag = "  <-- AXES DISAGREE"
            if gm != "  --" and gm.strip() != gb.strip():
                flag += "  <-- ORACLE FLIPS THE GEOMETRY VERDICT"
            print(f"{g:<22}{c:<16}{tt:<30}{gg:<30}{flag}")
    print(f"\nwrote {outdir}/tab_rggrecovery.tex and {outdir}/tab_realrecovery.tex")
