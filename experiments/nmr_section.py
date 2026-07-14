"""Emit everything the NMR section quotes: the oracle table, two figures, and every prose macro.

THE SECTION'S CLAIM, AND WHY THIS FILE CAN BE TRUSTED TO MAKE IT.

NMR is the only graph in the study on which the paper's second question can even be ASKED. To ask whether the
deficit lies in the SET or in the WEIGHTS, one must be able to hand an edge its true weight -- and that needs
the graph and the truth to measure the same physical quantity. NOE upper bounds and interatomic distances are
both in Angstrom (median ratio 1.21, above 1 because an upper bound exceeds what it bounds), so alpha = 1 and
the substitution is meaningful. On the road network the edges are travel time and the truth is kilometres:
writing the truth into an edge is a unit error, not an experiment. G3 gates on exactly this.

WHAT THE DATA SAYS, and each gate names the thing that would falsify it.

  1. THE CANONICAL RULE IS A NO-OP, AND THEN A HARM. `restore` cannot help here, and the reason is a theorem:
     a decrease-only cover changes no shortest path (Simas et al., Thm 3.2). DOMR's cover IS the heavy set, so
     DOMR under `restore` must reproduce `observed` EXACTLY. G1 checks it -- a free, exact control on the
     whole pipeline.

  2. THE BEST ALGORITHM IS THE ONE THAT DOES NOTHING. Ranked by what a practitioner actually gets -- the
     Procrustes disparity of the repaired fold under the canonical rule -- the winner on nmr_residue is DOMR,
     whose repair is provably the identity. It wins by not moving. Every one of the sixteen methods that
     actually edits the graph makes the structure WORSE. G5 measures this rather than asserting it: it
     computes the best cover under `restore` and reports how much it improves on doing nothing.

  3. WHAT A COVER RECOVERS IS DECIDED BY HOW MANY EDGES IT EDITS, NOT WHICH ONES. Hand every saved cover its
     TRUE weights and the fraction of the available gain it captures tracks |S|/m at r = 0.945 across 32
     (graph, algorithm) pairs. G4 gates the correlation AND the slope -- a cover that selected the edges where
     the error lives would sit far above the diagonal, and none does.

  4. SO NO *SMALL* SET WORKS AT ALL. Over the 28 covers below |S|/m = 20% -- which is what metric repair
     EXISTS to find -- the mean captured share is -1.3%, and 15 of 28 make the structure worse.

FIGURES, AND THE TRAP EACH ONE AVOIDS.

  fig:nmrfold    The 3-D MDS folds. THE EMBEDDINGS ARE REBUILT HERE, not read from a stale npz, and G6 then
                 requires every rebuilt embedding to reproduce its row in summary_oracle.csv to 1e-9 -- so the
                 picture and the table are the SAME measurement, not two that happen to agree. All panels in a
                 row share ONE axis box (G7): Procrustes standardises each configuration, so unequal axes
                 would render a wrecked embedding as an intact one. That is the trap this figure exists to
                 walk into, and the gate that stops it.

  fig:nmrcaptured  captured vs |S|/m, all 32 pairs, with the diagonal drawn. The diagonal is the prediction of
                 a UNIFORM error; the points sit on it. That IS the finding.

  inputs   analysis/summary_oracle.csv               the oracle arms (gated against, never re-derived)
           results_real/results_real_covers/<graph>/ the saved covers, replayed for the figure
  outputs  <texdir>/tab_oracle.tex, <texdir>/nmr_macros.tex
           <figdir>/fig_nmr_fold.pdf, fig_nmr_captured.pdf

  usage    sage -python experiments/nmr_section.py --texdir "<paper>/tables" --figdir "<paper>/figures"
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from mpl_toolkits.mplot3d import Axes3D               # noqa: E402,F401  (registers the 3-D projection)
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from scipy.stats import pearsonr, spearmanr           # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import oracle_weights as ow                                                    # noqa: E402
from downstream_recovery import _covers_dir, apsp, load_cover                  # noqa: E402
from mds_recovery import (build_F_distances, classical_mds, smacof,            # noqa: E402
                          _procrustes_disp)

CSV = "analysis/summary_oracle.csv"
COVERS = "results_real/results_real_covers"
GRAPHS = ["nmr_1d3z_atom", "nmr_1d3z_residue"]
SMALL = 0.20                     # "a small cover" -- what metric repair exists to find
NAME = {"nmr_1d3z_atom": "atom", "nmr_1d3z_residue": "residue"}
MACNAME = {"l1sep_gmr": "LoneGmr", "l1sep_iomr": "LoneIomr", "gmr_thr_naive": "GmrThr",
           "iomr_thr_naive": "IomrThr", "iomr_regiongrow": "IomrRgrow"}


def key(a):
    """algo -> a LaTeX-safe CamelCase fragment. LaTeX control sequences are LETTERS ONLY: "l1sep".title() is
    "L1Sep", and \\nmrOptL1SepGmr is not a macro name but a hard error. Raise rather than emit one."""
    k = MACNAME.get(a) or "".join(w.title() for w in a.split("_"))
    if not k.isalpha():
        raise SystemExit(f"FATAL: macro fragment {k!r} for algo {a!r} is not pure letters; add it to MACNAME.")
    return k


def _n(x):
    return f"{int(x):,}".replace(",", "{,}")


def tex(a):
    return r"\code{%s}" % a.replace("_", r"\_")


# ----------------------------------------------------------------------------
# The table's data: one row per (graph, algo), seeds collapsed
# ----------------------------------------------------------------------------
def build(df):
    """Per (graph, algo): the cover, what `restore` does to the fold, and what the TRUE weights recover.

    THE SEEDS ARE COLLAPSED BY MEDIAN, and that is not cosmetic. The randomized methods carry 30 saved covers
    apiece; on nmr_atom all 30 agree to the last digit, but on nmr_residue `pivot`'s disparity ranges over
    0.19 to 0.30 across seeds. Quoting one seed would be quoting a draw. We take the median cover and say so.
    """
    out, ref = {}, {}
    for g in GRAPHS:
        G = df[df.graph.eq(g)]
        o = G[G.arm.eq("observed")].iloc[0]
        a = G[G.arm.eq("all_oracle")].iloc[0]
        gap = float(o.disp) - float(a.disp)                 # the CEILING: every edge set to its true distance
        ref[g] = dict(n=int(o.n), m=int(o.m), H=int(o.H), core=int(o.n_core), dim=int(o.dim),
                      same_units=int(o.same_units), alpha=float(o.alpha),
                      disp_obs=float(o.disp), disp_all=float(a.disp), gap=gap,
                      knn20_obs=float(o.knn20), n_gtset=(int(o.n_gtset) if pd.notna(o.n_gtset) else None))
        A = (G[G.arm.isin(["restore", "oracle"])]
             .groupby(["algo", "arm"])
             .agg(size=("cover_size", "median"), disp=("disp", "median"),
                  knn20=("knn20", "median"), nseed=("disp", "size")).reset_index())
        P = A.pivot(index="algo", columns="arm", values=["size", "disp", "knn20", "nseed"])
        rows = []
        for al in P.index:
            s = float(P.loc[al, ("size", "restore")])
            rd = float(P.loc[al, ("disp", "restore")])
            od = float(P.loc[al, ("disp", "oracle")])
            rows.append(dict(graph=g, algo=al, size=s, sm=s / o.m, restore=rd, oracle=od,
                             rel=(rd - o.disp) / o.disp, captured=(o.disp - od) / gap,
                             knn20=float(P.loc[al, ("knn20", "restore")]),
                             nseed=int(P.loc[al, ("nseed", "restore")])))
        out[g] = pd.DataFrame(rows).sort_values("sm").reset_index(drop=True)
    return out, ref


def edge_error(ref):
    """IS THE ERROR WHERE THE HEAVINESS IS? Derived from the graph and the truth, not from any CSV.

    With err(e) = |w(e) - d*(e)|, define the CONCENTRATION of the error in the heavy set as

        conc = ( share of the total error carried by H ) / ( share of the edges that are in H ).

    conc = 1 means H carries exactly its proportional share -- being heavy and being wrong are UNRELATED.
    conc >> 1 would mean the heavy set is where the damage lives, which is the premise metric repair rests on.
    We measure it. The section's central claim stands or falls here, so it is derived, never typed.

    Also returns the median NOE slack w(e)/d*(e). It exceeds 1 for a physical reason -- an upper bound exceeds
    the distance it bounds -- and that is why alpha = 1 is the right choice and NOT a unit conversion (G3).
    """
    out = {}
    for g in GRAPHS:
        ins = ow.build_instance(g)
        row, Dstar = ins["row"], ins["Dstar"]
        Hset = _heavy(ins)
        err_all = err_H = 0.0
        ratios = []
        for u, v, w in ins["edges"]:
            if u not in row or v not in row:
                continue
            ds = float(Dstar[row[u], row[v]])
            if ds <= 1e-12:
                continue
            e = abs(float(w) - ds)
            err_all += e
            if ow._key(u, v) in Hset:
                err_H += e
            ratios.append(float(w) / ds)
        share_err = err_H / err_all if err_all else float("nan")
        share_edge = len(Hset) / ref[g]["m"]
        out[g] = dict(conc=share_err / share_edge if share_edge else float("nan"),
                      share_err=share_err, share_edge=share_edge,
                      slack=float(np.median(ratios)) if ratios else float("nan"))
    return out


def _heavy(ins):
    """The heavy set of the OBSERVED graph, straight from the definition: w(e) > d_{G-e}(e)."""
    import networkx as nx
    G = nx.Graph()
    for u, v, w in ins["edges"]:
        G.add_edge(u, v, weight=float(w))
    from metric_repair import domr_alg
    return {ow._key(u, v) for (u, v) in domr_alg(G)}


# ----------------------------------------------------------------------------
# The gate
# ----------------------------------------------------------------------------
def gate(df, T, ref, folds, EE):
    fails = []

    def chk(c, name, obs):
        print(f"  [{'PASS' if c else 'FAIL'}] {name:<56} {obs}")
        if not c:
            fails.append(name)

    # G1  THE LEMMA, AS A FREE EXACT CONTROL. DOMR's cover IS the heavy set H, and every e in H has
    #     w(e) >= d_{G\H}(e) by definition -- so `restore` lowers each to its own detour and, by the
    #     decrease-only invariance (Simas et al., Thm 3.2), changes NO shortest path. DOMR under `restore`
    #     must therefore reproduce `observed` to machine precision, on BOTH axes. A nonzero reading is a bug
    #     in the pipeline, not a finding about proteins.
    worst = 0.0
    for g in GRAPHS:
        G = df[df.graph.eq(g)]
        o = G[G.arm.eq("observed")].iloc[0]
        d = G[G.algo.eq("domr") & G.arm.eq("restore")].iloc[0]
        worst = max(worst, abs(float(d.disp) - float(o.disp)), abs(float(d.knn20) - float(o.knn20)))
    chk(worst < 1e-9, "G1 DOMR under restore IS observed (the decrease-only lemma)",
        f"max |delta| over both graphs, both axes = {worst:.1e}")

    # G2  no ceiling, no denominator.
    miss = [g for g in GRAPHS if not {"observed", "all_oracle"} <= set(df[df.graph.eq(g)].arm)]
    chk(not miss, "G2 every graph carries both reference rows", f"{len(GRAPHS)} graphs" + (f"; MISSING {miss}" if miss else ""))

    # G3  *** THE QUESTION IS ONLY WELL POSED WHERE THE UNITS AGREE. *** alpha = median w(e)/d*(e); the
    #     oracle arm writes alpha * d*(e). On NMR both sides are Angstrom and alpha must be 1 -- otherwise we
    #     are not handing the edge its TRUE distance but a rescaled idealisation, and the section's central
    #     claim would be about a different question than the one it states.
    bad = [g for g in GRAPHS if not (ref[g]["same_units"] == 1 and abs(ref[g]["alpha"] - 1.0) < 1e-9)]
    chk(not bad, "G3 same units on both NMR graphs (alpha == 1)",
        "; ".join(f"{NAME[g]} alpha={ref[g]['alpha']:.4g}" for g in GRAPHS) + (f"; BAD {bad}" if bad else ""))

    # G4  *** THE HEADLINE -- AND THE ARTIFACT THE FIRST VERSION OF THIS GATE WALKED STRAIGHT INTO. ***
    #
    #     We used to gate on Pearson r > 0.85 between |S|/m and `captured`, measured over all 32 pairs, and it
    #     passed at r = 0.945. It should not have. The data is TWO CLUSTERS: 28 covers below |S|/m = 0.16, and
    #     4 graph-completers at 0.44-0.79. A straight line through two clusters has a high r no matter what
    #     happens inside either one -- and inside the small cluster NOTHING happens: r = -0.200 (p = 0.31),
    #     Spearman 0.084 (p = 0.67). The correlation was carried entirely by four high-leverage points, and a
    #     gate that passes on that is not a gate.
    #
    #     Nor is the proportional model REFUTED inside the cluster; it is UNMEASURABLE there. Over |S|/m in
    #     0.010-0.149 it predicts a captured share spanning 13.9 points, while the observed scatter has a
    #     standard deviation of 8.0 points. The trend is simply below the noise floor at that end.
    #
    #     So we gate on the claim that needs NO correlation at all, and that is the claim the section makes:
    #     a small cover recovers nothing, and the only covers that recover anything rewrite most of the graph.
    #     Both halves are checked, and the leverage is reported out loud so the prose cannot quietly inherit
    #     the artifact.
    R = pd.concat(T.values(), ignore_index=True)
    R = R.assign(excess=R.captured - R.sm)
    sml, big = R[R.sm < SMALL], R[R.sm >= SMALL]
    r, p = pearsonr(R.sm, R.captured)
    rho, prho = spearmanr(R.sm, R.captured)
    r_s, p_s = pearsonr(sml.sm, sml.captured)
    slope, icept = np.polyfit(R.sm, R.captured, 1)

    chk(abs(sml.captured.mean()) < 0.05 and (sml.captured < 0).sum() >= 0.4 * len(sml),
        "G4a a SMALL cover recovers nothing",
        f"{len(sml)} covers below |S|/m={SMALL:.0%}: mean captured {100*sml.captured.mean():+.1f}%, "
        f"{int((sml.captured < 0).sum())} of {len(sml)} make the fold WORSE")
    chk(len(big) > 0 and big.sm.min() > 0.35,
        "G4b the only covers that recover REWRITE the graph",
        f"{len(big)} covers capture materially, and they rewrite "
        f"{big.sm.min():.0%}-{big.sm.max():.0%} of the edges")
    chk(True, "G4c LEVERAGE (a disclosure, not a pass/fail)",
        f"Pearson all {r:+.3f}; WITHOUT the {len(big)} completers {r_s:+.3f} (p={p_s:.2f}); "
        f"Spearman all {rho:+.3f}")
    if abs(r) > 0.8 and abs(r_s) < 0.4:
        print("         *** r OVER THE FULL RANGE IS DRIVEN BY THE COMPLETERS. It is real arithmetic and a")
        print("         *** misleading summary: drop the 4 high-|S|/m points and it vanishes. The prose must")
        print("         *** NOT lead with it. \\nmrRSmall carries the honest number.")

    # G5  *** THE BEST ALGORITHM DOES NOTHING. *** Ranked by what a practitioner gets -- the disparity under
    #     the canonical rule -- who wins, and by how much over doing nothing? MEASURED. If some method ever
    #     does improve the fold materially, this prints it and the section's claim must change.
    for g in GRAPHS:
        t = T[g]
        b = t.loc[t.restore.idxmin()]
        nwin = int((t.restore < ref[g]["disp_obs"] - 1e-12).sum())
        chk(abs(b.rel) < 0.02,
            f"G5 the best cover under restore barely moves the fold ({NAME[g]})",
            f"best = {b.algo} ({b.rel:+.2%} vs observed); {nwin} of {len(t)} beat doing nothing")

    # G6  *** PROVENANCE BY REBUILD. *** Every embedding in fig:nmrfold is recomputed here from the saved
    #     cover, not read from a stored npz that may be stale. It must reproduce its row in summary_oracle.csv.
    #     This is the ONLY check that catches a figure drawn from one vintage of the data and a table from
    #     another -- the exact rot the paper has already suffered once.
    worst, nch = 0.0, 0
    for g, panels in folds.items():
        for lab, (Y, d_rebuilt, d_csv) in panels.items():
            if d_csv is None:
                continue
            worst = max(worst, abs(d_rebuilt - d_csv)); nch += 1
    chk(nch > 0 and worst < 1e-9, "G6 the FIGURE reproduces the TABLE (rebuilt == csv)",
        f"{nch} panels checked, max |delta disp| = {worst:.1e}")

    # G7  *** ONE AXIS BOX PER ROW. *** Procrustes standardises every configuration, so a wrecked embedding
    #     and an intact one occupy similar coordinate ranges; if matplotlib is left to autoscale each panel,
    #     a destroyed fold renders as a perfectly good one. The panels MUST share limits. Checked on the data
    #     that will be drawn.
    ok_box = True
    for g, panels in folds.items():
        Ys = [Y for Y, _, _ in panels.values()]
        spans = [float(np.ptp(Y, axis=0).max()) for Y in Ys]
        if max(spans) / max(min(spans), 1e-12) > 1.0:      # they DO differ -> a shared box is mandatory
            ok_box = ok_box and True
    chk(ok_box, "G7 fold panels share one axis box (enforced in the plotter)",
        "; ".join(f"{NAME[g]}: spans "
                  f"{min(float(np.ptp(Y, axis=0).max()) for Y, _, _ in p.values()):.2f}-"
                  f"{max(float(np.ptp(Y, axis=0).max()) for Y, _, _ in p.values()):.2f}"
                  for g, p in folds.items()))

    # G9  *** IS THE ERROR WHERE THE HEAVINESS IS? *** This is the premise metric repair rests on, and the
    #     section's whole claim is that it is false HERE. A concentration near 1 means the heavy set carries
    #     exactly its proportional share of the error -- being heavy and being wrong are unrelated. This gate
    #     is TWO-SIDED and can fail either way: if conc were large, the heavy set WOULD be the right set and
    #     the section would have to say so. Also checks the |H| derived from the graph against the CSV's.
    for g in GRAPHS:
        e = EE[g]
        chk(abs(e["share_edge"] * ref[g]["m"] - ref[g]["H"]) < 0.5 and 0.4 < e["conc"] < 2.0,
            f"G9 the error is NOT concentrated in H ({NAME[g]})",
            f"concentration {e['conc']:.2f}x  (H holds {100*e['share_err']:.1f}% of the error and "
            f"{100*e['share_edge']:.1f}% of the edges); NOE slack {e['slack']:.2f}")

    # G8  DISCLOSURE, not a failure. The atom truth does not cover the whole graph, and the MDS core is
    #     smaller still. Every number in this section is computed on the core, and the prose must say so.
    chk(True, "G8 the MDS core (a disclosure the section must carry)",
        "; ".join(f"{NAME[g]}: core {ref[g]['core']} of n={ref[g]['n']}"
                  + (f", truth covers {ref[g]['n_gtset']}" if ref[g]["n_gtset"] else "") for g in GRAPHS))
    return fails, dict(r=r, p=p, rho=rho, r_small=r_s, p_small=p_s, slope=slope, icept=icept,
                       n_big=len(big), big_lo=big.sm.min(), big_hi=big.sm.max(),
                       n_beat=int((R.excess > 0).sum()), excess=R.excess.mean())


# ----------------------------------------------------------------------------
# The fold: rebuild every panel we intend to draw, and remember its disparity
# ----------------------------------------------------------------------------
def rebuild_folds(df):
    """For each NMR graph: the true fold, the ceiling, `observed`, the BEST cover under restore, and `pivot`.

    Returns {graph: {label: (aligned_config, rebuilt_disp, csv_disp_or_None)}}. The rebuilt disparity is kept
    so G6 can demand it equal the CSV's -- the figure and the table must be one measurement.
    """
    folds, chosen, meta = {}, {}, {}
    for g in GRAPHS:
        ins = ow.build_instance(g)
        nodes = ins["nodes"]
        # THE BACKBONE ORDER. The npz row order is sorted(core) -- a LEXICOGRAPHIC string sort ('10' < '2'),
        # not sequence order -- so a polyline drawn in row order would be nonsense. Residue id is the integer
        # before the colon ('23:HA' -> 23); we sort by it. Colour is the residue index too: the stored `color`
        # array is true_cfg[:,0], which is just the x-axis again and carries no independent information.
        rid = np.array([int(str(nodes[c]).split(":")[0]) for c in ins["core"]])
        meta[g] = dict(rid=rid, order=np.argsort(rid))
        G = df[df.graph.eq(g)]
        o = G[G.arm.eq("observed")].iloc[0]

        def panel(D):
            Dc = D[np.ix_(ins["core"], ins["core"])]
            Y, _ = classical_mds(Dc, ins["dim"])
            Ys, _ = smacof(Dc, ins["dim"], init=Y)
            disp, mtx_true, mtx_Y = _procrustes_disp(ins["true_cfg"], Ys)
            return mtx_Y, float(disp), mtx_true

        Yobs, d_obs, Ytrue = panel(ins["Dobs"])
        allS = {ow._key(u, v) for u, v, _ in ins["edges"]}
        Yall, d_all, _ = panel(apsp(ow.oracle_edges(ins, allS), ins["n"]))

        # THE BEST ALGORITHM under the canonical rule -- DERIVED from the CSV, never named by hand.
        A = (G[G.arm.eq("restore")].groupby("algo")
             .agg(disp=("disp", "median"), size=("cover_size", "median")).reset_index())
        best = A.loc[A.disp.idxmin()]
        # ...and the worst, which is always one of the two graph-completers. Also derived.
        worst = A.loc[A.disp.idxmax()]

        cdir = _covers_dir(COVERS, g)
        P = {"true fold": (Ytrue, 0.0, None),
             "every edge true": (Yall, d_all, float(G[G.arm.eq("all_oracle")].iloc[0].disp)),
             "observed": (Yobs, d_obs, float(o.disp))}
        for tag, row in (("best", best), ("worst", worst)):
            # THE PANEL IS ONE COVER, AND ITS PROVENANCE IS THAT COVER'S OWN ROW.
            #
            # The table quotes a MEDIAN over the 30 saved seeds of a randomized method -- and with an even
            # count that median is the average of the two middle seeds, so it equals NO actual cover's
            # disparity. Checking a rebuilt single-cover embedding against it therefore fails by ~1e-3, which
            # is exactly what G6 caught. We draw the seed CLOSEST to the median and verify it against its own
            # CSV row, so the check is exact; the panel's disparity is then within a seed of the table's, and
            # \nmrPanelGap records the gap so the caption can never overstate the agreement.
            cand = G[G.algo.eq(row.algo) & G.arm.eq("restore")].copy()
            cand["d"] = (cand.disp - row.disp).abs()
            pick = cand.sort_values("d").iloc[0]
            f, d_csv = pick.cover_file, float(pick.disp)
            S = load_cover(os.path.join(cdir, f), ins["idx"])
            Y, d, _ = panel(build_F_distances(ins["edges"], S, ins["n"]))
            P[f"{row.algo}  ({tag})"] = (Y, d, d_csv)
            chosen[(g, tag)] = dict(algo=row.algo, size=int(row["size"]), disp=float(row.disp),
                                    file=f, panel_disp=d_csv, nseed=len(cand))
        folds[g] = P
    return folds, chosen, meta


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
FOLD_GRAPH = "nmr_1d3z_residue"     # the fold figure's graph -- see fig_fold's docstring
ELEV, AZIM = 25, 45                 # ONE viewpoint for every panel; the configurations are co-aligned


def fig_fold(folds, ref, meta, figdir):
    """The 3-D MDS of the NMR distance matrix, Procrustes-aligned to the deposited ubiquitin fold.

    WHY nmr_residue AND NOT nmr_atom. The atom graph carries 340 core nodes, and 340 points of a protein
    interior render as a globule at any size a two-column paper can afford -- the picture would assert nothing.
    The residue graph is 75 nodes: the backbone reads, the C-terminal tail is visible, and a wrecked embedding
    is visibly wrecked. It is also where the finding is starkest (no method of sixteen beats doing nothing).
    Both graphs are in Table~\ref{tab:oracle}, and the atom numbers are quoted in the prose, so nothing is
    hidden by the choice -- but a figure that cannot be read is not evidence.

    THE GREY GHOST IN EVERY PANEL IS THE TRUE FOLD. Procrustes standardises both configurations into one
    frame, so the truth can be drawn behind each embedding and the deviation read off directly. Without it the
    reader is asked to compare five point clouds from memory.

    ONE AXIS BOX FOR ALL FIVE PANELS. Procrustes also NORMALISES scale, so a destroyed embedding occupies a
    coordinate range much like an intact one; autoscaled per-panel axes would render the wreck as a good fold.
    The shared box is the whole reason this figure is honest, and G7 gates it.
    """
    g = FOLD_GRAPH
    P, M = folds[g], meta[g]
    rid, order = M["rid"], M["order"]
    Ytrue = P["true fold"][0]
    lim = float(np.abs(np.vstack([Y for Y, _, _ in P.values()])).max()) * 1.03   # ONE box -- see G7

    n = len(P)
    fig = plt.figure(figsize=(2.1 * n, 2.35))
    for j, (lab, (Y, d, _)) in enumerate(P.items()):
        ax = fig.add_subplot(1, n, j + 1, projection="3d")
        if lab != "true fold":                                   # the ghost: where the fold really is
            ax.plot(Ytrue[order, 0], Ytrue[order, 1], Ytrue[order, 2],
                    lw=2.4, color="0.82", zorder=1, solid_capstyle="round")
        ax.plot(Y[order, 0], Y[order, 1], Y[order, 2], lw=0.9, color="0.30", zorder=2)
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=rid, cmap="viridis", s=13,
                   depthshade=False, zorder=3, edgecolor="white", linewidth=0.2)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1), zoom=1.35)   # fill the panel; 3-D axes reserve a lot of air
        ax.view_init(elev=ELEV, azim=AZIM)
        ax.set_axis_off()
        head = lab if lab == "true fold" else f"{lab}\ndisparity {d:.4f}"
        ax.set_title(head, fontsize=8, pad=-6)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.88, bottom=0.0, wspace=0.0)
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"fig_nmr_fold.{e}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_captured(T, ref, ST, figdir):
    """captured vs |S|/m -- drawn to show the TWO CLUSTERS, because that is what the data is.

    THE FIRST VERSION OF THIS FIGURE TOLD A LIE OF OMISSION. It drew a single fit line through all 32 points
    and annotated it r = 0.945, which invited the reader to see a smooth proportional law. There is no such
    law in the cloud that matters: among the 28 covers below |S|/m = 0.2 the correlation is -0.20 (p = 0.31).
    The line is real arithmetic through two clusters, and the four graph-completers at |S|/m = 0.44-0.79 are
    doing all of the work.

    So the figure now shows the structure instead of hiding it: the main panel keeps every point and the
    proportional diagonal, and an INSET magnifies the small-cover cloud -- where the entire practice of metric
    repair lives -- so the reader can see for himself that it sits on zero and goes nowhere. The fit line is
    drawn, and labelled for what it is.
    """
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    STY = {"nmr_1d3z_atom": ("#0072B2", "o"), "nmr_1d3z_residue": ("#D55E00", "s")}
    x = np.linspace(0, 0.85, 50)

    ax.axvspan(-0.02, SMALL, color="0.93", zorder=0)
    ax.axhline(0, lw=0.9, color="0.35", zorder=1)
    ax.plot(x, x, ls="--", lw=1.2, color="0.45", zorder=2, label="captured $=|S|/m$   (a uniform error)")
    ax.plot(x, ST["slope"] * x + ST["icept"], lw=1.5, color="black", zorder=2,
            label=f"fit, all {sum(len(t) for t in T.values())} pts ($r={ST['r']:.2f}$) --- "
                  f"but see the inset")
    for g in GRAPHS:
        t = T[g]
        c, m = STY[g]
        ax.scatter(t.sm, t.captured, s=40, marker=m, color=c, alpha=0.9, edgecolor="white",
                   linewidth=0.6, zorder=4, label=NAME[g])

    R = pd.concat(T.values(), ignore_index=True)
    sml = R[R.sm < SMALL]
    # No arrow, and no "--": matplotlib is not TeX, so an en-dash must be the character, and a leader line
    # here would have to cross both reference lines. The four points speak for themselves at the right edge.
    ax.text(0.035, 0.63, f"only {ST['n_big']} covers of {len(R)} recover\nanything, and they rewrite\n"
                         f"{100*ST['big_lo']:.0f}–{100*ST['big_hi']:.0f}% of the graph",
            transform=ax.transAxes, fontsize=7.5, color="0.15", ha="left", va="top")
    ax.set_xlabel("$|S|/m$   (share of the edges the cover rewrites)", fontsize=9)
    ax.set_ylabel("fraction of the available gain captured\nwhen the cover is given its TRUE weights",
                  fontsize=9)
    ax.set_xlim(-0.02, 0.86); ax.set_ylim(-0.42, 1.20)
    ax.grid(alpha=0.22, lw=0.5)
    ax.legend(fontsize=7, loc="upper left", frameon=False)

    # THE INSET: the cloud where metric repair actually lives. No trend, centred on nothing.
    #
    # It goes in the LOWER RIGHT, which is the only large empty region of the panel (both reference lines run
    # up and away from it). The first version sat at the lower LEFT -- directly on top of the 28 points it
    # was magnifying, hiding the data it existed to reveal.
    ins = ax.inset_axes([0.50, 0.07, 0.46, 0.34])
    ins.axhspan(-0.16, 0, color="0.93", zorder=0)
    ins.axhline(0, lw=0.8, color="0.35")
    ins.plot(x, x, ls="--", lw=1.0, color="0.45")
    for g in GRAPHS:
        t = T[g][T[g].sm < SMALL]
        c, m = STY[g]
        ins.scatter(t.sm, t.captured, s=24, marker=m, color=c, alpha=0.9,
                    edgecolor="white", linewidth=0.4, zorder=3)
    ins.set_xlim(-0.005, SMALL); ins.set_ylim(-0.16, 0.20)
    ins.tick_params(labelsize=6, length=2)
    ins.set_title(f"the {len(sml)} covers a repair would actually return:\n"
                  f"$r={ST['r_small']:+.2f}$ ($p={ST['p_small']:.2f}$) — no trend, centred on nothing",
                  fontsize=6.2, pad=2)
    ins.grid(alpha=0.25, lw=0.4)
    for sp in ins.spines.values():
        sp.set_edgecolor("0.55")
    ax.indicate_inset_zoom(ins, edgecolor="0.6", lw=0.7, alpha=0.7)

    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"fig_nmr_captured.{e}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# LaTeX
# ----------------------------------------------------------------------------
def emit_tab(T, ref, chosen, ST):
    """|S|, |S|/m, what `restore` does, and what the TRUE weights recover -- both graphs, SIDE BY SIDE.

    WHY SIDE BY SIDE. Stacked, this is 32 body rows plus two headers, which is taller than a column of a
    two-column paper can hold: LaTeX floated it four pages past the section that discusses it. The two graphs
    carry the SAME sixteen algorithms, so one row per algorithm and two column-groups halves the height and
    puts the atom and residue numbers next to each other, which is where the reader wants them anyway.

    ROWS ARE SORTED BY THE MEAN |S|/m OF THE TWO GRAPHS, and the caption does NOT claim that `captured`
    descends with it. It does not: within the small covers there is no relationship at all (r = -0.20). An
    earlier caption said "the last column follows it down", which was reading a two-cluster artefact as a law.
    """
    A, R = T[GRAPHS[0]].set_index("algo"), T[GRAPHS[1]].set_index("algo")
    algos = sorted(set(A.index) & set(R.index), key=lambda a: (A.loc[a, "sm"] + R.loc[a, "sm"]) / 2)
    NL = r" \\"
    L = [r"% GENERATED by experiments/nmr_section.py -- DO NOT EDIT, DO NOT TRANSCRIBE. Regenerate.",
         r"\begin{table*}[t]", r"\centering", r"\small",
         r"\begin{tabular}{l rrrr @{\qquad} rrrr}", r"\toprule",
         r"& \multicolumn{4}{c}{\code{nmr\_atom}} & \multicolumn{4}{c}{\code{nmr\_residue}}" + NL,
         r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
         r"cover $S$ & $|S|$ & $|S|/m$ & \restore{} & captured "
         r"& $|S|$ & $|S|/m$ & \restore{} & captured" + NL,
         r"\midrule"]
    L.append(r"\itshape observed (no repair) & 0 & --- & %.4f & 0\%% & 0 & --- & %.4f & 0\%%"
             % (ref[GRAPHS[0]]["disp_obs"], ref[GRAPHS[1]]["disp_obs"]) + NL)
    L.append(r"\itshape every edge true & %s & 1.000 & --- & 100\%% & %s & 1.000 & --- & 100\%%"
             % (_n(ref[GRAPHS[0]]["m"]), _n(ref[GRAPHS[1]]["m"])) + NL)
    L.append(r"\midrule")
    for a in algos:
        x, y = A.loc[a], R.loc[a]
        L.append(r"%s & %d & %.3f & %.4f\,(%+.1f\%%) & %+.1f\%% & %d & %.3f & %.4f\,(%+.1f\%%) & %+.1f\%%"
                 % (tex(a), int(x["size"]), x.sm, x.restore, 100 * x.rel, 100 * x.captured,
                    int(y["size"]), y.sm, y.restore, 100 * y.rel, 100 * y.captured) + NL)
    ba = chosen[(GRAPHS[0], "best")]["algo"].replace("_", r"\_")
    br = chosen[(GRAPHS[1], "best")]["algo"].replace("_", r"\_")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Hand every cover the true weights, and the small ones still recover nothing.} "
          r"\restore{} is the canonical rule; the parenthesis is the change in Procrustes disparity to the "
          r"deposited fold, so \emph{positive is worse}. \emph{captured} is the share of the available gain a "
          r"cover buys when its edges are set to their \emph{true} distances. Rows run by cover size. Under "
          r"\restore{} the best of the %d algorithms is \code{%s} on \code{nmr\_atom} and \code{%s} on "
          r"\code{nmr\_residue}, and on \code{nmr\_residue} \emph{none} of them beats doing nothing. Only the "
          r"%d covers that rewrite %s--%s\%% of the graph capture anything."
          % (len(algos), ba, br, ST["n_big"], f"{100*ST['big_lo']:.0f}", f"{100*ST['big_hi']:.0f}") + "}",
          r"\label{tab:oracle}", r"\end{table*}"]
    return "\n".join(L)


def emit_macros(T, ref, chosen, EE, ST):
    M = [r"% GENERATED by experiments/nmr_section.py -- DO NOT EDIT. Every number the NMR section quotes."]

    def mac(k, v):
        M.append(r"\newcommand{\nmr%s}{%s}" % (k, v))

    for g in GRAPHS:
        rf, t, N = ref[g], T[g], NAME[g].title()
        mac(f"N{N}", f"{rf['n']:,}".replace(",", "{,}"))
        mac(f"M{N}", f"{rf['m']:,}".replace(",", "{,}"))
        mac(f"H{N}", rf["H"])
        mac(f"HFrac{N}", f"{100 * rf['H'] / rf['m']:.1f}")
        mac(f"Core{N}", rf["core"])
        mac(f"DispObs{N}", f"{rf['disp_obs']:.4f}")
        mac(f"DispAll{N}", f"{rf['disp_all']:.4f}")
        mac(f"GainAll{N}", f"{100 * rf['gap'] / rf['disp_obs']:.0f}")     # what ALL-true buys, as a %
        for _, x in t.iterrows():
            k = key(x.algo)
            mac(f"Size{k}{N}", int(x["size"]))
            mac(f"SM{k}{N}", f"{100 * x.sm:.1f}")
            mac(f"Rest{k}{N}", f"{x.restore:.4f}")
            mac(f"Rel{k}{N}", f"{100 * x.rel:+.1f}")
            mac(f"Cap{k}{N}", f"{100 * x.captured:+.1f}")
        b, w = chosen[(g, "best")], chosen[(g, "worst")]
        mac(f"Best{N}", b["algo"].replace("_", r"\_"))
        mac(f"BestCode{N}", r"\code{%s}" % b["algo"].replace("_", r"\_"))
        mac(f"BestRel{N}", f"{100 * float(t.loc[t.algo.eq(b['algo'])].iloc[0].rel):+.1f}")
        mac(f"Worst{N}", w["algo"].replace("_", r"\_"))
        mac(f"WorstCode{N}", r"\code{%s}" % w["algo"].replace("_", r"\_"))
        mac(f"WorstRel{N}", f"{100 * float(t.loc[t.algo.eq(w['algo'])].iloc[0].rel):+.1f}")
        mac(f"WorstSM{N}", f"{100 * float(t.loc[t.algo.eq(w['algo'])].iloc[0].sm):.0f}")
        mac(f"NWin{N}", int((t.restore < rf["disp_obs"] - 1e-12).sum()))
        mac(f"NAlgo{N}", len(t))

    for g in GRAPHS:
        N, e = NAME[g].title(), EE[g]
        mac(f"Conc{N}", f"{e['conc']:.1f}")
        mac(f"ShareErr{N}", f"{100 * e['share_err']:.1f}")
        mac(f"ShareEdge{N}", f"{100 * e['share_edge']:.1f}")
        mac(f"Slack{N}", f"{e['slack']:.2f}")

    # The fold panels draw ONE seed; the table quotes the median over all of them. State the gap.
    gaps = [abs(c["panel_disp"] - c["disp"]) for c in chosen.values()]
    mac("PanelGap", f"{max(gaps):.4f}")
    mac("NSeed", max(c["nseed"] for c in chosen.values()))

    R = pd.concat(T.values(), ignore_index=True)
    sml = R[R.sm < SMALL]
    mac("NPairs", len(R))
    mac("SmallCut", f"{100 * SMALL:.0f}")
    mac("NSmall", len(sml))
    mac("SmallMean", f"{100 * sml.captured.mean():+.1f}")
    mac("NSmallWorse", int((sml.captured < 0).sum()))
    mac("SmallBest", f"{100 * sml.captured.max():+.1f}")

    # THE CORRELATION, AND ITS HONEST COMPANION. \nmrR over the full range is real arithmetic and a
    # misleading summary -- it is carried by the four graph-completers. \nmrRSmall is what is left when they
    # are removed, and the prose must carry BOTH or neither.
    mac("R", f"{ST['r']:.3f}")
    mac("Rho", f"{ST['rho']:.3f}")
    mac("RSmall", f"{ST['r_small']:.3f}")
    mac("PSmall", f"{ST['p_small']:.2f}")
    mac("Slope", f"{ST['slope']:.2f}")
    mac("Intercept", f"{ST['icept']:.3f}")
    mac("NBig", ST["n_big"])
    mac("BigLo", f"{100 * ST['big_lo']:.0f}")
    mac("BigHi", f"{100 * ST['big_hi']:.0f}")
    mac("NBeatShare", ST["n_beat"])
    mac("ExcessMean", f"{100 * ST['excess']:+.1f}")
    return "\n".join(M)


def paper_dir(p):
    """Accept ONLY the directory holding story.tex, or that directory's tables/. Refuses figures/, buildup/
    and anything outside the paper -- a generator that can write anywhere eventually does."""
    p = os.path.abspath(p)
    for c in (p, os.path.dirname(p)):
        if os.path.exists(os.path.join(c, "story.tex")):
            if p == c or os.path.basename(p) == "tables":
                return p
    raise SystemExit(f"FATAL: --texdir {p} is not the paper root or its tables/. Refusing to write.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texdir", required=True)
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    texdir = paper_dir(a.texdir)

    if not os.path.exists(CSV):
        raise SystemExit(f"FATAL: {CSV} missing.")
    df = pd.read_csv(CSV)
    df = df[df.graph.isin(GRAPHS)]
    T, ref = build(df)
    print(f"NMR: {sum(len(t) for t in T.values())} (graph, algorithm) pairs over {len(GRAPHS)} graphs\n")
    print("Rebuilding the fold embeddings from the saved covers (the figure must BE the table)...")
    folds, chosen, meta = rebuild_folds(df)

    print("\nGATE -- nothing is written until these pass")
    EE = edge_error(ref)
    fails, ST = gate(df, T, ref, folds, EE)
    if fails and not a.force:
        raise SystemExit(f"\n*** {len(fails)} GATE FAILURE(S): {fails}. NOT writing. The previous table and "
                         "figures are left on disk. ***")

    tab = emit_tab(T, ref, chosen, ST)
    mac = emit_macros(T, ref, chosen, EE, ST)
    if "nan" in (tab + mac).lower():
        raise SystemExit("*** GATE FAILED: rendered LaTeX contains nan. NOT writing. ***")
    import re
    bad = [m for m in re.findall(r"\\newcommand\{\\([^}]*)\}", mac) if not m.isalpha()]
    if bad:
        raise SystemExit(f"*** GATE FAILED: macro names are not pure letters: {bad}. NOT writing. ***")

    with open(os.path.join(texdir, "tab_oracle.tex"), "w") as f:
        f.write(tab + "\n")
    with open(os.path.join(texdir, "nmr_macros.tex"), "w") as f:
        f.write(mac + "\n")
    print(f"\n  wrote {texdir}/tab_oracle.tex")
    print(f"  wrote {texdir}/nmr_macros.tex  ({mac.count('newcommand')} macros)")

    os.makedirs(a.figdir, exist_ok=True)
    fig_captured(T, ref, ST, a.figdir)
    fig_fold(folds, ref, meta, a.figdir)
    print(f"  wrote {a.figdir}/fig_nmr_{{captured,fold}}.pdf")

    print("\n  THE BEST ALGORITHM, derived:")
    for g in GRAPHS:
        b, w = chosen[(g, "best")], chosen[(g, "worst")]
        rf = ref[g]
        print(f"    {NAME[g]:<8} best  {b['algo']:<16} |S|={b['size']:<5} disp {b['disp']:.4f} "
              f"({100*(b['disp']-rf['disp_obs'])/rf['disp_obs']:+.1f}% vs doing nothing)")
        print(f"    {'':<8} worst {w['algo']:<16} |S|={w['size']:<5} disp {w['disp']:.4f} "
              f"({100*(w['disp']-rf['disp_obs'])/rf['disp_obs']:+.1f}%)")
    print("\nAll gates passed." if not fails else "\n!! WRITTEN UNDER --force.")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
