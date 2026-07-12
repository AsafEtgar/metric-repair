"""mds_sweep.py -- per-ALGORITHM MDS geometry: Procrustes disparity for every repair algorithm, not
just one representative cover per variant (mds_recovery.py's default). Answers the sharper question:
does the geometry recovery depend on WHICH algorithm -- i.e. on cover quality/size?

  real graphs (ripe/nmr): every saved campaign cover -> disparity + cover size |S|/|H| (cheap: just a
                          build_F + MDS per cover). This is the headline comparison (a compact table),
                          and the disparity-vs-cover-size scatter is the appendix view.
                          dimacs_ny_d/_t join this tier: one node set, one geography, two edge weights --
                          road distance (exactly metric, |H|=0) and travel time (17 heavy edges of 6,017).
  RGG controls:           recompute each algorithm's cover -> disparity, and PERSIST the aligned point
                          clouds so mds_plots.py can draw the per-algorithm map grid.
  realrec (planted):      the hybrid, and the strongest of the three: a break PLANTED in a real metric base
                          (the NY road net), scored against ground truth we did not plant (the road map).
                          The RGG control has ground truth we planted; the pure-real tier has an injury we
                          did not. This has one of each.

Reuses mds_recovery.py's MDS + scoring helpers verbatim, so it cannot diverge from the main analysis, and
runs each RGG cover through the CAMPAIGN's own algorithm suite (harness.build_suite) rather than a private
re-implementation. Self-contained otherwise; never imported by a running task.

    sage -python experiments/mds_sweep.py --outdir analysis --plot
"""
import argparse
import csv
import glob
import os
import shutil
import sys

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                       # experiments/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # repo root
from mds_recovery import (classical_mds, smacof, _procrustes_disp, finite_core,       # noqa: E402
                          _nx_from_edges, _norm_cover, PURE_REAL_DIM, RGG_SPECS)
from downstream_recovery import (DOWNSTREAM_GRAPHS, load_graph, load_cover, _covers_dir,  # noqa: E402
                                 apsp, build_F_distances, true_distances, variant_of)
from graph_models import seed_all, random_geometric_metric_graph, break_metric_graph  # noqa: E402
from metric_repair import domr_alg                                                    # noqa: E402
# READ-ONLY import of the campaign harness (harness.py has no import-time side effects beyond setting the
# single-threaded-BLAS env vars, which is what we want anyway). We do NOT edit it.
from harness import build_suite, run_isolated, VERIFY                                 # noqa: E402

# ----------------------------------------------------------------------------
# The algorithm suite -- taken from the campaign, NOT re-implemented
# ----------------------------------------------------------------------------
# Every build_suite entry is fn(CC) -> (cover, info); every lambda this module used to hold returned a BARE
# cover. Passing a build_suite callable straight into _norm_cover therefore raises `TypeError: 'set' object is
# not subscriptable`, which the old bare `except Exception: continue` swallowed and printed as
# "[skip gmr_ilp]" -- a wiring bug masquerading as non-convergence, with the run still exiting 0. So every
# algorithm now goes through run_isolated, which speaks the (cover, info) contract, forks, and enforces a cap.
# There is no try/except around the parent-side plumbing any more: a wiring bug MUST crash.
SUITE_SEED = 1000                       # seeds *_bestofk / *_rand inside the suite (RGG_SPECS' seed too)
_SUITE = {name: (fn, variant, vkey) for name, variant, vkey, _n, _rg, fn in build_suite(seed=SUITE_SEED)}

# NB on `pivot`: MVD_Pivot draws its pivots from the GLOBAL numpy RNG (metric_repair.py:396). run_isolated
# FORKS, so the parent's global RNG is never advanced by any algorithm, and every algorithm below sees exactly
# the RNG state left by seed_all(seed) + graph generation -- identical to running pivot in-process, and
# independent of suite order. Verified: pivot |S| is unchanged from the pre-adapter run on all three RGGs.
SWEEP_ALGOS = [
    ("domr",           "DOMR"),
    ("gmr_bestofk",    "GMR"),      # DETERMINISTIC best-of-12 -- see the mds_recovery._bestofk note
    ("iomr_bestofk",   "IOMR"),
    ("spc_gmr",        "GMR"),
    ("spc_iomr",       "IOMR"),
    ("l1sep_gmr",      "GMR"),
    ("l1sep_iomr",     "IOMR"),
    ("pivot",          "GMR"),
    ("left_edge",      "IOMR"),
    # --- added: the exact solvers, the randomized rounding, and the plain threshold rounding ---
    ("gmr_ilp",        "GMR"),
    ("iomr_ilp",       "IOMR"),
    ("gmr_rand",       "GMR"),
    ("iomr_rand",      "IOMR"),
    ("gmr_thr_naive",  "GMR"),
    ("iomr_thr_naive", "IOMR"),
]

# Wall-clock cap per (algorithm, graph), seconds. The ILPs get 300s -- NOT harness.ALGO_TIMEOUT, which is 45s.
# RGG_SPECS (n=300, deg=10, frac=0.15, incl. `mixed`) sits OFF the campaign grid, so campaign convergence rates
# do not transfer; all six ILP runs on these exact instances were MEASURED to converge, the slowest at 74.5s
# (iomr_ilp / rgg_mixed). A 45s cap would kill precisely the panel the guard exists to protect. If 300s fires,
# investigate -- do not quietly drop the panel.
ILP_CAP = 300
DEFAULT_CAP = 600
ALGO_CAP = {"gmr_ilp": ILP_CAP, "iomr_ilp": ILP_CAP}

# Bigger instances (the n=1000 RGGs, the n=5000 road net) get a RAISED cap rather than a smaller ambition.
# The campaign drops both ILPs above n=500, so at n=1000 they may not converge at all -- but "may not" is not
# "does not", and the way to find out is to let them try. l1sep's separation LP is size-bound and *_bestofk
# pays 12 rounding rounds, so they are guarded too. A miss is NOT a silent drop: _failed_row emits a
# status!=ok row and mds_plots draws a labelled null panel.
BIG_N = 500
BIG_CAP = 900
BIG_CAP_ALGOS = {"gmr_ilp", "iomr_ilp", "l1sep_gmr", "l1sep_iomr", "gmr_bestofk", "iomr_bestofk"}

# The road net (n=5000, m=6017) is a different regime again, and the raised cap does NOT rescue it. The
# covering LP's cost is driven by |H|, not just n: the campaign solved this very graph when it carried 17
# heavy edges (results_real_covers/dimacs_ny_t/ has gmr_ilp, l1sep, *_bestofk), but a planted break leaves
# |H| ~ 400, and the LP family then does not land. MEASURED on dimacs_ny_d_inflate at the 900s cap:
# gmr_bestofk timed out (900.02s), as did the n=1000 iomr_ilp. So 900s is not a cap the LP family is close
# to missing -- it misses by an unknown margin, and paying 900s x 9 LP algorithms x 3 directions to relearn
# that costs ~5 hours and buys nothing. HUGE_CAP is therefore a laptop-sized attempt, honestly labelled:
# every algorithm still RUNS, and one that misses gets a status=timeout row and a red "NOT MEASURED" panel,
# never a silent drop. The combinatorial covers (domr, spc_*, pivot, left_edge) land comfortably inside it.
HUGE_N = 2000
HUGE_CAP = 300


def _cap_for(algo, n):
    """Wall-clock cap for one (algorithm, graph). At n <= BIG_N this returns EXACTLY the caps the n=300 RGG
    rows were measured under, so those rows cannot move; only the bigger instances see a different cap."""
    if n > HUGE_N:
        return HUGE_CAP
    if n > BIG_N and algo in BIG_CAP_ALGOS:
        return BIG_CAP
    return ALGO_CAP.get(algo, DEFAULT_CAP)


# Planted-break instances on a REAL metric base: (name, base, direction, frac, magnitude, seed).
# dimacs_ny_d is the base because it is EXACTLY METRIC (|H| = 0, verified in _planted_base): every bit of
# non-metricity in the corrupted graph is ours, so the true 2-D configuration (the DIMACS road geography)
# is a ground truth we did not have to assume. Magnitude and seed follow RGG_SPECS' convention.
#
# The fracs are NOT uniform, and deliberately so. inflate and mixed take the same dose (0.20); deflate takes
# 0.30 because a road network is HARD TO SHORTCUT -- break_metric_graph can only touch cycle edges, and
# deflate additionally needs a usable asymmetric 2-path, so of 6,017 edges (1,724 of them bridges, cycle space
# only 1,018) a frac of 0.10 deflates just 31 edges. 0.30 gives deflate its best chance; it still only lands
# 103. Measured observed-side disparity against the 0.0273 uncorrupted floor: inflate 0.3516 (12.9x),
# mixed 0.1382 (5.1x), deflate 0.0308 (1.1x -- flat, and structurally so).
REALREC_SPECS = [
    ("dimacs_ny_d_inflate", "dimacs_ny_d", "inflate", 0.20, 3.0, 1000),
    ("dimacs_ny_d_deflate", "dimacs_ny_d", "deflate", 0.30, 3.0, 1000),
    ("dimacs_ny_d_mixed",   "dimacs_ny_d", "mixed",   0.20, 3.0, 1000),
]
# NB the planted graphs are named `<base>_<direction>`, so they never collide with the UNCORRUPTED rows the
# pure-real path writes for `dimacs_ny_d` / `dimacs_ny_t` under the same (graph, algo) merge key.

# Provenance: what actually distinguishes one panel from another. `algo` names the cover; `cover_file` records
# WHICH saved cover on the real graphs (randomized algos have 30 seeds on disk and we take the first); the
# cover-side knobs are the ones the `_bestofk` bug misreported, so they are recorded rather than assumed.
SWEEP_FIELDS = ["dataset", "graph", "corruption", "algo", "variant", "status", "converged", "valid",
                "cover_size", "ratio_domr", "disp_classical", "disp_smacof", "neg_mass",
                "n", "n_used", "avg_degree", "frac", "magnitude", "seed", "radius",
                "cover_algo", "cover_file", "rounding", "oracle", "best_of_k", "lp_seed",
                "wall", "source", "base", "n_corrupted"]
KEY = ("graph", "algo")

# What harness.build_suite actually passes to covering_lp_cover for each of our members (harness.py:213-224).
# Recorded, not assumed -- these are the knobs the `_bestofk` bug got wrong.
_ROUNDING_OF = {"gmr_thr_naive": "deterministic", "iomr_thr_naive": "deterministic",
                "gmr_bestofk": "deterministic", "iomr_bestofk": "deterministic",
                "gmr_rand": "randomized", "iomr_rand": "randomized"}
_BESTOFK_OF = {"gmr_bestofk": 12, "iomr_bestofk": 12}

VAR_COLOR = {"observed": "#000000", "DOMR": "#888888", "GMR": "#0072B2", "IOMR": "#D55E00"}
REAL_TITLE = {"ripe_atlas": "ripe_atlas", "nmr_1d3z_atom": "nmr_1d3z_atom",
              "nmr_1d3z_residue": "nmr_1d3z_residue"}


# ----------------------------------------------------------------------------
# Scoring + row construction
# ----------------------------------------------------------------------------
def _score(D, true_cfg, dim, store=None, tag=None, algo=None):
    """Classical + SMACOF Procrustes disparity to true_cfg, plus neg_mass (a property of D).

    When `store` is given, the PROCRUSTES-ALIGNED coordinates are kept under
    `emb::<graph>::<corruption>::<algo>::<mds_algo>` -- the same key scheme mds_recovery uses, with the
    algorithm in the variant slot, so one parser reads both npz files. scipy's procrustes standardizes mtx1
    (centre, unit Frobenius norm) and returns mtx2 in THAT frame, so the aligned coords are only comparable to
    the identically standardized true configuration written by _store_true."""
    Yc, neg = classical_mds(D, dim)
    Ys, _ = smacof(D, dim, init=Yc)
    dc, _, aligned_c = _procrustes_disp(true_cfg, Yc)
    ds, _, aligned_s = _procrustes_disp(true_cfg, Ys)
    if store is not None and tag is not None:
        for mds_algo, aligned in (("classical", aligned_c), ("smacof", aligned_s)):
            if aligned is not None:
                store[f"emb::{tag}::{algo}::{mds_algo}"] = aligned
    return dc, ds, neg


def _store_true(store, tag, true_cfg):
    """The true configuration in the SAME standardized frame procrustes puts the aligned embeddings in
    (centre, then divide by the Frobenius norm) -- identical to mds_recovery.py's normalization. Plus the
    per-node colour = the true first coordinate."""
    t = np.asarray(true_cfg, float) - np.asarray(true_cfg, float).mean(0)
    t = t / (np.linalg.norm(t) or 1.0)
    store[f"true::{tag}"] = t
    store[f"color::{tag}"] = np.asarray(true_cfg, float)[:, 0]


def _row(dataset, graph, corruption, algo, variant, size, H, dc, ds, neg, **extra):
    """One CSV row. Keyword-form with **extra so both sweeps can attach their own provenance without a
    positional signature that has to grow in lockstep."""
    row = {k: "" for k in SWEEP_FIELDS}
    row.update({
        "dataset": dataset, "graph": graph, "corruption": corruption, "algo": algo, "variant": variant,
        "status": "ok", "cover_size": size,
        "ratio_domr": round(size / max(H, 1), 4),
        "disp_classical": round(dc, 6) if np.isfinite(dc) else "",
        "disp_smacof": round(ds, 6) if np.isfinite(ds) else "",
        "neg_mass": round(neg, 6) if np.isfinite(neg) else "",
        "cover_algo": algo, "source": "local",
    })
    row.update(extra)
    return row


def _failed_row(dataset, graph, corruption, algo, variant, status, **extra):
    """A panel that could NOT be measured. It gets a ROW -- never a silent skip -- so mds_plots draws a
    labelled empty panel and the reader can see the algorithm was tried and did not land."""
    row = {k: "" for k in SWEEP_FIELDS}
    row.update({"dataset": dataset, "graph": graph, "corruption": corruption, "algo": algo,
                "variant": variant, "status": status, "cover_algo": algo, "source": "local"})
    row.update(extra)
    return row


def _iter_saved(graph, idx, covers_root):
    """One representative saved cover per algorithm (first sorted file; randomized algos at one seed)."""
    cdir = _covers_dir(covers_root, graph) if covers_root else None
    if not cdir:
        return
    seen = set()
    for cf in sorted(glob.glob(os.path.join(cdir, "*.txt"))):
        algo = os.path.basename(cf)[:-4].split("__")[0]
        if algo in seen:
            continue
        seen.add(algo)
        yield algo, load_cover(cf, idx), os.path.basename(cf)


# ----------------------------------------------------------------------------
# Real graphs: score every SAVED campaign cover
# ----------------------------------------------------------------------------
def run_real_sweep(graph, covers_root, store=None):
    """Every saved cover of a real graph -> disparity per algorithm, scored against the external truth."""
    print(f"[sweep:real] {graph}", flush=True)
    nodes, idx, edges = load_graph(graph)
    n = len(nodes)
    gt_ix, Dtrue = true_distances(graph, nodes)
    dim = PURE_REAL_DIM[graph]
    Dobs_gt = apsp(edges, n)[np.ix_(gt_ix, gt_ix)]
    core = finite_core(Dobs_gt)
    true_cfg, _ = classical_mds(Dtrue[np.ix_(core, core)], dim)
    tag = f"{graph}::none"
    if store is not None:
        _store_true(store, tag, true_cfg)

    def restrict(D):
        return D[np.ix_(gt_ix, gt_ix)][np.ix_(core, core)]

    H = max(len(_norm_cover(domr_alg(_nx_from_edges(edges, n)))), 1)
    prov = {"n": n, "n_used": int(len(core)), "avg_degree": round(2.0 * len(edges) / max(n, 1), 3),
            "cover_file": ""}
    rows = [_row("pure_real", graph, "none", "observed", "observed", 0, H,
                 *_score(Dobs_gt[np.ix_(core, core)], true_cfg, dim, store, tag, "observed"),
                 **{**prov, "cover_algo": ""})]
    for algo, cover, fname in _iter_saved(graph, idx, covers_root):
        dc, ds, neg = _score(restrict(build_F_distances(edges, cover, n)), true_cfg, dim, store, tag, algo)
        rows.append(_row("pure_real", graph, "none", algo, variant_of(algo), len(cover), H, dc, ds, neg,
                         **{**prov, "cover_file": fname}))
    return rows


# ----------------------------------------------------------------------------
# RGG controls: recompute every algorithm's cover through the CAMPAIGN suite
# ----------------------------------------------------------------------------
def run_rgg_sweep(name, n, deg, direction, frac, mag, seed, store=None):
    """Recompute each algorithm's cover on a broken RGG -> disparity per algorithm + the aligned point cloud.

    NO try/except around the plumbing: run_isolated already contains every algorithm-side failure (it forks and
    returns status=timeout/oom/killed/error:...), so anything that raises HERE is a wiring bug in this file and
    must be loud. That is the whole lesson of the swallowed TypeError."""
    print(f"[sweep:rgg] {name}", flush=True)
    seed_all(seed)
    radius = float(np.sqrt(deg / (np.pi * max(n - 1, 1))))
    T = random_geometric_metric_graph(n, mode="radius", radius=radius)
    T = nx.convert_node_labels_to_integers(T.subgraph(max(nx.connected_components(T), key=len)).copy())
    C, _ = break_metric_graph(T, frac_q=frac, direction=direction, magnitude=mag)
    C = nx.convert_node_labels_to_integers(C.subgraph(max(nx.connected_components(C), key=len)).copy())
    nc = C.number_of_nodes()
    pos = np.array([C.nodes[i]["pos"] for i in range(nc)], dtype=float)
    edges = [(u, v, C[u][v]["weight"]) for u, v in C.edges()]
    Dobs = apsp(edges, nc)
    core = finite_core(Dobs)
    true_cfg = pos[core]
    tag = f"{name}::{direction}"
    if store is not None:
        _store_true(store, tag, true_cfg)

    def restrict(D):
        return D[np.ix_(core, core)]

    H = max(len(_norm_cover(domr_alg(C))), 1)
    prov = {"n": n, "n_used": int(len(core)), "avg_degree": deg, "frac": frac, "magnitude": mag,
            "seed": seed, "radius": round(radius, 6), "cover_file": "(recomputed)",
            "oracle": "naive", "lp_seed": SUITE_SEED}
    rows = [_row("rgg", name, direction, "observed", "observed", 0, H,
                 *_score(restrict(Dobs), true_cfg, 2, store, tag, "observed"),
                 **{**prov, "cover_algo": "", "cover_file": "", "oracle": "", "lp_seed": ""})]

    for algo, variant in SWEEP_ALGOS:
        fn, suite_variant, vkey = _SUITE[algo]
        assert suite_variant == variant, f"{algo}: suite says {suite_variant}, we say {variant}"
        out = run_isolated(fn, C, VERIFY[vkey], _cap_for(algo, nc))
        status = out.get("status", "error:no-status")
        wall = round(out.get("wall") or 0.0, 2)
        conv = out.get("converged")                    # None = the algorithm does not report convergence
        if status != "ok":
            print(f"    [{algo}] status={status} ({wall}s) -- panel marked non-converged", flush=True)
            rows.append(_failed_row("rgg", name, direction, algo, variant, status, **{**prov, "wall": wall}))
            continue
        if conv is False:
            # exact_metric_repair_ilp_separation returns a possibly-INVALID cover (not None) when it stops at
            # max rounds (metric_repair.py:1120); l1_separation does the same. A cover that is not certified is
            # not a result -- refuse it here rather than embed it and report a disparity for a broken repair.
            print(f"    [{algo}] converged=False ({wall}s) -- refusing an uncertified cover", flush=True)
            rows.append(_failed_row("rgg", name, direction, algo, variant, "not_converged",
                                    **{**prov, "wall": wall, "converged": 0, "valid": out.get("valid")}))
            continue
        cover = _norm_cover(out["cover"])
        dc, ds, neg = _score(restrict(build_F_distances(edges, cover, nc)), true_cfg, 2, store, tag, algo)
        rows.append(_row("rgg", name, direction, algo, variant, len(cover), H, dc, ds, neg,
                         **{**prov, "wall": wall, "valid": out.get("valid"),
                            "converged": "" if conv is None else int(conv),
                            "rounding": _ROUNDING_OF.get(algo, ""),
                            "best_of_k": _BESTOFK_OF.get(algo, "")}))
    return rows


# ----------------------------------------------------------------------------
# Planted break on a REAL metric base (dimacs road net): the controlled experiment, off the RGG
# ----------------------------------------------------------------------------
def _planted_base(base):
    """The uncorrupted base, LOADED END TO END THROUGH load_graph, with its ground truth welded to the nodes.

    TWO TRAPS live here, and both produce a plausible-but-meaningless disparity rather than a crash.

    Trap 1 -- RELABELING. run_rgg_sweep may call convert_node_labels_to_integers twice; that is safe for an
    RGG only because the true layout rides along as the node attribute `pos`. A real base has no `pos`, so we
    weld each node's ground-truth ROW INDEX (and its original label) on as node attributes HERE, before any
    relabeling. Read the truth back off the nodes afterwards and the relabel cannot permute it.

    Trap 2 -- NODE ORDERING. datasets.load_edgelist yields INT labels in CSV insertion order; this module's
    load_graph yields STR labels in LEXICOGRAPHIC order. Mixing them permutes true_cfg against the embedding
    -- a silent 3x inflation of the disparity, which looks exactly like a strong result. So: load_graph END TO
    END, never rgg_harness.generate_rgg / datasets.load_edgelist, and an assertion downstream that the truth
    row order IS the embedding row order.

    Returns (G, Dtrue, nodes, gt_ix, H0) with G on 0..n-1 = load_graph's index space."""
    nodes, idx, edges = load_graph(base)
    n = len(nodes)
    gt_ix, Dtrue = true_distances(base, nodes)
    row_of = {int(g): j for j, g in enumerate(gt_ix)}
    G = _nx_from_edges(edges, n)
    for i in range(n):
        G.nodes[i]["gtrow"] = row_of.get(i, -1)          # -> row of Dtrue; -1 = no ground truth
        G.nodes[i]["olabel"] = str(nodes[i])             # -> the original CSV label
    H0 = _norm_cover(domr_alg(G))
    print(f"    base {base}: n={n} m={len(edges)} |H|={len(H0)} gt={len(gt_ix)}/{n}", flush=True)
    # The planted experiment's premise: EVERY bit of non-metricity in the corrupted graph is ours. A base that
    # is already broken makes the "does repair undo the planted injury" question unanswerable.
    assert not H0, f"{base} is not exactly metric (|H|={len(H0)}) -- it cannot be a planted-break base"
    assert len(gt_ix) == n, f"{base}: only {len(gt_ix)}/{n} nodes carry ground truth"
    return G, Dtrue, nodes, gt_ix, H0


def run_realrec_sweep(name, base, direction, frac, mag, seed, store=None):
    """Plant a break in a real METRIC graph, repair it with every algorithm, and ask whether the repaired
    embedding returns to the real geography. The hybrid the RGG control and the pure-real experiment leave
    out: ground truth we did NOT plant (the road map) and an injury we DID."""
    print(f"[sweep:realrec] {name}  base={base} dir={direction} frac={frac} mag={mag}", flush=True)
    G0, Dtrue, nodes, gt_ix, _ = _planted_base(base)
    dim = PURE_REAL_DIM[base]

    seed_all(seed)
    C, corrupted = break_metric_graph(G0, frac_q=frac, direction=direction, magnitude=mag)
    C = nx.convert_node_labels_to_integers(C.subgraph(max(nx.connected_components(C), key=len)).copy())
    nc = C.number_of_nodes()
    gtrow = np.array([C.nodes[i]["gtrow"] for i in range(nc)], dtype=int)
    olabel = [C.nodes[i]["olabel"] for i in range(nc)]

    # TRAP 1: the truth survived the relabel only because it rode along as a node attribute.
    assert (gtrow >= 0).all(), "a surviving node carries no ground-truth row -- the relabel dropped the truth"
    # TRAP 2: row i of the embedding IS row gtrow[i] of Dtrue. Assert it against the labels, not by faith.
    assert olabel == [str(nodes[int(gt_ix[r])]) for r in gtrow], \
        "truth row order != embedding row order -- true_cfg is permuted against the embedding"

    edges = [(u, v, C[u][v]["weight"]) for u, v in C.edges()]
    Dobs = apsp(edges, nc)
    core = finite_core(Dobs)
    trows = gtrow[core]
    true_cfg, _ = classical_mds(Dtrue[np.ix_(trows, trows)], dim)
    assert true_cfg.shape[0] == len(core), "true configuration and embedding have different row counts"
    tag = f"{name}::{direction}"
    if store is not None:
        _store_true(store, tag, true_cfg)

    def restrict(D):
        return D[np.ix_(core, core)]

    H = max(len(_norm_cover(domr_alg(C))), 1)
    print(f"    broke {len(corrupted)} edge(s) of {G0.number_of_edges()}; |H|={H}; core {len(core)}/{nc}",
          flush=True)
    prov = {"n": G0.number_of_nodes(), "n_used": int(len(core)),
            "avg_degree": round(2.0 * G0.number_of_edges() / max(G0.number_of_nodes(), 1), 3),
            "frac": frac, "magnitude": mag, "seed": seed, "radius": "",
            "cover_file": "(recomputed)", "oracle": "naive", "lp_seed": SUITE_SEED,
            "base": base, "n_corrupted": len(corrupted)}
    rows = [_row("realrec", name, direction, "observed", "observed", 0, H,
                 *_score(restrict(Dobs), true_cfg, dim, store, tag, "observed"),
                 **{**prov, "cover_algo": "", "cover_file": "", "oracle": "", "lp_seed": ""})]

    for algo, variant in SWEEP_ALGOS:
        fn, suite_variant, vkey = _SUITE[algo]
        assert suite_variant == variant, f"{algo}: suite says {suite_variant}, we say {variant}"
        out = run_isolated(fn, C, VERIFY[vkey], _cap_for(algo, nc))
        status = out.get("status", "error:no-status")
        wall = round(out.get("wall") or 0.0, 2)
        conv = out.get("converged")
        if status != "ok":
            print(f"    [{algo}] status={status} ({wall}s) -- panel marked non-converged", flush=True)
            rows.append(_failed_row("realrec", name, direction, algo, variant, status,
                                    **{**prov, "wall": wall}))
            continue
        if conv is False:
            print(f"    [{algo}] converged=False ({wall}s) -- refusing an uncertified cover", flush=True)
            rows.append(_failed_row("realrec", name, direction, algo, variant, "not_converged",
                                    **{**prov, "wall": wall, "converged": 0, "valid": out.get("valid")}))
            continue
        cover = _norm_cover(out["cover"])
        dc, ds, neg = _score(restrict(build_F_distances(edges, cover, nc)), true_cfg, dim, store, tag, algo)
        print(f"    [{algo}] |S|={len(cover)} disp={dc:.4f} ({wall}s)", flush=True)
        rows.append(_row("realrec", name, direction, algo, variant, len(cover), H, dc, ds, neg,
                         **{**prov, "wall": wall, "valid": out.get("valid"),
                            "converged": "" if conv is None else int(conv),
                            "rounding": _ROUNDING_OF.get(algo, ""),
                            "best_of_k": _BESTOFK_OF.get(algo, "")}))
    return rows


# ----------------------------------------------------------------------------
# Figures + report
# ----------------------------------------------------------------------------
def fig_sweep_real(rows, outdir):
    """Appendix scatter: per real graph, SMACOF disparity vs cover size |S|/|H|, one point per
    algorithm, coloured by variant. Lower-left = edits little AND keeps geometry."""
    import matplotlib.pyplot as plt
    graphs = [g for g in REAL_TITLE if any(r["graph"] == g for r in rows)]
    if not graphs:
        print("    skip fig_sweep_real (no real rows)"); return
    fig, axes = plt.subplots(1, len(graphs), figsize=(4.3 * len(graphs), 3.8), squeeze=False)
    for ax, g in zip(axes[0], graphs):
        sub = [r for r in rows if r["graph"] == g and str(r.get("disp_smacof", "")) != ""]
        obs = next((float(r["disp_smacof"]) for r in sub if r["variant"] == "observed"), None)
        if obs is not None:
            ax.axhline(obs, color="black", lw=0.8, ls=":", label="observed / DOMR")
        for r in sub:
            if r["variant"] == "observed":
                continue
            x, y = float(r["ratio_domr"]), float(r["disp_smacof"])
            ax.scatter(x, y, s=42, color=VAR_COLOR.get(r["variant"], "#555"),
                       edgecolor="white", linewidth=0.5, zorder=3)
            ax.annotate(r["algo"], (x, y), fontsize=6, xytext=(3, 2),
                        textcoords="offset points", alpha=0.8)
        ax.set_title(REAL_TITLE.get(g, g), fontsize=9)
        ax.set_xlabel("cover size  $|S|/|H|$"); ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("Procrustes disparity (SMACOF)\n($\\downarrow$ closer to truth)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=VAR_COLOR[v], label=v) for v in ("GMR", "IOMR")]
    handles.append(plt.Line2D([], [], color="black", ls=":", label="observed / DOMR"))
    axes[0][-1].legend(handles=handles, fontsize=7, frameon=False, loc="best")
    fig.suptitle("Does editing less preserve geometry better?  Disparity vs cover size, per algorithm.",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig_mds_sweep_real.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    wrote fig_mds_sweep_real.pdf / .png")


def _report(rows):
    """Per-graph algorithm ranking by SMACOF disparity (best first), to eyeball the sweep."""
    print("\n=== per-graph disparity ranking (SMACOF; observed baseline first) ===")
    for g in dict.fromkeys(r["graph"] for r in rows):
        sub = [r for r in rows if r["graph"] == g]
        scored = [r for r in sub if str(r.get("disp_smacof", "")) != ""]
        obs = next((r for r in scored if r["variant"] == "observed"), None)
        ranked = sorted((r for r in scored if r["variant"] != "observed"),
                        key=lambda r: float(r["disp_smacof"]))
        base = f"obs={float(obs['disp_smacof']):.4f}" if obs else "obs=?"
        print(f"  {g:18s} {base}")
        for r in ranked:
            tag = "  <-- beats observed" if obs and float(r["disp_smacof"]) < float(obs["disp_smacof"]) else ""
            src = "" if r.get("source", "local") == "local" else f"  [{r['source']}]"
            print(f"      {r['algo']:16s} {r['variant']:5s} |S|/|H|={r['ratio_domr']:>7} "
                  f"disp={float(r['disp_smacof']):.4f}{tag}{src}")
        for r in sub:                                   # the panels that did not land are part of the result
            if str(r.get("disp_smacof", "")) == "":
                print(f"      {r['algo']:16s} {r['variant']:5s} NOT MEASURED  status={r.get('status')}")


# ----------------------------------------------------------------------------
# CSV merge -- NEVER drop a row we cannot reproduce
# ----------------------------------------------------------------------------
def _merge_with_existing(rows, path):
    """main() writes with open(path,"w") -- a TRUNCATING overwrite -- and _iter_saved only yields covers that
    exist ON THIS MACHINE. analysis/summary_mds_sweep.csv is gitignored, so a naive local run permanently
    destroys every row whose cover lives only on the cluster: ripe's `pivot` and `iomr_rand`, which no local
    run can rebuild (ripe's covering LP is a cluster job).

    So: MERGE. Any pre-existing (graph, algo) the local run did not produce is carried forward verbatim and
    stamped source=cluster; everything the local run did produce is source=local and wins."""
    if not os.path.exists(path):
        return rows
    shutil.copyfile(path, path + ".bak")
    with open(path, newline="") as f:
        old = list(csv.DictReader(f))
    fresh = {tuple(r[k] for k in KEY) for r in rows}
    kept = []
    for r in old:
        if tuple(r.get(k, "") for k in KEY) in fresh:
            continue
        row = {k: "" for k in SWEEP_FIELDS}
        row.update({k: v for k, v in r.items() if k in SWEEP_FIELDS})
        # Carry the RECORDED provenance forward. This used to hard-stamp source=cluster, which is only right
        # when the run that wrote the row was a cluster run: with `--only` / `--graphs` staging, a row an
        # EARLIER LOCAL run produced would be relabelled a cluster row -- a provenance lie in the CSV itself.
        if not row["source"]:
            row["source"] = "cluster"
        if not row["status"]:
            row["status"] = "ok"
        kept.append(row)
    if kept:
        print(f"\nmerged {len(kept)} pre-existing row(s) this run cannot reproduce (source=cluster):")
        for r in kept:
            print(f"    {r['graph']:18s} {r['algo']:16s} |S|={r['cover_size']:>8} disp={r['disp_smacof']}")
    print(f"backed up the previous CSV -> {path}.bak")
    return rows + kept


def _emb_tag(key):
    """The (graph::corruption) tag a store key belongs to: emb::TAG::algo::mds, true::TAG, color::TAG."""
    parts = key.split("::")
    return parts[1] if len(parts) > 1 else None


def _merge_npz(store, path):
    """The npz twin of _merge_with_existing, and just as load-bearing. np.savez_compressed TRUNCATES, so a
    staged run (`--only realrec`) would otherwise DELETE every array the previous stage wrote -- the RGG and
    ripe/nmr point clouds mds_plots draws from -- and the figures would come back empty, not wrong. Empty is
    a failure, not a pass.

    Rule: any key whose TAG this run did not touch is carried forward verbatim; every key of a tag this run
    DID touch is replaced wholesale, so a stale embedding of an algorithm that has since stopped converging
    cannot survive as a ghost panel."""
    if not os.path.exists(path):
        return store
    with np.load(path, allow_pickle=False) as z:
        old = {k: z[k] for k in z.files}
    touched = {_emb_tag(k) for k in store}
    carried = {k: v for k, v in old.items() if _emb_tag(k) not in touched}
    dropped = [k for k in old if _emb_tag(k) in touched and k not in store]
    shutil.copyfile(path, path + ".bak")
    print(f"\nnpz: {len(store)} fresh array(s); carried {len(carried)} from {len(old)} existing "
          f"(tags untouched by this run); dropped {len(dropped)} stale (tag regenerated)")
    print(f"backed up the previous npz -> {path}.bak")
    merged = dict(carried)
    merged.update(store)
    return merged


def _domr_selfcheck(rows):
    """DOMR's cover is exactly the heavy set, and reweighting it to the detour leaves every shortest path
    unchanged (Lemma 6.1): D_F == D_G, so DOMR's disparity MUST equal the observed disparity, per graph, per
    MDS algorithm. A gap is a pipeline bug -- most likely a permuted true configuration. This is the same
    invariant mds_recovery.main() checks; it is worth the four lines to check it here too."""
    print("\nDOMR self-check (disparity must equal observed; D_F == D_G by Lemma 6.1):")
    by = {}
    for r in rows:
        if r["algo"] in ("observed", "domr"):
            for col in ("disp_classical", "disp_smacof"):
                if str(r.get(col, "")) != "":
                    by.setdefault((r["graph"], col), {})[r["algo"]] = float(r[col])
    worst, checked = 0.0, 0
    for (g, col), d in sorted(by.items()):
        if "observed" in d and "domr" in d:
            gap = abs(d["observed"] - d["domr"])
            worst = max(worst, gap)
            checked += 1
            if gap >= 1e-6:
                print(f"    {g:22s} {col:15s} obs={d['observed']:.4f} domr={d['domr']:.4f} "
                      f"|gap|={gap:.2e}  <-- MISMATCH")
    print(f"  {checked} pair(s) checked; max |gap| = {worst:.2e}  "
          f"({'OK' if worst < 1e-6 else 'CHECK PIPELINE'})")
    return worst


def _default_covers_root():
    for cand in ("results_real_covers", os.path.join("results_real", "results_real_covers")):
        if os.path.isdir(cand):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="analysis")
    ap.add_argument("--covers-root", default=None)
    ap.add_argument("--only", choices=["real", "rgg", "realrec", "all"], default="all")
    ap.add_argument("--graphs", default=None,
                    help="comma-separated graph names to restrict this run to. Rows and embeddings for every "
                         "OTHER graph are carried forward from the existing CSV/npz, so the run is a stage, "
                         "not a truncation.")
    ap.add_argument("--plot", action="store_true", help="also write fig_mds_sweep_real (needs matplotlib)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    covers_root = a.covers_root or _default_covers_root()
    print(f"covers_root = {covers_root or '(none)'}")
    want = set(a.graphs.split(",")) if a.graphs else None

    rows, store = [], {}
    if a.only in ("real", "all"):
        for g in sorted(DOWNSTREAM_GRAPHS, key=lambda x: 0 if x.startswith("nmr") else 1):
            if want and g not in want:
                continue
            try:
                rows += run_real_sweep(g, covers_root, store)
            except FileNotFoundError as e:
                print(f"  [skip {g}] {e}")
    if a.only in ("rgg", "all"):
        for spec in RGG_SPECS:
            if want and spec[0] not in want:
                continue
            rows += run_rgg_sweep(*spec, store=store)
    if a.only in ("realrec", "all"):
        for spec in REALREC_SPECS:
            if want and spec[0] not in want:
                continue
            rows += run_realrec_sweep(*spec, store=store)

    _domr_selfcheck(rows)
    path = os.path.join(a.outdir, "summary_mds_sweep.csv")
    rows = _merge_with_existing(rows, path)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SWEEP_FIELDS)
        w.writeheader()
        w.writerows(rows)
    npz_path = os.path.join(a.outdir, "mds_sweep_embeddings.npz")
    np.savez_compressed(npz_path, **_merge_npz(store, npz_path))
    _report(rows)
    if a.plot:
        fig_sweep_real(rows, a.outdir)
    nfail = sum(1 for r in rows if r.get("status") not in ("ok", "", None))
    print(f"\nwrote {path} ({len(rows)} rows; {nfail} not measured)")
    print(f"wrote {npz_path} ({len(store)} arrays)")


if __name__ == "__main__":
    main()
