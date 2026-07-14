"""The COUPLING A/B -- the one experiment that can attribute the flip to the weight model, and nothing else.

WHAT IT ASKS. Appendix A reports that \\code{spc_gmr} and \\code{pivot} SWAP between the dense grid's two
sweeps: spc rewrites 0.225 of exp1 and 0.418 of exp2b; pivot rewrites 0.457 and 0.278. That is a real
observation and a useless one, because exp1 and exp2b differ in THREE things at once -- n (1000-1500 vs
2000), p (0.3/0.5 vs a sweep), and the weight model (coupled vs decoupled). The flip cannot be attributed to
any of them.

This array fixes exactly that. n is FIXED at 300. p is FIXED, point for point. The topology distribution is
identical. THE ONLY THING THAT CHANGES IS THE WEIGHT MODEL:

    coupled     random_geometric_weighted_graph(n, p)              weights Geom(1 - p)   -- spread tracks p
    decoupled   random_decoupled_geometric_weighted_graph(n, p)    weights Geom(0.5)     -- spread is FIXED

If the ranking still flips, the coupling alone did it. Nothing in the published campaign runs this comparison.

THE GRID, AND WHY THIS ALPHA RANGE. p = 2 n^-alpha, alpha from 0.2 to 0.7 (11 points), n = 300. Under the
coupled model the mean edge weight is 1/(1 - p): DENSITY IS THE WEIGHT SPREAD. Sweep p downward and the
weights collapse onto 1, and a graph whose edges all weigh 1 is metric by construction. Measured, before this
file was written:

    alpha    p       m        mean w    COUPLED |H|/m    DECOUPLED |H|/m
    0.2    0.639   28,628      2.77        0.4053            0.2465
    0.3    0.361   16,130      1.57        0.1276            0.2434
    0.4    0.204    9,087      1.26        0.0419            0.2399
    0.5    0.115    5,139      1.13        0.0121            0.1990
    0.6    0.065    2,904      1.07        0.0030            0.1542   <-- METRIC (the 3/5 onset)
    0.7    0.037    1,649      1.04        0.0006            0.1113   <-- METRIC

The sweep therefore spans the ONSET -- a 675x collapse in non-metricity -- while the decoupled control stays
flat. The two curves even CROSS near alpha = 0.25: coupling makes the graph MORE broken than the fixed model
at high density (its spread is wider there) and drives it METRIC at low density. That scissors IS the effect.

WHAT THIS SWEEP CANNOT GIVE, AND WE SAY SO UP FRONT. The exact optimum dies exactly where the coupling bites.
gmr_ilp converges on ~99% of instances when |H| < 200 and on 0% when |H| > 3000 -- and the coupled model
carries |H| ~ 11,600 at alpha = 0.2. So |S|/OPT is available only at the metric end. That is not a defect of
the design; it is the dense family's own pathology, the same one Appendix A reports for IOMR ("the optimum
exists only where there is nothing to repair"), and this array MEASURES it rather than inferring it. Below the
ILP's reach we report |S|/m, as Section 5 does.

THE BUDGET IS RAISED, AND IT IS NOT A TUNING KNOB. harness's small-grid TASK_BUDGET is 2 h. The alpha = 0.2
point costs ~2.25 core-h, so the budget WOULD fire -- and `run_one_task` then marks the REMAINING algorithms
`skipped_time`, walking build_suite in order, which ends:

    ... spc_gmr, spc_iomr, PIVOT, LEFT_EDGE

spc_gmr and pivot are the two methods whose flip this array exists to attribute. Under the 2 h budget they
would come back `skipped_time` at exactly the broken end where the coupling bites, and the experiment would
have destroyed its own headline. The suite's ceiling is 9.5 h (the sum of its per-algorithm caps), so BUDGET
is set to 10 h: it never binds, the per-algorithm cap is the only thing that can stop an algorithm, and
collect_coupling.py GATES on skipped_time == 0.

NOTHING IN harness.py IS TOUCHED. It is imported read-only; this file carries its own grid and its own runner,
reusing the harness's generator, suite, isolation, caps and CSV schema verbatim, so its rows concatenate with
the published small grid.

  one task = one graph = the whole suite -> results_coupling/task_NNNNNN.csv
  the grid = 11 alpha x 2 models x 30 seeds = 660 tasks

  count      sage -python experiments/coupling_harness.py --count
  preflight  sage -python experiments/coupling_harness.py --preflight
  one task   sage -python experiments/coupling_harness.py --task-index K --outdir results_coupling
"""
import argparse
import csv
import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness                                                          # noqa: E402
from harness import (                                                   # noqa: E402
    generate, build_suite, run_isolated, task_seed, VERIFY, RUN_FIELDS, CSV_FIELDS,
    REGION_H_MAX, TIMEOUT_S, ALGO_TIMEOUT, _norm, _aggregate,
)
from metric_repair import domr_alg                                      # noqa: E402

EXP = "coupling"              # a NEW label: collides with no collector, no published sweep
N = 300                       # matches the published small grid's exp2a/exp2b, so rows are comparable
ALPHA_LO, ALPHA_HI, N_ALPHA = 0.2, 0.7, 11        # p = 2 n^-alpha  =>  p = 0.639 .. 0.037
SEEDS = 30
MODELS = ["geometric", "decoupled_geometric"]     # coupled, then the FIXED-weight control

# The suite's ceiling is the sum of its per-algorithm caps. Above that the budget cannot bind, so the cap is
# the only thing that can stop an algorithm -- and a timeout means what it says. See the module docstring:
# under the 2 h default this array would have skipped spc_gmr and pivot, the two methods it exists to compare.
BUDGET_S = 10 * 3600


def points():
    """Every (alpha, model) pair. p depends ONLY on alpha, so the two models see the SAME p at every rung --
    that identity is the whole experiment, and preflight checks it."""
    pts = []
    for a in np.linspace(ALPHA_LO, ALPHA_HI, N_ALPHA):
        p = float(2.0 * N ** (-a))
        for mo in MODELS:
            pts.append(dict(exp=EXP, model=mo, n=N, p=p, alpha=float(a)))
    return pts


POINTS = points()


def all_tasks():
    return [(pt, s) for pt in POINTS for s in range(SEEDS)]


def run_one_task(task_index, outdir):
    """A faithful re-implementation of harness.run_one_task against THIS grid, with the budget raised.

    Everything that matters is the harness's: the same generator, the same DOMR-per-component non-metricity
    test, the same isolation, the same per-algorithm cap, the same aggregation, the same CSV_FIELDS. What
    differs is the grid and the budget, and nothing else -- a row that is not schema-identical to the
    campaign's cannot be read beside it."""
    pt, s = all_tasks()[task_index]
    seed = task_seed(pt, s)
    G = generate(pt, seed)
    comps = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    ws = [d["weight"] for _, _, d in G.edges(data=True)]
    giant = max((c.number_of_nodes() for c in comps), default=0)

    nonmetric, heavy_union = [], set()
    for CC in comps:
        hs = {_norm(u, v) for (u, v) in domr_alg(CC)}
        if hs:
            nonmetric.append(CC)
            heavy_union |= hs
    total_H = len(heavy_union)

    meta = dict(task=task_index, exp=pt["exp"], model=pt["model"], n=pt["n"],
                p=pt.get("p"), alpha=pt.get("alpha"), sample=s, seed=seed,
                V=G.number_of_nodes(), E=G.number_of_edges(),
                w_min=(min(ws) if ws else 0), w_max=(max(ws) if ws else 0),
                n_components=len(comps), giant=giant, H=total_H)

    rows, elapsed = [], 0.0
    for (name, variant, vkey, n_max, region_gated, fn) in build_suite(seed):
        row = {**meta, "algo": name, "variant": variant, **{k: None for k in RUN_FIELDS}}
        if elapsed >= BUDGET_S:
            row["status"] = "skipped_time"
        elif n_max is not None and giant > n_max:
            row["status"] = "skipped_n"
        elif region_gated and total_H > REGION_H_MAX:
            row["status"] = "skipped_H"
        elif not nonmetric:
            # THE METRIC END IS A REAL STATE, NOT AN ERROR. At alpha = 0.7 the coupled graph IS metric, the
            # empty cover is correct, and it costs nothing. That is the finding this array exists to show.
            row.update(status="ok", size=0, valid=1, cpu=0.0, wall=0.0, peak_mb=0.0)
        else:
            to = ALGO_TIMEOUT.get(name, TIMEOUT_S)
            results = [run_isolated(fn, CC, VERIFY[vkey], to) for CC in nonmetric]
            row.update(_aggregate(results))
            elapsed += row.get("wall") or 0.0
            covers = [set(r["cover"]) for r in results if r.get("cover") is not None]
            if covers:
                cu = set().union(*covers)
                row["light_frac"] = round(len(cu - heavy_union) / len(cu), 4) if cu else None
        rows.append(row)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"task_{task_index:06d}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_FIELDS})
    return path


def preflight(seeds=2):
    """Build every rung of the grid, for both models, and check the experiment is the experiment.

    Three things must hold or the array is not worth submitting:
      1. THE MODELS ARE MATCHED. Same n, same p at every rung. If they are not, the comparison confounds the
         weight model with the topology and the array answers nothing.
      2. THE COUPLED SWEEP CROSSES THE ONSET. It must be genuinely broken at one end and genuinely metric at
         the other, or there is no coupling effect to see.
      3. THE CONTROL IS FLAT. The decoupled model must NOT collapse. If it did, the gap between the two would
         not be the coupling -- it would be something else, and we would not know what."""
    print(f"PREFLIGHT -- {len(POINTS)} points x {SEEDS} seeds = {len(all_tasks())} tasks, n = {N}\n")
    print(f"  {'alpha':>6}{'p':>8}{'mean w':>8}   {'m (coupled)':>12}{'|H|/m':>9}   "
          f"{'m (decoup)':>12}{'|H|/m':>9}")
    print("  " + "-" * 72)
    rows = []
    for a in np.linspace(ALPHA_LO, ALPHA_HI, N_ALPHA):
        p = float(2.0 * N ** (-a))
        rec = {"alpha": a, "p": p}
        for mo in MODELS:
            pt = dict(exp=EXP, model=mo, n=N, p=p, alpha=a)
            hs, ms = [], []
            for s in range(seeds):
                G = generate(pt, task_seed(pt, s))
                m = G.number_of_edges()
                hs.append(len(domr_alg(G)) / m if m else 0.0)
                ms.append(m)
            rec[mo] = (float(np.mean(hs)), int(np.mean(ms)))
        rows.append(rec)
        c, d = rec["geometric"], rec["decoupled_geometric"]
        print(f"  {a:>6.2f}{p:>8.3f}{1/(1-p):>8.2f}   {c[1]:>12,}{c[0]:>9.4f}   {d[1]:>12,}{d[0]:>9.4f}")

    ok = True
    # 1 -- matched
    bad = [r for r in rows if abs(r["geometric"][1] - r["decoupled_geometric"][1]) / r["geometric"][1] > 0.10]
    print(f"\n  [{'PASS' if not bad else 'FAIL'}] the two models are MATCHED (same n, same p, same m to 10%)"
          f"   {len(rows) - len(bad)}/{len(rows)} rungs")
    ok = ok and not bad
    # 2 -- the coupled sweep crosses the onset
    ch = [r["geometric"][0] for r in rows]
    print(f"  [{'PASS' if (max(ch) > 0.20 and min(ch) < 0.01) else 'FAIL'}] the COUPLED sweep crosses the "
          f"onset   |H|/m {max(ch):.4f} -> {min(ch):.4f}  ({max(ch)/max(min(ch), 1e-9):.0f}x collapse)")
    ok = ok and max(ch) > 0.20 and min(ch) < 0.01
    # 3 -- the control is flat
    dh = [r["decoupled_geometric"][0] for r in rows]
    spread = max(dh) / max(min(dh), 1e-9)
    print(f"  [{'PASS' if spread < 5 else 'FAIL'}] the DECOUPLED control stays broken (does NOT collapse)"
          f"   |H|/m {max(dh):.4f} -> {min(dh):.4f}  ({spread:.1f}x)")
    ok = ok and spread < 5
    # 4 -- the budget cannot bind
    ceiling = sum(ALGO_TIMEOUT.get(e[0], TIMEOUT_S) for e in build_suite(0))
    print(f"  [{'PASS' if BUDGET_S > ceiling else 'FAIL'}] budget {BUDGET_S/3600:.0f}h > the "
          f"{ceiling/3600:.1f}h suite ceiling")
    print(f"         -> the per-algorithm cap is the ONLY thing that can stop an algorithm. Under the")
    print(f"            harness's 2h default the budget would have skipped spc_gmr and pivot -- the two")
    print(f"            methods this array exists to compare.")
    ok = ok and BUDGET_S > ceiling
    print(f"\n  NOTE: the exact optimum dies where the coupling bites (|H| ~ 11,600 at alpha=0.2, and")
    print(f"        gmr_ilp converges on 0% above |H| = 3,000). That is the dense family's own pathology,")
    print(f"        and MEASURING it is part of the point. Below the ILP's reach we report |S|/m.")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task-index", type=int, default=None)
    ap.add_argument("--outdir", default="results_coupling")
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    a = ap.parse_args()

    if a.count:
        print(len(all_tasks()))
        return
    if a.preflight:
        sys.exit(0 if preflight() else 1)
    if a.task_index is None:
        ap.error("--task-index is required (or --count / --preflight)")
    path = run_one_task(a.task_index, a.outdir)
    print(f"task {a.task_index} ({EXP}) -> {path}")


if __name__ == "__main__":
    main()
