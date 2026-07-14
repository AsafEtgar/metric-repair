"""The COUPLED density sweep at scale -- the arm the campaign never ran.

WHY IT EXISTS. The large dense grid has two sweeps and neither of them varies density under COUPLED weights:

    exp1   coupled   (Geom(1-p))    n = 1000-1500,  p pinned at {0.3, 0.5}     -- density does not move
    exp2b  DEcoupled (Geom(0.5))    n = 2000,       p = 2n^-alpha, 16 points   -- density moves, weights do not

So every statement the paper makes about density on the large grid is a statement about the DECOUPLED model.
This array supplies the missing cell: density moving, weights coupled.

THE THING THAT DECIDES THE GRID, AND IT IS NOT A DETAIL. Under the coupled model the edge weights are
Geometric(1 - p), so their mean is 1/(1 - p). Density IS the weight spread. At small p the weights collapse
onto 1, and a graph whose edges all weigh 1 is metric BY CONSTRUCTION -- there is nothing to repair. Measured
at n = 2000, before this array was written:

    alpha   p = 2n^-alpha      m        mean w     |H|/m
    0.667      0.0126        25,300      1.013     0.0000      <-- metric
    0.500      0.0447        89,593      1.047     0.0020      <-- metric
    0.370      0.120        239,153      1.136     0.0144
    0.274      0.250        499,466      1.333     0.0628
    0.229      0.350        699,741      1.538     0.1227      <-- the cap
    0.182      0.500        999,028      2.000     0.2503

The obvious mirror of exp2a (alpha in [1/2, 2/3]) would therefore have benchmarked repair algorithms on
graphs with 0-0.3% heavy edges: 320 tasks, ~800 core-hours, and nothing to repair in any of them. That is the
same mistake as the P2size jitter sweep the inversion table now drops (|H|/m = 0.004). The grid below runs
where the coupling actually bites.

    alpha 0.500 -> 0.229 (16 points), n = 2000, p = 2 n^-alpha  =>  p = 0.045 -> 0.35,  m = 90k -> 700k
    |H|/m sweeps 0.002 -> 0.123: the array SPANS THE ONSET, which is the point. It shows density and
    non-metricity moving together, because under coupling they are one knob -- and that entanglement is
    precisely what the decoupled model was invented to break.

The cap at p = 0.35 is deliberate. p = 0.5 gives ~1M edges, roughly twice the heaviest instance the campaign
has ever run (exp1 tops out at 563k), and harness.py:122 already flags it. The dense end would have bought a
column of timeouts.

WHAT IS DROPPED, AND WHY IT IS SAFE. gmr_bestofk and iomr_bestofk time out on 100% of exp2b's tasks -- 1800 s
each, every task, returning nothing, for 38% of the entire compute budget. They are dropped here. This is
NOT a free lunch and the collector must say so: a return rate is only comparable across sweeps if the
timeout cap is the same, and by not running them we forfeit their return-rate cells rather than measuring
them at 0%. Everything else in the large suite runs, at the same 1800 s cap, so every other cell IS
comparable with exp1 and exp2b.

NOTHING IN harness.py IS TOUCHED. It is imported read-only. Changing a task-import path mid-campaign would
invalidate 11,965 delivered tasks, so this file carries its own grid and its own runner, reusing the
harness's generator, suite, isolation and CSV schema verbatim -- the rows it writes are schema-identical to
the campaign's, which is what lets the collector concatenate them.

  one task   = one graph = the whole suite  -> results_coupled/task_NNNNNN.csv
  the grid   = 16 alpha x 20 seeds = 320 tasks

  count      sage -python experiments/coupled_harness.py --count
  one task   sage -python experiments/coupled_harness.py --task-index K --outdir results_coupled
  preflight  sage -python experiments/coupled_harness.py --preflight     (checks the grid IS non-metric)
  cluster    see --dsq, which prints the joblist
"""
import argparse
import csv
import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Read-only imports. harness.py is FROZEN: the published campaign's tasks import through it.
import harness                                                          # noqa: E402
from harness import (                                                   # noqa: E402
    seed_all, generate, build_suite, run_isolated, task_seed,
    DROP_LARGE, VERIFY, RUN_FIELDS, CSV_FIELDS, REGION_H_MAX,
    TIMEOUT_S, ALGO_TIMEOUT, _norm, _aggregate, task_budget,
)
from metric_repair import domr_alg                                      # noqa: E402

EXP = "exp2c"                 # a NEW label: never collides with exp1 / exp2a / exp2b in any collector
MODEL = "geometric"           # the COUPLED model -- weights Geometric(1 - p)
N = 2000
ALPHA_HI, ALPHA_LO, N_ALPHA = 0.5, 0.229, 16      # p = 2 n^-alpha  =>  p = 0.045 .. 0.35
SEEDS = 20

# Timing out on 100% of exp2b's tasks for 38% of its budget, and returning nothing. See the module docstring:
# this forfeits their return-rate cells rather than measuring them.
DROP_ALGOS = {"gmr_bestofk", "iomr_bestofk"}

# The grid is only worth running where the coupled model is actually broken. Preflight enforces it.
MIN_HM = 0.05                 # at least one grid point must carry >= 5% heavy edges
MAX_M = 750_000               # and none may exceed this (the cap that keeps us off the 1M-edge cliff)


def points():
    """16 alpha points, high to low: p = 2 n^-alpha, so alpha DOWN means p (and m, and |H|) UP."""
    return [dict(exp=EXP, model=MODEL, n=N, p=float(2.0 * N ** (-a)), alpha=float(a))
            for a in np.linspace(ALPHA_HI, ALPHA_LO, N_ALPHA)]


POINTS = points()


def all_tasks():
    return [(pt, s) for pt in POINTS for s in range(SEEDS)]


def suite_for(seed):
    """The large-grid suite, minus the two methods that only ever buy timeouts."""
    return [e for e in build_suite(seed) if e[0] not in DROP_LARGE and e[0] not in DROP_ALGOS]


def run_one_task(task_index, outdir):
    """A faithful re-implementation of harness.run_one_task against THIS grid.

    Every line that matters is the harness's: same generator, same DOMR-per-component non-metricity test,
    same isolation, same per-algorithm cap, same aggregation, same CSV_FIELDS. What differs is the grid and
    the two dropped algorithms -- and nothing else, because a row that is not schema-identical to the
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
    budget = task_budget("large")
    for (name, variant, vkey, n_max, region_gated, fn) in suite_for(seed):
        row = {**meta, "algo": name, "variant": variant, **{k: None for k in RUN_FIELDS}}
        if elapsed >= budget:
            row["status"] = "skipped_time"
        elif n_max is not None and giant > n_max:
            row["status"] = "skipped_n"
        elif region_gated and total_H > REGION_H_MAX:
            row["status"] = "skipped_H"
        elif not nonmetric:
            # The whole graph is already metric -> the empty cover is correct and costs nothing. This is a
            # REAL state on this grid, not an error: the sparse end of a coupled sweep IS metric, and that
            # is the finding the array exists to show.
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
    """REFUSE TO SUBMIT A SWEEP WITH NOTHING TO REPAIR.

    This is the gate that would have caught the original alpha in [1/2, 2/3] grid before it burned 800
    core-hours on metric graphs. It builds each grid point for real and measures |H|/m, and it fails if the
    sweep never gets meaningfully broken, or if it strays past the edge-count cap. It cannot pass vacuously:
    the numbers come from actually generating the graphs and running DOMR on them."""
    print(f"PREFLIGHT -- {len(POINTS)} grid points x {seeds} probe seeds, n = {N}, COUPLED weights\n")
    print(f"  {'alpha':>7}{'p':>9}{'m':>10}{'mean w':>9}{'|H|/m':>9}")
    print("  " + "-" * 46)
    hm_max, m_max, hm_lo = 0.0, 0, None
    for pt in POINTS:
        hs, ms = [], []
        for s in range(seeds):
            G = generate(pt, task_seed(pt, s))          # generate() seeds from the task seed itself
            m = G.number_of_edges()
            H = {_norm(u, v) for (u, v) in domr_alg(G)}
            hs.append(len(H) / m if m else 0.0)
            ms.append(m)
        hm, m = float(np.mean(hs)), int(np.mean(ms))
        if hm_lo is None:
            hm_lo = hm                                   # the sparse end, where the coupling collapses
        hm_max = max(hm_max, hm)
        m_max = max(m_max, m)
        print(f"  {pt['alpha']:>7.3f}{pt['p']:>9.4f}{m:>10,}{1 / (1 - pt['p']):>9.3f}{hm:>9.4f}")
    print()
    ok = True
    if hm_max < MIN_HM:
        print(f"  [FAIL] the sweep never exceeds |H|/m = {hm_max:.4f} (need >= {MIN_HM}). These graphs are "
              f"metric; there is nothing for a repair algorithm to do. REFUSING to submit.")
        ok = False
    else:
        print(f"  [PASS] the sweep reaches |H|/m = {hm_max:.4f} -- the coupled model is genuinely broken "
              f"at the dense end")
    if m_max > MAX_M:
        print(f"  [FAIL] heaviest point is {m_max:,} edges (cap {MAX_M:,}). REFUSING to submit.")
        ok = False
    else:
        print(f"  [PASS] heaviest point is {m_max:,} edges, under the {MAX_M:,} cap")
    print(f"  [INFO] the sparse end sits at |H|/m = {hm_lo:.4f} -- near-metric, BY DESIGN. That is the "
          f"onset, and it is the result the array exists to show, not a defect.")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task-index", type=int, default=None)
    ap.add_argument("--outdir", default="results_coupled")
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    a = ap.parse_args()

    if a.count:
        print(len(all_tasks()))
        return
    if a.preflight:
        sys.exit(0 if preflight() else 1)
    # The joblist is built by submit_coupled_dsq.sh and NOWHERE ELSE. A second generator here would emit
    # `sage -python` (right locally, wrong on the cluster, which runs conda) and the two would drift.
    if a.task_index is None:
        ap.error("--task-index is required (or --count / --preflight)")
    path = run_one_task(a.task_index, a.outdir)
    print(f"task {a.task_index} ({EXP}) -> {path}")


if __name__ == "__main__":
    main()
