"""The RGG SCALE array -- the sparse family pushed to n = 5000, in both corruption directions.

WHY. If the benchmark section is going to rest on the RGG (the only synthetic family with a planted corrupted
set AND true weights, and the one that actually resembles the real graphs), then the RGG has to carry the
scale claim too. The current large grid stops at n = 3000 and sweeps fraction and magnitude in ONE direction.
Neither is enough.

WHAT IS NEW, AND WHY EACH PIECE IS THERE.

  * n = 1000 .. 5000, step 200 (21 points), 30 seeds, INFLATE and DEFLATE.  The old ladder stopped at 3000
    with 20 seeds and no deflate arm at scale.

  * THE FRACTION AND MAGNITUDE SWEEPS NOW RUN IN BOTH DIRECTIONS.  P2df and P2dm are deflate-only in the
    published grid, and that is not survivable: the corruption direction does not shift the ranking, it
    INVERTS it. l1sep_gmr loses to spc_gmr on 100% of inflate tasks and beats it on 100% of deflate tasks.
    A "the effect of fraction" claim measured on deflate alone is a DEFLATE claim wearing a general label,
    and the section's own thesis is that the direction decides. So: P2if / P2im are added.

  * NO JITTER. P2size is gone. It plants no corruption, its graphs sit at |H|/m = 0.004 -- already metric --
    and every method scores ~0.01 there by doing nearly nothing. It was 220 of the old 1,040 "sparse" tasks
    and it dragged every median toward zero. A benchmark of repair algorithms on graphs that need no repair
    is not a benchmark.

THE BUDGET IS RAISED, AND THAT IS THE WHOLE POINT OF THIS FILE'S EXISTENCE.

The section wants to claim that algorithms hit their limits even on a SPARSE family. That claim is only worth
making if a timeout is an algorithmic fact. It would not have been. harness's per-grid TASK_BUDGET is 6 h,
and at n = 5000 the median task needs ~10 core-h. When the budget expires, `_run` marks every REMAINING
algorithm `skipped_time` -- walking build_suite_rgg in order, which ends:

    ... l1sep_gmr, l1sep_iomr, spc_gmr, spc_iomr, PIVOT, LEFT_EDGE

pivot and left_edge are LAST. They are also precisely the two methods whose limitation the section exists to
demonstrate (they COMPLETE the graph: at n = 5000 that is 12,497,500 edges for a graph that has ~30,000 --
a 450x blowup, and 5.5 GB of peak memory). Under a 6 h budget they would come back `skipped_time` and nobody
could tell whether they hit their 1800 s cap or were merely last in the queue when the clock ran out. The
finding would have been true BY CONSTRUCTION OF THE SUITE ORDER.

So the budget is raised above the theoretical ceiling instead. The RGG is connected (median 1 component,
giant >= 99.4% of nodes), so no algorithm exceeds its 1800 s cap, and a task is hard-bounded at
16 x 1800 s = 8 h. BUDGET = 9 h therefore never binds, the per-algorithm cap is the only constraint left, and
a timeout means what it says. `collect_rgg_scale.py` GATES on skipped_time == 0: if the budget ever fires,
the limitation claim is confounded and the collector refuses to print.

Today, for reference: skipped_time is 0 of 7,040 rows on the published large grid. This hazard is entirely
new to the larger n, which is exactly why it would have been missed.

NOTHING IN rgg_harness.py IS TOUCHED. It is imported read-only -- changing a task-import path mid-campaign
would invalidate 11,965 delivered tasks. `_run()` already takes `budget` as a parameter, so the raise costs
no edit. The grid, the seeds and the sweep set live here; the generator, suite, isolation, caps and CSV
schema are the harness's, verbatim, so these rows concatenate with the published ones.

  one task  = one graph = the whole suite  -> results_rgg_scale/task_NNNNNN.csv
  the grid  = 70 points x 30 seeds = 2,100 tasks

  count      sage -python experiments/rgg_scale_harness.py --count
  preflight  sage -python experiments/rgg_scale_harness.py --preflight
  one task   sage -python experiments/rgg_scale_harness.py --task-index K --outdir results_rgg_scale
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rgg_harness                                                      # noqa: E402
from rgg_harness import _base_p1, _base_p2, _run, task_seed, generate_rgg   # noqa: E402
from metric_repair import domr_alg                                      # noqa: E402

NS = tuple(range(1000, 5001, 200))          # 21 points: 1000, 1200, ..., 5000
SEEDS = 30
FRACS = (0.02, 0.05, 0.10, 0.20, 0.30)
MAGS = (2.0, 3.0, 5.0, 10.0)

# The ceiling for a CONNECTED graph is 16 algos x 1800 s = 8 h. 9 h therefore never binds, and the
# per-algorithm cap is the only thing that can stop an algorithm. See the module docstring: this is the
# difference between "pivot cannot do n=5000" and "pivot was last in the queue".
BUDGET_S = 9 * 3600


def points():
    pts = []
    # the size ladder, BOTH directions. deg stays at _base_p1's 12 -> m ~ 6n (30k edges at n=5000).
    for n in NS:
        a = _base_p1(); a.update(sweep="S1", n=n); pts.append(a)
        d = _base_p1(); d.update(sweep="S1d", n=n, direction="deflate"); pts.append(d)
    # density at n=2000, in both radius and kNN construction -- the RGG's own density sweep. With the dense
    # family in question, this is where a density story has to come from.
    for deg in (4, 8, 12, 20, 30, 40):
        c = _base_p1(); c.update(sweep="S2", n=2000, deg=deg); pts.append(c)
    for k in (8, 12, 20, 30):
        c = _base_p1(); c.update(sweep="S2k", n=2000, mode="knn", k=k); pts.append(c)
    # fraction and magnitude -- IN BOTH DIRECTIONS. The published grid runs these deflate-only.
    for direction, fs, ms in (("deflate", "P2df", "P2dm"), ("inflate", "P2if", "P2im")):
        for q in FRACS:
            e = _base_p2(); e.update(sweep=fs, n=1000, break_type="reweight",
                                     direction=direction, frac_q=q, magnitude=5.0); pts.append(e)
        for m in MAGS:
            e = _base_p2(); e.update(sweep=ms, n=1000, break_type="reweight",
                                     direction=direction, frac_q=0.10, magnitude=float(m)); pts.append(e)
    return pts


POINTS = points()


def all_tasks():
    return [(cfg, s) for cfg in POINTS for s in range(SEEDS)]


def run_one_task(task_index, outdir):
    """rgg_harness._run, with the ILPs dropped (they do not converge at this scale) and the budget RAISED
    above the per-algorithm ceiling so that the cap, not the queue, decides who fails."""
    cfg, s = all_tasks()[task_index]
    return _run(cfg, s, task_index, outdir, drop_ilp=True, budget=BUDGET_S)


def preflight(seeds=1):
    """Build the corners of the grid for real and check they are what the section will claim they are.

    Cheap, because it only touches the extremes: the biggest n (where the cost and memory claims live) and
    one point of each corruption direction (where |H|/m must differ, or the inversion has nothing to invert)."""
    print(f"PREFLIGHT -- {len(POINTS)} grid points x {SEEDS} seeds = {len(all_tasks())} tasks\n")
    by = {}
    for cfg in POINTS:
        by.setdefault(cfg["sweep"], []).append(cfg)
    print("  sweep    pts   what it varies")
    print("  " + "-" * 62)
    for sw in ("S1", "S1d", "S2", "S2k", "P2df", "P2dm", "P2if", "P2im"):
        if sw not in by:
            continue
        c = by[sw]
        d = sorted({x["direction"] for x in c})
        print(f"  {sw:<8} {len(c):>3}   n={min(x['n'] for x in c)}-{max(x['n'] for x in c)}  "
              f"direction={d}")
    assert "P2size" not in by, "jitter must not be in this grid"
    print("  (no P2size: the jitter sweep is DROPPED -- metric graphs cannot benchmark a repair)\n")

    ok = True
    # 1. both directions must actually break the graph DIFFERENTLY, or there is no inversion to report
    hm = {}
    for sw, n in (("S1", 1000), ("S1d", 1000)):
        cfg = [c for c in POINTS if c["sweep"] == sw and c["n"] == n][0]
        T, H, corrupted, _, _, _ = generate_rgg(cfg, task_seed(cfg, 0))
        m = H.number_of_edges()
        hm[sw] = len(domr_alg(H)) / m
        print(f"  {sw:<5} (n={n}, {cfg['direction']:<7}): m={m:,}  |H|/m={hm[sw]:.4f}  "
              f"planted={len(corrupted)}")
    sep = abs(hm["S1"] - hm["S1d"])
    print(f"\n  [{'PASS' if sep > 0.1 else 'FAIL'}] the two directions break the graph differently "
          f"(|H|/m differs by {sep:.3f})")
    ok = ok and sep > 0.1

    # 2. the top of the ladder must be buildable, and we quote its size in the paper
    cfg = [c for c in POINTS if c["sweep"] == "S1" and c["n"] == max(NS)][0]
    T, H, _, _, _, _ = generate_rgg(cfg, task_seed(cfg, 0))
    m = H.number_of_edges()
    comp = 5000 * 4999 // 2
    print(f"\n  [PASS] top of the ladder builds: n={max(NS)}  m={m:,}")
    print(f"         pivot/left_edge COMPLETE the graph -> {comp:,} edges, a {comp/m:.0f}x blowup.")
    print(f"         That is the limitation the section exists to show, and it needs this n to be visible.")

    # 3. THE BUDGET MUST NOT BIND. If it does, `skipped_time` decides who fails, in suite order.
    ceiling = 16 * rgg_harness.TIMEOUT_S
    print(f"\n  [{'PASS' if BUDGET_S > ceiling else 'FAIL'}] budget {BUDGET_S/3600:.1f}h > the "
          f"{ceiling/3600:.1f}h per-task ceiling (16 algos x {rgg_harness.TIMEOUT_S}s cap)")
    print(f"         -> the per-algorithm cap is the ONLY thing that can stop an algorithm, so a timeout")
    print(f"            is an algorithmic fact and not a queue position.")
    ok = ok and BUDGET_S > ceiling
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task-index", type=int, default=None)
    ap.add_argument("--outdir", default="results_rgg_scale")
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
    print(f"task {a.task_index} -> {path}")


if __name__ == "__main__":
    main()
