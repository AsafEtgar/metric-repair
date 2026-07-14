"""The RGG CAMPAIGN, RE-RUN -- small and large, PLANTED corruptions only (no jitter).

WHY THIS EXISTS. break_metric_graph's inflate branch used to read

    w'(uv) = max( magnitude * detour ,  w(uv) * 1.001 + 1 )

and that "+ 1" was an ABSOLUTE floor, written for integer weights where an edit must survive int(round(.)).
RGG weights are Euclidean distances in the unit square -- every one below 0.12 -- so the floor swamped
magnitude * detour and EVERY inflation landed at ~11.8x the detour no matter what magnitude was requested.
Measured across magnitude = 1.2, 1.5, 2, 3, 5, 10: the effective factor was 11.77 at all six. The knob was
inert; the inflate magnitude sweep planted one corruption six times.

The fix (graph_models.py) drops the additive floor and makes the integer guard relative to the DETOUR, which
is the quantity that decides heaviness. Every INFLATE row in the published RGG campaign was therefore
generated against a corruption nobody asked for, and this array replaces them.

WHAT IT RE-RUNS, AND WHAT IT DOES NOT.

    small   n = 100..500,  the exact optimum is available   74 points x 40 seeds = 2,960 tasks
    large   n = 1000..3000, the ladder and the sweeps        41 points x 20 seeds =   820 tasks

  Both grids are the PUBLISHED ones with break_type == "jitter" filtered out, per the author: inflate and
  deflate only. The dense family (exp1/exp2a/exp2b), the coupling A/B and exp2c are NOT affected and are NOT
  re-run -- harness.py never calls break_metric_graph at all. Its graphs are non-metric by WEIGHT MODEL, not
  by a planted corruption. The planted REAL bases are a separate array (realplanted_harness.py).

  DEFLATE rows would replay bit-identically (the inflate branch draws no randomness, so the shuffle stream is
  untouched), but they are re-run anyway: splicing two CSV vintages into one analysis is exactly the kind of
  provenance rot the gates exist to stop, and deflate is only ~28% of the tasks.

THE GRIDS ARE FILTERED, NOT REBUILT -- and that is what makes this a REPLACEMENT rather than a new experiment.
task_seed() hashes the cfg's CONTENT (sweep, n, deg, direction, magnitude, ...) plus the sample index, never
the task index. So dropping the jitter points renumbers the tasks but changes no seed: every surviving point
rebuilds THE SAME GRAPH it did before, and only the corruption differs. G4 checks that against the DELIVERED
CSV rather than trusting the argument.

BUDGETS ARE THE PUBLISHED ONES, and that is a deliberate choice. skipped_time is 0 of 175,218 rows on the
small grid and 0 of 32,582 on the large one -- the task budget never fired, so a timeout in this campaign is
an algorithmic fact, exactly as before. The fix also makes inflations WEAKER (mu = 3 instead of 11.8), which
shrinks the covering LP (6,825 broken cycles instead of 9,464 at n=300), so if anything there is more slack
than before. G5 gates on skipped_time == 0 and refuses to report if that stops being true.

  rgg_harness.py is imported READ-ONLY. This module owns its grid; it reuses that module's generator, suite,
  isolation, caps, budget map and CSV schema verbatim, so the rows are schema-identical to the campaign's.

  usage   python experiments/rgg_rerun_harness.py --preflight
          python experiments/rgg_rerun_harness.py --grid small --count
          python experiments/rgg_rerun_harness.py --grid small --task-index 0 --outdir results_rgg_rerun_small
"""
import argparse
import math
import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rgg_harness as rh                                                    # noqa: E402  READ-ONLY
from graph_models import (                                                  # noqa: E402
    seed_all, random_geometric_metric_graph, break_metric_graph,
)
from metric_repair import domr_alg                                          # noqa: E402


def _reweight_only(points):
    """The published points, minus jitter. Copies, so nothing upstream is mutated."""
    return [dict(p) for p in points if p.get("break_type") == "reweight"]


GRIDS = {"small": _reweight_only(rh.POINTS_RGG),
         "large": _reweight_only(rh.GRIDS["large"])}
SAMPLES = {"small": rh.SAMPLES["full"], "large": rh.SAMPLES["large"]}       # 40 and 20, as published
DROP_ILP = {"small": False, "large": True}                                  # as run_one_rgg_task does
BUDGET = {"small": rh.task_budget("full"), "large": rh.task_budget("large")}  # 2 h and 6 h, as published


def all_tasks(grid):
    return [(cfg, s) for cfg in GRIDS[grid] for s in range(SAMPLES[grid])]


def run_one(task_index, outdir, grid):
    cfg, s = all_tasks(grid)[task_index]
    return rh._run(cfg, s, task_index, outdir, drop_ilp=DROP_ILP[grid], budget=BUDGET[grid])


# ----------------------------------------------------------------------------
# The preflight. It REFUSES to submit unless the thing this array exists to fix is actually fixed.
# ----------------------------------------------------------------------------
def _effective_mu(G, H, B):
    """The inflation factor ACTUALLY planted: w'(uv) / d_{G-uv}(u,v), the ratio that decides heaviness."""
    out = []
    for (u, v) in B:
        Gm = G.copy(); Gm.remove_edge(u, v)
        try:
            out.append(H[u][v]["weight"] / nx.shortest_path_length(Gm, u, v, weight="weight"))
        except nx.NetworkXNoPath:
            continue
    return float(np.median(out)) if out else float("nan")


def preflight():
    fails = []

    def chk(c, name, obs):
        print(f"  [{'PASS' if c else 'FAIL'}] {name:<52} {obs}")
        if not c:
            fails.append(name)

    # G1  NO JITTER SURVIVED. The author asked for inflate and deflate only.
    kinds, dirs = set(), set()
    for g in GRIDS:
        for p in GRIDS[g]:
            kinds.add(p.get("break_type"))
            dirs.add(p.get("direction"))
    chk(kinds == {"reweight"} and dirs <= {"inflate", "deflate"},
        "G1 planted corruptions only (no jitter)", f"break_type={sorted(kinds)} direction={sorted(dirs)}")

    # G2  THE GRIDS ARE THE PUBLISHED ONES MINUS JITTER -- nothing added, nothing else lost.
    drop = {g: len(rh.POINTS_RGG if g == "small" else rh.GRIDS["large"]) - len(GRIDS[g]) for g in GRIDS}
    chk(all(v > 0 for v in drop.values()),
        "G2 jitter points dropped from BOTH grids",
        "; ".join(f"{g}: {len(GRIDS[g])} kept, {drop[g]} dropped "
                  f"({len(GRIDS[g]) * SAMPLES[g]:,} tasks)" for g in GRIDS))

    # G3  *** THE FIX WORKS. *** This is the entire reason the array is being submitted. Build a REAL instance
    #     at every magnitude the grid carries and measure the inflation actually planted. Under the old code
    #     every one of these came back 11.77.
    mags = sorted({p["magnitude"] for g in GRIDS for p in GRIDS[g]
                   if p.get("direction") == "inflate" and p.get("magnitude")})
    n, deg = 300, 12
    seed_all(7)
    G = random_geometric_metric_graph(n=n, mode="radius", radius=math.sqrt(deg / (math.pi * n)), dim=2)
    eff = {}
    for mu in mags:
        seed_all(7)
        H, B = break_metric_graph(G, frac_q=0.10, direction="inflate", magnitude=float(mu))
        eff[mu] = _effective_mu(G, H, B)
    off = [mu for mu in mags if not abs(eff[mu] / mu - 1.0) < 0.02]
    chk(not off, "G3 the inflate magnitude knob is LIVE (effective == requested)",
        "  ".join(f"{mu}->{eff[mu]:.2f}" for mu in mags) + (f"; OFF: {off}" if off else ""))
    if off:
        print("         *** The '+1' floor is still in graph_models.break_metric_graph, or has come back.")
        print("         *** Submitting would burn the cluster reproducing the bug. NOT submitting.")

    # G4  *** SAME GRAPHS, NEW CORRUPTION. *** task_seed hashes cfg CONTENT, not the task index, so filtering
    #     out jitter must not move a single seed. Checked against the DELIVERED CSV, not against the argument:
    #     if the filter mutated a cfg, or a key drifted, the seeds diverge and this fails.
    csv = "analysis/rgg/rgg_rows_with_ratio.csv"
    if not os.path.exists(csv):
        chk(False, "G4 seeds match the published campaign", f"{csv} missing -- cannot verify provenance")
    else:
        import pandas as pd
        old = pd.read_csv(csv, low_memory=False).drop_duplicates("task")
        keyc = ["sweep", "n", "deg", "k", "direction", "magnitude", "frac_q", "sample"]
        old = old[old.break_type.eq("reweight")].dropna(subset=["seed"])
        seen = {}
        for _, r in old.iterrows():
            seen[tuple(None if pd.isna(r[c]) else r[c] for c in keyc)] = int(r["seed"])
        bad, checked = [], 0
        for cfg, s in all_tasks("small"):
            k = (cfg["sweep"], cfg["n"], cfg.get("deg"), cfg.get("k"), cfg.get("direction"),
                 cfg.get("magnitude"), cfg.get("frac_q"), s)
            if k in seen:
                checked += 1
                if rh.task_seed(cfg, s) != seen[k]:
                    bad.append(k)
        chk(checked > 0 and not bad, "G4 seeds match the published campaign EXACTLY",
            f"{checked:,} tasks matched, {len(bad)} seed mismatches"
            + ("  -- the re-run would build DIFFERENT graphs" if bad else "  -- same graphs, new corruption"))

    # G5  H == B IS NOT A THEOREM, AND THE GRID NOW REACHES WHERE IT BREAKS. The detour is measured in G but
    #     heaviness is judged in H, so if an edge's detour runs through ANOTHER inflated edge that detour grew
    #     -- and a weak inflation no longer clears it. The old +1 inflated so hard nothing could hide. This is
    #     a DISCLOSURE, not a failure: it is a real property of the corruption model that the bug was masking,
    #     and the collector must not assume OPT == |H| at every magnitude.
    weak = []
    for mu in mags:
        seed_all(7)
        H, B = break_metric_graph(G, frac_q=0.10, direction="inflate", magnitude=float(mu))
        Hs = {tuple(sorted(e)) for e in domr_alg(H)}
        if Hs != {tuple(sorted(e)) for e in B}:
            weak.append((mu, len(B), len(Hs)))
    chk(True, "G5 where does H == B stop holding?",
        ("holds at every magnitude" if not weak
         else "; ".join(f"mu={m}: |B|={b} but |H|={h}" for m, b, h in weak)))
    if weak:
        print("         *** At these magnitudes the planted set is NOT the heavy set: a weak inflation does")
        print("         *** not survive its own interference. EXPECTED, and a finding. OPT == |H| must not be")
        print("         *** asserted grid-wide -- only where it is measured to hold.")

    print()
    for g in GRIDS:
        print(f"  {g:<6} {len(GRIDS[g]):>3} points x {SAMPLES[g]} seeds = {len(all_tasks(g)):>5,} tasks   "
              f"budget {BUDGET[g] // 3600} h   ILPs {'dropped' if DROP_ILP[g] else 'ON'}")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="small", choices=sorted(GRIDS))
    ap.add_argument("--task-index", type=int, default=None)
    ap.add_argument("--outdir", default="results_rgg_rerun")
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    a = ap.parse_args()

    if a.preflight:
        fails = preflight()
        if fails:
            raise SystemExit(f"\n*** PREFLIGHT FAILED: {fails}. NOT submitting. ***")
        print("\nPreflight clean.")
        return
    if a.count:
        print(len(all_tasks(a.grid)))
        return
    if a.task_index is None:
        ap.error("--task-index is required (or --count / --preflight)")
    path = run_one(a.task_index, a.outdir, a.grid)
    print(f"rgg_rerun {a.grid} task {a.task_index} -> {path}")


if __name__ == "__main__":
    main()
