"""PLANTED corruptions in REAL metric bases, re-run against the fixed inflate. A SEPARATE array.

This drives rgg_harness's EXISTING "realrec" grid -- 5 real bases x {inflate, deflate, mixed} x 4 fractions
x 15 seeds = 900 tasks. No grid is invented here; the module exists to add a PREFLIGHT (rgg_harness is
import-path code and must not be edited) and to keep the array's provenance in one file.

WHY THESE ROWS ARE SUSPECT, AND WHY ONLY SOME OF THEM ARE. The old inflate rule was

    w'(uv) = max( magnitude * detour ,  w(uv) * 1.001 + 1 )

The "+ 1" is an ABSOLUTE constant. Whether it swamped `magnitude * detour` depends entirely on THE SCALE OF
THE WEIGHTS, so the damage is per-base, not uniform:

    weights >> 1  (dimacs road networks: metres)   -- magnitude*detour dominates; the floor never bound;
                                                      those rows are UNCHANGED by the fix.
    weights <  1  (cosine distances, normalised)   -- the floor swamps everything; every inflation landed at
                                                      a fixed factor nobody asked for. Those rows are WRONG.

G3 measures this per base and prints which is which, from the real loaded graphs -- it is not inferred from
the name of the dataset. `mixed` is half inflate, so a damaged base damages its mixed arm too. `deflate` is
untouched on every base (that branch never had a floor).

  rgg_harness.py is imported READ-ONLY. Budget, suite, isolation, caps and CSV schema are its own.

  usage   python experiments/realplanted_harness.py --preflight
          python experiments/realplanted_harness.py --count
          python experiments/realplanted_harness.py --task-index 0 --outdir results_realplanted
"""
import argparse
import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rgg_harness as rh                                                    # noqa: E402  READ-ONLY
from graph_models import seed_all, break_metric_graph                       # noqa: E402

GRID = "realrec"


def all_tasks():
    return rh.all_tasks(GRID)


def run_one(task_index, outdir):
    return rh.run_one_rgg_task(task_index, outdir, GRID)


def _effective_mu(G, H, B):
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
        print(f"  [{'PASS' if c else 'FAIL'}] {name:<50} {obs}")
        if not c:
            fails.append(name)

    pts = rh.GRIDS[GRID]
    bases = sorted({p["base"] for p in pts})
    mags = sorted({p["magnitude"] for p in pts})
    dirs = sorted({p["direction"] for p in pts})
    chk(len(pts) > 0 and len(mags) == 1,
        "G1 the realrec grid is intact",
        f"{len(pts)} points x {rh.SAMPLES[GRID]} seeds = {len(all_tasks()):,} tasks; "
        f"bases={len(bases)} dirs={dirs} magnitude={mags}")

    # G2  every base actually loads. A missing raw file here is 15 dead tasks apiece, discovered on the
    #     cluster instead of in this shell.
    loaded, dead = {}, []
    for b in bases:
        try:
            cfg = next(p for p in pts if p["base"] == b)
            T, H, corr, jit, r, ja = rh.generate_rgg(cfg, rh.task_seed(cfg, 0))
            loaded[b] = T
        except Exception as e:
            dead.append(f"{b}: {type(e).__name__}")
    chk(not dead, "G2 every real base builds", f"{len(loaded)} of {len(bases)}"
        + (f"; DEAD {dead}" if dead else ""))

    # G3  *** WHICH BASES WERE ACTUALLY DAMAGED, AND IS THE KNOB LIVE NOW? ***
    #
    #     COUNTED PER EDGE, NOT MEDIANED. The first version of this gate compared the MEDIAN effective mu and
    #     reported "none damaged" -- and it was wrong. pbmc3k's median inflated edge is untouched by the old
    #     floor while 8.8% of its edges are not, some by a factor of 3. A floor is an absolute constant meeting
    #     a DISTRIBUTION of detours: it bites the short tail and leaves the middle alone, which is precisely
    #     the shape a median cannot see. Count the edges that MOVE.
    mu = float(mags[0])
    rows = []
    for b, T in loaded.items():
        ws = [d["weight"] for _, _, d in T.edges(data=True)]
        integer = all(isinstance(w, (int, np.integer)) for w in ws)
        seed_all(11)
        Hn, Bn = break_metric_graph(T, frac_q=0.10, direction="inflate", magnitude=mu)
        moved, worst = 0, 0.0
        for (u, v) in Bn:
            Gm = T.copy(); Gm.remove_edge(u, v)
            try:
                det = nx.shortest_path_length(Gm, u, v, weight="weight")
            except nx.NetworkXNoPath:
                continue
            old = max(det * mu, T[u][v]["weight"] * 1.001 + 1)
            old = max(1, int(round(old))) if integer else float(old)
            new = Hn[u][v]["weight"]
            if abs(old - new) > 1e-9:
                moved += 1
                worst = max(worst, abs(old - new) / max(new, 1e-12))
        rows.append((b, min(ws), max(ws), len(Bn), moved / max(len(Bn), 1), worst,
                     _effective_mu(T, Hn, Bn)))

    print(f"\n  {'base':<22}{'weight range':>20}{'|B|':>7}{'edges MOVED':>13}"
          f"{'worst rel':>11}{'new eff mu':>12}")
    print("  " + "-" * 86)
    damaged = []
    for b, lo, hi, nb, frac, worst, ne in rows:
        if frac > 0:
            damaged.append(b)
        print(f"  {b:<22}{f'[{lo:.3g}, {hi:.3g}]':>20}{nb:>7}{frac:>12.1%}{worst:>11.2f}{ne:>12.2f}")
    off = [b for b, _, _, _, _, _, ne in rows if not abs(ne / mu - 1.0) < 0.02]
    chk(not off, f"G3 the magnitude knob is LIVE on every base (asked {mu})",
        f"{len(rows) - len(off)} of {len(rows)} at {mu}" + (f"; STILL OFF: {off}" if off else ""))
    print(f"\n  DAMAGED by the old floor -- inflate and mixed arms must be re-run: {damaged or 'none'}")
    print(f"  BIT-IDENTICAL under the fix (the +1 never bound; re-running them changes nothing): "
          f"{[b for b, *_ in rows if b not in damaged] or 'none'}")
    print("  deflate is untouched on every base -- that branch never had an additive floor.")

    print(f"\n  {len(all_tasks()):,} tasks   budget {rh.task_budget(GRID) // 3600} h   ILPs dropped")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-index", type=int, default=None)
    ap.add_argument("--outdir", default="results_realplanted")
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
        print(len(all_tasks()))
        return
    if a.task_index is None:
        ap.error("--task-index is required (or --count / --preflight)")
    path = run_one(a.task_index, a.outdir)
    print(f"realplanted task {a.task_index} -> {path}")


if __name__ == "__main__":
    main()
