"""cost_law.py -- the controlled cost-law experiment the design promises: isolate runtime's dependence on
n, m, and |H| by holding two fixed and varying the third.

The main analyzers fit log(cpu) ~ n + w_max, which conflates n and m (in a fixed-degree RGG, m grows with n)
and never isolates |H|. This does the clean thing:

  sweep N : vary n, fix degree (so m grows with n), fix the corrupted-edge count  -> separates {n,m} from |H|
  sweep M : fix n, vary degree (m grows, n fixed), fix corrupted count            -> isolates m
  sweep H : fix n and degree (m fixed), vary corrupted count                      -> isolates |H|

Predictions (from the mechanism): pivot completes the graph first (Theta(n^2) edges) -> n-bound, FLAT in m and
|H|. bestofk does k roundings over the edges -> m-bound. l1sep recomputes all-pairs shortest paths per
separation round -> size-bound (n and m), with |H| setting the round count.

Writes cost_law.csv (one row per sweep-point: knob, n, m, |H|, seconds per algo) and, per algo, the log-log
slope against each swept parameter. With --plot (needs matplotlib) writes fig_cost_law: three panels, runtime
vs n / m / |H|, log-log, one line per algo.

Self-contained except for the algorithms themselves (metric_repair) and the RGG generator (graph_models);
those imports are READ-ONLY and this file is never imported by a running task. Run OFF the cluster's critical
path -- it is a one-time methodology run, a few minutes with the l1sep time-guard.

    sage -python experiments/cost_law.py --seeds 3 --plot --outdir analysis
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph_models import seed_all, random_geometric_metric_graph, break_metric_graph   # noqa: E402
from metric_repair import domr_alg, pivot_heuristic, l1_separation, covering_lp_cover, _norm  # noqa: E402

BEST_OF_K = 12
L1_GUARD_S = 90.0          # once l1sep exceeds this on a point, drop it from larger points in that sweep


def radius_for(n, deg):
    return math.sqrt(deg / (math.pi * max(n - 1, 1)))


def instance(n, deg, n_corrupt, seed):
    seed_all(seed)
    T = random_geometric_metric_graph(n, mode="radius", radius=radius_for(n, deg))
    T = nx.convert_node_labels_to_integers(T.subgraph(max(nx.connected_components(T), key=len)).copy())
    m = T.number_of_edges()
    frac = min(0.9, max(n_corrupt / max(m, 1), 1.0 / max(m, 1)))
    C, _ = break_metric_graph(T, frac_q=frac, direction="inflate", magnitude=3.0)
    C = nx.convert_node_labels_to_integers(C.subgraph(max(nx.connected_components(C), key=len)).copy())
    H = len({_norm(u, v) for (u, v) in domr_alg(C)})
    return C, C.number_of_nodes(), C.number_of_edges(), H


def _algos():
    return {
        "domr":        lambda G: domr_alg(G),
        "pivot":       lambda G: pivot_heuristic(G),
        "gmr_bestofk": lambda G: covering_lp_cover(G, solve="separation", rounding="randomized",
                                                   oracle="naive", seed=0, best_of_k=BEST_OF_K)[0],
        "l1sep_gmr":   lambda G: l1_separation(G, general=True),
    }


def run_sweep(name, points, seeds):
    """points: list of (knob_value, (n, deg, n_corrupt)). Returns list of dict rows."""
    print(f"\n=== sweep {name} ===", flush=True)
    algos = _algos()
    l1_alive = True                       # l1sep time-guard, per sweep
    rows = []
    for knob, (n, deg, ncorr) in points:
        per = {a: [] for a in algos}
        nn = mm = hh = 0
        for s in range(seeds):
            G, nn, mm, hh = instance(n, deg, ncorr, 1000 + s)
            for a, fn in algos.items():
                if a == "l1sep_gmr" and not l1_alive:
                    per[a].append(float("nan")); continue
                t = time.perf_counter()
                try:
                    fn(G)
                    dt = time.perf_counter() - t
                except Exception:                    # noqa: BLE001
                    dt = float("nan")
                per[a].append(dt)
                if a == "l1sep_gmr" and dt > L1_GUARD_S:
                    l1_alive = False
        med = {a: float(np.nanmedian(per[a])) if np.any(np.isfinite(per[a])) else float("nan") for a in algos}
        rows.append(dict(sweep=name, knob=knob, n=nn, m=mm, H=hh, **{f"cpu_{a}": med[a] for a in algos}))
        print("  %-8s n=%-5d m=%-6d |H|=%-5d  %s" % (name, nn, mm, hh,
              "  ".join("%s=%.3f" % (a, med[a]) for a in algos)), flush=True)
    return rows


def loglog_slope(xs, ys):
    xy = [(x, y) for x, y in zip(xs, ys) if x and x > 0 and y and y > 0 and np.isfinite(y)]
    if len(xy) < 2:
        return float("nan")
    lx = np.log([p[0] for p in xy]); ly = np.log([p[1] for p in xy])
    return float(np.polyfit(lx, ly, 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--plot", action="store_true", help="also write fig_cost_law (needs matplotlib)")
    ap.add_argument("--outdir", default="analysis")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    N = [(n, (n, 10, 12)) for n in (150, 250, 400, 600, 900)]     # vary n; m grows; |H| fixed
    M = [(d, (300, d, 12)) for d in (6, 10, 16, 24, 34)]          # vary m at fixed n; |H| fixed
    H = [(c, (300, 10, c)) for c in (4, 12, 30, 70, 140)]         # vary |H| at fixed n, m

    rows = run_sweep("N", N, a.seeds) + run_sweep("M", M, a.seeds) + run_sweep("H", H, a.seeds)
    import csv
    cols = list(rows[0].keys())
    path = os.path.join(a.outdir, "cost_law.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(r)

    algos = list(_algos())
    byS = {s: [r for r in rows if r["sweep"] == s] for s in ("N", "M", "H")}
    print("\n=== log-log slopes  d(log time)/d(log knob) ===")
    print("%-12s %8s %8s %8s" % ("algo", "vs n (N)", "vs m (M)", "vs |H| (H)"))
    slopes = {}
    for al in algos:
        sn = loglog_slope([r["n"] for r in byS["N"]], [r[f"cpu_{al}"] for r in byS["N"]])
        sm = loglog_slope([r["m"] for r in byS["M"]], [r[f"cpu_{al}"] for r in byS["M"]])
        sh = loglog_slope([max(r["H"], 1) for r in byS["H"]], [r[f"cpu_{al}"] for r in byS["H"]])
        slopes[al] = (sn, sm, sh)
        print("%-12s %8.2f %8.2f %8.2f" % (al, sn, sm, sh))
    print(f"\nwrote {path}")

    if a.plot:
        import matplotlib.pyplot as plt
        color = {"domr": "#888888", "pivot": "#0072B2", "gmr_bestofk": "#009E73", "l1sep_gmr": "#D55E00"}
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for ax, (S, xk, xl) in zip(axes, (("N", "n", "$n$ (vertices; $m$ grows, $|H|$ fixed)"),
                                          ("M", "m", "$m$ (edges; $n$, $|H|$ fixed)"),
                                          ("H", "H", "$|H|$ (broken edges; $n$, $m$ fixed)"))):
            d = byS[S]
            for al in algos:
                xs = [r[xk] for r in d]; ys = [r[f"cpu_{al}"] for r in d]
                pts = [(x, y) for x, y in zip(xs, ys) if x and y and np.isfinite(y) and y > 0]
                if len(pts) >= 2:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4,
                            color=color[al], label=al)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(xl); ax.grid(alpha=0.25, which="both")
        axes[0].set_ylabel("runtime (s)")
        axes[0].legend(fontsize=8, frameon=False)
        fig.suptitle("Cost laws: pivot is flat in $m$ and $|H|$ ($n$-bound); "
                     "bestofk grows in $m$; l1sep grows in $n$ and $m$.", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(a.outdir, f"fig_cost_law.{ext}"), dpi=150, bbox_inches="tight")
        print(f"wrote {a.outdir}/fig_cost_law.pdf / .png")


if __name__ == "__main__":
    main()
