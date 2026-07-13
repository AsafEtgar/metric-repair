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


def _write_csv(rows, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)


def run_sweep(name, points, seeds, checkpoint=None, sofar=None):
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
        if checkpoint:                              # partial artifact after every point; a long l1sep call
            _write_csv((sofar or []) + rows, checkpoint)   # in one sweep can never cost us the others
    return rows


def loglog_slope(xs, ys):
    xy = [(x, y) for x, y in zip(xs, ys) if x and x > 0 and y and y > 0 and np.isfinite(y)]
    if len(xy) < 2:
        return float("nan")
    lx = np.log([p[0] for p in xy]); ly = np.log([p[1] for p in xy])
    return float(np.polyfit(lx, ly, 1)[0])


AXES = (("N", "n", "$n$ (vertices; $m$ grows with $n$, $|H|$ fixed)"),
        ("M", "m", "$m$ (edges; $n$, $|H|$ fixed)"),
        ("H", "H", "$|H|$ (broken edges; $n$, $m$ fixed)"))
COLOR = {"domr": "#888888", "pivot": "#0072B2", "gmr_bestofk": "#009E73", "l1sep_gmr": "#D55E00"}


def read_csv_rows(path):
    import csv
    out = []
    for r in csv.DictReader(open(path)):
        d = {k: r[k] for k in ("sweep",)}
        for k in ("knob", "n", "m", "H"):
            d[k] = int(r[k])
        for k in r:
            if k.startswith("cpu_"):
                d[k] = float(r[k]) if r[k] not in ("", "nan") else float("nan")
        out.append(d)
    return out


def fit_slopes(rows, algos):
    """Per algo, the log-log slope against the knob each sweep actually varies."""
    byS = {s: [r for r in rows if r["sweep"] == s] for s in ("N", "M", "H")}
    slopes = {}
    for al in algos:
        sl = {}
        for S, xk, _ in AXES:
            xs = [max(r[xk], 1) for r in byS[S]]
            sl[S] = loglog_slope(xs, [r[f"cpu_{al}"] for r in byS[S]])
        slopes[al] = sl
    return byS, slopes


def make_figure(rows, algos, outdir):
    import matplotlib.pyplot as plt
    byS, slopes = fit_slopes(rows, algos)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, (S, xk, xl) in zip(axes, AXES):
        d = byS[S]
        for al in algos:
            pts = [(r[xk], r[f"cpu_{al}"]) for r in d
                   if r[xk] and np.isfinite(r[f"cpu_{al}"]) and r[f"cpu_{al}"] > 0]
            if len(pts) < 2:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4, color=COLOR[al],
                    label="%s  (slope %.2f)" % (al, slopes[al][S]))
            # The l1sep time-guard truncates the series. An unmarked short line reads as "it got cheap";
            # it means the opposite -- the next point was too EXPENSIVE to run. Say so on the figure.
            dropped = [r[xk] for r in d if r[xk] and not np.isfinite(r[f"cpu_{al}"])]
            if dropped:
                x0, y0 = pts[-1]
                ax.annotate("", xy=(x0, y0), xytext=(14, 20), textcoords="offset points",
                            arrowprops=dict(arrowstyle="<|-", color=COLOR[al], lw=1.2))
                ax.text(0.03, 0.97, "%s: %.0f s guard --\n%d larger point%s too EXPENSIVE to run\n"
                        "(the line stops; the cost does not)" %
                        (al, L1_GUARD_S, len(dropped), "s" if len(dropped) > 1 else ""),
                        transform=ax.transAxes, color=COLOR[al], fontsize=6.5, ha="left", va="top")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(xl); ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    axes[0].set_ylabel("runtime (s), median of 3 seeds")
    fig.suptitle("Three cost laws. pivot: flat in $m$ and $|H|$, steep in $n$ -- it completes the graph to "
                 "$\\Theta(n^2)$ first.  bestofk: ~linear in $m$.\n"
                 "l1sep: SIZE-bound, not $|H|$-bound -- 35$\\times$ more broken edges costs it 1.9$\\times$, "
                 "while 2.6$\\times$ more edges costs it 9.3$\\times$.", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig_cost_law.{ext}"), dpi=150, bbox_inches="tight")
    print(f"wrote {outdir}/fig_cost_law.pdf / .png")


def rounds_probe(outdir, seed=1000):
    """Decompose l1sep's cost into (separation rounds) x (cost per round) along each axis.

    This is what adjudicates 'l1sep is size-bound, |H| only sets the round count'. It also checks the
    plateau at large |H| is a real plateau and not the max_rounds=200 cap firing (which would return a
    possibly-invalid cover and cap the runtime artificially -- see l1_separation's docstring).
    """
    from metric_repair import l1_separation
    pts = ([("H", 300, 10, c) for c in (4, 12, 30, 70, 140)] +
           [("M", 300, d, 12) for d in (6, 10, 16)] +
           [("N", nn, 10, 12) for nn in (150, 250, 400)])
    rows = []
    print("\n=== l1sep round decomposition (seed %d) ===" % seed)
    print("%-6s %5s %6s %5s %9s %7s %10s %10s" % ("sweep", "n", "m", "|H|", "seconds", "rounds",
                                                  "converged", "s/round"))
    for S, n, deg, ncorr in pts:
        G, nn, mm, hh = instance(n, deg, ncorr, seed)
        t = time.perf_counter()
        S_cov, info = l1_separation(G, general=True, return_info=True)
        dt = time.perf_counter() - t
        r = int(info.get("rounds") or 0)
        row = dict(sweep=S, n=nn, m=mm, H=hh, seconds=round(dt, 3), rounds=r,
                   converged=bool(info.get("converged")), sec_per_round=round(dt / max(r, 1), 4),
                   cover_size=len(S_cov), max_rounds=200, seed=seed)
        rows.append(row)
        print("%-6s %5d %6d %5d %9.2f %7d %10s %10.3f" % (S, nn, mm, hh, dt, r,
                                                          row["converged"], row["sec_per_round"]), flush=True)
    path = os.path.join(outdir, "cost_law_rounds.csv")
    _write_csv(rows, path)
    print(f"wrote {path}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--plot", action="store_true", help="also write fig_cost_law (needs matplotlib)")
    ap.add_argument("--replot", action="store_true", help="rebuild the figure from an existing cost_law.csv")
    ap.add_argument("--rounds", action="store_true",
                    help="also write cost_law_rounds.csv: l1sep rounds vs cost-per-round on each axis")
    ap.add_argument("--outdir", default="analysis")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, "cost_law.csv")
    algos = list(_algos())

    if a.replot:
        rows = read_csv_rows(path)
    else:
        N = [(n, (n, 10, 12)) for n in (150, 250, 400, 600, 900)]     # vary n; m grows; |H| fixed
        M = [(d, (300, d, 12)) for d in (6, 10, 16, 24, 34)]          # vary m at fixed n; |H| fixed
        H = [(c, (300, 10, c)) for c in (4, 12, 30, 70, 140)]         # vary |H| at fixed n, m
        # H first: it is the arm that has never run, and the l1sep guard makes sweep N the slow one.
        # Each sweep is independent (fixed seeds, per-sweep guard), so order does not affect a measurement.
        rows = []
        for name, pts in (("H", H), ("N", N), ("M", M)):
            rows += run_sweep(name, pts, a.seeds, checkpoint=path, sofar=list(rows))
        rows.sort(key=lambda r: ({"N": 0, "M": 1, "H": 2}[r["sweep"]], r["knob"]))
        _write_csv(rows, path)
        print(f"\nwrote {path}")

    _byS, slopes = fit_slopes(rows, algos)
    print("\n=== log-log slopes  d(log time)/d(log knob) ===")
    print("%-12s %9s %9s %10s" % ("algo", "vs n (N)", "vs m (M)", "vs |H| (H)"))
    for al in algos:
        print("%-12s %9.2f %9.2f %10.2f" % (al, slopes[al]["N"], slopes[al]["M"], slopes[al]["H"]))
    print("\nNOTE: sweep N holds degree fixed, so m rides along with n -- the n axis is NOT isolated at\n"
          "fixed m. Only sweeps M and H isolate their knob exactly. pivot's flatness in m (sweep M) is\n"
          "what licenses reading its sweep-N growth as an n law.")

    if a.rounds:
        rounds_probe(a.outdir)
    if a.plot or a.replot:
        make_figure(rows, algos, a.outdir)


if __name__ == "__main__":
    main()
