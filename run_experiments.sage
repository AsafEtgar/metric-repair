# ============================================================================
# run_experiments.sage  --  headless, parametrized metric-repair experiment runner
#
# One task -> one tidy CSV in results/.  Designed for batch / cluster use:
# no plotting, no display, fully driven by CLI args, reproducible from --seed.
#
# Usage (local):
#   sage run_experiments.sage --n 100 --p 0.5 --reps 30 --algo all --seed 0
#
# On a cluster, launch one of these per array task with a distinct --seed, then
# merge everything with:   sage -python process_results.py
#
# To define an actual experiment, edit run_one() below (marked TODO).
# ============================================================================
import argparse, os, sys, time, subprocess
import pandas as pd

load("graph_models.sage")
load("metric_repair.sage")

ALGORITHMS = {
    "domr_alg":            domr_alg,
    "shortest_path_cover": shortest_path_cover,
    "left_edge_heuristic": left_edge_heuristic,
    "pivot_heuristic":     pivot_heuristic,
    "l1_min_heuristic":    l1_min_heuristic,
}

GENERATORS = {
    "geometric":   random_geometric_weighted_graph,
    "exponential": random_exponential_weighted_graph,
    "metric":      random_metric_graph,
    "uniform":     random_weighted_graph,
}


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def run_one(G, algo_name):
    """Run one algorithm on one graph and return a dict of measurements.

    TODO: this template records cover size, runtime and validity. Replace / extend with whatever
    your experiment actually measures (e.g. ratio to OPT, demand-satisfaction, hyperbolicity, ...)."""
    f = ALGORITHMS[algo_name]
    t0 = time.perf_counter()
    S = f(G)
    runtime = time.perf_counter() - t0
    return {"cover_size": len(S), "runtime_s": runtime, "valid": int(verifier(G, S))}


def main():
    ap = argparse.ArgumentParser(description="Metric-repair experiment runner")
    ap.add_argument("--n", type=int, required=True, help="number of vertices")
    ap.add_argument("--p", type=float, required=True, help="edge / weight parameter")
    ap.add_argument("--reps", type=int, default=30, help="graphs per parameter point")
    ap.add_argument("--algo", choices=list(ALGORITHMS) + ["all"], default="all")
    ap.add_argument("--generator", choices=list(GENERATORS), default="geometric")
    ap.add_argument("--seed", type=int, default=0, help="seeds Sage+NumPy+Python (see seed_all)")
    ap.add_argument("--require-connected", action="store_true", default=True)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    seed_all(args.seed)
    algos = list(ALGORITHMS) if args.algo == "all" else [args.algo]
    gen = GENERATORS[args.generator]
    commit, sver = git_commit(), version()

    rows = []
    for rep in range(args.reps):
        G = gen(args.n, args.p)
        tries = 0
        while args.require_connected and not G.is_connected() and tries < 100:
            G = gen(args.n, args.p); tries += 1
        for algo in algos:
            meas = run_one(G, algo)
            rows.append({
                "experiment": "template", "generator": args.generator, "algo": algo,
                "n": args.n, "p": args.p, "rep": rep, "seed": args.seed,
                "git": commit, "sage": sver, **meas,
            })

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"{args.generator}_n{args.n}_p{args.p}_seed{args.seed}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
