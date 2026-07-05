"""
run_experiments.py  --  the metric-repair experiment harness (pure Python).

This single module is used in two ways:

  * imported by the experiments notebook for interactive/local sweeps, and
  * run as a CLI for headless / cluster jobs:

      python run_experiments.py --generator geometric --n 100 --p 0.3 --reps 30 --seed 0

Because the harness lives in an importable module (not in the notebook), ``multiprocessing`` works
under both fork (Linux) and spawn (macOS): a worker just re-imports run_experiments. The algorithms
are timed one at a time inside each instance, so per-algorithm ``runtime_s`` stays comparable even
while many instances run in parallel.

Design principle: measure per instance, store long/tidy, aggregate at analysis time. Every row is one
(instance, component, algorithm) observation -- nothing is pre-averaged.
"""

import argparse
import datetime
import multiprocessing as mp
import os
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import networkx as nx

from graph_models import (
    seed_all, random_geometric_weighted_graph, random_exponential_weighted_graph,
    random_uniform_weighted_graph, random_weighted_graph,
)
from metric_repair import (
    complete, domr_alg, pivot_heuristic, left_edge_heuristic, shortest_path_cover,
    l1_min_heuristic, l1_separation, covering_lp_cover, verifier, iomr_verifier,
    exact_metric_repair_ilp_separation,
)

GENERATORS = {
    "geometric":   random_geometric_weighted_graph,
    "exponential": random_exponential_weighted_graph,
    "uniform":     random_uniform_weighted_graph,
    "gnp_int":     random_weighted_graph,
}

# (name, variant, model, fn(CC, comCC)).  variant: on_G runs on the component, complete runs on its
# completion.  model picks the validity checker (general -> verifier, iomr -> iomr_verifier).
ALGORITHMS = [
    ("domr",        "on_G",     "general", lambda CC, comCC: domr_alg(CC)),
    ("pivot",       "on_G",     "general", lambda CC, comCC: pivot_heuristic(CC)),
    ("pivot",       "complete", "general", lambda CC, comCC: pivot_heuristic(comCC)),
    ("left_edge",   "on_G",     "iomr",    lambda CC, comCC: left_edge_heuristic(CC)),
    ("left_edge",   "complete", "iomr",    lambda CC, comCC: left_edge_heuristic(comCC)),
    ("spc_general", "on_G",     "general", lambda CC, comCC: shortest_path_cover(CC)),
    ("spc_general", "complete", "general", lambda CC, comCC: shortest_path_cover(comCC)),
    ("spc_iomr",    "on_G",     "iomr",    lambda CC, comCC: shortest_path_cover(CC, general=False)),
    ("spc_iomr",    "complete", "iomr",    lambda CC, comCC: shortest_path_cover(comCC, general=False)),
    ("l1",          "on_G",     "general", lambda CC, comCC: l1_min_heuristic(CC)),
    # Cutting-plane L1 (no enumeration/completion; runs on the component). general=True keeps repaired
    # weights strictly positive (min_weight=1); general=False is the increase-only (IOMR) variant.
    ("l1_sep",      "on_G",     "general", lambda CC, comCC: l1_separation(CC, general=True)),
    ("l1_sep",      "on_G",     "iomr",    lambda CC, comCC: l1_separation(CC, general=False)),
    # Covering-LP threshold rounding via separation (scales; f-approx). best_of_k=12 rounds several
    # optimal-face vertices -- the strongest IOMR approximation heuristic (closes much of the LP gap).
    ("cover_thr",     "on_G",   "iomr",    lambda CC, comCC: covering_lp_cover(CC, solve="separation", rounding="deterministic", iomr=True)[0]),
    ("cover_thr_bok", "on_G",   "iomr",    lambda CC, comCC: covering_lp_cover(CC, solve="separation", rounding="deterministic", iomr=True, best_of_k=12, seed=0)[0]),
    # Exact ground-truth baselines (cutting-plane ILP; scales past enumeration -- see OVERVIEW.md sec 6).
    # These are the SLOW rows: comment them out for pure-heuristic timing runs. Each returns (cover,info),
    # so we take [0]. exact_iomr forces a hit on a light edge of every broken cycle (increase-only).
    ("exact_general", "on_G",   "general", lambda CC, comCC: exact_metric_repair_ilp_separation(CC)[0]),
    ("exact_iomr",    "on_G",   "iomr",    lambda CC, comCC: exact_metric_repair_ilp_separation(CC, iomr=True)[0]),
]


# ----------------------------------------------------------------------------
# Core: run every algorithm on one freshly generated graph -> tidy rows
# ----------------------------------------------------------------------------

def run_instance(generator, n, p, seed, trial, compute_valid=True):
    """Run every algorithm on one graph and return a list of tidy row dicts (one per
    (component, algorithm)). Each algorithm is timed in isolation, so runtimes stay comparable."""
    seed_all(seed)                                       # reproducible; the seed is in every row
    G = GENERATORS[generator](int(n), float(p))
    rows = []
    for ci, comp in enumerate(nx.connected_components(G)):
        CC = G.subgraph(comp).copy()
        comCC = complete(CC)
        n_broken = len(domr_alg(CC))                     # broken edges in this component
        for name, variant, model, fn in ALGORITHMS:
            t0 = perf_counter()
            S = fn(CC, comCC)
            dt = perf_counter() - t0
            valid = -1
            if compute_valid:
                tgt = CC if variant == "on_G" else comCC
                valid = int(iomr_verifier(tgt, S) if model == "iomr" else verifier(tgt, S))
            rows.append(dict(
                generator=generator, n=int(n), p=float(p), seed=int(seed), trial=int(trial),
                component=ci, algorithm=name, variant=variant, model=model,
                cover_size=len(S), runtime_s=dt, valid=valid,
                comp_n=CC.number_of_nodes(), comp_edges=CC.number_of_edges(), n_broken=n_broken))
    return rows


def _worker(arg):                                        # module-level so it pickles under spawn & fork
    task, compute_valid = arg
    return run_instance(*task, compute_valid=compute_valid)


# ----------------------------------------------------------------------------
# Sweep / parallelism / persistence
# ----------------------------------------------------------------------------

def build_tasks(generators, ns, ps=None, p_of_n=None, trials=1, base_seed=0):
    """Cartesian sweep generators x ns x (ps | p_of_n(n)) x trials -> list of task tuples.
    Pass ps (a list of p values) OR p_of_n (a function n -> p)."""
    tasks = []
    for gen in generators:
        for n in ns:
            pvals = [float(p_of_n(n))] if p_of_n is not None else [float(x) for x in ps]
            for p in pvals:
                for trial in range(int(trials)):
                    tasks.append((gen, int(n), float(p), int(base_seed) + len(tasks), int(trial)))
    return tasks


def run_sweep(tasks, parallel=True, n_jobs=None, compute_valid=True):
    """Run all tasks; return one tidy DataFrame. Parallel across instances; serial and parallel give
    identical results (the only difference is the runtime_s column)."""
    work = [(t, compute_valid) for t in tasks]
    if parallel:
        with mp.Pool(n_jobs or mp.cpu_count()) as pool:
            res = pool.map(_worker, work)
    else:
        res = [_worker(w) for w in work]
    return pd.DataFrame([r for sub in res for r in sub])


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "nogit"


def save_results(df, name, outdir="results"):
    """Write the tidy rows to results/<name>_<timestamp>.csv.gz, stamped with the git commit."""
    os.makedirs(outdir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = df.copy()
    out["code_version"] = git_commit()
    out["run_timestamp"] = stamp
    path = os.path.join(outdir, f"{name}_{stamp}.csv.gz")
    out.to_csv(path, index=False)
    print(f"saved {len(out)} rows -> {path}  (code {git_commit()})")
    return path


# ----------------------------------------------------------------------------
# Analysis helpers (aggregate at analysis time, never during the run)
# ----------------------------------------------------------------------------

def per_instance(df):
    """Collapse components: one row per (instance, algorithm) with the per-graph total cover/runtime."""
    keys = ["generator", "n", "p", "seed", "trial", "algorithm", "variant", "model"]
    return df.groupby(keys, as_index=False).agg(
        cover_size=("cover_size", "sum"),
        runtime_s=("runtime_s", "sum"),
        valid=("valid", "min"))


def summarize(df, by=("generator", "algorithm", "variant", "n", "p")):
    """Mean/std cover size, mean runtime (ms), and validity rate over trials."""
    inst = per_instance(df)
    out = inst.groupby(list(by)).agg(
        cover_mean=("cover_size", "mean"),
        cover_std=("cover_size", "std"),
        runtime_ms=("runtime_s", lambda s: 1000.0 * s.mean()),
        valid_rate=("valid", "mean"),
        reps=("cover_size", "size"))
    return out.round(3)


# ----------------------------------------------------------------------------
# CLI (headless / cluster): one invocation -> one CSV in results/
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Metric-repair experiment runner (one task -> one CSV)")
    ap.add_argument("--generator", choices=list(GENERATORS), default="geometric")
    ap.add_argument("--n", type=int, required=True, help="number of vertices")
    ap.add_argument("--p", type=float, required=True, help="edge / weight parameter")
    ap.add_argument("--reps", type=int, default=30, help="graphs (trials) per parameter point")
    ap.add_argument("--seed", type=int, default=0, help="base seed (seeds NumPy + Python; see seed_all)")
    ap.add_argument("--no-valid", action="store_true", help="skip the verifier check (faster)")
    ap.add_argument("--serial", action="store_true", help="disable multiprocessing")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    tasks = build_tasks([args.generator], ns=[args.n], ps=[args.p], trials=args.reps, base_seed=args.seed)
    df = run_sweep(tasks, parallel=not args.serial, compute_valid=not args.no_valid)
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"{args.generator}_n{args.n}_p{args.p}_seed{args.seed}.csv")
    df["code_version"] = git_commit()
    df.to_csv(path, index=False)
    print(f"wrote {len(df)} rows -> {path}  (python {sys.version.split()[0]}, "
          f"networkx {nx.__version__}, code {git_commit()})")


if __name__ == "__main__":
    main()
