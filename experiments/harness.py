"""harness.py -- POC experiment harness for the metric-repair algorithm comparison.

Defines: the experiment grid (Exp 1 / 2a / 2b), a reproducible seed per task, the algorithm suite, and a
fork-isolated per-algorithm runner that enforces a wall-clock cap and records size / validity / CPU+wall
time / peak RSS / convergence / cuts / LP bound / etc. One task = one graph; run_one_task writes one CSV
whose rows are the per-algorithm results aggregated over the graph's connected components.

See EXPERIMENTS.md for the exact experimental design. Cluster usage: experiments/RUN.md.
"""
import os
# single-threaded BLAS: fork-safe (we fork per algorithm), deterministic CPU timing, matches the cluster.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import csv
import time
import zlib
import math
import resource
import multiprocessing as mp

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> metric_repair
from graph_models import (                                              # noqa: E402
    seed_all, random_geometric_weighted_graph, random_decoupled_geometric_weighted_graph,
)
from metric_repair import (                                            # noqa: E402
    domr_alg, exact_metric_repair_ilp_separation, metric_repair_lp_separation, covering_lp_cover,
    l1_separation, shortest_path_cover, pivot_heuristic, left_edge_heuristic,
    verifier, iomr_verifier,
)

# ----------------------------------------------------------------------------
# Experiment grid
# ----------------------------------------------------------------------------
N_SAMPLES = 40
TIMEOUT_S = 30 * 60          # per (algorithm, instance) wall-clock cap
# Per-algorithm overrides. Both exact ILP-separation solvers are NP-hard and blow up at large n/OPT.
# Cap each at 45s: they stay EXACT wherever they converge quickly (small n, sparse) and report
# converged=False (no exact_opt) otherwise -- the analysis then falls back to the LP LOWER BOUND for that
# MR variant (rgg_analyze/analyze _task_refs), so a timeout never leaves a config without a reference. This
# bounds the runtime tail (iomr_ilp was ~93% of the RGG poc compute at the old 180s cap). NB: shared by the
# geometric harness too, where GMR still recovers its exact optimum from the integral gmr_lp_rsp on timeout.
ALGO_TIMEOUT = {"iomr_ilp": 45, "gmr_ilp": 45}
TASK_BUDGET_S = 120 * 60     # per-task (all algorithms on one graph) budget; later algos -> skipped_time
                             # so a slow graph can't blow the SLURM per-task limit and lose the whole CSV.
REGION_H_MAX = 200           # region growing only when total broken-edge count |H| is small (it's O(V*E)/pair)
BEST_OF_K = 12
WEIGHT_P = 0.5               # fixed weight parameter for the decoupled model (Geometric(1-0.5))


def _ints(a):
    return [int(round(x)) for x in a]


def _points():
    """The list of experiment 'points'; each point gets N_SAMPLES sampled graphs."""
    pts = []
    # Exp 1: fixed p, sweep n in [100,500] (20 points), coupled geometric.
    for p in (0.3, 0.5):
        for n in _ints(np.linspace(100, 500, 20)):
            pts.append(dict(exp="exp1", model="geometric", n=n, p=float(p), alpha=None))
    # Exp 2a: n=500, coupled geometric, alpha in [1/2, 2/3] (onset of non-metricity at 3/5; all connected).
    for a in np.linspace(0.5, 2.0 / 3.0, 20):
        n = 500
        pts.append(dict(exp="exp2a", model="geometric", n=n, p=float(n ** (-a)), alpha=float(a)))
    # Exp 2b: n=500, DECOUPLED geometric (fixed weights), alpha in [1/2, 4/5] (density sweep; crosses
    # connectivity ~0.706 -> disconnected at the sparse end).
    for a in np.linspace(0.5, 0.8, 20):
        n = 500
        pts.append(dict(exp="exp2b", model="decoupled_geometric", n=n, p=float(n ** (-a)), alpha=float(a)))
    return pts


def _points_small():
    """Smaller POC grid: Exp1 n=100..300 (step 10), Exp2a/2b at n=300 -- memory-safe. 30 seeds (SAMPLE_COUNT)."""
    pts = []
    for p in (0.3, 0.5):
        for n in _ints(np.linspace(100, 300, 21)):
            pts.append(dict(exp="exp1", model="geometric", n=n, p=float(p), alpha=None))
    for a in np.linspace(0.5, 2.0 / 3.0, 20):
        n = 300
        pts.append(dict(exp="exp2a", model="geometric", n=n, p=float(n ** (-a)), alpha=float(a)))
    for a in np.linspace(0.5, 0.8, 20):
        n = 300
        pts.append(dict(exp="exp2b", model="decoupled_geometric", n=n, p=float(n ** (-a)), alpha=float(a)))
    return pts


# Large-scale grid (see EXPERIMENT_REGISTRY.md §3): no exact solver at this scale -> DROP_LARGE strips the ILP
# AND the rsp methods (O(w_max*n^2)); the suite becomes heuristics + LP-bound references + domr. exp1 sweeps a
# large n-ladder at three densities; exp2 (decoupled) sits at n=2000 with p = 2*n^-alpha, alpha 4/5 -> 1/3
# (the x2 keeps it connected while densifying to ~320k edges, to surface algorithm separation on dense graphs).
LARGE_NS = tuple(range(1000, 3001, 200))               # 1000..3000 step 200 -> 11 points
DROP_LARGE = {"gmr_ilp", "iomr_ilp", "gmr_lp_rsp", "iomr_lp_rsp", "iomr_thr_rsp"}


def _points_large():
    pts = []
    # exp1: coupled geometric, planted breaks. p is an EDGE PROBABILITY, so these graphs are DENSE -> the
    # ladder is capped at n<=1500 (n=2000,p=0.5 would be ~1M edges, where even domr times out; probe finding).
    # p in {0.3, 0.5}; ~edges = p*n^2/2 -> at most 0.5*1500^2/2 = 562k (heavy but tractable; domr+LP land).
    for p in (0.3, 0.5):
        for n in _ints(np.linspace(1000, 1500, 10)):   # finer 10-point mesh over the tractable dense range
            pts.append(dict(exp="exp1", model="geometric", n=n, p=float(p), alpha=None))
    # exp2: decoupled geometric density ONSET at FIXED n=2000, p = 2*n^-alpha (alpha 4/5 -> 1/3, 16 pts).
    # alpha-controlled density: sparse-alpha points are light (~9k edges); only the dense end (alpha~1/3 ->
    # ~317k edges) is heavy, where heuristics fall back to the LP bound (domr ~9s still lands).
    n2 = 2000
    for a in np.linspace(4.0 / 5.0, 1.0 / 3.0, 16):
        pts.append(dict(exp="exp2b", model="decoupled_geometric", n=n2,
                        p=float(2.0 * n2 ** (-a)), alpha=float(a)))
    return pts


POINTS = _points()
GRIDS = {"full": POINTS, "small": _points_small(), "large": _points_large()}
SAMPLE_COUNT = {"full": N_SAMPLES, "small": 30, "large": 20}


def all_tasks(grid="full"):
    """Flat list of (point, sample_index). One task -> one graph -> one CSV file."""
    return [(pt, s) for pt in GRIDS[grid] for s in range(SAMPLE_COUNT[grid])]


def task_seed(pt, s):
    """Deterministic, distinct seed per task (reproducible; recorded in the output)."""
    key = f"{pt['exp']}|{pt['model']}|{pt['n']}|{pt.get('p')}|{pt.get('alpha')}|{s}"
    return zlib.crc32(key.encode()) & 0x7FFFFFFF


def generate(pt, seed):
    seed_all(seed)
    if pt["model"] == "geometric":
        return random_geometric_weighted_graph(pt["n"], pt["p"])
    if pt["model"] == "decoupled_geometric":
        return random_decoupled_geometric_weighted_graph(pt["n"], pt["p"], WEIGHT_P)
    raise ValueError(pt["model"])


# ----------------------------------------------------------------------------
# Algorithm suite.  Each fn(CC) -> (cover_set_or_None, info_dict).  cover=None marks a pure LOWER BOUND
# (the IOMR LP), whose value goes in info['lp_bound'] rather than a cover size.
# ----------------------------------------------------------------------------

def _lp(CC, iomr, oracle):
    val, y, D, ncuts, nrounds = metric_repair_lp_separation(CC, iomr=iomr, oracle=oracle, return_rounds=True)
    info = {"lp_bound": float(val), "oracle": oracle, "cuts": int(ncuts), "rounds": int(nrounds)}
    # Only the EXACT (rsp) GMR LP is feasible for all cycles AND integral, so only its support is a valid
    # cover. The naive GMR LP (canonical cycles only) and the IOMR LP (fractional) are LOWER BOUNDS only.
    if (not iomr) and oracle == "rsp":
        inv = {i: e for e, i in D.items()}
        return {inv[i] for i in np.nonzero(y > 0.5)[0]}, info
    return None, info


def _exact(CC, iomr):
    cover, info = exact_metric_repair_ilp_separation(CC, iomr=iomr)
    return cover, {"converged": bool(info["converged"]), "rounds": int(info["rounds"]),
                   "cuts": int(info["constraints"]),
                   "exact_opt": (len(cover) if info["converged"] else None)}


def _cov(CC, rounding, iomr, oracle, seed=None, best_of_k=1):
    cover, info = covering_lp_cover(CC, solve="separation", rounding=rounding, iomr=iomr,
                                    oracle=oracle, seed=seed, best_of_k=best_of_k)
    return cover, {"lp_bound": info.get("lp_value"), "oracle": oracle, "guaranteed": info.get("guaranteed"),
                   "rounds": info.get("rounds"),        # separation-oracle rounds (None for solve="enum")
                   "full_separation": info.get("full_separation"), "min_pair_dist": info.get("min_pair_dist")}


def build_suite(seed):
    """(name, variant, verify_key, n_max, region_gated, fn(CC)).  n_max gates on the GIANT component size;
    region_gated gates on total |H| <= REGION_H_MAX.  verify_key: gmr/iomr/none(=lower bound)."""
    return [
        ("domr",           "DOMR", "gmr",  None, False, lambda CC: (domr_alg(CC), {})),
        ("gmr_lp_rsp",     "GMR",  "gmr",  None, False, lambda CC: _lp(CC, False, "rsp")),
        ("gmr_lp_naive",   "GMR",  "gmr",  None, False, lambda CC: _lp(CC, False, "naive")),
        ("gmr_ilp",        "GMR",  "gmr",  None, False, lambda CC: _exact(CC, False)),
        # GMR analogues of the covering-LP rounding family (iomr=False). No gmr_thr_rsp (the rsp GMR LP is
        # already integral -> thresholding is trivial/redundant with gmr_lp_rsp) and no gmr_regiongrow
        # (region growing is an IOMR light-edge construction). Useful on FLOAT data where rsp is dropped, so
        # gmr_lp_rsp's exact integral cover is unavailable and these give fast valid GMR covers.
        ("gmr_thr_naive",  "GMR",  "gmr",  None, False, lambda CC: _cov(CC, "deterministic", False, "naive")),
        ("gmr_bestofk",    "GMR",  "gmr",  None, False,
         lambda CC: _cov(CC, "deterministic", False, "naive", seed=seed, best_of_k=BEST_OF_K)),
        ("gmr_rand",       "GMR",  "gmr",  None, False, lambda CC: _cov(CC, "randomized", False, "naive", seed=seed)),
        ("iomr_ilp",       "IOMR", "iomr", None, False, lambda CC: _exact(CC, True)),
        ("iomr_lp_naive",  "IOMR", "none", None, False, lambda CC: _lp(CC, True, "naive")),
        ("iomr_lp_rsp",    "IOMR", "none", 150,  False, lambda CC: _lp(CC, True, "rsp")),
        ("iomr_thr_naive", "IOMR", "iomr", None, False, lambda CC: _cov(CC, "deterministic", True, "naive")),
        ("iomr_thr_rsp",   "IOMR", "iomr", 150,  False, lambda CC: _cov(CC, "deterministic", True, "rsp")),
        ("iomr_bestofk",   "IOMR", "iomr", None, False,
         lambda CC: _cov(CC, "deterministic", True, "naive", seed=seed, best_of_k=BEST_OF_K)),
        ("iomr_rand",      "IOMR", "iomr", None, False, lambda CC: _cov(CC, "randomized", True, "naive", seed=seed)),
        ("iomr_regiongrow", "IOMR", "iomr", None, True, lambda CC: _cov(CC, "region_growing", True, "naive")),
        ("l1sep_gmr",      "GMR",  "gmr",  None, False, lambda CC: (l1_separation(CC, general=True), {})),
        ("l1sep_iomr",     "IOMR", "iomr", None, False, lambda CC: (l1_separation(CC, general=False), {})),
        ("spc_gmr",        "GMR",  "gmr",  None, False, lambda CC: (shortest_path_cover(CC, general=True), {})),
        ("spc_iomr",       "IOMR", "iomr", None, False, lambda CC: (shortest_path_cover(CC, general=False), {})),
        ("pivot",          "GMR",  "gmr",  None, False, lambda CC: (pivot_heuristic(CC), {})),
        ("left_edge",      "IOMR", "iomr", None, False, lambda CC: (left_edge_heuristic(CC), {})),
    ]


VERIFY = {"gmr": verifier, "iomr": iomr_verifier, "none": None}
_RSS_MB = (1.0 / 1024.0 / 1024.0) if sys.platform == "darwin" else (1.0 / 1024.0)  # ru_maxrss: bytes(mac)/KB(linux)

# fields produced per (algorithm, component) run
RUN_FIELDS = ("status", "size", "valid", "cpu", "wall", "peak_mb", "lp_bound", "exact_opt",
              "converged", "rounds", "cuts", "oracle", "guaranteed", "full_separation", "min_pair_dist",
              # light_frac = |cover \ heavy(DOMR)| / |cover|: fraction of the cover that is LIGHT (w<=detour).
              # 0 => the cover is all heavy edges (GMR coincides with DOMR, e.g. inflate); >0 => GMR departs
              # from DOMR by raising light edges (shortcut/deflate/jitter). Per-instance, computed from the
              # cover union vs the heavy-edge union (not in _aggregate).
              "light_frac")


def _norm(u, v):
    return (int(u), int(v)) if int(u) <= int(v) else (int(v), int(u))


def _child(fn, CC, verify_fn, conn):
    try:
        t_cpu, t_wall = time.process_time(), time.perf_counter()
        cover, info = fn(CC)
        cpu, wall = time.process_time() - t_cpu, time.perf_counter() - t_wall
        if cover is None:
            size, valid, cov = None, None, None               # pure lower bound
        else:
            size = len(cover)
            valid = int(verify_fn(CC, cover)) if verify_fn else None
            cov = [_norm(u, v) for (u, v) in cover]            # for the per-instance light_frac (vs DOMR)
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RSS_MB
        out = {"status": "ok", "size": size, "valid": valid, "cpu": cpu, "wall": wall, "peak_mb": peak_mb,
               "cover": cov}
        for k in ("lp_bound", "exact_opt", "converged", "rounds", "cuts", "oracle", "guaranteed",
                  "full_separation", "min_pair_dist"):
            out[k] = info.get(k)
        conn.send(out)
    except MemoryError:
        conn.send({"status": "oom"})
    except Exception as e:                                     # noqa: BLE001
        conn.send({"status": "error:" + repr(e)[:120]})
    finally:
        conn.close()


def run_isolated(fn, CC, verify_fn, timeout_s):
    """Run fn(CC) in a forked child with a wall-clock cap; return a dict with RUN_FIELDS (subset).

    DRAIN-AS-YOU-WAIT: _child now ships the cover, which on large/dense instances (n=3000) exceeds the ~64KB
    OS pipe buffer. poll()+recv() (not p.join() first) lets the parent start reading so the child's send()
    can't block -> avoids the deadlock that truncated the real-data run (see real_harness.run_isolated)."""
    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(fn, CC, verify_fn, child_conn))
    t0 = time.perf_counter()
    p.start()
    child_conn.close()                                        # read-only parent -> poll() sees EOF if child dies
    if not parent_conn.poll(timeout_s):                       # nothing within the cap -> still computing -> kill
        p.terminate(); p.join(5)
        if p.is_alive():
            p.kill(); p.join()
        return {"status": "timeout", "wall": time.perf_counter() - t0}
    try:
        out = parent_conn.recv()                              # child mid-send -> our read drains + unblocks it
    except EOFError:                                          # child died before sending (segfault / OOM-kill)
        out = {"status": "killed"}
    p.join()
    return out


# ----------------------------------------------------------------------------
# Aggregate a per-algorithm result across the graph's (non-metric) components
# ----------------------------------------------------------------------------

_WORST = {"ok": 0, "skipped_n": 0, "skipped_H": 0, "timeout": 3, "oom": 4, "killed": 4}


def _worse(a, b):
    return b if _WORST.get(b.split(":")[0], 5) >= _WORST.get(a.split(":")[0], 5) else a


def _aggregate(results):
    """Combine per-component runs of ONE algorithm. size/cpu/wall/rounds/cuts/lp_bound/exact_opt sum;
    peak_mb max; valid/converged/guaranteed/full_separation AND; min_pair_dist min; status = worst."""
    agg = {k: None for k in RUN_FIELDS}
    agg["status"] = "ok"
    sums = {"size": None, "cpu": 0.0, "wall": 0.0, "rounds": None, "cuts": None,
            "lp_bound": None, "exact_opt": None}
    peak = 0.0
    for r in results:
        agg["status"] = _worse(agg["status"], r.get("status", "ok"))
        for k in ("cpu", "wall"):
            sums[k] += r.get(k) or 0.0
        for k in ("size", "rounds", "cuts", "lp_bound", "exact_opt"):
            v = r.get(k)
            if v is not None:
                sums[k] = (sums[k] or 0) + v
        peak = max(peak, r.get("peak_mb") or 0.0)
        for k in ("valid", "converged", "guaranteed", "full_separation"):
            v = r.get(k)
            if v is not None:
                agg[k] = v if agg[k] is None else (int(agg[k]) and int(v))
        for k in ("oracle",):
            if r.get(k) is not None:
                agg[k] = r[k]
        if r.get("min_pair_dist") is not None:
            agg["min_pair_dist"] = min(agg["min_pair_dist"], r["min_pair_dist"]) \
                if agg["min_pair_dist"] is not None else r["min_pair_dist"]
    agg.update(sums)
    agg["peak_mb"] = round(peak, 1)
    for k in ("cpu", "wall"):
        agg[k] = round(sums[k], 4)
    return agg


META_FIELDS = ("task", "exp", "model", "n", "p", "alpha", "sample", "seed",
               "V", "E", "w_min", "w_max", "n_components", "giant", "H")
CSV_FIELDS = list(META_FIELDS) + ["algo", "variant"] + list(RUN_FIELDS)


def run_one_task(task_index, outdir, grid="full"):
    pt, s = all_tasks(grid)[task_index]
    seed = task_seed(pt, s)
    G = generate(pt, seed)
    comps = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    ws = [d["weight"] for _, _, d in G.edges(data=True)]
    giant = max((c.number_of_nodes() for c in comps), default=0)

    nonmetric, heavy_union = [], set()             # a component needs repair iff domr(CC) != empty
    for CC in comps:                               # heavy_union = the DOMR set (edges w>detour) across comps
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

    suite = build_suite(seed)
    if grid == "large":                             # scale suite: no exact solver, no rsp (see DROP_LARGE)
        suite = [e for e in suite if e[0] not in DROP_LARGE]
    rows = []
    elapsed = 0.0
    for (name, variant, vkey, n_max, region_gated, fn) in suite:
        row = {**meta, "algo": name, "variant": variant, **{k: None for k in RUN_FIELDS}}
        if elapsed >= TASK_BUDGET_S:               # ran out of per-task budget -> skip the rest
            row["status"] = "skipped_time"
        elif n_max is not None and giant > n_max:
            row["status"] = "skipped_n"
        elif region_gated and total_H > REGION_H_MAX:
            row["status"] = "skipped_H"
        elif not nonmetric:                        # whole graph already metric -> empty cover
            row.update(status="ok", size=0, valid=1, cpu=0.0, wall=0.0, peak_mb=0.0)
        else:
            to = ALGO_TIMEOUT.get(name, TIMEOUT_S)
            results = [run_isolated(fn, CC, VERIFY[vkey], to) for CC in nonmetric]
            row.update(_aggregate(results))
            elapsed += row.get("wall") or 0.0
            covers = [set(r["cover"]) for r in results if r.get("cover") is not None]
            if covers:                                 # light_frac = share of the cover that is NOT a heavy edge
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
