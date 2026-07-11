"""real_harness.py -- run the metric-repair suite on the REAL datasets (see REAL_EXPERIMENTS.md).

One task = (graph, mode[, seed]) -> one CSV row per algorithm (aggregated over connected components), plus
the union cover saved to `covers/` for the ground-truth-recovery pass (R2/R3). Modes:

  det   : the 11 DETERMINISTIC non-ILP algorithms, once (seed fixed).
  rand  : the 5 RANDOMIZED algorithms, once per seed (the array runs 30 seeds).
  gmr_ilp / iomr_ilp : one exact ILP, with the 17h cap (overrides the shared 45s ALGO_TIMEOUT).

Drops the 3 rsp methods (float weights / intractable w_max). Reuses harness.build_suite (now incl. the GMR
covering analogues), VERIFY, _aggregate, RUN_FIELDS. Keyed on GRAPH NAME, not a synthetic config.

Local test (this Mac lacks a system numpy):  sage -python experiments/run_real_task.py --array heur --task-index K
"""
import os
import sys
import csv
import time
import resource
import multiprocessing as mp

import networkx as nx

# pivot/MVD_Pivot recurses ~n deep on the completed graph; real graphs reach n=2700 (pbmc3k) while Python's
# default limit is 1000 -> RecursionError. Forked task children inherit this. 2700 frames is well within the
# C stack, so no segfault risk at these sizes. (The synthetic runs never hit it -- pivot only saw n<=800.)
sys.setrecursionlimit(100000)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # experiments/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
from graph_models import seed_all                                                 # noqa: E402
from datasets import load_edgelist                                                # noqa: E402
from metric_repair import domr_alg                                                # noqa: E402
from harness import build_suite, _aggregate, VERIFY, RUN_FIELDS, REGION_H_MAX, _RSS_MB   # noqa: E402

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DROP_RSP = {"gmr_lp_rsp", "iomr_lp_rsp", "iomr_thr_rsp"}
RAND_ALGOS = {"pivot", "iomr_rand", "iomr_bestofk", "gmr_rand", "gmr_bestofk"}
ILP_ALGOS = {"gmr_ilp", "iomr_ilp"}
ILP_TIMEOUT_S = int(float(os.environ.get("ILP_TIMEOUT_S", 17 * 3600)))   # 17h cap; env-overridable for the
                                                            # long high-mem ILP rerun (set e.g. 169200 = 47h)
N_SEEDS = 30                                                # randomized algos averaged over 30 seeds

# harness.TIMEOUT_S (1800s) is tuned for the SMALL synthetic graphs (n<=500) where every heuristic finishes
# fast. On the big real graphs it is catastrophic: on ripe_atlas (n=999, 442k edges, 95% broken) EVERY one of
# the 11 deterministic heuristics runs the full 1800s (measured), so a det task needs ~5.5h and dies at the
# 2.5h SLURM walltime with NO csv written -- the cause of the truncated heur run (only ~286/589 csvs landed).
# Cap each heuristic tighter, and add a per-task budget net so the csv is ALWAYS written within walltime.
REAL_HEUR_TIMEOUT_S = 600                                   # per-(algo, component) wall cap for non-ILP heuristics

# PER-ALGORITHM CAPS. At a flat 600s the three expensive algorithms all read `timeout`, which is a statement
# about the cap, not about them. The paper's claim is which algorithms work in practice, so each gets enough
# room that a timeout means "intractable", not "we were impatient". The measured 600s timeouts had three
# distinct causes, so the caps are set per cause:
#
#   l1sep_*        |H|-bound. Post-A1 it converges properly instead of stopping early on a blind oracle:
#                  522s @ |H|=210, 696s @ 1761, 1041s @ 3273 (and tol=1e-9 adds cuts). It timed out on the
#                  four graphs with |H| >= 3376. 3600s should capture ~|H| <= 10k.
#   *_bestofk      m-bound (k independent roundings of the covering LP): 208 of the 276 timeout rows. Dies
#                  above ~14k edges.
#   pivot          n-bound -- it COMPLETES the graph first, so cost follows n^2 edges regardless of |H|.
#                  Timed out on all 30 seeds of dimacs_ny_t (n=5000) whose |H| is only ~18.
#
# Worst case: det = 9*600 + 2*3600 = 12600s (3.5h); rand = 5*1800 = 9000s (2.5h). Both inside the 4h budget
# below, which is inside the 6h walltime that submit_real_dsq.sh now requests. The budget must stay under the
# walltime: a task the budget stops still writes its CSV, one SLURM kills does not.
REAL_ALGO_TIMEOUT = {
    "l1sep_gmr": 3600, "l1sep_iomr": 3600,
    "gmr_bestofk": 1800, "iomr_bestofk": 1800,
    "pivot": 1800, "gmr_rand": 1800, "iomr_rand": 1800,
}

REAL_TASK_BUDGET_S = 4 * 3600                               # once a task's summed algo wall exceeds this, the
                                                            # remaining algos are marked skipped_time (csv still lands)

BASE_GRAPHS = ["dimacs_ny_d", "dimacs_ny_t", "ripe_atlas", "nmr_1d3z_atom", "nmr_1d3z_residue",
               "pbmc3k_cosine_knn", "cassiopeia_barcode_knn"]
INVERSION_BASES = ["bct_coactivation", "flycns_male", "fish1_ten"]                # each -> +{_lin,_log,_raw}
CONVERSIONS = ["", "_lin", "_log", "_raw"]


def real_graphs():
    """The 19 processed graphs: 7 non-inverted + 3 inversion bases x 4 conversions."""
    gs = list(BASE_GRAPHS)
    for b in INVERSION_BASES:
        gs += [b + c for c in CONVERSIONS]
    return gs


RAW_VARIANTS = {b + "_raw" for b in INVERSION_BASES}         # inverted-semantics -> characterization only


def dist_sensible():
    """Graphs whose weight is a genuine distance (exclude the _raw inversion variants) -> get the ILP."""
    return [g for g in real_graphs() if g not in RAW_VARIANTS]


# ----------------------------------------------------------------------------
# Task enumeration (two independent arrays)
# ----------------------------------------------------------------------------

def heur_tasks():
    """Array H: per graph, one det bundle + 30 randomized seeds -> 19 * 31 = 589 tasks."""
    t = []
    for g in real_graphs():
        t.append((g, "det", 0))
        for s in range(N_SEEDS):
            t.append((g, "rand", s))
    return t


def ilp_tasks():
    """Array I: one exact ILP per (distance-sensible graph, variant) -> 16 * 2 = 32 tasks."""
    t = []
    for g in dist_sensible():
        t.append((g, "gmr_ilp", 0))
        t.append((g, "iomr_ilp", 0))
    return t


def all_tasks(array):
    return heur_tasks() if array == "heur" else ilp_tasks()


# ----------------------------------------------------------------------------
# Fork-isolated runner that also returns the cover (harness._child drops it)
# ----------------------------------------------------------------------------

def _norm(u, v):
    return (int(u), int(v)) if int(u) <= int(v) else (int(v), int(u))


def _child(fn, CC, verify_fn, conn):
    try:
        t_cpu, t_wall = time.process_time(), time.perf_counter()
        cover, info = fn(CC)
        cpu, wall = time.process_time() - t_cpu, time.perf_counter() - t_wall
        if cover is None:
            size, valid, cov = None, None, None                # pure lower bound (naive LP)
        else:
            size = len(cover)
            valid = int(verify_fn(CC, cover)) if verify_fn else None
            cov = [_norm(u, v) for (u, v) in cover]
        out = {"status": "ok", "size": size, "valid": valid, "cpu": cpu, "wall": wall,
               "peak_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RSS_MB, "cover": cov}
        for k in ("lp_bound", "exact_opt", "converged", "rounds", "cuts", "oracle", "guaranteed",
                  "full_separation", "min_pair_dist"):
            out[k] = info.get(k)
        conn.send(out)
    except MemoryError:
        conn.send({"status": "oom"})
    except Exception as e:                                      # noqa: BLE001
        conn.send({"status": "error:" + repr(e)[:120]})
    finally:
        conn.close()


def run_isolated(fn, CC, verify_fn, timeout_s):
    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(fn, CC, verify_fn, child_conn))
    t0 = time.perf_counter()
    p.start()
    child_conn.close()                         # parent is read-only -> poll() sees EOF the instant the child dies
    # DRAIN-AS-YOU-WAIT (do NOT p.join() first). _child sends the full cover, which on the big-H real graphs
    # (ripe: 421k edges) far exceeds the ~64KB OS pipe buffer. If we joined first, the child would block in
    # send() waiting for a reader while we block in join() waiting for it to exit -> deadlock, and the algo
    # burns its ENTIRE timeout (this is what killed the heur run: 11 algos x 1800s deadlock >> walltime -> no
    # csv). poll() lets recv() start reading, which unblocks the child.
    #
    # All three harnesses now use this pattern; harness.run_isolated ships a cover too (for light_frac).
    # The cap applies to time-to-FIRST-BYTE, not to the whole transfer.
    if not parent_conn.poll(timeout_s):        # no result within the cap -> genuinely still computing -> kill
        p.terminate(); p.join(5)
        if p.is_alive():                       # stuck in a C call / uninterruptible -> SIGKILL so the cap is hard
            p.kill(); p.join()
        return {"status": "timeout", "wall": time.perf_counter() - t0}
    try:
        out = parent_conn.recv()               # child is mid-send -> our read drains the pipe and unblocks it
    # EOFError = died before sending; OSError = died mid-frame (OOM-killed while pickling a large cover).
    except (EOFError, OSError):
        out = {"status": "killed"}
    p.join()
    return out


# ----------------------------------------------------------------------------
# One task
# ----------------------------------------------------------------------------

META_FIELDS = ("graph", "mode", "seed", "algo", "variant", "n", "m", "n_components", "giant",
               "H", "nonmetric_frac", "w_min", "w_max")
CSV_FIELDS = list(META_FIELDS) + list(RUN_FIELDS)


def _suite_for(mode, seed):
    suite = [e for e in build_suite(seed) if e[0] not in DROP_RSP]
    if mode == "det":
        return [e for e in suite if e[0] not in RAND_ALGOS and e[0] not in ILP_ALGOS]
    if mode == "rand":
        return [e for e in suite if e[0] in RAND_ALGOS]
    return [e for e in suite if e[0] == mode]                   # ilp mode: mode is the algo name


def _save_cover(covers_dir, graph, mode, seed, algo, cover):
    d = os.path.join(covers_dir, graph)
    os.makedirs(d, exist_ok=True)
    tag = f"{mode}_s{seed}" if mode == "rand" else mode
    with open(os.path.join(d, f"{algo}__{tag}.txt"), "w") as f:
        for (u, v) in sorted(cover):
            f.write(f"{u} {v}\n")


def run_real(graph, mode, seed, outdir, covers_dir):
    G0 = load_edgelist(os.path.join("data", "processed", f"{graph}.csv"))
    # Relabel to contiguous ints 0..n-1: several repair algos index nodes as int array positions and break
    # on string labels (nmr_atom "13:HG2#") or sparse ints. Covers are mapped back to the ORIGINAL labels
    # on save (below), so GT alignment (ground_truth.py) still matches the graph's node IDs.
    G = nx.convert_node_labels_to_integers(G0, label_attribute="orig")
    orig = {i: G.nodes[i]["orig"] for i in G.nodes()}
    seed_all(seed)                                             # seeds the global RNG (pivot) + build_suite(seed)
    comps = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    giant = max((c.number_of_nodes() for c in comps), default=0)
    nonmetric, total_H = [], 0
    for CC in comps:
        h = len(domr_alg(CC)); total_H += h
        if h > 0:
            nonmetric.append(CC)
    ws = [d["weight"] for _, _, d in G.edges(data=True)]
    m = G.number_of_edges()
    meta = dict(graph=graph, mode=mode, seed=seed, n=G.number_of_nodes(), m=m,
                n_components=len(comps), giant=giant, H=total_H,
                nonmetric_frac=(round(total_H / m, 6) if m else 0.0),
                w_min=(min(ws) if ws else 0), w_max=(max(ws) if ws else 0))

    rows = []
    elapsed = 0.0
    for (name, variant, vkey, n_max, region_gated, fn) in _suite_for(mode, seed):
        base = {**meta, "algo": name, "variant": variant, **{f: None for f in RUN_FIELDS}}
        # The ILP is EXEMPT from the task budget on purpose: it is a once-ever exact reference given its own
        # 17h array and 17:30 walltime, so a 3h budget would abort it before it can converge. Consequence
        # (AUDIT_REPORT.md B5, NOT fixed): the ILP path has NO task-level cap at all.
        if name not in ILP_ALGOS and elapsed >= REAL_TASK_BUDGET_S:   # budget spent -> skip rest (csv still lands)
            base["status"] = "skipped_time"
        elif n_max is not None and giant > n_max:
            base["status"] = "skipped_n"
        elif region_gated and total_H > REGION_H_MAX:          # region growing is O(V*E)/pair -> only for small |H|
            base["status"] = "skipped_H"
        elif not nonmetric:                                    # already metric -> empty cover (control case)
            base.update(status="ok", size=0, valid=1, cpu=0.0, wall=0.0, peak_mb=0.0)
        else:
            to = ILP_TIMEOUT_S if name in ILP_ALGOS else REAL_ALGO_TIMEOUT.get(name, REAL_HEUR_TIMEOUT_S)
            # `to` is charged PER NON-METRIC COMPONENT, not per algorithm. One algorithm on a graph with k
            # such components can therefore burn k*to of wall before the budget above is ever re-checked
            # (it is only tested BETWEEN algorithms). Safe today because every processed graph has exactly one
            # non-metric component. See AUDIT_REPORT.md B4 (NOT fixed).
            results = [run_isolated(fn, CC, VERIFY[vkey], to) for CC in nonmetric]
            base.update(_aggregate(results))
            # NOTE: this sums the CHILD-measured wall, which stops before verify_fn runs. verifier() does a full
            # dense APSP, so `elapsed` under-counts real time on big graphs (B13, NOT fixed).
            elapsed += base.get("wall") or 0.0
            covers = [{_norm(u, v) for (u, v) in r["cover"]} for r in results if r.get("cover") is not None]
            if covers:
                cu = set().union(*covers)                     # {(int,int)} -> map back to original node labels
                _save_cover(covers_dir, graph, mode, seed, name, {(orig[u], orig[v]) for (u, v) in cu})
        rows.append(base)

    os.makedirs(outdir, exist_ok=True)
    fname = f"{graph}__{mode}" + (f"_s{seed:02d}" if mode == "rand" else "") + ".csv"
    path = os.path.join(outdir, fname)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_FIELDS})
    return path


def run_one_real_task(array, task_index, outdir, covers_dir):
    graph, mode, seed = all_tasks(array)[task_index]
    return run_real(graph, mode, seed, outdir, covers_dir)
