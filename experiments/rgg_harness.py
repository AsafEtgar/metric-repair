"""rgg_harness.py -- RGG ground-truth experiments (sibling of harness.py; see RGG_EXPERIMENTS.md).

Part 1 (regime characterization, OFAT) and Part 2 (kNN recovery on jitter breaks) on Random Geometric
Graphs with FLOAT weights (exactly metric -> clean planted `corrupted` set; the 3 rsp methods drop, 15/18
run). Reuses harness.py's algorithm suite, fork isolation, per-component aggregation, and CSV machinery;
adds RGG knobs, edit-precision/recall (cover vs `corrupted`), and the T/C/F kNN metrics.

One task = one graph = one CSV (rows aggregated over connected components). Part 2 rows are LONG: one row
per (instance, algorithm, k).
"""
import os
import sys
import csv
import math
import time
import zlib
import resource
import multiprocessing as mp

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # experiments/ -> harness
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> models
from graph_models import (                                                       # noqa: E402
    seed_all, random_geometric_metric_graph, break_metric_graph, jitter_points,
)
from metric_repair import domr_alg                                              # noqa: E402
from harness import (                                                            # noqa: E402
    build_suite, _aggregate, VERIFY, RUN_FIELDS, _RSS_MB,
    N_SAMPLES, TIMEOUT_S, ALGO_TIMEOUT, TASK_BUDGET_S, REGION_H_MAX,
)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DROP_RSP = {"gmr_lp_rsp", "iomr_lp_rsp", "iomr_thr_rsp"}   # need integer weights; float RGG -> drop
K_LIST = (5, 10, 20, 50)                                    # kNN sizes (Part 2)
TRIPLET_MULT = 20                                           # #triples = TRIPLET_MULT * V (Part 2)


def _radius(deg, n):
    """radius giving expected degree `deg` for n uniform points in the unit square (2D)."""
    return math.sqrt(deg / (math.pi * n))


# baseline n=300 for the OFAT (non-size) sweeps; the size sweeps span n=100..500 (step 20) on their own.
def _base_p1():
    return dict(part="p1", mode="radius", n=300, deg=12, k=None, dim=2,
                break_type="reweight", direction="inflate", frac_q=0.10, magnitude=3.0,
                n_jitter=4, jitter_r=1.5, subset_s=0.5)


def _base_p2():
    return dict(part="p2", mode="radius", n=300, deg=12, k=None, dim=2,
                break_type="jitter", direction=None, frac_q=None, magnitude=None,
                n_jitter=8, jitter_r=2.0, subset_s=0.5)


SIZE_NS = tuple(range(100, 501, 20))                       # size sweep: n=100..500 step 20 (21 points)


def _points_part1():
    pts = []

    def add(sweep, base=_base_p1, **over):
        c = base(); c.update(over); c["sweep"] = sweep; pts.append(c)

    for n in SIZE_NS:                                         # S1 size sweep (inflate; density fixed deg=12)
        add("S1", n=n)
    for deg in (4, 8, 12, 20, 30, 40):                        # S2 density (radius)
        add("S2", deg=deg)
    for k in (4, 8, 12, 20, 30):                              # S2' density (knn)
        add("S2k", mode="knn", k=k)
    for m in (1.2, 1.5, 2, 3, 5, 10):                         # S3 inflate magnitude
        add("S3", direction="inflate", magnitude=float(m))
    for m in (1.2, 1.5, 2, 3, 5, 10):                         # S3' deflate magnitude
        add("S3d", direction="deflate", magnitude=float(m))
    for q in (0.01, 0.02, 0.05, 0.10, 0.20, 0.30):            # S4 fraction (inflate & deflate)
        add("S4i", direction="inflate", frac_q=q)
    for q in (0.01, 0.02, 0.05, 0.10, 0.20, 0.30):
        add("S4d", direction="deflate", frac_q=q)
    for nj in (1, 2, 4, 8, 16):                               # S5a jitter count
        add("S5a", break_type="jitter", n_jitter=nj, jitter_r=1.5, subset_s=0.5)
    for jr in (0.5, 1.0, 1.5, 2.5, 4.0):                      # S5b jitter magnitude (units of radius)
        add("S5b", break_type="jitter", n_jitter=4, jitter_r=jr, subset_s=0.5)
    for sfrac in (0.1, 0.25, 0.5, 0.75, 0.9):                 # S5c jitter subset (metric<->non-metric dial)
        add("S5c", break_type="jitter", n_jitter=4, jitter_r=1.5, subset_s=sfrac)
    for m in (1.5, 3, 6):                                     # S6 magnitude x frac_q interaction (inflate)
        for q in (0.05, 0.1, 0.2):
            add("S6", direction="inflate", magnitude=float(m), frac_q=q)
    return pts


def _points_part2():
    pts = []

    def add(sweep, **over):
        c = _base_p2(); c.update(over); c["sweep"] = sweep; pts.append(c)

    for n in SIZE_NS:                                         # size sweep under jitter (kNN recovery vs n)
        add("P2size", n=n)
    for sfrac in (0.1, 0.25, 0.5, 0.75, 0.9):                 # subset_s sweep
        add("P2s", subset_s=sfrac)
    for jr in (0.5, 1.0, 2.0, 3.0, 4.0):                      # jitter magnitude sweep
        add("P2j", jitter_r=jr)
    for nj in (2, 4, 8, 16, 32):                              # jitter count sweep
        add("P2n", n_jitter=nj)
    return pts


POINTS_RGG = _points_part1() + _points_part2()


def _points_poc():
    """Small proof-of-concept grid: size sweep n=100..250 (step 10), both break flavors, n<=250 so it
    fits comfortably in memory. Each n gets an inflate reweight (part1-style: ratios + edit metrics) and a
    jitter (part2-style: ratios + edit metrics + kNN recovery). Averaged over SAMPLES['poc'] seeds."""
    pts = []
    for n in range(100, 251, 10):
        a = _base_p1(); a.update(part="p1", sweep="POCsize_inflate", n=n); pts.append(a)
        b = _base_p2(); b.update(part="p2", sweep="POCsize_jitter", n=n); pts.append(b)
    return pts


LARGE_NS = tuple(range(1000, 3001, 200))                   # 1000..3000 step 200 -> 11 points


def _points_large():
    """Large-scale grid (see EXPERIMENT_REGISTRY.md §3): the n-ladder swept at a fixed baseline (P1 inflate +
    P2 jitter), plus density at n=2000 in both radius and knn modes. Reuses the S1/P2size/S2/S2k sweep ids so
    rgg_analyze/rgg_plots work unchanged. No ILP at this scale (dropped in build_suite_rgg via drop_ilp)."""
    pts = []
    for n in LARGE_NS:                                      # size ladder: P1 inflate + P2 jitter (kNN)
        a = _base_p1(); a.update(sweep="S1", n=n); pts.append(a)
        b = _base_p2(); b.update(sweep="P2size", n=n); pts.append(b)
    for deg in (4, 8, 12, 20, 30, 40):                     # density (radius) at n=2000
        c = _base_p1(); c.update(sweep="S2", n=2000, deg=deg); pts.append(c)
    for k in (8, 12, 20, 30):                              # density (knn) at n=2000
        c = _base_p1(); c.update(sweep="S2k", n=2000, mode="knn", k=k); pts.append(c)
    return pts


GRIDS = {"full": POINTS_RGG, "poc": _points_poc(), "large": _points_large()}
SAMPLES = {"full": N_SAMPLES, "poc": 30, "large": 20}


def all_tasks(grid="full"):
    return [(cfg, s) for cfg in GRIDS[grid] for s in range(SAMPLES[grid])]


def task_seed(cfg, s):
    keys = ("part", "sweep", "mode", "n", "deg", "k", "dim", "break_type", "direction",
            "magnitude", "frac_q", "n_jitter", "jitter_r", "subset_s")
    key = "|".join(str(cfg.get(x)) for x in keys) + f"|{s}"
    return zlib.crc32(key.encode()) & 0x7FFFFFFF


def build_suite_rgg(seed, drop_ilp=False):
    drop = DROP_RSP | ({"gmr_ilp", "iomr_ilp"} if drop_ilp else set())
    return [e for e in build_suite(seed) if e[0] not in drop]


# ----------------------------------------------------------------------------
# RGG instance generation (true graph T + broken graph H + ground truth)
# ----------------------------------------------------------------------------

def generate_rgg(cfg, seed):
    """Return (T, H, corrupted, jittered, radius, jitter_abs). T = true metric RGG; H = broken copy."""
    seed_all(seed)
    n, dim = cfg["n"], cfg["dim"]
    if cfg["mode"] == "radius":
        radius = _radius(cfg["deg"], n)
        T = random_geometric_metric_graph(n, mode="radius", radius=radius, dim=dim, weight_scale=None)
        unit = radius
    else:
        T = random_geometric_metric_graph(n, mode="knn", k=cfg["k"], dim=dim, weight_scale=None)
        radius = None
        ws = [d["weight"] for _, _, d in T.edges(data=True)]
        unit = float(np.median(ws)) if ws else 1.0            # local edge scale for jitter units
    if cfg["break_type"] == "reweight":
        H, corrupted = break_metric_graph(T, frac_q=cfg["frac_q"], direction=cfg["direction"],
                                          magnitude=cfg["magnitude"])
        return T, H, corrupted, None, radius, None
    jitter_abs = cfg["jitter_r"] * unit
    H, corrupted, jittered = jitter_points(T, n_jitter=cfg["n_jitter"], jitter=jitter_abs,
                                           subset_s=cfg["subset_s"])
    return T, H, corrupted, jittered, radius, jitter_abs


# ----------------------------------------------------------------------------
# Fork-isolated runner that also returns the cover (harness._child drops it)
# ----------------------------------------------------------------------------

def _norm(u, v):
    return (int(u), int(v)) if int(u) <= int(v) else (int(v), int(u))


def _child_rgg(fn, CC, verify_fn, conn):
    try:
        t_cpu, t_wall = time.process_time(), time.perf_counter()
        cover, info = fn(CC)
        cpu, wall = time.process_time() - t_cpu, time.perf_counter() - t_wall
        if cover is None:
            size, valid, cov = None, None, None               # pure lower bound (naive LP)
        else:
            size = len(cover)
            valid = int(verify_fn(CC, cover)) if verify_fn else None
            cov = [_norm(u, v) for (u, v) in cover]
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RSS_MB
        out = {"status": "ok", "size": size, "valid": valid, "cpu": cpu, "wall": wall,
               "peak_mb": peak_mb, "cover": cov}
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


def run_isolated_rgg(fn, CC, verify_fn, timeout_s):
    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child_rgg, args=(fn, CC, verify_fn, child_conn))
    t0 = time.perf_counter()
    p.start()
    child_conn.close()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate(); p.join()
        return {"status": "timeout", "wall": time.perf_counter() - t0}
    if parent_conn.poll():
        return parent_conn.recv()
    return {"status": "killed" if (p.exitcode and p.exitcode != 0) else "error:noresult"}


# ----------------------------------------------------------------------------
# kNN machinery (Part 2): all-pairs shortest paths -> per-node kNN sets -> comparison
# ----------------------------------------------------------------------------

def _apsp(edges, n):
    """Dense all-pairs shortest-path distance matrix from an index edge list [(iu, iv, w), ...]."""
    if not edges:
        D = np.full((n, n), np.inf); np.fill_diagonal(D, 0.0); return D
    r, c, w = [], [], []
    for iu, iv, wt in edges:
        r += [iu, iv]; c += [iv, iu]; w += [wt, wt]
    A = csr_matrix((w, (r, c)), shape=(n, n))
    return dijkstra(A, directed=False)


def _knn_all(D, K_list):
    """K -> {v: set of its K nearest (finite-distance, excluding self)}, one sort per node for all K."""
    Kmax = max(K_list)
    out = {K: {} for K in K_list}
    for i in range(D.shape[0]):
        row = D[i]
        nb = []
        for j in np.argsort(row, kind="stable"):
            if j == i:
                continue
            if not np.isfinite(row[j]):
                break
            nb.append(int(j))
            if len(nb) >= Kmax:
                break
        for K in K_list:
            out[K][i] = set(nb[:K])
    return out


def _knn_cmp(knnT, knnX):
    """(mean Jaccard, mean recall) of knnX vs the reference knnT, over nodes with a nonempty true set."""
    js, rs = [], []
    for v, t in knnT.items():
        if not t:
            continue
        x = knnX.get(v, set())
        inter = len(t & x); uni = len(t | x)
        js.append(inter / uni if uni else 1.0)
        rs.append(inter / len(t))
    return (float(np.mean(js)) if js else None, float(np.mean(rs)) if rs else None)


def _triples(n, m, rng):
    a, b, c = rng.integers(0, n, m), rng.integers(0, n, m), rng.integers(0, n, m)
    ok = (a != b) & (a != c) & (b != c)
    return a[ok], b[ok], c[ok]


def _triplet_acc(DT, DX, tri):
    a, b, c = tri
    tb, tc, xb, xc = DT[a, b], DT[a, c], DX[a, b], DX[a, c]
    fin = np.isfinite(tb) & np.isfinite(tc) & np.isfinite(xb) & np.isfinite(xc) & (tb != tc)
    if not fin.any():
        return None
    return float(((tb[fin] < tc[fin]) == (xb[fin] < xc[fin])).mean())


# ----------------------------------------------------------------------------
# One task
# ----------------------------------------------------------------------------

RGG_META_FIELDS = ("task", "part", "sweep", "model", "mode", "n", "dim", "radius", "k", "deg",
                   "break_type", "direction", "magnitude", "frac_q", "n_jitter", "jitter", "subset_s",
                   "sample", "seed", "V", "E", "w_min", "w_max", "n_components", "giant", "H", "n_corrupted")
RGG_RUN_FIELDS = list(RUN_FIELDS) + ["edit_precision", "edit_recall"]
KNN_FIELDS = ["knn_k", "jaccard_TC", "jaccard_TF", "recall_TF", "lift", "triplet_acc_C", "triplet_acc_F"]
RGG_CSV_FIELDS = list(RGG_META_FIELDS) + ["algo", "variant"] + RGG_RUN_FIELDS + KNN_FIELDS


def _run(cfg, s, task_index, outdir, drop_ilp=False):
    seed = task_seed(cfg, s)
    T, H, corrupted, jittered, radius, jitter_abs = generate_rgg(cfg, seed)
    corr = {_norm(u, v) for (u, v) in corrupted}

    comps = [H.subgraph(c).copy() for c in nx.connected_components(H)]
    giant = max((c.number_of_nodes() for c in comps), default=0)
    nonmetric, total_H = [], 0
    for CC in comps:
        h = len(domr_alg(CC)); total_H += h
        if h > 0:
            nonmetric.append(CC)
    ws = [d["weight"] for _, _, d in H.edges(data=True)]

    meta = dict(task=task_index, part=cfg["part"], sweep=cfg["sweep"], model="rgg", mode=cfg["mode"],
                n=cfg["n"], dim=cfg["dim"], radius=(round(radius, 6) if radius is not None else None),
                k=cfg.get("k"), deg=cfg.get("deg"), break_type=cfg["break_type"],
                direction=cfg.get("direction"), magnitude=cfg.get("magnitude"), frac_q=cfg.get("frac_q"),
                n_jitter=cfg.get("n_jitter"), jitter=(round(jitter_abs, 6) if jitter_abs else None),
                subset_s=cfg.get("subset_s"), sample=s, seed=seed,
                V=H.number_of_nodes(), E=H.number_of_edges(),
                w_min=(min(ws) if ws else 0), w_max=(max(ws) if ws else 0),
                n_components=len(comps), giant=giant, H=total_H, n_corrupted=len(corr))

    # Part 2 kNN prep: index the shared node set; distance matrices for T (true) and C (corrupted).
    part2 = cfg["part"] == "p2"
    knnT = knnC = jacTC = tacc_C = tri = DT = None
    nodes = index = edgesH = None
    if part2:
        nodes = sorted(H.nodes()); index = {u: i for i, u in enumerate(nodes)}
        edgesH = [(index[u], index[v], float(d["weight"])) for u, v, d in H.edges(data=True)]
        edgesT = [(index[u], index[v], float(d["weight"])) for u, v, d in T.edges(data=True)]
        DT, DC = _apsp(edgesT, len(nodes)), _apsp(edgesH, len(nodes))
        knnT = _knn_all(DT, K_LIST)
        knnC = _knn_all(DC, K_LIST)
        jacTC = {K: _knn_cmp(knnT[K], knnC[K])[0] for K in K_LIST}
        tri = _triples(len(nodes), TRIPLET_MULT * len(nodes), np.random.default_rng(seed))
        tacc_C = _triplet_acc(DT, DC, tri)

    rows, elapsed = [], 0.0
    for (name, variant, vkey, n_max, region_gated, fn) in build_suite_rgg(seed, drop_ilp):
        base = {**meta, "algo": name, "variant": variant,
                **{f: None for f in RGG_RUN_FIELDS}, **{f: None for f in KNN_FIELDS}}
        cover_union = None
        if elapsed >= TASK_BUDGET_S:
            base["status"] = "skipped_time"
        elif n_max is not None and giant > n_max:
            base["status"] = "skipped_n"
        elif region_gated and total_H > REGION_H_MAX:
            base["status"] = "skipped_H"
        elif not nonmetric:                                    # already metric -> empty cover
            base.update(status="ok", size=0, valid=1, cpu=0.0, wall=0.0, peak_mb=0.0)
        else:
            to = ALGO_TIMEOUT.get(name, TIMEOUT_S)
            results = [run_isolated_rgg(fn, CC, VERIFY[vkey], to) for CC in nonmetric]
            base.update(_aggregate(results))
            elapsed += base.get("wall") or 0.0
            covers = [{_norm(u, v) for (u, v) in r["cover"]} for r in results if r.get("cover") is not None]
            if covers:
                cover_union = set().union(*covers)
                inter = len(cover_union & corr)
                base["edit_precision"] = (inter / len(cover_union)) if cover_union else None
                base["edit_recall"] = (inter / len(corr)) if corr else None

        # Part 2: emit one LONG row per k with T/C/F kNN metrics (only when a repaired graph F exists).
        if part2 and cover_union is not None and base["status"] == "ok":
            cov_idx = {tuple(sorted((index[u], index[v]))) for (u, v) in cover_union
                       if u in index and v in index}
            DF = _apsp([e for e in edgesH if tuple(sorted((e[0], e[1]))) not in cov_idx], len(nodes))
            tacc_F = _triplet_acc(DT, DF, tri)
            knnF = _knn_all(DF, K_LIST)
            for K in K_LIST:
                jacTF, recTF = _knn_cmp(knnT[K], knnF[K])
                row = dict(base)
                row.update(knn_k=K, jaccard_TC=jacTC[K], jaccard_TF=jacTF, recall_TF=recTF,
                           lift=(None if (jacTF is None or jacTC[K] is None) else jacTF - jacTC[K]),
                           triplet_acc_C=tacc_C, triplet_acc_F=tacc_F)
                rows.append(row)
        else:
            rows.append(base)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"task_{task_index:06d}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RGG_CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in RGG_CSV_FIELDS})
    return path


def run_one_rgg_task(task_index, outdir, grid="full"):
    cfg, s = all_tasks(grid)[task_index]
    return _run(cfg, s, task_index, outdir, drop_ilp=(grid == "large"))
