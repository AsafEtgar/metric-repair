"""
Ad-hoc scaling benchmark: every standard repair algorithm on 10 geometric G(n,p) graphs.

n=200, p=0.3. Guards: 25-min hard wall, per-(graph,algorithm) task isolation (an OOM/error kills only
that task), a memory watchdog that aborts a phase if free RAM < 1.5 GB, l1 (memory-heavy: it solves an
LP over ~C(n,3) triangle constraints on the K_n completion) quarantined to low parallelism, and
incremental CSV so partial results always survive. Enumeration-based exact ILP / broken-cycle rounding
are intentionally excluded (they blow up well below n=200). Run: `sage -python bench_n200.py`.
"""
import os, sys, csv, time, datetime, threading, subprocess
import multiprocessing as mp
from multiprocessing import TimeoutError as MPTimeout
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N = int(os.environ.get("BENCH_N", 200))
P = float(os.environ.get("BENCH_P", 0.3))
NGRAPHS = int(os.environ.get("BENCH_GRAPHS", 10))
DEADLINE_S = int(os.environ.get("BENCH_DEADLINE_S", 25 * 60))
MEM_FLOOR_GB = float(os.environ.get("BENCH_MEM_FLOOR_GB", 1.5))
OUT = os.path.join("results", "bench_n200_p03_%s.csv" %
                   datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# key -> (display name, variant, model, needs_completion). Phase 1 = low memory; phase 2 = l1 (heavy).
ALGO_TABLE = {
    "domr":     ("domr",        "on_G",     "general", False),
    "pivot_onG":("pivot",       "on_G",     "general", False),
    "pivot_com":("pivot",       "complete", "general", True),
    "left_onG": ("left_edge",   "on_G",     "iomr",    False),
    "left_com": ("left_edge",   "complete", "iomr",    True),
    "spci_onG": ("spc_iomr",    "on_G",     "iomr",    False),
    "spcg_onG": ("spc_general", "on_G",     "general", False),
    "spci_com": ("spc_iomr",    "complete", "iomr",    True),
    "spcg_com": ("spc_general", "complete", "general", True),
    "l1_onG":   ("l1",          "on_G",     "general", False),
}
PHASE1 = ["domr", "pivot_onG", "pivot_com", "left_onG", "left_com",
          "spci_onG", "spcg_onG", "spci_com", "spcg_com"]
PHASE2 = ["l1_onG"]
FIELDS = ["seed", "key", "algorithm", "variant", "model", "cover_size",
          "runtime_s", "valid", "comp_n", "comp_edges", "status"]


def _apply(key, CC, com):
    import metric_repair as mr
    if key == "domr":      return mr.domr_alg(CC)
    if key == "pivot_onG": return mr.pivot_heuristic(CC)
    if key == "pivot_com": return mr.pivot_heuristic(com)
    if key == "left_onG":  return mr.left_edge_heuristic(CC)
    if key == "left_com":  return mr.left_edge_heuristic(com)
    if key == "spcg_onG":  return mr.shortest_path_cover(CC)
    if key == "spcg_com":  return mr.shortest_path_cover(com)
    if key == "spci_onG":  return mr.shortest_path_cover(CC, general=False)
    if key == "spci_com":  return mr.shortest_path_cover(com, general=False)
    if key == "l1_onG":    return mr.l1_min_heuristic(CC)
    raise KeyError(key)


def run_one(arg):
    seed, key = arg
    name, variant, model, needs_com = ALGO_TABLE[key]
    base = dict(seed=seed, key=key, algorithm=name, variant=variant, model=model,
                cover_size=-1, runtime_s=-1, valid=-1, comp_n=-1, comp_edges=-1, status="ok")
    try:
        import networkx as nx
        import metric_repair as mr
        from graph_models import seed_all, random_geometric_weighted_graph
        from time import perf_counter
        seed_all(seed)
        G = random_geometric_weighted_graph(N, P)
        comp = max(nx.connected_components(G), key=len)
        CC = G.subgraph(comp).copy()
        com = mr.complete(CC) if needs_com else None            # completion NOT timed (matches harness)
        t0 = perf_counter()
        S = _apply(key, CC, com)
        dt = perf_counter() - t0
        tgt = com if variant == "complete" else CC
        valid = int(mr.iomr_verifier(tgt, S) if model == "iomr" else mr.verifier(tgt, S))
        base.update(cover_size=len(S), runtime_s=round(dt, 3), valid=valid,
                    comp_n=CC.number_of_nodes(), comp_edges=CC.number_of_edges(), status="ok")
    except MemoryError:
        base["status"] = "OOM"
    except Exception as e:
        base["status"] = "ERR:%s:%s" % (type(e).__name__, str(e)[:70])
    return base


def available_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        pass
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
        page = 4096
        free = inactive = spec = 0
        for ln in out.splitlines():
            if "page size of" in ln:
                page = int(ln.split("page size of")[1].split("bytes")[0])
            elif ln.startswith("Pages free:"):
                free = int(ln.split(":")[1].strip().rstrip("."))
            elif ln.startswith("Pages inactive:"):
                inactive = int(ln.split(":")[1].strip().rstrip("."))
            elif ln.startswith("Pages speculative:"):
                spec = int(ln.split(":")[1].strip().rstrip("."))
        return (free + inactive + spec) * page / 1e9
    except Exception:
        return 999.0                                            # can't tell -> never abort on memory


def run_phase(label, tasks, workers, deadline, writer, fout, results):
    if not tasks or time.time() >= deadline:
        print("[%s] skipped (no time/tasks)" % label, flush=True)
        return None
    abort = {"reason": None}
    pool = mp.Pool(workers)
    stop = threading.Event()

    def watchdog():
        while not stop.is_set():
            a = available_gb()
            if a < MEM_FLOOR_GB:
                abort["reason"] = "low memory (%.1f GB free)" % a
                try: pool.terminate()
                except Exception: pass
                return
            stop.wait(3)
    threading.Thread(target=watchdog, daemon=True).start()

    print("[%s] %d tasks on %d workers" % (label, len(tasks), workers), flush=True)
    it = pool.imap_unordered(run_one, tasks)
    done, total = 0, len(tasks)
    while done < total:
        rem = deadline - time.time()
        if rem <= 0:
            abort["reason"] = abort["reason"] or "25-min time cap"
            break
        if abort["reason"]:
            break
        try:
            r = it.next(timeout=min(rem, 5))
        except MPTimeout:
            continue                                            # re-check deadline / memory every 5 s
        except Exception as e:
            abort["reason"] = abort["reason"] or "pool error (%s)" % type(e).__name__
            break
        done += 1
        results.append(r)
        writer.writerow(r); fout.flush()
        print("[%s] %2d/%d  %-12s %-9s seed%-2d size=%-5s t=%-8s %s"
              % (label, done, total, r["algorithm"], r["variant"], r["seed"],
                 r["cover_size"], r["runtime_s"], r["status"]), flush=True)
    stop.set()
    try:
        pool.terminate() if abort["reason"] else pool.close()
        pool.join()
    except Exception:
        pass
    if abort["reason"]:
        print("[%s] ABORTED: %s" % (label, abort["reason"]), flush=True)
    return abort["reason"]


def main():
    cpu = mp.cpu_count()
    try:
        total_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    except Exception:
        total_gb = 8.0
    p1_workers = max(2, min(8, cpu - 1))
    p2_workers = max(1, min(2, int(available_gb() // 4)))       # l1 is memory-heavy; scale to FREE RAM
    print("machine: %d cores, %.0f GB RAM, free %.1f GB | phase1 %d workers, phase2 %d workers"
          % (cpu, total_gb, available_gb(), p1_workers, p2_workers), flush=True)
    print("benchmark: n=%d p=%.2f, %d graphs, deadline %d min -> %s\n"
          % (N, P, NGRAPHS, DEADLINE_S // 60, OUT), flush=True)

    os.makedirs("results", exist_ok=True)
    t0 = time.time()
    deadline = t0 + DEADLINE_S
    results = []
    with open(OUT, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=FIELDS); writer.writeheader()
        tasks1 = [(s, k) for k in PHASE1 for s in range(NGRAPHS)]
        tasks2 = [(s, k) for k in PHASE2 for s in range(NGRAPHS)]
        ab1 = run_phase("P1-cheap", tasks1, p1_workers, deadline, writer, fout, results)
        if ab1 and "memory" in ab1:
            print("\nskipping phase 2 (l1) after a memory abort in phase 1", flush=True)
        else:
            run_phase("P2-l1", tasks2, p2_workers, deadline, writer, fout, results)

    # summary
    print("\n==================== SUMMARY (elapsed %.1f min) ===================="
          % ((time.time() - t0) / 60), flush=True)
    print("%-12s %-9s %4s %10s %10s %6s %4s %s"
          % ("algorithm", "variant", "ok", "mean_cover", "mean_time_s", "valid", "oom", "err"), flush=True)
    agg = defaultdict(list)
    for r in results:
        agg[(r["algorithm"], r["variant"])].append(r)
    for k in sorted(agg):
        rows = agg[k]
        ok = [r for r in rows if r["status"] == "ok"]
        oom = sum(1 for r in rows if r["status"] == "OOM")
        err = sum(1 for r in rows if r["status"].startswith("ERR"))
        valid = sum(1 for r in ok if r["valid"] == 1)
        mc = (sum(r["cover_size"] for r in ok) / len(ok)) if ok else float("nan")
        mt = (sum(r["runtime_s"] for r in ok) / len(ok)) if ok else float("nan")
        print("%-12s %-9s %4d %10.1f %10.2f %6s %4d %d"
              % (k[0], k[1], len(ok), mc, mt, "%d/%d" % (valid, len(ok)), oom, err), flush=True)
    print("\nrows written: %d -> %s" % (len(results), OUT), flush=True)
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
