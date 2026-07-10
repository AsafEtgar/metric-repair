"""Run ONE downstream-recovery task (one real graph, all its saved covers) -> one CSV.

    sage -python experiments/run_downstream_task.py --graph ripe_atlas --covers results_real_covers --outdir results_downstream
    sage -python experiments/run_downstream_task.py --task-index 0 --outdir results_downstream        # for dSQ
    sage -python experiments/run_downstream_task.py --count                                            # -> 3

Self-contained: imports only downstream_recovery (which imports nothing from the running campaign's harnesses).
One task per graph; there are only a few. Output CSV name mirrors the graph so a dSQ array is idempotent.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from downstream_recovery import DOWNSTREAM_GRAPHS, FIELDS, run_one_graph   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=None, help="one of the DOWNSTREAM_GRAPHS keys")
    ap.add_argument("--task-index", type=int, default=None, help="index into sorted(DOWNSTREAM_GRAPHS)")
    ap.add_argument("--covers", default="results_real_covers", help="root holding <graph>/<algo>__<tag>.txt")
    ap.add_argument("--outdir", default="results_downstream")
    ap.add_argument("--count", action="store_true", help="print the number of graphs and exit")
    a = ap.parse_args()

    graphs = sorted(DOWNSTREAM_GRAPHS)
    if a.count:
        print(len(graphs)); return
    if a.graph is None and a.task_index is None:
        ap.error("pass --graph or --task-index (or --count)")
    graph = a.graph if a.graph is not None else graphs[a.task_index]
    if graph not in DOWNSTREAM_GRAPHS:
        ap.error(f"{graph!r} is not a downstream graph; choices: {graphs}")

    os.makedirs(a.outdir, exist_ok=True)
    rows = run_one_graph(graph, covers_root=a.covers)
    path = os.path.join(a.outdir, f"{graph}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    ncov = rows[0]["n_covers_seen"] if rows else 0
    print(f"{graph}: {len(rows)} rows from {ncov} covers -> {path}")
    if not rows:
        print(f"  WARNING: no covers found under {a.covers}/{graph}/ -- nothing scored.")


if __name__ == "__main__":
    main()
