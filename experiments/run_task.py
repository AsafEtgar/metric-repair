"""Run ONE experiment task (one graph, the whole algorithm suite) -> one CSV in --outdir.

Cluster:  python experiments/run_task.py --task-index K
Local test (this Mac lacks a system numpy):  sage -python experiments/run_task.py --task-index K
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_one_task, all_tasks   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-index", type=int, default=None)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--grid", default="full", choices=["full", "small", "large"], help="which task grid to run")
    ap.add_argument("--count", action="store_true", help="print the total number of tasks and exit")
    a = ap.parse_args()
    if a.count:
        print(len(all_tasks(a.grid)))
        return
    if a.task_index is None:
        ap.error("--task-index is required (or pass --count)")
    path = run_one_task(a.task_index, a.outdir, a.grid)
    print(f"task {a.task_index} ({a.grid}) -> {path}")


if __name__ == "__main__":
    main()
