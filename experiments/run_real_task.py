"""Run ONE real-dataset task -> one CSV in --outdir (+ covers in --covers). See REAL_EXPERIMENTS.md.

Cluster:  python experiments/run_real_task.py --array {heur,ilp} --task-index K
Local:    sage -python experiments/run_real_task.py --array heur --task-index K
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_harness import run_one_real_task, all_tasks   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--array", choices=["heur", "ilp"], required=True)
    ap.add_argument("--task-index", type=int, default=None)
    ap.add_argument("--outdir", default="results_real")
    ap.add_argument("--covers", default="results_real_covers")
    ap.add_argument("--count", action="store_true", help="print the number of tasks in this array and exit")
    a = ap.parse_args()
    if a.count:
        print(len(all_tasks(a.array)))
        return
    if a.task_index is None:
        ap.error("--task-index is required (or pass --count)")
    path = run_one_real_task(a.array, a.task_index, a.outdir, a.covers)
    print(f"real task {a.array}/{a.task_index} -> {path}")


if __name__ == "__main__":
    main()
