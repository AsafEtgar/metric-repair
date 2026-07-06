"""Write joblist.txt: one command per task, for Dead-Simple-Queue (dSQ) on the cluster.

    python experiments/make_joblist.py [--python python] [--outdir results] [--joblist joblist.txt]

Each line is a standalone shell command `python experiments/run_task.py --task-index K --outdir ...`.
Feed joblist.txt to dSQ (see experiments/RUN.md).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import all_tasks   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default="python", help="interpreter on the cluster (conda env python)")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--joblist", default="joblist.txt")
    a = ap.parse_args()
    n = len(all_tasks())
    with open(a.joblist, "w") as f:
        for i in range(n):
            f.write(f"{a.python} experiments/run_task.py --task-index {i} --outdir {a.outdir}\n")
    print(f"{n} tasks -> {a.joblist}  (outdir={a.outdir})")


if __name__ == "__main__":
    main()
