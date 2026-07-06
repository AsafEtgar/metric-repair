"""Write joblist.txt: one command per task, for Dead-Simple-Queue (dSQ) on the cluster.

    python experiments/make_joblist.py [--python python] [--outdir results] [--joblist joblist.txt]
        [--setup 'module load miniconda && conda activate metricrepair']

Each line is a standalone shell command `python experiments/run_task.py --task-index K --outdir ...`,
optionally prefixed by --setup so every task self-activates its conda env on the compute node (dSQ runs
each line in a bare shell that does NOT inherit your login-shell env). Feed joblist.txt to dSQ (RUN.md).
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
    ap.add_argument("--setup", default="", help="shell prefix prepended (with &&) to every task command, "
                    "e.g. 'module load miniconda && conda activate metricrepair'")
    a = ap.parse_args()
    n = len(all_tasks())
    prefix = f"{a.setup} && " if a.setup else ""
    with open(a.joblist, "w") as f:
        for i in range(n):
            f.write(f"{prefix}{a.python} experiments/run_task.py --task-index {i} --outdir {a.outdir}\n")
    print(f"{n} tasks -> {a.joblist}  (outdir={a.outdir})")


if __name__ == "__main__":
    main()
